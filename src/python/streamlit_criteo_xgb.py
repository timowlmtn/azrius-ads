#!/usr/bin/env python3
import streamlit as st
import duckdb
import numpy as np
import pandas as pd
from pyiceberg.catalog import load_catalog
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt

# Catalog / table config
CATALOG_NAME = "local"  # or "hive"/your sql catalog
TABLE_IDENTIFIER = "criteo.ad_challenge_v1"
VIEW_NAME = "criteo_ad"

NUM_COLS = [f"I{i}" for i in range(1, 14)]
CAT_COLS = [f"C{i}" for i in range(1, 27)]


# ---------------------------
# Data access helpers
# ---------------------------


@st.cache_resource(show_spinner=True)
def create_duckdb_view():
    """
    Load Iceberg table via PyIceberg and register as a DuckDB view.

    NOTE: This currently loads the whole Iceberg table into memory as Arrow.
    Keep sample sizes small (e.g. 1–5% or <= 1M rows) while exploring.
    """
    catalog = load_catalog(CATALOG_NAME)
    table = catalog.load_table(TABLE_IDENTIFIER)

    arrow_table = table.scan().to_arrow()
    con = duckdb.connect()
    con.register(VIEW_NAME, arrow_table)
    return con


def get_train_test_counts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    query = f"""
        SELECT
          CASE WHEN Label IS NULL THEN 'test' ELSE 'train' END AS split,
          COUNT(*) AS n_rows
        FROM {VIEW_NAME}
        GROUP BY 1
        ORDER BY split
    """
    return con.execute(query).df()


def load_train_sample(
    con: duckdb.DuckDBPyConnection,
    sample_frac: float,
    max_rows: int,
) -> pd.DataFrame:
    """
    Load a random sample of training rows (Label IS NOT NULL).
    sample_frac is a percent, e.g. 1.0 = 1% of table.
    """
    query = f"""
        SELECT *
        FROM {VIEW_NAME}
        WHERE Label IS NOT NULL
        USING SAMPLE {sample_frac} PERCENT
        LIMIT {max_rows}
    """
    return con.execute(query).df()


def load_test_sample(
    con: duckdb.DuckDBPyConnection,
    max_rows: int,
) -> pd.DataFrame:
    """
    Load a small sample of test rows (Label IS NULL).
    """
    query = f"""
        SELECT *
        FROM {VIEW_NAME}
        WHERE Label IS NULL
        LIMIT {max_rows}
    """
    return con.execute(query).df()


# ---------------------------
# Feature preprocessing
# ---------------------------


def hash_categorical_series(s: pd.Series) -> pd.Series:
    """
    Naive, intuitive categorical handling:
    - Treat each distinct string as its own category
    - Map categories to dense integers via factorize.

    This is not production-grade, but it's easy to reason about.
    """
    s = s.fillna("__NA__").astype(str)
    codes, _ = pd.factorize(s)
    return pd.Series(codes, index=s.index, dtype="int32")


def preprocess_features(df: pd.DataFrame):
    """
    For training rows: return (X, y).
    """
    y = df["Label"].astype("int8").to_numpy()

    num_df = df[NUM_COLS].copy()
    num_df = num_df.fillna(0).astype("float32")

    cat_df = pd.DataFrame(index=df.index)
    for col in CAT_COLS:
        cat_df[col] = hash_categorical_series(df[col])

    X = pd.concat([num_df, cat_df], axis=1)
    return X, y


def preprocess_features_test(df: pd.DataFrame):
    """
    For test rows: return (X, Id_series).
    """
    ids = df["Id"]

    num_df = df[NUM_COLS].copy()
    num_df = num_df.fillna(0).astype("float32")

    cat_df = pd.DataFrame(index=df.index)
    for col in CAT_COLS:
        cat_df[col] = hash_categorical_series(df[col])

    X = pd.concat([num_df, cat_df], axis=1)
    return X, ids


# ---------------------------
# Modeling
# ---------------------------


def train_xgboost_with_history(
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train XGBoost and capture training / validation logloss history.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=params.get("n_jobs", 8),
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        random_state=random_state,
    )

    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False,
    )

    # Manual logloss as a sanity check
    val_pred = model.predict_proba(X_val)[:, 1]
    val_logloss = log_loss(y_val, val_pred)

    evals_result = model.evals_result()
    # evals_result structure:
    # {
    #   'validation_0': {'logloss': [ ... per iteration ... ]},
    #   'validation_1': {'logloss': [ ... per iteration ... ]}
    # }

    return model, evals_result, val_logloss


# ---------------------------
# Streamlit UI
# ---------------------------


def main():
    st.title("Criteo XGBoost Exploration (Iceberg + DuckDB + Streamlit)")

    st.markdown(
        """
        This app lets you:
        - Sample training data from the **criteo.ad_challenge_v1** Iceberg table  
        - Train a simple **XGBoost** model  
        - Visualize **logloss over boosting rounds** and **feature importance**  

        ⚠️ For now, this loads the full Iceberg table into memory.  
        Keep samples small (e.g., 1–5% or <= 500k rows) while exploring.
        """
    )

    # Sidebar controls
    st.sidebar.header("Sampling")
    sample_frac = st.sidebar.slider(
        "Sample fraction (%) of training data",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
    )
    max_rows = st.sidebar.number_input(
        "Max training rows",
        min_value=10_000,
        max_value=2_000_000,
        value=200_000,
        step=10_000,
    )

    st.sidebar.header("XGBoost Parameters")
    n_estimators = st.sidebar.slider("n_estimators (trees)", 50, 500, 200, 50)
    max_depth = st.sidebar.slider("max_depth", 2, 12, 6, 1)
    learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.5, 0.1, 0.01)
    subsample = st.sidebar.slider("subsample", 0.5, 1.0, 0.8, 0.05)
    colsample_bytree = st.sidebar.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.05)

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "n_jobs": 8,
    }

    st.sidebar.header("Actions")
    train_button = st.sidebar.button("Load sample & train")

    # Connect once
    with st.spinner("Connecting to Iceberg catalog and creating DuckDB view..."):
        con = create_duckdb_view()

    # Show basic counts
    st.subheader("Dataset overview")
    counts_df = get_train_test_counts(con)
    st.dataframe(counts_df, use_container_width=True)

    if train_button:
        # Load sample
        with st.spinner(
            f"Loading training sample: {sample_frac}% (max {max_rows} rows)..."
        ):
            df_train = load_train_sample(
                con, sample_frac=sample_frac, max_rows=max_rows
            )

        st.write(f"Loaded **{len(df_train):,}** training rows.")
        if len(df_train) == 0:
            st.error(
                "No training rows loaded. Try increasing sample fraction or max_rows."
            )
            return

        # Preprocess
        with st.spinner("Preprocessing features..."):
            X, y = preprocess_features(df_train)
        st.write(f"Feature matrix shape: `{X.shape}`, labels shape: `{y.shape}`")

        # Basic label balance
        unique, counts = np.unique(y, return_counts=True)
        label_dist = pd.DataFrame({"Label": unique, "Count": counts})
        st.write("Label distribution in sample:")
        st.dataframe(label_dist, use_container_width=True)

        # Train
        with st.spinner("Training XGBoost model..."):
            model, evals_result, val_logloss = train_xgboost_with_history(X, y, params)

        st.success(f"Training complete. Validation logloss: **{val_logloss:.6f}**")

        # Logloss curves
        train_logloss = evals_result["validation_0"]["logloss"]
        val_logloss_history = evals_result["validation_1"]["logloss"]
        rounds = list(range(1, len(train_logloss) + 1))
        logloss_df = pd.DataFrame(
            {
                "round": rounds,
                "train_logloss": train_logloss,
                "val_logloss": val_logloss_history,
            }
        )

        st.subheader("Logloss over boosting rounds")
        st.line_chart(logloss_df.set_index("round")[["train_logloss", "val_logloss"]])

        # Feature importance
        st.subheader("Feature importance (gain)")
        importance = model.get_booster().get_score(importance_type="gain")

        # XGBoost uses feature names like "f0", "f1", ... in order of columns
        feature_names = list(X.columns)
        imp_rows: list[dict] = []
        for fname, score in importance.items():
            # fname is like 'f0', 'f1', ...
            idx = int(fname[1:])
            if 0 <= idx < len(feature_names):
                col_name = feature_names[idx]
            else:
                col_name = fname
            imp_rows.append({"feature": col_name, "gain": score})

        if imp_rows:
            imp_df = (
                pd.DataFrame(imp_rows).sort_values("gain", ascending=False).head(30)
            )
            st.dataframe(imp_df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(imp_df["feature"], imp_df["gain"])
            ax.invert_yaxis()
            ax.set_xlabel("Gain")
            ax.set_ylabel("Feature")
            ax.set_title("Top 30 features by XGBoost gain")
            st.pyplot(fig)
        else:
            st.info("No feature importance info available (empty model.get_score()).")

        # Optional: small prediction sample
        st.subheader("Prediction sample on held-out validation set")
        # For intuition: show a few predicted probabilities
        # We'll just sample rows from df_train and recompute preds
        sample_rows = df_train.sample(n=min(10, len(df_train)), random_state=42)
        X_sample, _ = preprocess_features(sample_rows)
        probs = model.predict_proba(X_sample)[:, 1]
        sample_out = sample_rows[["Id", "Label"]].copy()
        sample_out["Predicted_Prob"] = probs
        st.dataframe(sample_out, use_container_width=True)


if __name__ == "__main__":
    main()
