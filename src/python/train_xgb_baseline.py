#!/usr/bin/env python3
"""
Naive XGBoost baseline for Criteo Display Ad Challenge.

Assumptions:
- You have an Iceberg table "criteo.ad_challenge_v1" in a "local" (sql) catalog.
- Table columns include: Id, Label (nullable), I1..I13, C1..C26
- Label is NULL for test rows, 0/1 for train rows.

This script:
- samples training rows,
- does very simple feature preprocessing,
- trains an XGBoost model,
- prints validation logloss,
- optionally writes a small submission CSV for test rows.
"""

import argparse
from typing import Tuple, Optional, List

import duckdb
import numpy as np
import pandas as pd
from pyiceberg.catalog import load_catalog
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import xgboost as xgb


CATALOG_NAME = "local"
TABLE_IDENTIFIER = "criteo.ad_challenge_v1"
VIEW_NAME = "criteo_ad"

NUM_COLS = [f"I{i}" for i in range(1, 14)]
CAT_COLS = [f"C{i}" for i in range(1, 27)]


# ---------------------------
# Data loading / sampling
# ---------------------------


def create_duckdb_view() -> duckdb.DuckDBPyConnection:
    """Load the Iceberg table via PyIceberg and register it as a DuckDB view."""
    catalog = load_catalog(CATALOG_NAME)
    table = catalog.load_table(TABLE_IDENTIFIER)

    arrow_table = table.scan().to_arrow()
    con = duckdb.connect()
    con.register(VIEW_NAME, arrow_table)
    return con


def load_train_sample(
    con: duckdb.DuckDBPyConnection,
    sample_frac: float,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a random sample of training rows (Label is not NULL).

    We use DuckDB's SAMPLE clause; size is controlled by sample_frac and/or max_rows.
    """
    base_query = f"""
        SELECT *
        FROM {VIEW_NAME}
        WHERE Label IS NOT NULL
        USING SAMPLE {sample_frac} PERCENT
    """
    if max_rows is not None:
        base_query += f" LIMIT {max_rows}"

    df = con.execute(base_query).df()
    return df


def load_test_sample(
    con: duckdb.DuckDBPyConnection,
    max_rows: int = 100_000,
) -> pd.DataFrame:
    """
    Load a sample of test rows (Label IS NULL).
    This is just for getting an intuition + writing a toy submission.
    """
    query = f"""
        SELECT *
        FROM {VIEW_NAME}
        WHERE Label IS NULL
        LIMIT {max_rows}
    """
    df = con.execute(query).df()
    return df


# ---------------------------
# Feature preprocessing
# ---------------------------


def hash_categorical_series(s: pd.Series) -> pd.Series:
    """
    Very naive hashing: map each categorical string to a 32-bit integer.

    This keeps feature space bounded and lets us feed it into XGBoost
    as a numeric column. It's *not* optimal, but it's simple/intuitive.
    """
    # Fill missing with a sentinel
    s = s.fillna("__NA__").astype(str)
    # Python's hash is salted per process; use a stable hash via pandas' factorize
    # or a simple custom hash. Here: factorize -> dense integers.
    codes, _ = pd.factorize(s)
    return pd.Series(codes, index=s.index, dtype="int32")


def preprocess_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Split into X (features) and y (label), doing simple numeric + hashed-categorical prep.
    """
    # Target
    y = df["Label"].astype("int8").to_numpy()

    # Numeric features: fill missing with 0
    num_df = df[NUM_COLS].copy()
    num_df = num_df.fillna(0).astype("float32")

    # Categorical -> hashed ints
    cat_df = pd.DataFrame(index=df.index)
    for col in CAT_COLS:
        cat_df[col] = hash_categorical_series(df[col])

    # Combine
    X = pd.concat([num_df, cat_df], axis=1)
    return X, y


def preprocess_features_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Same as preprocess_features but for test rows (no Label).
    Returns (X, Id_series).
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


def train_xgboost(
    X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2, random_state: int = 42
):
    """
    Train a naive XGBoost model, print validation logloss, and return the fitted model.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=8,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    # Evaluate
    val_pred = model.predict_proba(X_val)[:, 1]
    ll = log_loss(y_val, val_pred)
    print(f"Validation logloss: {ll:.6f}")

    return model


def write_submission_csv(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    ids: pd.Series,
    path: str,
):
    """
    Write a simple Kaggle-style submission CSV: Id,Predicted
    """
    probs = model.predict_proba(X_test)[:, 1]
    sub_df = pd.DataFrame(
        {
            "Id": ids.values,
            "Predicted": probs,
        }
    )
    sub_df.to_csv(path, index=False)
    print(f"Wrote submission sample to: {path}")


# ---------------------------
# CLI / main
# ---------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Naive XGBoost baseline for Criteo DAC using Iceberg + DuckDB."
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Percentage of training rows to sample (e.g., 1.0 = 1%%, 5.0 = 5%%).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=500_000,
        help="Maximum number of training rows to load (after sampling).",
    )
    parser.add_argument(
        "--submission-path",
        type=str,
        default=None,
        help="If set, also score a sample of test rows and write a submission-style CSV to this path.",
    )
    parser.add_argument(
        "--test-max-rows",
        type=int,
        default=100_000,
        help="Maximum number of test rows to use for submission sample.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Connect + view
    con = create_duckdb_view()

    # 2) Load training sample
    print(f"Loading training sample: {args.sample_frac}% (max {args.max_rows} rows)...")
    df_train = load_train_sample(
        con, sample_frac=args.sample_frac, max_rows=args.max_rows
    )
    print(f"Loaded {len(df_train)} training rows.")

    # 3) Preprocess
    print("Preprocessing training features...")
    X, y = preprocess_features(df_train)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # 4) Train model + logloss
    model = train_xgboost(X, y)

    # 5) Optional: build a small submission from test rows
    if args.submission_path:
        print(f"Loading up to {args.test_max_rows} test rows for submission sample...")
        df_test = load_test_sample(con, max_rows=args.test_max_rows)
        print(f"Loaded {len(df_test)} test rows.")

        print("Preprocessing test features...")
        X_test, ids = preprocess_features_test(df_test)
        print(f"X_test shape: {X_test.shape}")

        write_submission_csv(model, X_test, ids, args.submission_path)


if __name__ == "__main__":
    main()
