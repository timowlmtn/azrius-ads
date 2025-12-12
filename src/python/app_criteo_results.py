#!/usr/bin/env python3
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

# Paths – adjust if needed
ICEBERG_PATH = "data/iceberg/criteo/ad_challenge_v1"
DEFAULT_RESULTS_PATH = "data/2025-12-06-110817.csv"

NUM_COLS = [f"I{i}" for i in range(1, 14)]
CAT_COLS = [f"C{i}" for i in range(1, 27)]


@st.cache_resource(show_spinner=True)
def load_predictions(results_path: str) -> pd.DataFrame:
    """Load the submission / predictions CSV."""
    return pd.read_csv(results_path)


@st.cache_resource(show_spinner=True)
def load_features_for_ids(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Use DuckDB to load only the rows from the Iceberg-backed Parquet data
    that correspond to the predicted Ids.
    """
    ids_df = pred_df[["Id"]].copy()

    con = duckdb.connect()
    con.register("pred_ids", ids_df)

    # Assumes Iceberg table data files live under data/ subdir
    parquet_glob = str(Path(ICEBERG_PATH) / "data" / "*.parquet")

    query = f"""
        WITH ad AS (
          SELECT *
          FROM read_parquet('{parquet_glob}')
        )
        SELECT a.*
        FROM ad a
        JOIN pred_ids p ON a.Id = p.Id
    """
    features_df = con.execute(query).df()
    con.close()
    return features_df


def main():
    st.title("Criteo Ad Challenge – Prediction Exploration")

    st.markdown(
        """
        This app visualizes model predictions from a training run and links them
        back to the original features.

        - **Left sidebar**: key statistical summaries  
        - **Main area**: visual exploration of prediction scores and feature relationships
        """
    )

    # --- Sidebar: file selection & stats ---
    st.sidebar.header("Data Sources")

    results_path = st.sidebar.text_input(
        "Results CSV path",
        value=DEFAULT_RESULTS_PATH,
        help="Path to the submission/prediction CSV (e.g. data/YYYY-MM-DD-HHMMSS.csv)",
    )

    if not os.path.exists(results_path):
        st.sidebar.error(f"Results file not found: {results_path}")
        st.stop()

    with st.spinner("Loading predictions..."):
        pred_df = load_predictions(results_path)

    st.sidebar.write(f"Loaded **{len(pred_df):,}** predictions")

    # Basic prediction stats
    st.sidebar.subheader("Prediction statistics")
    preds = pred_df["Predicted"].values
    st.sidebar.metric("Mean predicted CTR", f"{preds.mean():.4f}")
    st.sidebar.metric("Min predicted", f"{preds.min():.4f}")
    st.sidebar.metric("Max predicted", f"{preds.max():.4f}")

    q10, q50, q90 = np.quantile(preds, [0.1, 0.5, 0.9])
    st.sidebar.write("Quantiles:")
    st.sidebar.write(f"10th percentile: `{q10:.4f}`")
    st.sidebar.write(f"50th percentile: `{q50:.4f}`")
    st.sidebar.write(f"90th percentile: `{q90:.4f}`")

    frac_low = (preds < 0.1).mean()
    frac_high = (preds > 0.9).mean()
    st.sidebar.write(f"Fraction < 0.1: `{frac_low:.2%}`")
    st.sidebar.write(f"Fraction > 0.9: `{frac_high:.2%}`")

    # --- Load features for those Ids ---
    st.subheader("Join predictions back to feature data")

    with st.spinner("Loading matching feature rows from Iceberg Parquet data..."):
        features_df = load_features_for_ids(pred_df)

    if features_df.empty:
        st.error("No matching features found for the predicted Ids. Check that Ids line up.")
        st.stop()

    st.write(f"Matched **{len(features_df):,}** rows with features.")

    # Join preds and features on Id
    full_df = features_df.merge(pred_df, on="Id", how="inner")
    st.write("Preview of joined data:")
    st.dataframe(full_df.head(), use_container_width=True)

    # --- Main visualizations ---
    st.subheader("Prediction distribution")
    st.caption("Histogram of predicted probabilities across the selected run.")

    st.bar_chart(
        pd.DataFrame(
            {"Predicted": preds},
        ),
        x=None,
        y="Predicted",
    )

    # Better histogram with binning
    hist_values, bin_edges = np.histogram(preds, bins=30, range=(0.0, 1.0))
    hist_df = pd.DataFrame(
        {
            "bin_left": bin_edges[:-1],
            "bin_right": bin_edges[1:],
            "count": hist_values,
        }
    )
    hist_df["bin_center"] = (hist_df["bin_left"] + hist_df["bin_right"]) / 2.0
    st.altair_chart(
        __import__("altair").Chart(hist_df)
        .mark_bar()
        .encode(
            x=__import__("altair").X("bin_center:Q", title="Predicted CTR"),
            y=__import__("altair").Y("count:Q", title="Count"),
        )
        .properties(height=300),
        use_container_width=True,
    )

    st.subheader("Explore predictions by feature")

    # Choose feature to group by / visualize
    all_features = NUM_COLS + CAT_COLS
    feature = st.selectbox(
        "Select a feature to explore",
        options=all_features,
        index=0,
    )

    # Numeric vs categorical handling
    if feature in NUM_COLS:
        st.write(f"Exploring numeric feature **{feature}**")

        # Bin numeric feature into quantiles
        df_non_null = full_df[full_df[feature].notnull()].copy()
        if df_non_null.empty:
            st.warning(f"No non-null values for {feature}")
        else:
            df_non_null["bin"] = pd.qcut(
                df_non_null[feature],
                q=10,
                duplicates="drop",
            )
            grouped = (
                df_non_null.groupby("bin")["Predicted"]
                .agg(["count", "mean"])
                .reset_index()
                .rename(columns={"mean": "mean_predicted"})
            )
            st.write("Binned summary:")
            st.dataframe(grouped, use_container_width=True)

            # Plot mean predicted by bin
            st.altair_chart(
                __import__("altair").Chart(grouped)
                .mark_bar()
                .encode(
                    x=__import__("altair").X("bin:N", title=f"{feature} bin"),
                    y=__import__("altair").Y("mean_predicted:Q", title="Mean predicted CTR"),
                    tooltip=["bin", "count", "mean_predicted"],
                )
                .properties(height=300),
                use_container_width=True,
            )
    else:
        st.write(f"Exploring categorical feature **{feature}**")

        # Group by category: top N by count
        grouped = (
            full_df.groupby(feature)["Predicted"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"mean": "mean_predicted"})
            .sort_values("count", ascending=False)
        )

        top_n = st.slider("Show top N categories by count", 5, 50, 20)
        top_grouped = grouped.head(top_n)

        st.write("Top categories by count:")
        st.dataframe(top_grouped, use_container_width=True)

        st.altair_chart(
            __import__("altair").Chart(top_grouped)
            .mark_bar()
            .encode(
                x=__import__("altair").X(f"{feature}:N", sort="-y"),
                y=__import__("altair").Y("mean_predicted:Q", title="Mean predicted CTR"),
                tooltip=[feature, "count", "mean_predicted"],
            )
            .properties(height=300),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
