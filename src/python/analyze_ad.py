#!/usr/bin/env python3
import argparse

import duckdb
from pyiceberg.catalog import load_catalog


VIEW_NAME = "criteo_ad"
CATALOG_NAME = "local"  # adjust if needed
TABLE_IDENTIFIER = "criteo.ad_challenge_v1"


def create_connection_with_view():
    """Load Iceberg table via PyIceberg and register it as a DuckDB view."""
    catalog = load_catalog(CATALOG_NAME)
    table = catalog.load_table(TABLE_IDENTIFIER)

    arrow_table = table.scan().to_arrow()
    con = duckdb.connect()
    con.register(VIEW_NAME, arrow_table)
    return con


def run_counts(con: duckdb.DuckDBPyConnection):
    """Count train vs test rows."""
    query = f"""
        SELECT
          CASE WHEN Label IS NULL THEN 'test' ELSE 'train' END AS split,
          COUNT(*) AS n_rows
        FROM {VIEW_NAME}
        GROUP BY 1
        ORDER BY split
    """
    df = con.execute(query).df()
    print(df)


def run_describe(con: duckdb.DuckDBPyConnection):
    """Run a DESCRIBE on the view (schema-level describe)."""
    query = f"DESCRIBE SELECT * FROM {VIEW_NAME}"
    df = con.execute(query).df()
    print(df)


def run_sql(con: duckdb.DuckDBPyConnection, sql: str):
    """Run an arbitrary SQL statement against the view."""
    df = con.execute(sql).df()
    print(df)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Criteo Iceberg data exploration utility"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--counts",
        action="store_true",
        help="Show train vs test row counts",
    )
    group.add_argument(
        "--describe",
        action="store_true",
        help="Describe the criteo_ad view schema",
    )
    group.add_argument(
        "--sql",
        type=str,
        help="Run an arbitrary SQL query (the view is called 'criteo_ad')",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    con = create_connection_with_view()

    if args.sql:
        run_sql(con, args.sql)
    elif args.describe:
        run_describe(con)
    else:
        # Default action: counts
        run_counts(con)


if __name__ == "__main__":
    main()
