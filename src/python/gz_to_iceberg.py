#!/usr/bin/env python3
import argparse
import gzip
from pathlib import Path
from typing import Dict, List, Generator, Optional

import pyarrow as pa
from pyiceberg.catalog import load_catalog
from pyiceberg.exceptions import NoSuchTableError, NoSuchNamespaceError


# ---------- Schema helpers ----------

NUM_COLS = [f"I{i}" for i in range(1, 14)]
CAT_COLS = [f"C{i}" for i in range(1, 27)]

ARROW_SCHEMA = pa.schema(
    [pa.field("Id", pa.int64(), nullable=False)]
    + [pa.field("Label", pa.int32(), nullable=True)]
    + [pa.field(name, pa.float64(), nullable=True) for name in NUM_COLS]
    + [pa.field(name, pa.string(), nullable=True) for name in CAT_COLS]
)


def parse_criteo_line(
    line: str,
    has_label: bool,
    row_id: int,
) -> Dict[str, Optional[object]]:
    """
    Parse a single Criteo DAC line (TSV) into a dict matching ARROW_SCHEMA.

    train.txt(.gz):
        label \t I1..I13 \t C1..C26  => 40 fields

    test.txt(.gz):
        I1..I13 \t C1..C26         => 39 fields, no label
    """
    parts = line.rstrip("\n").split("\t")

    if has_label and len(parts) != 40:
        raise ValueError(f"Expected 40 columns for train row, got {len(parts)}")
    if not has_label and len(parts) != 39:
        raise ValueError(f"Expected 39 columns for test row, got {len(parts)}")

    row: Dict[str, Optional[object]] = {}
    row["Id"] = row_id

    if has_label:
        row["Label"] = int(parts[0])
        int_start = 1
    else:
        row["Label"] = None
        int_start = 0

    int_end = int_start + 13
    int_vals = parts[int_start:int_end]
    cat_vals = parts[int_end : int_end + 26]

    # numeric fields
    for i, val in enumerate(int_vals, start=1):
        key = f"I{i}"
        if val == "" or val is None:
            row[key] = None
        else:
            # could use int() if you prefer; float gives you room for later transforms
            row[key] = float(val)

    # categorical fields
    for i, val in enumerate(cat_vals, start=1):
        key = f"C{i}"
        if val == "" or val is None:
            row[key] = None
        else:
            row[key] = val

    return row


def iter_arrow_batches_from_gz(
    path: Path,
    has_label: bool,
    start_id: int,
    batch_size: int = 500_000,
) -> Generator[pa.Table, None, int]:
    """
    Stream a gzipped Criteo file and yield PyArrow Tables of size batch_size.
    Returns the next available id after finishing.
    """
    cur_id = start_id
    batch: Dict[str, List[Optional[object]]] = {name: [] for name in ARROW_SCHEMA.names}

    def flush_batch() -> Optional[pa.Table]:
        if not batch["Id"]:
            return None
        # build Arrow table with explicit schema
        table = pa.table(batch, schema=ARROW_SCHEMA)
        # reset
        for k in batch.keys():
            batch[k] = []
        return table

    with gzip.open(path, mode="rt", encoding="utf-8") as f:
        for line in f:
            row = parse_criteo_line(line, has_label=has_label, row_id=cur_id)
            cur_id += 1
            for col in ARROW_SCHEMA.names:
                batch[col].append(row[col])

            if len(batch["Id"]) >= batch_size:
                table = flush_batch()
                if table is not None:
                    yield table

    # final partial batch
    table = flush_batch()
    if table is not None:
        yield table

    return cur_id


def main():
    parser = argparse.ArgumentParser(
        description="Stream Criteo *.txt.gz into an Iceberg table (Parquet-backed)."
    )
    parser.add_argument(
        "--catalog",
        default="local",
        help="PyIceberg catalog name (default: local, configured in ~/.pyiceberg.yaml)",
    )
    parser.add_argument(
        "--table",
        default="criteo.ad_challenge_v1",
        help="Table identifier, e.g. 'criteo.dac_v1'",
    )
    parser.add_argument(
        "--train",
        type=Path,
        required=True,
        help="Path to train.txt.gz (or .gzip)",
    )
    parser.add_argument(
        "--test",
        type=Path,
        required=True,
        help="Path to test.txt.gz (or .gzip)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500_000,
        help="Number of rows per Arrow batch / append",
    )

    args = parser.parse_args()

    catalog = load_catalog("local")

    # --- TRAIN: create table if needed with first batch ---
    start_id = 1
    train_batches = iter_arrow_batches_from_gz(
        args.train, has_label=True, start_id=start_id, batch_size=args.batch_size
    )

    try:
        # try loading (maybe it already exists)
        table = catalog.load_table(args.table)
        print(f"Loaded existing Iceberg table: {args.table}")
    except NoSuchTableError:
        print(
            f"Table {args.table} does not exist yet, creating from first train batch..."
        )
        first_batch = next(train_batches, None)
        if first_batch is None:
            raise RuntimeError("Train file produced no rows; cannot create table.")

        namespace_str, table_name = args.table.rsplit(".", 1)

        # --- NEW: ensure namespace exists ---
        try:
            catalog.load_namespace_properties(namespace_str)
        except NoSuchNamespaceError:

            print(f"Namespace {namespace_str!r} does not exist, creating...")
            catalog.create_namespace(namespace_str)

        # Create table with schema derived from Arrow
        table = catalog.create_table(
            identifier=args.table,
            schema=first_batch.schema,
        )
        print(f"Created table {args.table} at {table.location()}")

        # Append first batch
        print(f"Appending initial train batch with {first_batch.num_rows} rows...")
        table.append(first_batch)

    # Append remaining train batches
    for batch in train_batches:
        print(f"Appending train batch with {batch.num_rows} rows...")
        table.append(batch)

    # Compute next starting id (rough but fine: scan table length)
    # Alternatively, track id from generator return; here we just rescan once:
    current_count = table.scan().to_arrow().num_rows
    start_id = current_count + 1

    # --- TEST: append unlabeled data ---
    print(f"Streaming test data starting at Id={start_id} ...")
    test_batches = iter_arrow_batches_from_gz(
        args.test, has_label=False, start_id=start_id, batch_size=args.batch_size
    )

    for batch in test_batches:
        print(f"Appending test batch with {batch.num_rows} rows...")
        table.append(batch)

    total_rows = table.scan().to_arrow().num_rows
    print(f"Done. Total rows in Iceberg table {args.table}: {total_rows}")


if __name__ == "__main__":
    main()
