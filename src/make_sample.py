"""Stream a user-level sample of the REES46 CSV out of archive.zip into parquet.

Sampling by user_id (not by row) keeps every sampled user's full history and
session baskets intact.

Usage:
    .venv/bin/python src/make_sample.py archive.zip 2019-Nov.csv data/nov_u5.parquet --pct 5
"""
import argparse
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DTYPES = {
    "event_type": "category",
    "product_id": "int64",
    "category_id": "int64",
    "category_code": "string",
    "brand": "string",
    "price": "float32",
    "user_id": "int64",
    "user_session": "string",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    ap.add_argument("member")
    ap.add_argument("out")
    ap.add_argument("--pct", type=int, default=5, help="percent of users to keep")
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer, rows, total = None, 0, 0
    # accepts either archive.zip (+ member name) or an already-extracted .csv (member ignored)
    src = (
        open(args.zip_path, "rb")
        if args.zip_path.endswith(".csv")
        else zipfile.ZipFile(args.zip_path).open(args.member)
    )
    with src as f:
        reader = pd.read_csv(
            f, chunksize=args.chunksize, dtype=DTYPES, usecols=["event_time", *DTYPES]
        )
        for i, chunk in enumerate(reader):
            total += len(chunk)
            # multiplicative hash so consecutive ids do not map to consecutive buckets
            keep = (chunk["user_id"] * 2654435761 % 100) < args.pct
            chunk = chunk[keep].copy()
            chunk["event_time"] = pd.to_datetime(
                chunk["event_time"].str.replace(" UTC", "", regex=False)
            )
            chunk["event_type"] = chunk["event_type"].astype("string")
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(args.out, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(chunk)
            if i % 10 == 0:
                print(f"chunk {i}: read {total:,} kept {rows:,}", flush=True)
    if writer:
        writer.close()
    print(f"done: read {total:,} kept {rows:,} -> {args.out}")


if __name__ == "__main__":
    main()
