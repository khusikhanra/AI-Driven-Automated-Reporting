"""
partition.py — Split cleaned data into daily Hive-style partitions:

    data/partitioned/product_sales/year=YYYY/month=MM/day=DD/orders.parquet
    data/partitioned/cancellations/year=YYYY/month=MM/day=DD/orders.parquet

WHY partition by day, and why this exact layout:

1. This is the standard "Hive partitioning" convention. DuckDB, Athena,
   and Spark all understand year=/month=/day= directories natively and
   can prune partitions at query time (i.e. "give me last 7 days" only
   reads 7 files, not the whole dataset). This matters for a real
   pipeline, not just as a nicety — at scale, unpartitioned data means
   every query scans everything.

2. It gives a 1:1 mental model with the eventual S3 bucket. When Day 3
   moves this to s3://bucket/product_sales/year=.../, the folder
   structure — and therefore every downstream query — doesn't change.
   Only the storage backend does. This is the main reason to build it
   this way NOW rather than dumping one big Parquet file: it removes an
   entire class of "worked locally, broke in the cloud" bugs.

3. Daily grain (not hourly/weekly) matches the reporting cadence — one
   Lambda run = one new day of data = one new partition appended. This
   makes "has today's data landed yet" a trivial existence check on a
   single expected partition path, which the pipeline uses for its
   failure handling in Day 4.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import paths  # noqa: E402


def write_daily_partitions(df: pd.DataFrame, base_dir: Path, dataset_name: str) -> int:
    """
    Writes one Parquet file per calendar day found in df['InvoiceDate'].
    Returns the number of partitions written.
    """
    df = df.copy()
    df["_date"] = df["InvoiceDate"].dt.date

    partitions_written = 0
    for day, day_df in df.groupby("_date"):
        partition_dir = Path(base_dir) / paths.partition_relpath(dataset_name, day)
        partition_dir.mkdir(parents=True, exist_ok=True)
        day_df.drop(columns=["_date"]).to_parquet(partition_dir / "orders.parquet", index=False)
        partitions_written += 1

    return partitions_written


if __name__ == "__main__":
    root = paths.PROJECT_ROOT
    partitioned = paths.partitioned_dir(root)

    product_sales = pd.read_parquet(paths.product_sales_path(root))
    cancellations = pd.read_parquet(paths.cancellations_path(root))

    n_sales = write_daily_partitions(product_sales, partitioned, paths.PRODUCT_SALES)
    n_cancel = write_daily_partitions(cancellations, partitioned, paths.CANCELLATIONS)

    print(f"product_sales: {n_sales} daily partitions written")
    print(f"cancellations: {n_cancel} daily partitions written")
    print(f"Output root: {partitioned}")
