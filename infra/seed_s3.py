"""
seed_s3.py — One-time seeding of the S3 bucket with the REAL historical state.
The Lambda's synthetic generator then continues the series one trading day per run.

Usage (with your AWS credentials configured):
    python infra/seed_s3.py --bucket YOUR_BUCKET

Uploads:
  data/partitioned/...      real partitions with date <= cutoff (default 2011-12-09,
                            i.e. the FULL history)
  data/cleaned/*.parquet    cleaned datasets (source for the generator's profile)
  data/profile/profile.json statistical profile the generator samples from
                            (built from the cleaned data if not already present)
  data/history_cache/history.parquet   history rows for dates <= cutoff

The generator refuses to run unless the full history is present (it would
otherwise generate "synthetic" days that overlap real ones), so a cutoff earlier
than 2011-12-09 is only useful for testing that safeguard.
"""

import argparse
import re
import sys
from datetime import date
from pathlib import Path

import boto3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import paths  # noqa: E402

DATE_RE = re.compile(r"year=(\d+)[/\\]month=(\d+)[/\\]day=(\d+)")


def partition_date(path: Path):
    m = DATE_RE.search(str(path))
    return date(*(int(x) for x in m.groups())) if m else None


def seed(bucket: str, cutoff: date, s3=None, root: Path = ROOT) -> dict:
    s3 = s3 or boto3.client("s3")
    uploaded = {"partitions": 0, "cleaned": 0, "history_rows": 0, "profile": 0}

    for f in sorted((root / "data/partitioned").rglob("*.parquet")):
        d = partition_date(f)
        if d and d <= cutoff:
            s3.upload_file(str(f), bucket, f.relative_to(root).as_posix())
            uploaded["partitions"] += 1

    for name in ("product_sales.parquet", "cancellations.parquet"):
        s3.upload_file(str(root / "data/cleaned" / name), bucket, f"data/cleaned/{name}")
        uploaded["cleaned"] += 1

    # Profile: the generator's learned fingerprint of the real data.
    from pipeline.synthetic_profile import load_or_build_profile
    load_or_build_profile(root)  # builds data/profile/profile.json from the cleaned data if absent
    s3.upload_file(str(paths.profile_path(root)), bucket, "data/profile/profile.json")
    uploaded["profile"] = 1

    # History cache: reuse the local full cache filtered to the cutoff if present
    # (per-day rows are independent of later days), otherwise it is rebuilt lazily
    # by the Lambda on first run (slow: ~80s, so build locally first).
    local_cache = root / "data/history_cache/history.parquet"
    if local_cache.exists():
        h = pd.read_parquet(local_cache)
        h["date"] = pd.to_datetime(h["date"]).dt.date
        h = h[h["date"] <= cutoff]
        tmp = root / "data/history_cache/_seed_history.parquet"
        h.to_parquet(tmp, index=False)
        s3.upload_file(str(tmp), bucket, "data/history_cache/history.parquet")
        tmp.unlink()
        uploaded["history_rows"] = len(h)
    else:
        print("WARNING: no local history cache; run pipeline/history_cache.py first "
              "or the first Lambda run will take ~80s and may time out.")
    return uploaded


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--cutoff", default="2011-12-09")
    a = ap.parse_args()
    print(seed(a.bucket, date.fromisoformat(a.cutoff)))
