"""
Shared fixtures. Tests that GENERATE data always write into a temporary
sandbox copy — never into the real data/partitioned — so the suite cannot
pollute the Day 1 historical state.

The suite needs the Day 1 outputs (data/cleaned, data/partitioned). If they
are absent, dependent tests SKIP with instructions instead of failing:
    python pipeline/clean.py && python pipeline/partition.py
"""

import shutil
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "infra"))

HIST_END = date(2011, 12, 9)


@pytest.fixture(scope="session")
def real_root():
    needed = [ROOT / "data/cleaned/product_sales.parquet",
              ROOT / "data/cleaned/cancellations.parquet",
              ROOT / "data/partitioned/product_sales"]
    if not all(p.exists() for p in needed):
        pytest.skip("Day 1 outputs missing. Run: python pipeline/clean.py && python pipeline/partition.py")
    from pipeline.synthetic_profile import load_or_build_profile
    load_or_build_profile(ROOT)  # builds data/profile/profile.json once (deterministic, derived data)
    return ROOT


def _copy_day(src_root: Path, dst_root: Path, d: date):
    for ds in ("product_sales", "cancellations"):
        rel = Path("data/partitioned") / ds / f"year={d.year}" / f"month={d.month:02d}" / f"day={d.day:02d}"
        if (src_root / rel).exists():
            shutil.copytree(src_root / rel, dst_root / rel)


def build_sandbox(real_root: Path, dst: Path, full: bool = False, last_day: date = HIST_END) -> Path:
    """Minimal sandbox: cleaned data + profile + ONE partition (enough to locate 'latest')."""
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "data").mkdir(exist_ok=True)
    shutil.copytree(real_root / "data/cleaned", dst / "data/cleaned")
    shutil.copytree(real_root / "data/profile", dst / "data/profile")
    if full:
        shutil.copytree(real_root / "data/partitioned", dst / "data/partitioned")
        if (real_root / "data/history_cache").exists():
            shutil.copytree(real_root / "data/history_cache", dst / "data/history_cache")
    else:
        _copy_day(real_root, dst, last_day)
    return dst


@pytest.fixture
def sandbox(tmp_path, real_root):
    """Factory: sandbox() -> fresh isolated project root."""
    counter = {"n": 0}

    def _make(full=False, last_day=HIST_END):
        counter["n"] += 1
        return build_sandbox(real_root, tmp_path / f"sb{counter['n']}", full=full, last_day=last_day)
    return _make


@pytest.fixture(scope="session")
def model(real_root):
    from pipeline.synthetic_profile import load_or_build_profile
    return load_or_build_profile(real_root)


@pytest.fixture(scope="session")
def full_ro_root(tmp_path_factory, real_root):
    """Full copy of the project data (incl. history cache) for READ-ONLY regression tests."""
    return build_sandbox(real_root, tmp_path_factory.mktemp("full_ro"), full=True)
