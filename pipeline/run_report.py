"""
run_report.py — Single entry point for the full pipeline: assumes
clean.py and partition.py have already been run (Day 1, run once when new
raw data lands), then does anomaly detection -> narrative -> report for
one date. This is deliberately the function Day 3's Lambda handler will
call directly, so it's written as a plain function, not just a __main__
block, with explicit error handling around the one step that can fail for
reasons outside this code's control (the LLM call).
"""

import sys
import json
import logging
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.metrics import get_connection, compute_daily_metrics
from pipeline.report import build_report_with_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_report")


def run(target_date: date, root: Path) -> dict:
    """
    Returns a result dict with status, output path, and cost info — never
    raises for expected failure modes (missing data, LLM failure), so a
    Lambda caller can inspect .status rather than catching exceptions for
    control flow. Unexpected errors still raise, since those should
    surface loudly rather than be swallowed.
    """
    logger.info(f"Starting report run for {target_date}")

    con = get_connection(root / "data" / "partitioned")
    metrics = compute_daily_metrics(con, target_date)

    if not metrics.get("has_data"):
        logger.warning(f"No data found for {target_date} — nothing to report.")
        return {"status": "no_data", "date": str(target_date)}

    try:
        html, context = build_report_with_context(target_date, root)
    except Exception as e:
        # The LLM call (inside build_report -> generate_narrative) is the
        # one step in this pipeline that depends on an external service
        # and can fail for reasons outside this code's control (timeout,
        # rate limit, API outage). Everything upstream of it (clean,
        # partition, metrics, anomaly detection) is deterministic local
        # computation and failing there would indicate a real bug, not a
        # transient condition — so only this step gets a soft failure path.
        logger.error(f"Report generation failed for {target_date}: {e}")
        return {"status": "error", "date": str(target_date), "error": str(e)}

    out_dir = root / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"report_{target_date}.html"
    out_path.write_text(html, encoding="utf-8")

    logger.info(f"Report written to {out_path}")
    return {"status": "ok", "date": str(target_date), "output_path": str(out_path), "context": context}


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    target = date(2011, 12, 9)
    if len(sys.argv) > 1:
        target = date.fromisoformat(sys.argv[1])

    result = run(target, root)
    print(json.dumps(result, indent=2))
