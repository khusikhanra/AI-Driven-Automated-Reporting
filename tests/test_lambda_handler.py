"""
Integration test for lambda_handler against MOCKED S3 (moto) and a local fake
Slack webhook, driving the synthetic generator (which can always produce another day). Uses the real dataset and real pipeline/generator
code — only AWS and Slack are faked. Run: python -m pytest tests/test_lambda_handler.py -v -s
"""

import json
import os
import sys
import threading
import time
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "infra"))

BUCKET = "test-reporting-bucket"
SLACK_MESSAGES = []


class _SlackHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers["Content-Length"])
        SLACK_MESSAGES.append(json.loads(self.rfile.read(n))["text"])
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def slack_server():
    srv = HTTPServer(("127.0.0.1", 0), _SlackHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/hook"
    srv.shutdown()


@pytest.fixture
def env(slack_server, tmp_path):
    """Function-scoped (not module) so WORK_DIR is fresh per test: each test's
    handler calls get a clean /tmp, like separate Lambda invocations could,
    so there is no cross-test state leakage via a stale work directory."""
    os.environ.update({
        "BUCKET_NAME": BUCKET,
        "SLACK_WEBHOOK_URL": slack_server,
        "WORK_DIR": str(tmp_path / "work"),
        "AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test",
        "AWS_DEFAULT_REGION": "us-east-1",
        "USE_MOCK_LLM": "true",
    })
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("GENERATOR_SEED", None)
    os.environ.pop("GENERATOR_EVENT", None)


def _keys(s3, prefix):
    out = []
    for p in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        out += [o["Key"] for o in p.get("Contents", [])]
    return out


@pytest.fixture
def seeded_bucket(env):
    """Real full history seeded (required: the generator refuses partial history)."""
    import seed_s3
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        seeded = seed_s3.seed(BUCKET, date(2011, 12, 9), s3=s3)
        assert seeded["partitions"] > 0 and seeded["history_rows"] > 0 and seeded["profile"] == 1
        yield s3


def test_first_invocation_generates_and_delivers_a_synthetic_day(seeded_bucket):
    from lambda_handler import handler
    s3 = seeded_bucket
    seeded_partitions = set(_keys(s3, "data/partitioned/"))

    SLACK_MESSAGES.clear()
    r1 = handler({}, None, s3_client=s3)

    assert r1["status"] == "ok" and r1["date"] == "2011-12-11"     # Sat 12-10 closed, skipped
    assert r1["data_source"] == "synthetic" and isinstance(r1["seed"], int)
    assert "reports/report_2011-12-11.html" in _keys(s3, "reports/")
    assert r1["slack"] == "ok_200" and len(SLACK_MESSAGES) == 1
    assert "2011-12-11" in SLACK_MESSAGES[0] and "simulated data" in SLACK_MESSAGES[0]

    html = s3.get_object(Bucket=BUCKET, Key=r1["report_key"])["Body"].read().decode()
    assert "Daily Sales Report" in html and "data:image/png;base64" in html
    assert "Simulated data" in html and f"seed {r1['seed']}" in html

    new_parts = set(_keys(s3, "data/partitioned/")) - seeded_partitions
    assert any("day=11" in k and "month=12" in k for k in new_parts)
    assert "data/synthetic_manifest/date=2011-12-11.json" in _keys(s3, "data/synthetic_manifest/")
    # Every stored key is '/'-separated. On Windows str(path) would use backslashes and the new
    # day would be uploaded under keys no later prefix listing finds (root cause of a real bug).
    assert not [k for k in _keys(s3, "") if "\\" in k]

    # The key helper is OS-independent: check it with Windows path semantics on any platform.
    from pathlib import PureWindowsPath
    from lambda_handler import s3_key
    work = PureWindowsPath("C:/Users/someone/AppData/Local/Temp/reporting-work")
    assert s3_key(work / "data/synthetic_manifest/date=2011-12-11.json", work) == \
        "data/synthetic_manifest/date=2011-12-11.json"


def test_state_persists_across_invocations_advancing_one_day_each(seeded_bucket):
    from lambda_handler import handler
    s3 = seeded_bucket
    r1 = handler({}, None, s3_client=s3)
    r2 = handler({}, None, s3_client=s3)
    assert (r1["date"], r2["date"]) == ("2011-12-11", "2011-12-12")
    assert r1["seed"] != r2["seed"]                       # fresh entropy per live run


def test_regeneration_by_date_does_not_advance_state(seeded_bucket):
    from lambda_handler import handler
    s3 = seeded_bucket
    handler({}, None, s3_client=s3)                       # -> 2011-12-11
    before = set(_keys(s3, "data/partitioned/"))
    r2 = handler({"target_date": "2011-12-11"}, None, s3_client=s3)
    assert r2["status"] == "ok" and r2["date"] == "2011-12-11"
    assert set(_keys(s3, "data/partitioned/")) == before   # no new partition written
    r3 = handler({}, None, s3_client=s3)                   # still advances from 12-11, not 12-11 again
    assert r3["date"] == "2011-12-12"


def test_explicit_seed_and_event_are_honoured_and_reproducible(seeded_bucket):
    from lambda_handler import handler
    s3 = seeded_bucket
    SLACK_MESSAGES.clear()
    r1 = handler({"seed": 777, "synthetic_event": "bulk_order"}, None, s3_client=s3)
    assert r1["seed"] == 777 and r1["synthetic_event"] == "bulk_order"
    assert "concentration risk" in SLACK_MESSAGES[-1]      # lowercase: Slack's own wording (see format_slack)
    html = s3.get_object(Bucket=BUCKET, Key=r1["report_key"])["Body"].read().decode()
    assert "Concentration risk" in html                    # Title Case: the HTML report's own wording

    # Same seed replayed via env var against a SECOND bucket, inside the SAME
    # active moto mock context (nesting a second `with mock_aws():` here would
    # share the same underlying mock backend as the outer one and silently
    # continue the first bucket's timeline instead of isolating state).
    import seed_s3
    bucket2 = BUCKET + "-replay"
    s3b = boto3.client("s3", region_name="us-east-1")
    s3b.create_bucket(Bucket=bucket2)
    seed_s3.seed(bucket2, date(2011, 12, 9), s3=s3b)
    os.environ["BUCKET_NAME"] = bucket2
    os.environ["GENERATOR_SEED"] = "777"
    os.environ["GENERATOR_EVENT"] = "bulk_order"
    try:
        r2 = handler({}, None, s3_client=s3b)
    finally:
        os.environ["BUCKET_NAME"] = BUCKET
        del os.environ["GENERATOR_SEED"], os.environ["GENERATOR_EVENT"]
    assert r2["date"] == "2011-12-11" and r2["seed"] == 777   # independent bucket -> same starting point
    html1 = s3.get_object(Bucket=BUCKET, Key=r1["report_key"])["Body"].read().decode()
    html2 = s3b.get_object(Bucket=bucket2, Key=r2["report_key"])["Body"].read().decode()
    import re
    strip_ts = lambda h: re.sub(r"Generated [\d\-T:.+]+", "Generated <ts>", h)
    assert strip_ts(html1) == strip_ts(html2)   # identical except the wall-clock "Generated" timestamp


def test_failure_path_is_loud_and_leaves_s3_untouched(seeded_bucket):
    from lambda_handler import handler
    s3 = seeded_bucket
    SLACK_MESSAGES.clear()
    keys_before = set(_keys(s3, ""))
    with pytest.raises(Exception):                        # GeneratorError: 2099 is far past the horizon
        handler({"target_date": "2099-01-01"}, None, s3_client=s3)
    assert any("FAILED" in m for m in SLACK_MESSAGES)
    assert set(_keys(s3, "")) == keys_before


def test_cold_start_reuses_persisted_history_and_profile(seeded_bucket, tmp_path):
    """Simulates a fresh container (empty /tmp) picking up where a prior one left
    off, using ONLY what the prior invocation uploaded to S3 -- this is what makes
    the history-cache speed-up (Day 3) actually hold across real Lambda cold
    starts, not just within one warm container."""
    from lambda_handler import handler
    s3 = seeded_bucket
    handler({}, None, s3_client=s3)

    os.environ["WORK_DIR"] = str(tmp_path / "work_cold")   # brand-new /tmp
    t0 = time.time()
    r2 = handler({}, None, s3_client=s3)
    elapsed = time.time() - t0

    assert r2["date"] == "2011-12-12"
    assert elapsed < 30, f"cold-start run took {elapsed:.1f}s; history cache may not be persisting via S3"
    assert "data/history_cache/history.parquet" in _keys(s3, "data/history_cache/")


def test_repeated_runs_stay_healthy_over_a_short_horizon(seeded_bucket):
    """Unlike the Day 3 replay shim, the generator never exhausts. Run enough
    invocations to cross the Christmas closure and confirm every run succeeds
    with a strictly increasing date."""
    from lambda_handler import handler
    s3 = seeded_bucket
    dates = [handler({}, None, s3_client=s3)["date"] for _ in range(10)]
    assert dates == sorted(dates) and len(set(dates)) == 10
    assert len(_keys(s3, "reports/")) == 10
