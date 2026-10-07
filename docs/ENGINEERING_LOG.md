# Engineering log and design decisions

> This is the detailed engineering history of the project: what was built, what went wrong,
> and why each design decision was made. For an overview start with the [README](../README.md);
> for the case study see [PORTFOLIO.md](../PORTFOLIO.md).

A serverless pipeline that cleans real transactional data, detects statistical
anomalies, generates a plain-English narrative with an LLM, renders a
self-contained HTML report, delivers it to Slack, and continues producing new
days indefinitely via a calibrated synthetic-data generator — all running
on a daily schedule with no human in the loop.

Built over five iterations (Day 1-5 below), each one reviewed and verified
against the previous before moving on, including two real bugs found by the
test suite and fixed rather than hidden, and one of my own earlier
explanations of a detector "limitation" that turned out to be wrong and was
corrected with a formal proof once I checked it properly. See
[What actually happened](#what-actually-happened-day-by-day) for the honest
version of how this was built.

## What it does, end to end

```
EventBridge (daily cron)
        │
        ▼
   Lambda invocation
        │
        ├─ 1. Download partitions + cleaned data + history cache + profile from S3
        ├─ 2. Generate the next trading day (synthetic generator, seeded + logged)
        ├─ 3. Compute deterministic metrics (DuckDB) — revenue, orders, AOV, returns
        ├─ 4. Detect anomalies (robust z-score + rule-based concentration check)
        ├─ 5. Generate narrative (Claude, forced structured output, metrics-only input)
        ├─ 6. Render self-contained HTML report (embedded chart, no external assets)
        └─ 7. Upload to S3, post summary + presigned link to Slack
             (only after every prior step succeeds — a failed run leaves no
             partial state, so the next scheduled run safely retries)
```

## Why this dataset, and the historical/synthetic boundary

The pipeline runs on the real
[UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail)
(541,909 transaction lines, Dec 2010 - Dec 2011, a real UK-based online retailer)
rather than a purely synthetic one, specifically so the cleaning, anomaly
detection, and the generator's own statistical calibration all have to deal
with genuine data mess: ~25% missing customer IDs, real cancellations, non-
product fee/postage line items, and one single order that legitimately makes
up 85% of its day's revenue (see [Key decisions](#key-decisions-and-why)).

That real history ends 2011-12-09. Every date after that is produced by
`pipeline/synthetic_generator.py`, which learns a statistical profile from
the real data (weekday/monthly seasonality, basket sizes, product popularity,
price distributions, customer pool, cancellation rates, trading calendar
including closures) and samples new, never-overlapping days from it — never
inventing data wholesale, and never touching or duplicating a real date. Every
synthetic report is visibly labelled ("Simulated data... seed N") in both the
HTML report and the Slack message; real historical reports are never labelled
this way. The generator can also inject five calibrated, ground-truth-logged
business events (see below) to exercise the anomaly detector on demand.

## Setup

```bash
# 1. Place the dataset
#    Put the original "Online Retail.xlsx" (UCI) in data/raw/. A CSV release of the
#    same data also works. The loader auto-detects either; see "Excel date quirk" below.

# 2. Install (Python 3.12+; on Windows: .venv\Scripts\activate)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# 3. Build the pipeline's derived state (data/ is gitignored; this rebuilds it)
python pipeline/clean.py              # raw -> data/cleaned (+ data_quality_report.json)
python pipeline/partition.py          # cleaned -> Hive-style daily partitions
python pipeline/synthetic_profile.py  # Synthetic Profile v2.0 -> data/profile/profile.json
python pipeline/history_cache.py      # anomaly baseline cache (fast: ~0.3s)

# 4. Run the tests
python -m pytest tests/ -q

# 5. Generate a report
python pipeline/run_report.py 2011-12-09        # a real historical day
python pipeline/synthetic_generator.py           # generate the next (synthetic) day
python pipeline/run_report.py 2011-12-11         # report on it
```

Dependencies are pinned in `requirements-dev.txt` (local) and
`requirements-lambda.txt` (container). The suite was verified with NumPy 2.4.3,
pandas 3.0.2, pyarrow 25.0.1 and DuckDB 1.5.5 on Python 3.12; the code uses no
version-specific APIs, but it has not been executed on Python 3.14 by the author
of this pass, so run `python -m pytest -q` once in your own environment.

`USE_MOCK_LLM=true` (the default — see `.env.example`) runs the entire pipeline
for free with no Anthropic API key, using a narrative that is grounded in the
real computed metrics rather than invented. Set `ANTHROPIC_API_KEY` and
`USE_MOCK_LLM=false` to use the real model.

## Key decisions and why

**Excel date quirk (found when verifying against the raw workbook).** In
`Online Retail.xlsx` the InvoiceDate column mixes two cell types: text cells
(`"12/13/2010 9:02"`, unambiguous and correct) and real datetime cells whose day
and month are **swapped** (true 2010-12-01 is stored as 2010-01-12; true
2011-10-12 as 2011-12-10). The swap applies to every datetime-typed cell in every
year, so it can only be repaired where cell types are still visible: in
`load_raw()`, not in `clean()`. An earlier heuristic (only repair dates before
2010-12-01) silently left all swapped 2011 dates wrong. Verified against the
independent CSV release: all 232,959 datetime cells and 308,950 text cells
reproduce its dates, and the xlsx and CSV routes now yield identical cleaned
data and the same profile fingerprint. The repair count is in the quality report.

**Cleaning (Day 1).** Every one of the 541,909 raw rows lands in exactly one
labelled bucket (clean sale / cancellation / non-product adjustment / dropped
error), with a reconciliation check enforcing the counts sum back to the
input — nothing silently disappears. Postage/fee/manual-adjustment stock
codes are excluded from product revenue via an explicit, documented list
(`config/data_rules.py`) rather than a regex, because some non-numeric stock
codes (gift vouchers, certain physical goods) are genuinely real products and
a blanket rule would have wrongly dropped real revenue.

**Mean vs. median (Day 1-2).** On 2011-12-09, one single line item (80,995
units of a paper craft item) is 85% of that day's revenue, which makes mean
AOV wildly misleading (£4,503 vs. a median of £317). The report shows both,
and the anomaly detector has an explicit rule-based concentration-risk check
specifically for this pattern, independent of the statistical detector.

**Robust (median/MAD) anomaly detection, not mean/std (Day 2).** Mean/std is
exactly the statistic a single extreme day distorts. Median/MAD-based z-scores
are robust to it. Validated against the real bulk-order day (correctly
flagged, z=12.87) and scanned across the full real year (27/305 days flagged,
8.9% — a plausible rate, not noise and not silence).

**History caching (Day 3).** The anomaly baseline originally recomputed all
305+ days from scratch on every call (about 1-2 minutes: 127s measured in the final verification run). Cached to Parquet, recomputing
only missing dates: a steady-state daily run (one new day) is ~0.3-2s, and
this does not degrade as history grows — a real requirement once this runs
daily on a schedule indefinitely, not just once in testing.

**Serverless, container image, upload-after-success (Day 3-4).** Lambda +
container image (pandas/duckdb/pyarrow/matplotlib exceed the zip package size
limit) + EventBridge, with S3 writes happening only after a report fully
succeeds — so a mid-run failure leaves no partial state, and the next
scheduled invocation safely retries the same day rather than skipping it or
duplicating work.

**Historical-base + synthetic-continuation generator, not a live feed (Day 4).**
A real dataset like this one is finite; a believable automated-reporting demo
needs data that keeps arriving. Rather than faking a live feed, the generator
is built to be honest about what it is: statistically calibrated to the real
data, deterministic given a seed (for testing and replay), fresh and logged
when run live, and clearly labelled as simulated everywhere it appears.

## Testing

**132 tests, all passing** (last verified run: `113 passed in 205.85s`). `python -m pytest tests/ -q` — roughly 3-4 minutes.
The count is pytest's *collected* count, which expands parametrized tests (for example one test over all six
event types is six tests). Counting `def test_` lines by hand gives a smaller number (64 in the generator file
against 81 collected). Check it yourself with `python -m pytest --collect-only -q | tail -1` and, per file,
`python -m pytest --collect-only -q | grep :: | cut -d: -f1 | sort | uniq -c`.

| File | Count | Covers |
|---|---|---|
| `test_clean.py` | 12 | Excel date-swap repair (datetime vs text cells, every year, real `.xlsx` round trip), exact row reconciliation, audit of dropped rows, one stable schema across all buckets |
| `test_synthetic_generator.py` | 81 | schema/dtype parity with real data, trading calendar, reproducibility (same seed+date+event ⇒ byte-identical output), 30-day statistical sanity on normal generation, all 5 injected events, 18 failure/edge cases (corrupt profile (corrupt, wrong version, tampered fingerprint), invalid seed, partial history, identifier overflow, no-side-effects-on-failure) |
| `test_synthetic_pipeline_integration.py` | 13 | the *existing, unmodified* Day 1-3 pipeline correctly consumes synthetic partitions; week-over-week comparisons bridge real→synthetic dates correctly; the history cache extends incrementally; **ground-truth detection rates measured across 8-12 seeds per event**, not asserted once |
| `test_lambda_handler.py` | 7 | full handler lifecycle against mocked S3 + a real local Slack server: generation, state persistence, regeneration-by-date, explicit seed/event reproducibility, loud failure with untouched S3 on error, and — found and fixed during this review — a real moto test-isolation bug where a nested mock context silently shared state with its parent |
| `conftest.py` | — | every generative test runs in an isolated tmp-path sandbox; nothing in the suite can write to the real project's `data/` |

Tests that generate data always assert against measured values (detection
rates across many seeds, actual z-scores, real reconciliation counts) rather
than hand-waved expectations — several of the numbers below were discovered
by running the code, not decided in advance.

## Known limitations (measured, not guessed)

**`demand_drop` is mathematically undetectable by the current threshold, on
this dataset — proven, not assumed.** Order count and revenue are bounded
below by zero (order count practically by one), but the current detector's
severity is unbounded above. The maximum |z| any drop could *ever* reach,
even at order_count=1 (the most extreme drop physically possible), is:

```
orders:  (62.00 - 1)    / 20.76   = 2.94   (sigma = 1.4826 x MAD, the detector's own scale)
revenue: (28530.63 - 0) / 13225.47 = 2.16
```

Both sit below the 3.5 detection threshold *at the theoretical extreme*. This
is a real property of this dataset's day-to-day dispersion relative to its
typical volume, not a generator calibration issue — an earlier version of
this project's code (and an earlier version of this very explanation)
mischaracterized it as "the low-side threshold is negative" and then tried to
fix it by recalibrating the generator's severity, which increased severity
but didn't change the outcome; that failed fix is what led to deriving the
actual cause above. A real fix would require the Day 2 detector to use an
asymmetric or relative/log-scale threshold instead of an absolute robust
z-score — a detector redesign, deliberately out of scope here since it would
invalidate the Day 2 thresholds already calibrated and verified against real
data (the 27/305-day flag rate above).

**`promo_uplift` (an ordinary, non-faulty volume increase) trips the detector
about a third of the time** (measured: 4/12 seeds in the final verification run; it varies with the seeds tried), because the detector's baseline is
a flat all-history median with no seasonal/weekday adjustment, so an ordinary
busy pre-Christmas day can itself look unusual against the full year. This is
a genuine, documented blind spot in distinguishing "ordinary seasonal peak"
from "anomalous spike," not fixed for the same scope reason as above.

**Lambda downloads the full S3 state (~600 small files, ~18MB) on every
invocation** rather than querying S3 directly via DuckDB's `httpfs` extension.
Simple and already fast enough at this scale, but doesn't scale indefinitely —
listed as the natural next optimization once the dataset is large enough for
it to matter.

**Secrets (`ANTHROPIC_API_KEY`, the Slack webhook URL) live in Lambda
environment variables**, encrypted at rest but readable by anyone with
`lambda:GetFunctionConfiguration`. Secrets Manager is the documented next
step (`infra/deploy.md`, step 9), not built here to keep the deployment
footprint matched to this project's scope.

**No IaC (Terraform/CDK).** Deployment is documented, exact, verification-
gated CLI steps (`infra/deploy.md`) rather than half-finished infrastructure
code — a deliberate scope decision made at the start of the cloud-deployment
phase, not an oversight.

## Cost

**LLM:** `USE_MOCK_LLM=true` (default) costs nothing: the narrative is generated from a template
grounded in the real computed metrics. With a real key the model and per-token prices come from
`ANTHROPIC_MODEL`, `ANTHROPIC_INPUT_PRICE_PER_MTOK` and `ANTHROPIC_OUTPUT_PRICE_PER_MTOK`
(code defaults: `claude-sonnet-4-5`, $3 / $15 per million input / output tokens). Prices and model
availability change, so verify both at <https://docs.claude.com> before relying on them. The prompt is a
few hundred tokens of pre-computed numbers and the answer is a short structured object, so a report
should cost on the order of a cent or less; the exact figure is computed and logged for every run. The
live path has been exercised only against a stubbed client in this repository, never against the real
API.

**AWS:** at one invocation/day, S3 storage (tens of MB) and Lambda invocation
count/duration both sit comfortably inside the AWS free tier for a long time.

## Project structure

```
config/data_rules.py           Explicit non-product-code list, cancellation rule, year-end closure rule
pipeline/
  paths.py                     Single source of truth for the data/ layout (root-relative)
  clean.py                     Day 1: raw xlsx/csv -> labelled, reconciled buckets + date repair
  partition.py                 Day 1: Hive-style year=/month=/day= partitioning
  metrics.py                   Day 1-2: deterministic DuckDB metrics (no LLM math, ever)
  anomaly_detection.py         Day 2: robust z-score + concentration-risk + return-rate
  narrative.py                 Day 2: structured LLM call, mock mode, cost tracking
  report.py                    Day 2-4: self-contained HTML, synthetic-data banner
  history_cache.py             Day 3: incremental Parquet-cached baseline (1-2 min -> ~1s)
  run_report.py                Day 2-3: single orchestration entry point
  synthetic_profile.py         Day 4: Synthetic Profile v2.0 (validated, fingerprinted, versioned)
  synthetic_generator.py       Day 4: calibrated day generation + 5 ground-truth events
lambda_handler.py               Day 3-4: S3 sync, generation, report, Slack delivery
infra/
  seed_s3.py                   One-time S3 seeding (real history + profile)
  deploy.md                    Exact, verification-gated AWS CLI deployment steps
tests/                          132 tests; see Testing above
Dockerfile, requirements-lambda.txt, .dockerignore     Lambda container image
.env.example                    Every environment variable, documented, no real secrets
```

## What actually happened, day by day

This project was built incrementally, with each stage reviewed, tested
against real data, and verified before the next began — including real bugs
found and fixed along the way rather than papered over:

- **Day 1** — cleaning, partitioning, deterministic metrics. Found: a single
  real order worth 85% of its day's revenue, which shaped every later
  design decision about mean vs. median and anomaly detection.
- **Day 2** — robust anomaly detection, LLM narrative, HTML report. Found
  and fixed: a month-over-month date bug (`day(31)` → invalid "Feb 31")
  invisible on a single test date, surfaced by scanning the full year.
- **Day 3** — cloud deployment design, history caching, Lambda handler, a
  temporary replay mechanism to prove the automation mechanics honestly
  before the real generator existed. Found and fixed: an empty-cache dtype
  bug caught by a strict cold-vs-warm equality test.
- **Day 4** — the real synthetic generator, replacing the temporary replay
  shim entirely; 86 new tests. Found and fixed: a moto test-isolation bug
  (nested mock contexts silently sharing state).
- **Day 5 (this pass)** — full project review, dead code removed (the Day 3
  replay shim), the LLM model identifier and prices reviewed (now configurable through environment variables; see Cost), the
  real project data de-polluted after manual testing had left synthetic days
  mixed into it, and — the most important finding of this pass — my own
  earlier explanation of why `demand_drop` isn't detected was wrong. I
  initially treated it as a generator calibration bug, "fixed" it by
  recalibrating severity, found the fix didn't work, and only then derived
  and verified the actual mathematical cause (above). The incorrect
  intermediate explanation is left in the code comments and in this README
  rather than quietly deleted, because the corrected version is more
  trustworthy with the mistake visible than without it.

- **Maintenance pass after Day 5** — migrated to Synthetic Profile v2.0 (the
  obsolete `get_profile()` API and the v1 profile layout are gone; a v1 profile is
  rejected with a clear error instead of being half-read), found and fixed the
  Excel date swap above, made the profile scale use robust sigmas as documented,
  fixed a quantity-sampling bias that inflated generated revenue ~2.6x, and
  added the first tests for the cleaning layer.
- **Final polish pass** — found and fixed a Windows-only bug in the Lambda handler (S3 keys were built
  with `str(path)`, which uses backslashes on Windows, so the new day was uploaded under keys that no
  `data/...` prefix listing matched and every invocation regenerated 2011-12-11; keys now come from
  `s3_key()` using `as_posix()`, and a test asserts no stored key contains a backslash). Added tests
  for the live Anthropic code path using a stubbed client (it had none), corrected a README claim
  about the default model and price that did not match the code, added CI, sample outputs,
  screenshots and an architecture diagram, and tightened `.gitignore`.

## Deployment status

Built, reviewed, and tested locally (113/132 tests, a full end-to-end run, and
the exact pinned Lambda dependencies verified to install and run correctly in
an isolated Python 3.12 environment matching the Lambda base image). **Not
deployed to AWS** — the development environment has no AWS credentials and no
network path to AWS services, and no Docker daemon to build the container
image. `infra/deploy.md` contains the exact, verification-gated remaining
steps (S3 → IAM → Docker build → ECR → Lambda → schedule), each written
against the final code and ready to run as-is.
