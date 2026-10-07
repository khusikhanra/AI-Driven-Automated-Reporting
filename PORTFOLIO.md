# Case Study: Automated Daily Sales Reporting with Data Engineering, Analytics and AI

**One line:** a pipeline that cleans real e-commerce data, spots unusual days statistically, has an LLM
explain them in plain English, and delivers a report every day, built so that every number is checked,
reproducible and honestly labelled.

<p align="center">
  <img src="docs/images/architecture.svg" alt="Architecture diagram" width="900">
</p>

| | |
|---|---|
| **Domain** | E-commerce analytics and reporting automation |
| **Data** | UCI Online Retail: 541,909 transactions, 1 Dec 2010 to 9 Dec 2011, UK gift retailer |
| **Stack** | Python, pandas, DuckDB, Parquet, matplotlib, pytest, moto; Anthropic API, AWS Lambda, S3, ECR, EventBridge, CloudWatch, Slack |
| **Output** | Self-contained HTML report and Slack message per day, plus an ever-growing synthetic continuation of the data |
| **Status** | Fully working and tested locally; AWS deployment scripted but not deployed (see below) |

## The problem

Operations and finance teams want a short, trustworthy morning summary: what sold, what changed,
and whether anything looks wrong. Typical hand-built versions fail in three ways. The data has silent
quality problems, "anomaly" rules fire on every busy day, and when an AI is added it is allowed to do
arithmetic, so a confident sentence can contain a wrong number.

## Goals

1. Account for every input row. Nothing may disappear silently.
2. Flag unusual days with a method that single extreme orders cannot break.
3. Use an LLM for what it is good at (explaining) and never for what it is not (calculating).
4. Run unattended every day, including when no new real data exists.
5. Make every output reproducible and every test meaningful.

## What was built

| Stage | Description |
|---|---|
| Clean | Rules-based cleaning splits rows into clean sales, cancellations, non-product adjustments (postage, fees), data errors and missing descriptions, and a reconciliation check fails the run if the buckets do not sum to the input |
| Partition | One Parquet file per trading day in a Hive-style layout that DuckDB reads natively |
| Analyse | Deterministic metrics (revenue, orders, median and mean order value, returns, top products) with week-over-week and month-over-month comparison |
| Detect | Robust z-scores (median and MAD) on revenue, orders and returns, plus a concentration check that flags a single product above 50% of the day |
| Explain | The LLM receives only the computed numbers and must answer through a forced structured schema |
| Deliver | HTML report with charts and alerts, Slack message, structured logs |
| Continue | A synthetic generator learns a versioned profile from the real history and produces new, labelled trading days with ground-truth anomaly events |
| Automate | A container Lambda syncs state from S3, advances one day, reports, and persists state only after success |

## Key engineering decisions

**Median over mean, robust over standard.** On 9 Dec 2011 a single order of 80,995 units was 85% of the
day's revenue. The mean order value was £4,502.60; the median was £317.38. Mean and standard deviation
are themselves distorted by such days, so the detector uses the median and the median absolute
deviation, which keep the baseline stable. The report also surfaces the concentration separately, because
a revenue "spike" caused by one order is a different business story from broad demand.

**The model never touches the arithmetic.** All figures are computed before the LLM is called. The prompt
carries only those results (never raw rows), the system prompt forbids inventing or changing numbers, and
the response must arrive through a tool-use schema whose fields are validated. A free, deterministic mock
mode exists, so the full pipeline, tests and CI run without a key or any cost.

**Synthetic data that behaves like the real thing, and says so.** To keep the automation running without
a live feed, the generator samples from a statistical profile of the real data and sends each generated
day through the same cleaning rules. Every day carries a manifest (seed, event, profile fingerprint,
reconciliation) and every report shows a "Simulated data" banner. The same seed and date always produce
identical output.

**Fail loudly, write nothing partial.** A wrong-version or tampered profile is rejected, not silently
rebuilt. A failed generation removes any partial output. State is uploaded to S3 only after the report
succeeds, so a failed run cannot corrupt the next one.

**Document limits instead of tuning them away.** One injected event type (a demand drop) cannot be
detected at the chosen threshold on this data. Rather than quietly retune the generator, the cause is
proven (order count cannot fall below 1, so the most extreme drop reaches only |z| = 2.94 against a
threshold of 3.5) and written down.

## Problems found and what they taught

| Finding | Root cause | Fix and lesson |
|---|---|---|
| **Wrong dates in the raw Excel file** | Day and month were swapped in every datetime-typed cell, in 2011 as well as 2010, while text-typed cells were fine. An earlier date-range heuristic missed the 2011 dates, so cleaned data peaked on the wrong day | Repair at load time, where cell types are visible. Verified against an independent CSV release: all 232,959 cells match. Lesson: check data against a second source, and repair where the information still exists |
| **Lambda regenerated the same day forever on Windows** | S3 keys were built with `str(path)`, which uses backslashes on Windows, so uploaded state was invisible to later prefix listings. Linux never showed it | Keys are built with `as_posix()`; a test asserts no stored key contains a backslash. Lesson: portability bugs hide in string-ified paths, and the CI platform is not the user's platform |
| **Generated revenue 2.6x too high** | Per-product quantity summaries included each product's maximum at fixed intervals, over-weighting the largest orders | Quantile-midpoint summaries; revenue now tracks the real distribution |
| **Event severity measured on the wrong scale** | The profile used standard deviation while the detector uses MAD | Both now use robust sigma, so "5 sigma" means the same thing everywhere |
| **The live LLM path had no tests** | Every test forced mock mode | Added tests with a stubbed client for forced tool use, incomplete responses, missing key and SDK, and cost arithmetic, then confirmed they fail when the code is deliberately broken |
| **Documentation drifted from the code** | A claimed model and price change was never made in the code | Docs now match the code, and unverifiable price claims were removed |

## Results

- 541,909 input rows fully reconciled; revenue of £10,272,118.87 independently recomputed and matched to
  the penny.
- 27 of 305 days flagged (8.9%) on the real data; the three detectable injected event types were flagged in
  8 of 8 trials.
- Baseline history for anomaly detection rebuilt in 0.3 s instead of 127 s.
- 132 automated tests passing on Python 3.12 (Linux); an earlier 113-test version of the suite also
  passed on Python 3.14 on Windows.

## AI, automation, analytics and cloud: how each is used

| Component | Role | Notes |
|---|---|---|
| **AI** | Turns computed results into a short narrative with a recommendation | Anthropic Claude via forced tool use; mock mode default; cost estimated per run; live path tested with a stub only |
| **Automation** | Daily unattended run: generate, analyse, report, deliver | Scheduler triggers a container Lambda; explicit-date regeneration for reruns; structured JSON logs |
| **Analytics** | Metrics, comparisons, anomaly and concentration detection | DuckDB over partitioned Parquet; robust statistics; history cache |
| **Cloud** | State and execution | S3 for partitions, profile, cache, manifests and reports; ECR plus Lambda for compute; CloudWatch logs; Slack for delivery |

### Honest status of the cloud part

The AWS pieces are designed, containerised and tested against a mocked S3 (including fresh-container
starts that rely only on persisted state), and the deployment guide is written against the final code.
They have **not** been deployed: the development environment had no AWS credentials and no Docker
daemon. The live Anthropic call has likewise not been run against the real API. The next step is to
follow `infra/deploy.md` in a real account.

## What would come next

1. Deploy to AWS and add alarms on failed runs.
2. Move the Slack webhook and API key to Secrets Manager.
3. Redesign the detector with asymmetric or log-scale thresholds so demand drops become detectable.
4. Add a small web dashboard that reads the same partitions.
5. Run the dataset-dependent tests in CI by fetching the dataset in the workflow.

## Skills demonstrated

Data cleaning and quality assurance, reconciliation design, Parquet and DuckDB analytics, robust
statistics and anomaly detection, LLM integration with structured output and cost control, synthetic data
generation and reproducibility, serverless and container deployment on AWS, test design (failure paths,
mocked cloud services, deliberately broken code to prove tests can fail), cross-platform debugging, and technical writing.

## Repository guide

- [`README.md`](README.md): overview, quick start, results
- [`docs/ENGINEERING_LOG.md`](docs/ENGINEERING_LOG.md): detailed build history and rationale
- [`docs/samples/`](docs/samples): real report, manifest and stage outputs
- [`infra/deploy.md`](infra/deploy.md): AWS deployment steps
