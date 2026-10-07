# Deployment runbook (manual, AWS CLI)

Manual on purpose: documented CLI steps with an explicit verification check after
each beat half-finished Terraform for a project this size. IaC (CDK/Terraform) is
listed as a next step in README.md.

**Status of each step:** everything in this file was written against the final,
tested code (87/87 local tests pass: Day 1-4 pipeline, synthetic generator, and
the Lambda handler against mocked S3/Slack). It was reviewed carefully but **not
executed against real AWS** — the development environment has no AWS credentials
and no network path to AWS. Steps 1-9 below are the exact remaining manual work.
Each has a verification check; do not proceed past one that fails.

Set once:
```bash
export AWS_REGION=eu-west-2            # choose your region
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export BUCKET=sales-reporting-$ACCOUNT_ID
export FN=daily-sales-report
export REPO=daily-sales-report
```

## 1. S3 bucket (private)
```bash
aws s3api create-bucket --bucket $BUCKET --region $AWS_REGION \
  --create-bucket-configuration LocationConstraint=$AWS_REGION
aws s3api put-public-access-block --bucket $BUCKET \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```
(us-east-1: omit `--create-bucket-configuration`.)
Reports are never public; they are shared via time-limited presigned URLs.

## 2. Build local artifacts, then seed the bucket with the full real history
```bash
python pipeline/clean.py && python pipeline/partition.py   # if data/ isn't already built
python pipeline/history_cache.py                           # builds the history cache (~80s)
python pipeline/synthetic_profile.py                        # builds the generator's profile
python infra/seed_s3.py --bucket $BUCKET                    # default cutoff = full history
```
**Verify:** `aws s3 ls s3://$BUCKET/data/ --recursive | wc -l` is ~610+ (305+302
real partitions, 2 cleaned files, 1 profile, 1 history cache);
`aws s3 ls s3://$BUCKET/data/profile/` shows `profile.json`.

The generator refuses to run against partial history (it would otherwise generate
"synthetic" days that overlap real ones), so always seed the full history — the
`--cutoff` flag on `seed_s3.py` exists for testing that safeguard, not for normal use.

## 3. IAM execution role (least privilege)
```bash
cat > trust.json <<EOF
{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
 "Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}
EOF
aws iam create-role --role-name $FN-role --assume-role-policy-document file://trust.json
aws iam attach-role-policy --role-name $FN-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole   # CloudWatch logs only

cat > s3policy.json <<EOF
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":["s3:GetObject","s3:PutObject"],"Resource":"arn:aws:s3:::$BUCKET/*"},
 {"Effect":"Allow","Action":"s3:ListBucket","Resource":"arn:aws:s3:::$BUCKET"}]}
EOF
aws iam put-role-policy --role-name $FN-role --policy-name bucket-access --policy-document file://s3policy.json
```
No `DeleteObject`, no wildcard buckets: the function cannot delete data or touch
any other bucket. **Verify:** `aws iam get-role-policy --role-name $FN-role --policy-name bucket-access`.

## 4. Build the image locally, smoke-test, push to ECR
```bash
docker build -t $REPO .
```
**Verify:** build succeeds. The pinned dependency set in `requirements-lambda.txt`
was confirmed to install cleanly and run the full pipeline correctly in an
isolated Python 3.12 virtualenv (matching the Lambda base image's Python version)
during development — but the actual `docker build` itself was not run (no Docker
in the development sandbox). This is the first real test of the container image;
if it fails on dependency size or a missing wheel, that is the first place to look.

```bash
# Smoke test with the Lambda Runtime Interface Emulator BEFORE pushing to ECR
# (needs your AWS creds in env; this run WILL write a real synthetic day + report
# to your bucket, same as a real invocation):
docker run --rm -p 9000:8080 -e BUCKET_NAME=$BUCKET \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN -e AWS_REGION $REPO &
curl -s -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{}'
```
**Verify:** JSON response with `"status": "ok"`, `"data_source": "synthetic"`,
and a `report_key`. First call should report `2011-12-11` (Dec 10 is a closed
Saturday, skipped automatically).

```bash
aws ecr create-repository --repository-name $REPO --region $AWS_REGION
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
docker tag $REPO:latest $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
```

## 5. Create the Lambda function
```bash
aws lambda create-function --function-name $FN --package-type Image \
  --code ImageUri=$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest \
  --role arn:aws:iam::$ACCOUNT_ID:role/$FN-role \
  --memory-size 1024 --timeout 120 --region $AWS_REGION \
  --environment "Variables={BUCKET_NAME=$BUCKET,USE_MOCK_LLM=true,SLACK_WEBHOOK_URL=<your-slack-webhook>}"
```
Sizing: 1024MB also buys CPU for pandas/DuckDB. Timeout is 120s: with the history
cache, a steady-state run is a few seconds of compute, but downloading ~600 S3
objects plus cold start need headroom. Tighten after measuring real durations in
CloudWatch (`run_complete` log lines report `llm_cost_usd`; add timing if you
want to tune this further).

Optional environment variables, add to the `Variables={...}` map above as needed:
- `ANTHROPIC_API_KEY=<key>` + `USE_MOCK_LLM=false` — use the real LLM instead of
  the free, metrics-grounded mock narrative. Costs real money per run (see
  README.md "Cost" section); `USE_MOCK_LLM=true` is the safe default.
- `GENERATOR_SEED=<int>` / `GENERATOR_EVENT=<name>` — force a specific
  reproducible synthetic day instead of a fresh live one each run. Useful for a
  demo (e.g. force `bulk_order` to always show the concentration-risk alert) but
  NOT for normal operation, since it would generate the SAME day forever rather
  than advancing. Leave both unset for real daily automation.

Env vars are encrypted at rest but visible to anyone with
`lambda:GetFunctionConfiguration`; Secrets Manager is the production-grade next
step for `ANTHROPIC_API_KEY` and the Slack webhook URL.

## 6. Manual invoke test (BEFORE scheduling)
```bash
aws lambda invoke --function-name $FN --cli-binary-format raw-in-base64-out \
  --payload '{}' out.json --region $AWS_REGION && cat out.json
aws logs tail /aws/lambda/$FN --since 10m --region $AWS_REGION
```
**Verify all of:** `out.json` has `"status":"ok"`; `aws s3 ls s3://$BUCKET/reports/`
lists the report; Slack received the message (tagged `(simulated data)`); the
presigned link opens the report and shows the "Simulated data" banner with a
seed number; the log shows `run_complete` with duration well under the timeout.
Invoke once more and confirm the date advances by one trading day (state persisted
via S3, not local to the container) — this is the real-world version of what
`tests/test_lambda_handler.py::test_state_persists_across_invocations...` and
`test_cold_start_reuses_persisted_history_and_profile` already proved locally
against mocked S3.

## 7. Failure-path test
```bash
aws lambda invoke --function-name $FN --cli-binary-format raw-in-base64-out \
  --payload '{"target_date":"2099-01-01"}' out.json --region $AWS_REGION
```
**Verify:** invocation reports a function error (date too far ahead of the
current data), Slack shows a `:x: ... FAILED` alert, and S3 contents (partition
count, history cache) are unchanged from before this call.

## 8. Daily schedule (EventBridge Scheduler)
```bash
cat > sched-trust.json <<EOF
{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
 "Principal":{"Service":"scheduler.amazonaws.com"},"Action":"sts:AssumeRole"}]}
EOF
aws iam create-role --role-name $FN-scheduler-role --assume-role-policy-document file://sched-trust.json
aws iam put-role-policy --role-name $FN-scheduler-role --policy-name invoke --policy-document \
 "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"lambda:InvokeFunction\",\"Resource\":\"arn:aws:lambda:$AWS_REGION:$ACCOUNT_ID:function:$FN\"}]}"

aws scheduler create-schedule --name $FN-daily --region $AWS_REGION \
  --schedule-expression "cron(0 7 * * ? *)" --schedule-expression-timezone "Europe/London" \
  --flexible-time-window Mode=OFF \
  --target "{\"Arn\":\"arn:aws:lambda:$AWS_REGION:$ACCOUNT_ID:function:$FN\",\"RoleArn\":\"arn:aws:iam::$ACCOUNT_ID:role/$FN-scheduler-role\",\"Input\":\"{}\"}"
```
**Verify:** temporarily set the cron a few minutes ahead and confirm a scheduled
invocation appears in the logs, then restore the daily time.

The synthetic generator (unlike the retired Day 3 replay shim) never exhausts — it produces a new trading day indefinitely, so this
schedule can run forever with no manual intervention once set up. The only
ongoing operational costs to watch are LLM calls (if `USE_MOCK_LLM=false`) and
S3/Lambda usage, both of which stay within AWS free-tier limits at this scale for
a long time (see README.md "Cost").

## 9. (Optional) Secrets Manager for API key and webhook
Not done in this pass (documented as a next step rather than built, consistent
with the project's original scope decisions). To upgrade: store
`ANTHROPIC_API_KEY` and `SLACK_WEBHOOK_URL` in Secrets Manager, grant the
execution role `secretsmanager:GetSecretValue` on those two secret ARNs only,
and have `lambda_handler.py` fetch them at the top of `handler()` instead of
reading `os.environ` directly for those two values.

## Cleanup
Delete the schedule, Lambda, ECR repo (`--force`), IAM roles, and empty+delete the
bucket to stop any charges.
