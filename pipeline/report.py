"""
report.py — Renders one day's metrics + anomalies + narrative into a
single self-contained HTML file (chart embedded as a base64 PNG, no
external assets). Self-containment matters because Day 3 delivers this
file via Slack/email, where it needs to open correctly with zero
dependency on anything else being reachable.
"""

import base64
import io
from pathlib import Path
from datetime import date, timedelta
import sys
import json

import matplotlib
matplotlib.use("Agg")  # no display backend needed/available in this environment
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.metrics import get_connection, compute_daily_metrics
from pipeline.anomaly_detection import detect
from pipeline.history_cache import get_history
from pipeline.narrative import generate_narrative
from pipeline.paths import manifest_path

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Daily Sales Report — {date}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
          max-width: 760px; margin: 40px auto; color: #1a1a1a; line-height: 1.5; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-bottom: 28px; font-size: 14px; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 28px; }}
  .metric-card {{ background: #f6f6f7; border-radius: 8px; padding: 14px; }}
  .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.03em; }}
  .metric-value {{ font-size: 20px; font-weight: 600; margin-top: 4px; }}
  .metric-delta {{ font-size: 12px; margin-top: 2px; }}
  .delta-up {{ color: #15803d; }}
  .delta-down {{ color: #b91c1c; }}
  .section {{ margin-bottom: 28px; }}
  .section h2 {{ font-size: 15px; text-transform: uppercase; letter-spacing: 0.04em;
                 color: #444; border-bottom: 1px solid #e5e5e5; padding-bottom: 6px; }}
  .alert {{ background: #fef3c7; border-left: 4px solid #d97706; padding: 12px 16px;
            border-radius: 4px; margin-bottom: 10px; font-size: 14px; }}
  .alert-title {{ font-weight: 600; margin-bottom: 4px; }}
  ul {{ padding-left: 20px; }}
  li {{ margin-bottom: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #eee; }}
  .sim-banner {{ background: #e0e7ff; border-left: 4px solid #4f46e5; color: #312e81;
                 padding: 10px 14px; border-radius: 4px; margin-bottom: 20px; font-size: 13px; }}
  .footer {{ font-size: 12px; color: #999; margin-top: 40px; border-top: 1px solid #eee; padding-top: 12px; }}
  img {{ max-width: 100%; border-radius: 8px; }}
</style>
</head>
<body>
  <h1>Daily Sales Report</h1>
  <div class="subtitle">{date} · Generated {generated_at}</div>
  {data_banner}

  <div class="metrics-grid">
    <div class="metric-card">
      <div class="metric-label">Revenue</div>
      <div class="metric-value">£{revenue:,.2f}</div>
      {revenue_delta}
    </div>
    <div class="metric-card">
      <div class="metric-label">Orders</div>
      <div class="metric-value">{order_count}</div>
      {order_delta}
    </div>
    <div class="metric-card">
      <div class="metric-label">Median Order</div>
      <div class="metric-value">£{median_order_value:,.2f}</div>
      <div class="metric-delta">mean: £{mean_aov:,.2f}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Unique Customers</div>
      <div class="metric-value">{unique_customers}</div>
    </div>
  </div>

  <div class="section">
    <h2>Summary</h2>
    <p>{summary}</p>
  </div>

  {alerts_html}

  <div class="section">
    <h2>Key Drivers</h2>
    <ul>{drivers_html}</ul>
  </div>

  <div class="section">
    <h2>Revenue — Trailing 30 Days</h2>
    <img src="data:image/png;base64,{chart_b64}" alt="Revenue trend chart">
  </div>

  <div class="section">
    <h2>Top Products Today</h2>
    <table>
      <tr><th>Product</th><th>Revenue</th></tr>
      {products_html}
    </table>
  </div>

  <div class="section">
    <h2>Recommendation</h2>
    <p>{recommendation}</p>
  </div>

  <div class="footer">
    Automated report · Narrative mode: {narrative_mode}
    {cost_line}
  </div>
</body>
</html>
"""


def render_chart(history, target_date: date) -> str:
    window_start = target_date - timedelta(days=30)
    window = history[(history["date"] >= window_start) & (history["date"] <= target_date)].sort_values("date")

    fig, ax = plt.subplots(figsize=(7, 2.8), dpi=130)
    ax.plot(window["date"], window["revenue"], color="#2563eb", linewidth=1.8)
    ax.fill_between(window["date"], window["revenue"], color="#2563eb", alpha=0.08)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel("Revenue (£)", fontsize=9)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def delta_html(pct, label) -> str:
    if pct is None:
        return f'<div class="metric-delta">{label}: n/a</div>'
    cls = "delta-up" if pct >= 0 else "delta-down"
    sign = "+" if pct >= 0 else ""
    return f'<div class="metric-delta {cls}">{label}: {sign}{pct}%</div>'


def build_report_with_context(target_date: date, root: Path):
    """Returns (html, context). context carries headline facts for delivery (Slack)."""
    con = get_connection(root / "data" / "partitioned")
    cache_path = root / "data" / "history_cache" / "history.parquet"
    history = get_history(con, cache_path)

    metrics = compute_daily_metrics(con, target_date)
    if not metrics.get("has_data"):
        raise ValueError(f"No data for {target_date}")

    anomalies = detect(con, target_date, history)
    result = generate_narrative(metrics, anomalies)
    narrative = result["narrative"]
    cost = result["cost"]

    # Synthetic days are labelled honestly. Only the seed is shown: the injected
    # event (ground truth) stays out of the report so it cannot leak into it.
    man_file = manifest_path(root, target_date)
    is_synthetic = man_file.exists()
    data_banner = ""
    if is_synthetic:
        seed = json.loads(man_file.read_text(encoding="utf-8")).get("seed")
        data_banner = (f'<div class="sim-banner"><b>Simulated data.</b> This day was generated by the '
                       f'synthetic-data generator (seed {seed}), continuing the real 2010-2011 series. '
                       f'It is not real trading.</div>')

    alerts = []
    conc = anomalies.get("concentration_risk", {})
    if conc.get("flagged"):
        alerts.append(
            f'<div class="alert"><div class="alert-title">⚠ Concentration risk</div>'
            f'"{conc["product"]}" made up {conc["pct_of_day_revenue"]}% of today\'s revenue.</div>'
        )
    for a in anomalies.get("statistical_anomalies", []):
        if a["metric"] == "return_rate":
            alerts.append(
                f'<div class="alert"><div class="alert-title">⚠ Return rate spike</div>'
                f'Return rate ({a["value"]:.1%}) is well above the trailing median ({a["baseline_median"]:.1%}).</div>'
            )
        elif "robust_zscore" in a:
            alerts.append(
                f'<div class="alert"><div class="alert-title">⚠ Unusual {a["metric"]}</div>'
                f'{a["metric"].replace("_", " ").title()} is {abs(a["robust_zscore"])} robust standard '
                f'deviations {a["direction"]} the trailing baseline.</div>'
            )
    alerts_html = "\n".join(alerts)

    drivers_html = "".join(f"<li>{d}</li>" for d in narrative["key_drivers"])
    products_html = "".join(
        f'<tr><td>{p["product"]}</td><td>£{p["revenue"]:,.2f}</td></tr>'
        for p in metrics["top_products"]
    )

    chart_b64 = render_chart(history, target_date)

    cost_line = ""
    if cost["mode"] == "live":
        cost_line = f' · {cost["input_tokens"]}+{cost["output_tokens"]} tokens · ${cost["cost_usd"]:.6f}'

    html = TEMPLATE.format(
        date=metrics["date"],
        data_banner=data_banner,
        generated_at=result["generated_at"],
        revenue=metrics["revenue"],
        revenue_delta=delta_html(metrics["comparisons"]["wow"]["revenue_pct_change"], "vs last week"),
        order_count=metrics["order_count"],
        order_delta=delta_html(metrics["comparisons"]["wow"]["order_count_pct_change"], "vs last week"),
        median_order_value=anomalies.get("median_order_value", 0),
        mean_aov=metrics["aov"],
        unique_customers=metrics["unique_customers"],
        summary=narrative["summary"],
        alerts_html=alerts_html,
        drivers_html=drivers_html,
        chart_b64=chart_b64,
        products_html=products_html,
        recommendation=narrative["recommendation"],
        narrative_mode=cost["mode"],
        cost_line=cost_line,
    )
    context = {
        "date": metrics["date"],
        "revenue": metrics["revenue"],
        "order_count": metrics["order_count"],
        "median_order_value": anomalies.get("median_order_value", 0),
        "summary": narrative["summary"],
        "recommendation": narrative["recommendation"],
        "concentration_flagged": bool(conc.get("flagged")),
        "anomaly_metrics": [a["metric"] for a in anomalies.get("statistical_anomalies", [])],
        "data_source": "synthetic" if is_synthetic else "historical",
        "narrative_mode": cost["mode"],
        "llm_cost_usd": cost["cost_usd"],
    }
    return html, context


def build_report(target_date: date, root: Path) -> str:
    return build_report_with_context(target_date, root)[0]


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    target = date(2011, 12, 9)
    html = build_report(target, root)

    out_dir = root / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"report_{target}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Report written to {out_path}")
