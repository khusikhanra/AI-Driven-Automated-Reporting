
"""
narrative.py

Turns deterministic metrics + anomaly flags into an executive narrative.

Pipeline:

    metrics.py
        ↓
    anomaly_detection.py
        ↓
    narrative.py
        ↓
    report.py

The LLM receives only pre-computed metrics and anomaly information.

It does not receive raw transaction rows and does not perform business
calculations.

Two modes are supported:

1. MOCK MODE
   USE_MOCK_LLM=true

   - No API call
   - No API key required
   - Free and deterministic
   - Useful for testing

2. LIVE MODE
   USE_MOCK_LLM=false
   ANTHROPIC_API_KEY=<your key>

   - Calls Anthropic
   - Uses structured tool output
   - Produces an AI-generated executive narrative
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


# ============================================================================
# LOAD .ENV FOR LOCAL DEVELOPMENT
# ============================================================================

try:
    from dotenv import load_dotenv

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    ENV_FILE = PROJECT_ROOT / ".env"

    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)

except ImportError:
    # python-dotenv is optional in Lambda.
    # Lambda normally receives environment variables directly.
    pass


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL = os.environ.get(
    "ANTHROPIC_MODEL",
    "claude-sonnet-4-5",
)

PRICE_PER_MTOK_INPUT = float(
    os.environ.get(
        "ANTHROPIC_INPUT_PRICE_PER_MTOK",
        "3.00",
    )
)

PRICE_PER_MTOK_OUTPUT = float(
    os.environ.get(
        "ANTHROPIC_OUTPUT_PRICE_PER_MTOK",
        "15.00",
    )
)


# ============================================================================
# STRUCTURED OUTPUT SCHEMA
# ============================================================================

NARRATIVE_TOOL = {
    "name": "submit_report_narrative",
    "description": (
        "Submit the structured executive narrative for the daily "
        "e-commerce report."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": (
                    "A concise 2-3 sentence plain-language summary "
                    "of the day's performance."
                ),
            },
            "key_drivers": {
                "type": "array",
                "items": {
                    "type": "string",
                },
                "description": (
                    "2-4 short bullet points describing what drove "
                    "the day's results. Use only supplied metrics."
                ),
            },
            "anomaly_explanation": {
                "type": "string",
                "description": (
                    "Explain flagged anomalies in plain language. "
                    "If none were flagged, explicitly say so."
                ),
            },
            "recommendation": {
                "type": "string",
                "description": (
                    "One concrete and actionable recommendation "
                    "for a business stakeholder."
                ),
            },
        },
        "required": [
            "summary",
            "key_drivers",
            "anomaly_explanation",
            "recommendation",
        ],
    },
}


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """
You are a retail analytics narrator.

You are given pre-computed, already-correct metrics for one day of an
e-commerce business.

Your job is ONLY to explain the supplied numbers clearly for a business
stakeholder.

Rules:

- Never invent a number.
- Never calculate a new business metric.
- Never change or reinterpret supplied numbers.
- Do not claim a cause unless the supplied data provides evidence for it.
- Use the supplied anomaly and concentration flags when discussing unusual
  performance.
- Prefer median order value when describing a typical order.
- Keep the writing concise and concrete.
- Avoid filler such as "robust", "leverage", "synergy", and "game-changing".
- Write like a professional business analyst.
- If there are no anomalies, explicitly state that no anomalies were flagged.
- Return the answer ONLY through the submit_report_narrative tool.
"""


# ============================================================================
# PROMPT BUILDER
# ============================================================================

def build_user_prompt(metrics: dict, anomalies: dict) -> str:
    """
    Convert deterministic pipeline output into the structured payload
    sent to the LLM.
    """

    payload = {
        "date": metrics.get("date"),
        "revenue": metrics.get("revenue"),
        "order_count": metrics.get("order_count"),
        "units_sold": metrics.get("units_sold"),
        "unique_customers": metrics.get("unique_customers"),
        "mean_aov": metrics.get("aov"),
        "median_order_value": anomalies.get(
            "median_order_value"
        ),
        "returns": {
            "count": metrics.get("return_count"),
            "value": metrics.get("return_value"),
        },
        "top_products": metrics.get(
            "top_products",
            [],
        ),
        "comparisons": metrics.get(
            "comparisons",
            {},
        ),
        "concentration_risk": anomalies.get(
            "concentration_risk",
            {},
        ),
        "statistical_anomalies": anomalies.get(
            "statistical_anomalies",
            [],
        ),
    }

    return (
        "Here are today's deterministic metrics.\n"
        "Do not calculate additional metrics.\n"
        "Call submit_report_narrative with the executive narrative.\n\n"
        + json.dumps(
            payload,
            indent=2,
            default=str,
        )
    )


# ============================================================================
# MOCK NARRATIVE
# ============================================================================

def _mock_response(
    metrics: dict,
    anomalies: dict,
) -> dict:
    """
    Deterministic metrics-grounded response.

    This allows the complete reporting pipeline to be tested without
    an external API call.
    """

    concentration = anomalies.get(
        "concentration_risk",
        {},
    )

    statistical_anomalies = anomalies.get(
        "statistical_anomalies",
        [],
    )

    revenue = metrics.get(
        "revenue",
        0,
    )

    order_count = metrics.get(
        "order_count",
        0,
    )

    median_order_value = anomalies.get(
        "median_order_value",
        0,
    )

    drivers = [
        (
            f"Revenue was £{revenue:,.2f} across "
            f"{order_count} orders."
        )
    ]

    # Concentration driver
    if concentration.get("flagged"):
        product = concentration.get(
            "product",
            "the leading product",
        )

        pct = concentration.get(
            "pct_of_day_revenue",
            0,
        )

        concentration_revenue = concentration.get(
            "revenue",
            0,
        )

        drivers.append(
            f"'{product}' accounted for "
            f"{pct}% of the day's revenue "
            f"(£{concentration_revenue:,.2f})."
        )

    # Top product driver
    top_products = metrics.get(
        "top_products",
        [],
    )

    if top_products:
        top_product = top_products[0].get(
            "product",
            "the top product",
        )

        drivers.append(
            f"Top product by revenue: {top_product}."
        )

    # Anomaly explanation
    if statistical_anomalies:
        explanations = []

        for anomaly in statistical_anomalies:
            metric = anomaly.get(
                "metric",
                "Metric",
            )

            if "robust_zscore" in anomaly:
                zscore = anomaly.get(
                    "robust_zscore",
                    "N/A",
                )

                direction = anomaly.get(
                    "direction",
                    "off",
                )

                explanations.append(
                    f"{metric} was {zscore} "
                    f"standard deviations ({direction} baseline)."
                )

            else:
                note = anomaly.get(
                    "note",
                    "flagged as unusual",
                )

                explanations.append(
                    f"{metric}: {note}"
                )

        anomaly_explanation = " ".join(
            explanations
        )

    else:
        anomaly_explanation = (
            "No statistically significant anomalies "
            "were flagged for this date."
        )

    # Recommendation
    if concentration.get("flagged"):
        product = concentration.get(
            "product",
            "the concentrated product",
        )

        recommendation = (
            f"Review the '{product}' contribution before "
            "treating today's revenue as representative "
            "of typical demand."
        )

    elif statistical_anomalies:
        recommendation = (
            "Review the flagged metrics against recent "
            "operational changes such as promotions, "
            "stock availability, or site issues."
        )

    else:
        recommendation = (
            "No immediate action is indicated; "
            "performance is within the normal range."
        )

    return {
        "summary": (
            f"On {metrics.get('date')}, revenue was "
            f"£{revenue:,.2f} from {order_count} orders "
            f"(median order value "
            f"£{median_order_value:,.2f})."
        ),
        "key_drivers": drivers,
        "anomaly_explanation": anomaly_explanation,
        "recommendation": recommendation,
    }


# ============================================================================
# LIVE ANTHROPIC CALL
# ============================================================================

def _live_response(
    metrics: dict,
    anomalies: dict,
    api_key: str,
) -> dict:
    """
    Call Anthropic and force structured output through tool use.
    """

    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "The anthropic package is not installed.\n"
            "Run:\n"
            "python -m pip install anthropic"
        ) from exc

    client = anthropic.Anthropic(
        api_key=api_key,
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=[
            NARRATIVE_TOOL
        ],
        tool_choice={
            "type": "tool",
            "name": "submit_report_narrative",
        },
        messages=[
            {
                "role": "user",
                "content": build_user_prompt(
                    metrics,
                    anomalies,
                ),
            }
        ],
    )

    # Find structured tool response
    tool_use_block = next(
        (
            block
            for block in response.content
            if getattr(
                block,
                "type",
                None,
            ) == "tool_use"
        ),
        None,
    )

    if tool_use_block is None:
        raise RuntimeError(
            "Anthropic returned no structured "
            "submit_report_narrative tool response."
        )

    narrative = tool_use_block.input

    # Validate required fields
    required_fields = [
        "summary",
        "key_drivers",
        "anomaly_explanation",
        "recommendation",
    ]

    missing = [
        field
        for field in required_fields
        if field not in narrative
    ]

    if missing:
        raise RuntimeError(
            "Anthropic returned an incomplete narrative. "
            f"Missing fields: {missing}"
        )

    # Token usage
    input_tokens = getattr(
        response.usage,
        "input_tokens",
        0,
    )

    output_tokens = getattr(
        response.usage,
        "output_tokens",
        0,
    )

    # Estimated cost
    cost = (
        (input_tokens / 1_000_000)
        * PRICE_PER_MTOK_INPUT
        +
        (output_tokens / 1_000_000)
        * PRICE_PER_MTOK_OUTPUT
    )

    return {
        "narrative": narrative,
        "cost": {
            "mode": "live",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(
                cost,
                6,
            ),
        },
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# ============================================================================
# PUBLIC API
# ============================================================================

def generate_narrative(
    metrics: dict,
    anomalies: dict,
    use_mock: bool | None = None,
) -> dict:
    """
    Generate the report narrative.

    Behavior:

    use_mock=True
        Always use mock mode.

    use_mock=False
        Always use live Anthropic mode.

    use_mock=None
        Read USE_MOCK_LLM from environment.
    """

    api_key = os.environ.get(
        "ANTHROPIC_API_KEY"
    )

    environment_mock = os.environ.get(
        "USE_MOCK_LLM",
        "true",
    ).strip().lower()

    if use_mock is None:
        use_mock = environment_mock not in {
            "false",
            "0",
            "no",
            "off",
        }

    # ------------------------------------------------------------------------
    # MOCK MODE
    # ------------------------------------------------------------------------

    if use_mock:
        narrative = _mock_response(
            metrics,
            anomalies,
        )

        return {
            "narrative": narrative,
            "cost": {
                "mode": "mock",
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            },
            "generated_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

    # ------------------------------------------------------------------------
    # LIVE MODE VALIDATION
    # ------------------------------------------------------------------------

    if not api_key:
        raise RuntimeError(
            "USE_MOCK_LLM=false but ANTHROPIC_API_KEY "
            "is missing.\n\n"
            "For local testing:\n"
            "1. Create .env from .env.example\n"
            "2. Set USE_MOCK_LLM=false\n"
            "3. Set ANTHROPIC_API_KEY=<your key>"
        )

    return _live_response(
        metrics,
        anomalies,
        api_key,
    )


# ============================================================================
# STANDALONE SMOKE TEST
# ============================================================================

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from pipeline import paths
    from pipeline.anomaly_detection import detect
    from pipeline.history_cache import get_history
    from pipeline.metrics import compute_daily_metrics, get_connection
    from pipeline.synthetic_generator import latest_partitioned_date

    root = paths.PROJECT_ROOT
    con = get_connection(paths.partitioned_dir(root))
    history = get_history(con, paths.history_cache_path(root))

    target = latest_partitioned_date(paths.partitioned_dir(root))
    metrics = compute_daily_metrics(con, target)
    anomalies = detect(con, target, history)

    print(json.dumps(generate_narrative(metrics, anomalies), indent=2, ensure_ascii=False, default=str))
