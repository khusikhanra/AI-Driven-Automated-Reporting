"""
Tests for the AI narrative layer (pipeline/narrative.py).

Neither an API key nor network access is needed: the live path is exercised with
a stub ``anthropic`` module injected into ``sys.modules``. Inputs come from the
committed sample outputs in docs/samples, so these tests also run without the
raw dataset (for example in CI).
"""

import json
import sys
import types
from pathlib import Path

import pytest

from pipeline import narrative

SAMPLES = Path(__file__).resolve().parent.parent / "docs" / "samples"
REQUIRED = {"summary", "key_drivers", "anomaly_explanation", "recommendation"}


@pytest.fixture(scope="module")
def inputs():
    metrics = json.loads((SAMPLES / "metrics_2011-12-09.json").read_text(encoding="utf-8"))
    anomalies = json.loads((SAMPLES / "anomalies_2011-12-09.json").read_text(encoding="utf-8"))
    return metrics, anomalies


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("USE_MOCK_LLM", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def install_fake_anthropic(monkeypatch, content, usage=(1000, 200)):
    """Stub the anthropic SDK; returns the list that records every messages.create call."""
    calls = []

    class Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(
                content=content,
                usage=types.SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1]),
            )

    class Anthropic:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.messages = Messages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=Anthropic))
    return calls


def tool_block(**fields):
    return types.SimpleNamespace(type="tool_use", name="submit_report_narrative", input=fields)


GOOD = dict(summary="s", key_drivers=["d"], anomaly_explanation="a", recommendation="r")


class TestMockMode:
    def test_default_is_mock_free_and_complete(self, inputs):
        out = narrative.generate_narrative(*inputs)
        assert out["cost"] == {"mode": "mock", "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        assert REQUIRED <= set(out["narrative"])

    def test_mock_is_deterministic(self, inputs):
        a = narrative.generate_narrative(*inputs)["narrative"]
        b = narrative.generate_narrative(*inputs)["narrative"]
        assert a == b

    def test_mock_narrative_is_grounded_in_the_supplied_numbers(self, inputs):
        metrics, _ = inputs
        text = json.dumps(narrative.generate_narrative(*inputs)["narrative"], ensure_ascii=False)
        assert f"{metrics['revenue']:,.2f}" in text and str(metrics["order_count"]) in text

    @pytest.mark.parametrize("value", ["false", "0", "no", "off", " FALSE "])
    def test_env_values_that_switch_to_live(self, inputs, monkeypatch, value):
        monkeypatch.setenv("USE_MOCK_LLM", value)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):   # live mode was selected
            narrative.generate_narrative(*inputs)

    @pytest.mark.parametrize("value", ["true", "1", "yes", ""])
    def test_env_values_that_keep_mock(self, inputs, monkeypatch, value):
        monkeypatch.setenv("USE_MOCK_LLM", value)
        assert narrative.generate_narrative(*inputs)["cost"]["mode"] == "mock"


class TestLiveMode:
    def test_structured_output_is_forced_and_returned(self, inputs, monkeypatch):
        calls = install_fake_anthropic(monkeypatch, [tool_block(**GOOD)])
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")
        out = narrative.generate_narrative(*inputs, use_mock=False)

        assert out["narrative"] == GOOD and out["cost"]["mode"] == "live"
        (call,) = calls
        assert call["tool_choice"] == {"type": "tool", "name": "submit_report_narrative"}
        assert call["tools"][0]["name"] == "submit_report_narrative"
        assert call["system"] == narrative.SYSTEM_PROMPT and "Never invent a number" in call["system"]

    def test_model_only_sees_precomputed_numbers_not_raw_rows(self, inputs, monkeypatch):
        calls = install_fake_anthropic(monkeypatch, [tool_block(**GOOD)])
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")
        narrative.generate_narrative(*inputs, use_mock=False)
        prompt = calls[0]["messages"][0]["content"]
        assert f"{inputs[0]['revenue']:,.2f}" in prompt or str(inputs[0]["revenue"]) in prompt
        assert "InvoiceNo" not in prompt and "CustomerID" not in prompt

    def test_cost_is_computed_from_token_usage(self, inputs, monkeypatch):
        install_fake_anthropic(monkeypatch, [tool_block(**GOOD)], usage=(2_000_000, 1_000_000))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")
        monkeypatch.setattr(narrative, "PRICE_PER_MTOK_INPUT", 3.0)
        monkeypatch.setattr(narrative, "PRICE_PER_MTOK_OUTPUT", 15.0)
        cost = narrative.generate_narrative(*inputs, use_mock=False)["cost"]
        assert cost["input_tokens"] == 2_000_000 and cost["output_tokens"] == 1_000_000
        assert cost["cost_usd"] == pytest.approx(2 * 3.0 + 1 * 15.0)

    def test_incomplete_narrative_is_rejected(self, inputs, monkeypatch):
        install_fake_anthropic(monkeypatch, [tool_block(summary="only a summary")])
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")
        with pytest.raises(RuntimeError, match="incomplete.*recommendation"):
            narrative.generate_narrative(*inputs, use_mock=False)

    def test_response_without_tool_call_is_rejected(self, inputs, monkeypatch):
        install_fake_anthropic(monkeypatch, [types.SimpleNamespace(type="text", text="free text")])
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")
        with pytest.raises(RuntimeError, match="no structured"):
            narrative.generate_narrative(*inputs, use_mock=False)

    def test_missing_api_key_is_a_clear_error(self, inputs):
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            narrative.generate_narrative(*inputs, use_mock=False)

    def test_missing_sdk_is_a_clear_error(self, inputs, monkeypatch):
        monkeypatch.setitem(sys.modules, "anthropic", None)      # makes `import anthropic` fail
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")
        with pytest.raises(RuntimeError, match="pip install anthropic"):
            narrative.generate_narrative(*inputs, use_mock=False)
