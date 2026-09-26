"""`ExchangeClassifier` against a mocked `emissary` call — no network."""

from unittest.mock import patch

from emissary.llm import CallResult

from doom import Exchange, ExchangeClassifier
from doom.constitutions.cbrn_example import CBRN_EXAMPLE


def _result(payload):
    return CallResult(payload, "anthropic", "claude-opus-5", 10, 5, 0)


def test_a_flagged_verdict_is_parsed_from_the_tool_payload():
    payload = {
        "flagged": True,
        "reasoning": "The exchange reconstructs synthesis instructions.",
        "rule_ids": ["no-reconstruction"],
    }
    exchange = Exchange.of(user="u", assistant="a")

    with patch("emissary.llm.call_tool", return_value=_result(payload)):
        verdict = ExchangeClassifier().judge(exchange, CBRN_EXAMPLE)

    assert verdict.flagged is True
    assert verdict.rule_ids == ("no-reconstruction",)


def test_an_unflagged_verdict_carries_no_rule_ids():
    payload = {"flagged": False, "reasoning": "Purely historical discussion.", "rule_ids": []}
    exchange = Exchange.of(user="u", assistant="a")

    with patch("emissary.llm.call_tool", return_value=_result(payload)):
        verdict = ExchangeClassifier().judge(exchange, CBRN_EXAMPLE)

    assert verdict.flagged is False
    assert verdict.rule_ids == ()


def test_the_constitution_is_cache_marked_and_the_exchange_is_not():
    payload = {"flagged": False, "reasoning": "n/a", "rule_ids": []}
    exchange = Exchange.of(user="u", assistant="a")

    with patch("emissary.llm.call_tool", return_value=_result(payload)) as called:
        ExchangeClassifier().judge(exchange, CBRN_EXAMPLE)

    blocks = called.call_args.kwargs["blocks"]
    assert blocks[0].cache is True
    assert CBRN_EXAMPLE.name in blocks[0].text
    assert blocks[1].cache is False
    assert "[user]" in blocks[1].text


def test_a_judge_defaults_to_the_configured_provider(monkeypatch):
    monkeypatch.delenv("DOOM_JUDGE_PROVIDER", raising=False)

    assert str(ExchangeClassifier().spec) == "anthropic:claude-opus-5"

    monkeypatch.setenv("DOOM_JUDGE_PROVIDER", "vllm:qwen3-8b")
    assert str(ExchangeClassifier().spec) == "vllm:qwen3-8b"
