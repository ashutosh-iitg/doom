"""The cascade's routing policy — what escalates, what doesn't, and who decides.

Mocked throughout. The screen's scoring path has never run against a live
model (no vLLM server), so these pin the routing logic given a score, not the
score itself.
"""

from unittest.mock import patch

import pytest
from emissary import ChoiceResult, ProviderError, parse_spec

from doom import Cascade, Exchange, Screen, Verdict
from doom.cascade import CascadeVerdict
from doom.constitutions.cbrn_example import CBRN_EXAMPLE

EXCHANGE = Exchange.of(user="u", assistant="a")


class _StubScreen:
    def __init__(self, score: float):
        self._score = score
        self.calls = 0

    def score(self, exchange, constitution) -> float:
        self.calls += 1
        return self._score


class _StubJudge:
    def __init__(self, flagged: bool = True):
        self._flagged = flagged
        self.calls = 0

    def judge(self, exchange, constitution) -> Verdict:
        self.calls += 1
        return Verdict(flagged=self._flagged, reasoning="judged", rule_ids=("no-cbrn-uplift",))


def _cascade(score, *, flagged=True, threshold=0.15):
    screen, judge = _StubScreen(score), _StubJudge(flagged)
    return Cascade(screen, judge, CBRN_EXAMPLE, threshold=threshold), screen, judge


def test_a_low_score_never_reaches_the_expensive_judge():
    """The entire cost saving: the second stage does not run."""
    cascade, screen, judge = _cascade(0.02)

    out = cascade.evaluate(EXCHANGE)

    assert judge.calls == 0
    assert screen.calls == 1
    assert out.escalated is False
    assert out.flagged is False


def test_a_high_score_escalates_and_the_judge_decides():
    cascade, _, judge = _cascade(0.9, flagged=True)

    out = cascade.evaluate(EXCHANGE)

    assert judge.calls == 1
    assert out.escalated is True
    assert out.flagged is True
    assert out.verdict.rule_ids == ("no-cbrn-uplift",)


def test_an_escalated_exchange_the_judge_clears_is_not_flagged():
    """Escalation is not a verdict — the screen only buys a second look, and
    the judge is free to clear what it sees."""
    cascade, _, judge = _cascade(0.9, flagged=False)

    out = cascade.evaluate(EXCHANGE)

    assert judge.calls == 1
    assert out.escalated is True
    assert out.flagged is False


def test_the_threshold_is_inclusive_at_its_boundary():
    cascade, _, judge = _cascade(0.15, threshold=0.15)

    assert cascade.evaluate(EXCHANGE).escalated is True
    assert judge.calls == 1


def test_lowering_the_threshold_escalates_more_traffic():
    """The compute/robustness knob: same exchange, same score, different route."""
    strict, _, strict_judge = _cascade(0.2, threshold=0.5)
    lenient, _, lenient_judge = _cascade(0.2, threshold=0.1)

    assert strict.evaluate(EXCHANGE).escalated is False
    assert lenient.evaluate(EXCHANGE).escalated is True
    assert (strict_judge.calls, lenient_judge.calls) == (0, 1)


def test_the_score_is_kept_on_both_routes_for_auditing():
    """Tuning the threshold requires seeing the scores of exchanges that were
    screened out, not just the ones that escalated."""
    screened_out, _, _ = _cascade(0.03)
    escalated, _, _ = _cascade(0.77)

    assert screened_out.evaluate(EXCHANGE).score == 0.03
    assert escalated.evaluate(EXCHANGE).score == 0.77


def test_a_screened_out_result_is_distinguishable_from_a_judge_clearance():
    """Both are `flagged=False`; only `escalated` says whether a judge ever
    looked. Flag-rate auditing depends on telling them apart."""
    screened_out = _cascade(0.01)[0].evaluate(EXCHANGE)
    cleared = _cascade(0.9, flagged=False)[0].evaluate(EXCHANGE)

    assert (screened_out.flagged, cleared.flagged) == (False, False)
    assert (screened_out.escalated, cleared.escalated) == (False, True)


class TestScreen:
    def _screen(self):
        return Screen(parse_spec("vllm:qwen3-8b"))

    def test_the_screen_returns_the_flag_probability(self):
        result = ChoiceResult(
            probabilities={"SAFE": 0.3, "FLAG": 0.7},
            provider="vllm",
            model="qwen3-8b",
            input_tokens=100,
            output_tokens=1,
            cached_input_tokens=0,
        )

        with patch("emissary.call_choice", return_value=result):
            assert self._screen().score(EXCHANGE, CBRN_EXAMPLE) == 0.7

    def test_the_constitution_is_cache_marked_and_the_exchange_is_not(self):
        result = ChoiceResult({"SAFE": 1.0, "FLAG": 0.0}, "vllm", "m", 1, 1, 0)

        with patch("emissary.call_choice", return_value=result) as called:
            self._screen().score(EXCHANGE, CBRN_EXAMPLE)

        blocks = called.call_args.kwargs["blocks"]
        assert blocks[0]["cache"] is True
        assert blocks[1]["cache"] is False
        assert called.call_args.kwargs["labels"] == ["SAFE", "FLAG"]

    def test_the_screen_reads_its_provider_from_the_environment(self, monkeypatch):
        """Anthropic exposes no logprobs, so the screen cannot share the
        judge's provider — it gets its own variable."""
        monkeypatch.setenv("DOOM_SCREEN_PROVIDER", "vllm:qwen3-8b")

        assert str(Screen().spec) == "vllm:qwen3-8b"

    def test_an_unconfigured_screen_fails_with_an_actionable_message(self, monkeypatch):
        """There is no sensible default model for an arbitrary local server,
        so refusing to guess one is the honest behaviour — but the error has
        to say what to set."""
        monkeypatch.delenv("DOOM_SCREEN_PROVIDER", raising=False)

        with pytest.raises(ProviderError, match="name one as 'vllm:<model-id>'"):
            Screen()


def test_cascade_verdict_flagged_mirrors_the_underlying_verdict():
    verdict = CascadeVerdict(Verdict(flagged=True, reasoning="r"), score=0.9, escalated=True)

    assert verdict.flagged is True
