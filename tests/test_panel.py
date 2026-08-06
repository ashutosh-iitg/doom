"""`Panel`'s v0 policy: flagged if any judge flags."""

from doom import Exchange, Panel, Verdict
from doom.constitutions.cbrn_example import CBRN_EXAMPLE


class _StubJudge:
    def __init__(self, verdict: Verdict):
        self._verdict = verdict

    def judge(self, exchange, constitution) -> Verdict:
        return self._verdict


def _panel(*flagged: bool) -> Panel:
    judges = tuple(_StubJudge(Verdict(flagged=f, reasoning="stub", rule_ids=())) for f in flagged)
    return Panel(judges=judges, constitution=CBRN_EXAMPLE)


def test_flagged_is_false_when_no_judge_flags():
    panel = _panel(False, False)

    assert panel.flagged(Exchange.of(user="u", assistant="a")) is False


def test_flagged_is_true_if_any_judge_flags():
    panel = _panel(False, True, False)

    assert panel.flagged(Exchange.of(user="u", assistant="a")) is True


def test_evaluate_returns_every_judges_verdict_in_order():
    panel = _panel(True, False, True)

    verdicts = panel.evaluate(Exchange.of(user="u", assistant="a"))

    assert [v.flagged for v in verdicts] == [True, False, True]
