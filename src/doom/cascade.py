"""The paper's §4 two-stage cascade: screen everything, escalate the few.

A cheap `Screen` scores every exchange; only those at or above `threshold`
reach the expensive `ExchangeClassifier`. The paper reports ~10% of traffic
escalating, with the first stage accounting for most of the remaining cost —
the saving comes from the second stage almost never running, not from the
first stage being free.

`threshold` is the compute/robustness knob (the paper's Figure 3c): lower it
and more traffic escalates, raising both cost and recall. It is a knob, not a
calibration — turning it honestly requires a labelled set to measure the
escalation rate and missed-violation rate against.
"""

from dataclasses import dataclass

from .constitution import Constitution
from .exchange import Exchange
from .judge import ExchangeClassifier
from .screen import Screen
from .verdict import Verdict

DEFAULT_THRESHOLD = 0.15
"""Deliberately low, per the paper's §4 calibration: the first stage's
threshold is set so the vast majority of known-bad examples are flagged,
accepting a high false-positive rate because flagged traffic is escalated
rather than refused. **Unvalidated** — a real value comes from measuring
against labelled exchanges, not from taste."""


@dataclass(frozen=True)
class CascadeVerdict:
    """What the cascade decided, and how it got there.

    Carries the route as well as the ruling because the two stages do not
    speak with the same authority: `escalated=False` means only the screen
    ever saw this exchange, and `verdict.flagged` is then always False. A
    caller auditing flag rates or tuning `threshold` needs to tell "the judge
    cleared it" from "the screen never sent it".
    """

    verdict: Verdict
    score: float
    escalated: bool

    @property
    def flagged(self) -> bool:
        return self.verdict.flagged


class Cascade:
    """Screen every exchange; escalate the suspicious ones to a real judge."""

    def __init__(
        self,
        screen: Screen,
        judge: ExchangeClassifier,
        constitution: Constitution,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.screen = screen
        self.judge = judge
        self.constitution = constitution
        self.threshold = threshold

    def evaluate(self, exchange: Exchange) -> CascadeVerdict:
        score = self.screen.score(exchange, self.constitution)
        if score < self.threshold:
            # Not "the screen cleared it" — the screen is not entitled to
            # clear anything on its own. It is "nothing here justified the
            # cost of a closer look", which is the only claim a first stage
            # calibrated for recall can honestly make.
            return CascadeVerdict(
                verdict=Verdict(
                    flagged=False,
                    reasoning=f"Screened out below threshold (score {score:.3f}).",
                ),
                score=score,
                escalated=False,
            )

        # The judge's ruling is the decision of record. The screen's score got
        # us here and is kept for auditing, but it never overrides or blends
        # with the verdict — averaging a calibrated-for-recall screen into a
        # considered judgement would corrupt both.
        return CascadeVerdict(
            verdict=self.judge.judge(exchange, self.constitution),
            score=score,
            escalated=True,
        )
