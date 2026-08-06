"""`Panel`: a sequence of judges evaluated over the same exchange and constitution.

Deliberately thin in v0 — every judge evaluates, `flagged` is true if any one
does. This is the seam later work extends: the paper's §4 two-stage cascade
(cheap judge screens, expensive judge escalates) or §6's weighted probe+
classifier ensemble become a routing/weighting policy in front of
`evaluate`, not a change to `Verdict` or `ExchangeClassifier`.
"""

from dataclasses import dataclass

from .constitution import Constitution
from .exchange import Exchange
from .judge import ExchangeClassifier
from .verdict import Verdict


@dataclass(frozen=True)
class Panel:
    judges: tuple[ExchangeClassifier, ...]
    constitution: Constitution

    def evaluate(self, exchange: Exchange) -> tuple[Verdict, ...]:
        return tuple(judge.judge(exchange, self.constitution) for judge in self.judges)

    def flagged(self, exchange: Exchange) -> bool:
        return any(verdict.flagged for verdict in self.evaluate(exchange))
