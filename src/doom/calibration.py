"""§4 calibration: measure, don't guess, `Cascade`'s `threshold`.

`DEFAULT_THRESHOLD` in `cascade.py` is an explicit placeholder — a real value
comes from sweeping candidate thresholds against labelled exchanges and
reading off the escalation rate / missed-violation rate trade-off. This
module does the sweep; it does not itself change `DEFAULT_THRESHOLD` — that
still needs a real labelled dataset, which does not exist yet.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .constitution import Constitution
from .dataset import LabelledExchange
from .exchange import Exchange


class ScreenLike(Protocol):
    def score(self, exchange: Exchange, constitution: Constitution) -> float: ...


@dataclass(frozen=True)
class ThresholdStats:
    """Escalation rate and missed-violation rate at one candidate threshold.

    Both are rates, not counts, so thresholds are comparable across
    differently-sized labelled sets.
    """

    threshold: float
    escalation_rate: float
    missed_violation_rate: float
    n: int


def sweep_thresholds(
    screen: ScreenLike,
    constitution: Constitution,
    labelled: Sequence[LabelledExchange],
    thresholds: Sequence[float],
) -> tuple[ThresholdStats, ...]:
    """Score every labelled exchange once, then evaluate each candidate
    threshold against those scores.

    A threshold escalates an exchange when `score >= threshold` — the same
    boundary rule `Cascade.evaluate` uses, so a threshold read off this sweep
    behaves in `Cascade` exactly as measured here.
    """
    scored = [(le, screen.score(le.exchange, constitution)) for le in labelled]
    n_flagged = sum(1 for le in labelled if le.flagged)

    stats = []
    for threshold in thresholds:
        n_escalated = sum(1 for _, score in scored if score >= threshold)
        n_missed = sum(1 for le, score in scored if le.flagged and score < threshold)
        stats.append(
            ThresholdStats(
                threshold=threshold,
                escalation_rate=n_escalated / len(scored) if scored else 0.0,
                missed_violation_rate=n_missed / n_flagged if n_flagged else 0.0,
                n=len(labelled),
            )
        )
    return tuple(stats)
