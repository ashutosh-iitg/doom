"""`JuryVerdict`: the unifying Phase 2 output — the same shape whether it
came from `Consolidator` (one call) or `Deliberator` (a multi-turn debate).
Fields only deliberation produces (`deliberation_rounds`, `termination_reason`,
`transcript`, `dissenting_views`) default to their "not applicable" value for
consolidation rather than a fabricated one — `confidence` is `None`, not a
made-up 1.0, when no mechanism actually computed it.
"""

from dataclasses import dataclass

from .defect import Defect


@dataclass(frozen=True)
class JuryVerdict:
    negatives: tuple[Defect, ...]
    confidence: float | None = None
    dissenting_views: tuple[str, ...] = ()
    deliberation_rounds: int = 0
    termination_reason: str | None = None
    phase1_judgements: tuple[tuple[str, tuple[Defect, ...]], ...] = ()
    transcript: tuple[str, ...] = ()
    failed_jurors: tuple[str, ...] = ()
