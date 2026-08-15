"""`Defect`: one step-grounded, severity-tagged, evidence-backed reasoning
weakness — the same shape whether it's a raw Phase 1 finding or a Phase 2
consensus finding, mirroring `Verdict`'s own "one uniform shape" philosophy
rather than a parallel raw/consensus type hierarchy. `vote_count`/`voters`
default to "not applicable" (0/()) for Phase 1's own independent findings,
becoming meaningful once a Phase 2 policy has run.
"""

from dataclasses import dataclass
from typing import Literal

Impact = Literal["neutral", "minor", "major", "fatal"]


@dataclass(frozen=True)
class Defect:
    statement_refs: tuple[str, ...]
    what_went_wrong: str
    impact: Impact
    evidence: str
    correct_value: str | None = None
    vote_count: int = 0
    voters: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: dict) -> "Defect":
        """The one shared parser for a defect's tool-call payload, used by
        Phase 1, Consolidation, and Deliberation alike."""
        return cls(
            statement_refs=tuple(payload["statement_refs"]),
            what_went_wrong=str(payload["what_went_wrong"]),
            impact=payload["impact"],
            evidence=str(payload["evidence"]),
            correct_value=payload.get("correct_value"),
            vote_count=int(payload.get("vote_count", 0)),
            voters=tuple(payload.get("voters", ())),
        )
