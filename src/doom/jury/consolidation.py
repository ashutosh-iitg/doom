"""`Consolidator`: Phase 2's cheap mode (paper §2.2.2, Listing 4) — one extra
call that re-performs the Phase 1 task with every juror's independent
findings attached as unverified candidates, and emits the final consensus
verdict with vote metadata.
"""

from typing import Any

import emissary

from .defect import Defect
from .phase1 import SYSTEM_PROMPT
from .trace import ReasoningTrace
from .verdict import JuryVerdict

RECORD_CONSENSUS_TOOL: dict[str, Any] = {
    "name": "record_consensus",
    "description": "Record the consolidated, verified list of trace-grounded reasoning defects.",
    "input_schema": {
        "type": "object",
        "properties": {
            "negatives": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "statement_refs": {"type": "array", "items": {"type": "string"}},
                        "what_went_wrong": {"type": "string"},
                        "impact": {
                            "type": "string",
                            "enum": ["neutral", "minor", "major", "fatal"],
                        },
                        "evidence": {"type": "string"},
                        "correct_value": {"type": "string"},
                        "vote_count": {
                            "type": "integer",
                            "description": (
                                "How many panel auditors' candidates support this finding "
                                "(0 if it is yours alone)."
                            ),
                        },
                        "voters": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "IDs of the supporting jurors; empty for own findings.",
                        },
                    },
                    "required": [
                        "statement_refs",
                        "what_went_wrong",
                        "impact",
                        "evidence",
                        "vote_count",
                        "voters",
                    ],
                },
            },
        },
        "required": ["negatives"],
    },
}


def _render_panel_candidates(phase1: tuple[tuple[str, tuple[Defect, ...]], ...]) -> str:
    lines = [
        (
            f"## PANEL CANDIDATE FINDINGS (unverified -- leads, not conclusions)\n\n"
            f"Before you, {len(phase1)} independent auditors examined this same trace under the "
            "exact instructions above. Their raw findings are reproduced below, attributed by "
            "auditor id. Treat them as CANDIDATES, not established facts: auditors can be "
            "wrong, can duplicate one another, and can miss defects entirely.\n\n"
            "==== begin panel findings ===="
        )
    ]
    for juror_id, defects in phase1:
        for d in defects:
            lines.append(
                f"[{juror_id}] statement_refs={list(d.statement_refs)} impact={d.impact}\n"
                f"  what_went_wrong: {d.what_went_wrong}\n"
                f"  evidence: {d.evidence}"
            )
    lines.append("==== end panel findings ====")
    return "\n".join(lines)


class Consolidator:
    """Phase 2 policy: one moderator call verifies, merges, and fills gaps
    in the panel's Phase 1 findings."""

    def __init__(self, spec: emissary.Spec | None = None):
        self.spec = spec or emissary.resolve_spec(
            env_var="DOOM_JURY_MODERATOR_PROVIDER", default="anthropic"
        )

    def run(
        self,
        trace: ReasoningTrace,
        phase1: tuple[tuple[str, tuple[Defect, ...]], ...],
    ) -> JuryVerdict:
        result = emissary.call_tool(
            self.spec,
            system=SYSTEM_PROMPT,
            blocks=[
                {"text": trace.render(), "cache": True},
                {"text": _render_panel_candidates(phase1), "cache": False},
            ],
            tool=RECORD_CONSENSUS_TOOL,
        )
        negatives = tuple(Defect.from_payload(d) for d in result.payload["negatives"])
        return JuryVerdict(negatives=negatives, phase1_judgements=phase1)
