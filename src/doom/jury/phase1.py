"""`Juror`: an independent reasoning-trace auditor.

Phase 1 (paper §2.2.1, Listing 3): reads a trace and returns a self-contained
list of trace-grounded defects, with no visibility into any other juror's
findings. Also plays the juror's role in Phase 2 deliberation (`contribute`,
Listing 6) — grounded natural-language argument, not a structured verdict,
so it's wrapped in a single-field tool call rather than a new emissary
capability (emissary exposes no free-text completion call; see CLAUDE.md's
jury decision on why this stays inside `doom.jury`, not `emissary`).
"""

from typing import Any

import emissary

from .defect import Defect
from .trace import ReasoningTrace

RECORD_DEFECTS_TOOL: dict[str, Any] = {
    "name": "record_defects",
    "description": "Record the trace-grounded reasoning defects found in this trace.",
    "input_schema": {
        "type": "object",
        "properties": {
            "negatives": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "statement_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Step IDs where the issue manifests, e.g. ['STEP-14'].",
                        },
                        "what_went_wrong": {
                            "type": "string",
                            "description": (
                                "Specific description embedding [STEP-x] references and direct "
                                "quotes from the trace."
                            ),
                        },
                        "impact": {
                            "type": "string",
                            "enum": ["neutral", "minor", "major", "fatal"],
                        },
                        "evidence": {
                            "type": "string",
                            "description": "Direct quotes from the trace, prefixed with step IDs.",
                        },
                        "correct_value": {
                            "type": "string",
                            "description": "What the correct reasoning/value/conclusion should be.",
                        },
                    },
                    "required": ["statement_refs", "what_went_wrong", "impact", "evidence"],
                },
            },
        },
        "required": ["negatives"],
    },
}

RECORD_ARGUMENT_TOOL: dict[str, Any] = {
    "name": "record_argument",
    "description": "Record this turn's natural-language contribution to the deliberation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "argument": {
                "type": "string",
                "description": "Agree, disagree with evidence, raise a new defect, or concede.",
            },
        },
        "required": ["argument"],
    },
}

SYSTEM_PROMPT = """\
You are a reasoning-trace auditor. You identify concrete weaknesses in reasoning traces by \
pointing to specific steps. Judge the *weaknesses* of the provided reasoning trace by pointing \
to specific bad reasoning moves.

Critical constraints:
- Do NOT write a fresh full response to the problem.
- Avoid generic feedback. Every critique must be trace-grounded: name the exact quantity, \
expression, or claim involved, and reference which [STEP-x] it occurs in.

Grounding requirement: name the specific step(s), quote the exact problematic claim, and \
explain why it is wrong and what the correct value/reasoning should be.

Genericness test: for every point, ask "could this comment apply to a totally different problem \
with no edits?" If yes, rewrite it to name a specific artifact and operation from a specific \
[STEP-x].

Specificity self-check (1-5, only include issues scoring >= 4): 5 cites an exact formula/number \
and the correct value; 4 cites a specific claim and names the error type; 3 or below references \
a step but describes the issue generically or could apply to a different problem — discard \
these.

Defect propagation: any step that is based on an error in a previous step is itself considered \
incorrect and inherits the same severity as the upstream defect.

Call `record_defects` exactly once with your findings."""


class Juror:
    """One independent judge: lists a trace's defects (Phase 1), and argues
    a point in deliberation when called on (Phase 2)."""

    def __init__(self, spec: emissary.Spec | None = None, juror_id: str | None = None):
        self.spec = spec or emissary.resolve_spec(
            env_var="DOOM_JURY_JUROR_PROVIDER", default="anthropic"
        )
        self.juror_id = juror_id or str(self.spec)

    def judge(self, trace: ReasoningTrace) -> tuple[Defect, ...]:
        result = emissary.call_tool(
            self.spec,
            system=SYSTEM_PROMPT,
            blocks=[{"text": trace.render(), "cache": True}],
            tool=RECORD_DEFECTS_TOOL,
        )
        return tuple(Defect.from_payload(d) for d in result.payload["negatives"])

    def contribute(
        self, trace: ReasoningTrace, transcript: tuple[str, ...], instruction: str
    ) -> str:
        prompt = (
            f"You are {self.juror_id}. Your prior contributions are marked with your id below.\n\n"
            "DELIBERATION SO FAR:\n==== begin deliberation so far ====\n"
            + "\n".join(transcript)
            + "\n==== end deliberation so far ====\n\n"
            "MODERATOR INSTRUCTION FOR YOU:\n==== begin moderator instructions ====\n"
            f"{instruction}\n==== end moderator instructions ====\n\n"
            "Critical rules: every claim you make MUST reference specific [STEP-x] markers from "
            "the reasoning trace above. If you cannot ground a claim in a specific step, do not "
            "make it. You may agree, disagree with evidence, or raise a new issue. Be concise, "
            'and focus on the strongest argument. If you have nothing new to add, say "I have no '
            'new arguments to present."'
        )
        result = emissary.call_tool(
            self.spec,
            system=SYSTEM_PROMPT,
            blocks=[
                {"text": trace.render(), "cache": True},
                {"text": prompt, "cache": False},
            ],
            tool=RECORD_ARGUMENT_TOOL,
        )
        return str(result.payload["argument"])
