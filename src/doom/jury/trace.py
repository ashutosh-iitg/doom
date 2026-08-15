"""`ReasoningTrace`: the unit Reasoning Jury audits — a problem, a
step-segmented reasoning trace, and a final solution. Genuinely different
from `Exchange`/`Constitution`: this asks whether a chain of reasoning is
*correct*, not whether an exchange violates a policy (see CLAUDE.md's jury
decision).

Segmentation follows the paper's §2.1 regex: an end-of-sentence character
followed by a blank line, which survives code blocks and multi-line
derivations better than a bare double-newline split.
"""

import re
from dataclasses import dataclass

_STEP_BOUNDARY = re.compile(r"(?<=[.!?])\s*\n\s*\n+")


def segment_trace(raw_trace: str) -> tuple[str, ...]:
    """Split a raw reasoning trace into ordered step segments."""
    return tuple(s.strip() for s in _STEP_BOUNDARY.split(raw_trace.strip()) if s.strip())


@dataclass(frozen=True)
class ReasoningTrace:
    problem: str
    steps: tuple[str, ...]
    final_solution: str

    @classmethod
    def of(cls, *, problem: str, raw_trace: str, final_solution: str) -> "ReasoningTrace":
        return cls(problem=problem, steps=segment_trace(raw_trace), final_solution=final_solution)

    def render(self) -> str:
        """The input block every juror, the consolidator, and every
        deliberation turn are shown — built once here, not duplicated
        per call site."""
        steps_text = "\n\n".join(f"[STEP-{i}] {step}" for i, step in enumerate(self.steps, start=1))
        return (
            "### PROBLEM\n"
            "==== begin problem ====\n"
            f"{self.problem}\n"
            "==== end problem ====\n\n"
            "### REASONING TRACE (the trace to audit)\n"
            "==== begin reasoning trace ====\n"
            f"{steps_text}\n"
            "==== end reasoning trace ====\n\n"
            "### Final response\n"
            "==== begin final response ====\n"
            f"{self.final_solution}\n"
            "==== end final response ===="
        )
