"""Phase 2 — Deliberation (paper §2.2.2, Listings 5-6): a moderator that is
deliberately blind to the problem/trace/solution runs a multi-turn debate
among jurors until consensus or a hard turn cap, then extracts a final
verdict. Separating "control of the discussion" from "access to the source
material" is what keeps the moderator from becoming a covert, biased extra
juror.

Fault tolerance (Appendix F): a juror failing mid-deliberation has that turn
skipped and is dropped after `MAX_CONSECUTIVE_FAILURES` in a row; a
moderator failure falls back to deterministic round-robin turn order rather
than stalling; nothing here silently drops a failure — every dropped juror
ends up in the returned `JuryVerdict.failed_jurors`.
"""

import logging
from dataclasses import dataclass, replace
from typing import Any

import emissary

from .defect import Defect
from .phase1 import Juror
from .trace import ReasoningTrace
from .verdict import JuryVerdict

log = logging.getLogger(__name__)

MAX_TURNS = 40
"""Hard backstop regardless of termination logic — never let a stalled
deliberation run forever."""

MAX_CONSECUTIVE_FAILURES = 2
"""A juror is dropped from the panel after this many consecutive failed
turns (Appendix F)."""

MODERATOR_SYSTEM_PROMPT = """\
You are a deliberation moderator for a panel of reasoning-trace auditors.

CRITICAL CONSTRAINTS:
- You have NO access to the problem, reasoning trace, or solution under review.
- You can ONLY see the conversation between panel members.
- You must NEVER speculate about the content of the reasoning trace.
- Your role is purely procedural: manage turns, identify disagreements, and detect convergence.

YOUR RESPONSIBILITIES:

0. CONSENSUS TRACKING: produce one high-fidelity consolidated judgement by clustering \
overlapping findings into canonical, problem-specific items and preserving maximum specificity \
from the most detailed source judgement.

1. TURN SELECTION: choose which panelist speaks next. Ensure every panelist speaks at least \
once per logical round, prioritise panelists involved in unresolved disagreements, and recall \
any panelist silent for two or more turns.

2. INSTRUCTION GENERATION: give the selected panelist a specific, process-oriented instruction \
— point them to specific disagreements, ask them to clarify, defend, or concede a point. NEVER \
suggest what the "right" answer about the trace content is.

3. TERMINATION DETECTION: signal that deliberation should end when a full logical round passes \
with no new substantive argument, all panelists have explicitly signalled agreement on all \
points, or arguments are cycling without resolution.

Call `record_decision` exactly once with your decision."""

RECORD_MODERATOR_DECISION_TOOL: dict[str, Any] = {
    "name": "record_decision",
    "description": "Record this turn's moderation decision and updated consensus state.",
    "input_schema": {
        "type": "object",
        "properties": {
            "should_terminate": {"type": "boolean"},
            "termination_reason": {"type": "string"},
            "next_speaker": {"type": "string", "description": "The agent-id chosen to speak next."},
            "instruction": {"type": "string"},
            "consensus_negatives": {
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
                        "vote_count": {"type": "integer"},
                        "voters": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["statement_refs", "what_went_wrong", "impact", "evidence"],
                },
            },
            "confidence": {
                "type": "number",
                "description": "Degree of agreement across the panel, in [0, 1].",
            },
            "dissenting_views": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["should_terminate", "consensus_negatives"],
    },
}


@dataclass(frozen=True)
class ModeratorDecision:
    should_terminate: bool
    next_speaker: str | None
    instruction: str | None
    consensus: JuryVerdict
    termination_reason: str | None = None


def _render_consensus(verdict: JuryVerdict) -> str:
    if not verdict.negatives:
        return "(empty -- this is the first turn)"
    return "\n".join(
        f"- [{d.impact}] {d.what_went_wrong} (refs={list(d.statement_refs)})"
        for d in verdict.negatives
    )


class Moderator:
    """Runs the deliberation loop's per-turn decisions — blind to the trace,
    sees only the transcript."""

    def __init__(self, spec: emissary.Spec | None = None):
        self.spec = spec or emissary.resolve_spec(
            env_var="DOOM_JURY_MODERATOR_PROVIDER", default="anthropic"
        )

    def decide(
        self,
        transcript: tuple[str, ...],
        all_juror_ids: tuple[str, ...],
        spoken_this_round: tuple[str, ...],
        previous_consensus: JuryVerdict,
        total_turns: int,
        logical_round: int,
    ) -> ModeratorDecision:
        silent = tuple(j for j in all_juror_ids if j not in spoken_this_round)
        prompt = (
            "DELIBERATION STATUS:\n"
            f"- Total turns so far: {total_turns}\n"
            f"- Current logical round: {logical_round}\n"
            f"- Panelists who have spoken this round: {list(spoken_this_round)}\n"
            f"- Panelists who have NOT spoken this round: {list(silent)}\n"
            f"- All panelist IDs: {list(all_juror_ids)}\n\n"
            "TRANSCRIPT:\n==== begin transcript ====\n"
            + "\n".join(transcript)
            + "\n==== end transcript ====\n\n"
            "YOUR PREVIOUS CONSENSUS STATE:\n"
            "==== begin previous consensus state ====\n"
            + _render_consensus(previous_consensus)
            + "\n==== end previous consensus state ===="
        )
        result = emissary.call_tool(
            self.spec,
            system=MODERATOR_SYSTEM_PROMPT,
            blocks=[{"text": prompt, "cache": False}],
            tool=RECORD_MODERATOR_DECISION_TOOL,
        )
        payload = result.payload
        consensus = JuryVerdict(
            negatives=tuple(Defect.from_payload(d) for d in payload.get("consensus_negatives", ())),
            confidence=payload.get("confidence"),
            dissenting_views=tuple(payload.get("dissenting_views", ())),
        )
        return ModeratorDecision(
            should_terminate=bool(payload["should_terminate"]),
            next_speaker=payload.get("next_speaker"),
            instruction=payload.get("instruction"),
            consensus=consensus,
            termination_reason=payload.get("termination_reason"),
        )


def _round_robin_next(active_ids: tuple[str, ...], turn: int) -> str:
    return active_ids[turn % len(active_ids)]


def _round_robin_fallback(
    active_ids: tuple[str, ...], turn: int, consensus: JuryVerdict
) -> ModeratorDecision:
    return ModeratorDecision(
        should_terminate=False,
        next_speaker=_round_robin_next(active_ids, turn),
        instruction=(
            "Continue the discussion — moderator unavailable, deterministic turn order in effect."
        ),
        consensus=consensus,
        termination_reason=None,
    )


@dataclass(frozen=True)
class Deliberator:
    """Phase 2 policy: multi-turn debate among jurors, moderated by a
    content-blind `Moderator`, with fault tolerance per Appendix F."""

    jurors: tuple[Juror, ...]
    moderator: Moderator
    max_turns: int = MAX_TURNS

    def run(
        self,
        trace: ReasoningTrace,
        phase1: tuple[tuple[str, tuple[Defect, ...]], ...],
    ) -> JuryVerdict:
        jurors_by_id = {j.juror_id: j for j in self.jurors}
        all_ids = tuple(jurors_by_id)
        transcript: tuple[str, ...] = ()
        consensus = JuryVerdict(negatives=())
        spoken_this_round: set[str] = set()
        logical_round = 1
        consecutive_failures: dict[str, int] = dict.fromkeys(all_ids, 0)
        failed_jurors: list[str] = []
        termination_reason = None

        for turn in range(self.max_turns):
            active_ids = tuple(j for j in all_ids if j not in failed_jurors)
            if not active_ids:
                termination_reason = "all_jurors_failed"
                break

            try:
                decision = self.moderator.decide(
                    transcript, active_ids, tuple(spoken_this_round), consensus, turn, logical_round
                )
            # Any moderator failure degrades to round-robin, per Appendix F.
            except Exception as exc:  # noqa: BLE001
                log.warning(f"moderator failed on turn {turn}: {exc}")
                decision = _round_robin_fallback(active_ids, turn, consensus)

            consensus = decision.consensus
            if decision.should_terminate:
                termination_reason = decision.termination_reason
                break

            speaker_id = decision.next_speaker
            if speaker_id not in jurors_by_id or speaker_id in failed_jurors:
                speaker_id = _round_robin_next(active_ids, turn)
            juror = jurors_by_id[speaker_id]

            try:
                argument = juror.contribute(trace, transcript, decision.instruction or "")
                consecutive_failures[speaker_id] = 0
            # Any juror failure degrades — turn skipped, dropped after repeated failures (Appendix F).
            except Exception as exc:  # noqa: BLE001
                log.warning(f"juror {speaker_id} failed to contribute on turn {turn}: {exc}")
                consecutive_failures[speaker_id] += 1
                if consecutive_failures[speaker_id] >= MAX_CONSECUTIVE_FAILURES:
                    failed_jurors.append(speaker_id)
                    log.warning(f"juror {speaker_id} dropped after repeated failures")
                continue

            transcript += (f"[{speaker_id}]: {argument}",)
            spoken_this_round.add(speaker_id)
            if set(active_ids) <= spoken_this_round:
                spoken_this_round = set()
                logical_round += 1
        else:
            termination_reason = "max_turns_reached"

        return replace(
            consensus,
            deliberation_rounds=logical_round,
            termination_reason=termination_reason,
            transcript=transcript,
            phase1_judgements=phase1,
            failed_jurors=tuple(failed_jurors),
        )
