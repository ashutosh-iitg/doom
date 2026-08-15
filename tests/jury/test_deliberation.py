"""`Deliberator`: the turn loop, moderator blindness, and fault tolerance
(Appendix F) — all exercised with stub jurors/moderators, no real model call.
"""

from doom.jury.deliberation import Deliberator, ModeratorDecision
from doom.jury.verdict import JuryVerdict
from tests.jury.fixtures import DEFECTIVE_TRACE


class _StubJuror:
    def __init__(self, juror_id, arguments=None, fail_after=None):
        self.juror_id = juror_id
        self._arguments = iter(arguments or [])
        self._fail_after = fail_after
        self._calls = 0

    def contribute(self, trace, transcript, instruction):
        self._calls += 1
        if self._fail_after is not None and self._calls > self._fail_after:
            raise RuntimeError("stub juror failure")
        return next(self._arguments, "I have no new arguments to present.")


class _StubModerator:
    def __init__(self, decisions):
        self._decisions = iter(decisions)

    def decide(
        self,
        transcript,
        all_juror_ids,
        spoken_this_round,
        previous_consensus,
        total_turns,
        logical_round,
    ):
        return next(self._decisions)


def _decision(should_terminate=False, next_speaker=None, negatives=(), termination_reason=None):
    return ModeratorDecision(
        should_terminate=should_terminate,
        next_speaker=next_speaker,
        instruction="respond",
        consensus=JuryVerdict(negatives=negatives),
        termination_reason=termination_reason,
    )


def test_deliberation_terminates_when_the_moderator_says_so():
    moderator = _StubModerator([_decision(should_terminate=True, termination_reason="converged")])
    jurors = (_StubJuror("jury-0"),)
    deliberator = Deliberator(jurors=jurors, moderator=moderator)

    verdict = deliberator.run(DEFECTIVE_TRACE, phase1=())

    assert verdict.termination_reason == "converged"
    assert verdict.failed_jurors == ()


def test_deliberation_appends_each_jurors_turn_to_the_transcript():
    moderator = _StubModerator(
        [
            _decision(next_speaker="jury-0"),
            _decision(should_terminate=True, termination_reason="converged"),
        ]
    )
    jurors = (_StubJuror("jury-0", arguments=["I disagree, see STEP-2."]),)
    deliberator = Deliberator(jurors=jurors, moderator=moderator)

    verdict = deliberator.run(DEFECTIVE_TRACE, phase1=())

    assert any("I disagree, see STEP-2." in turn for turn in verdict.transcript)


def test_a_stalled_deliberation_stops_at_max_turns():
    moderator = _StubModerator(_decision(next_speaker="jury-0") for _ in range(1000))
    jurors = (_StubJuror("jury-0", arguments=["arg"] * 1000),)
    deliberator = Deliberator(jurors=jurors, moderator=moderator, max_turns=5)

    verdict = deliberator.run(DEFECTIVE_TRACE, phase1=())

    assert verdict.termination_reason == "max_turns_reached"
    assert verdict.deliberation_rounds >= 1


def test_a_juror_is_dropped_after_repeated_consecutive_failures():
    moderator = _StubModerator(_decision(next_speaker="jury-0") for _ in range(10))
    jurors = (_StubJuror("jury-0", fail_after=0),)

    deliberator = Deliberator(jurors=jurors, moderator=moderator, max_turns=10)

    verdict = deliberator.run(DEFECTIVE_TRACE, phase1=())

    assert "jury-0" in verdict.failed_jurors


def test_a_moderator_failure_falls_back_to_round_robin():
    class _FlakyModerator:
        def __init__(self):
            self.calls = 0

        def decide(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("moderator down")
            return _decision(should_terminate=True, termination_reason="converged")

    jurors = (_StubJuror("jury-0", arguments=["ok"]),)
    deliberator = Deliberator(jurors=jurors, moderator=_FlakyModerator(), max_turns=5)

    verdict = deliberator.run(DEFECTIVE_TRACE, phase1=())

    assert any("ok" in turn for turn in verdict.transcript)
    assert verdict.termination_reason == "converged"


def test_phase1_judgements_are_carried_through_to_the_final_verdict():
    from doom.jury import Defect

    moderator = _StubModerator([_decision(should_terminate=True, termination_reason="converged")])
    jurors = (_StubJuror("jury-0"),)
    deliberator = Deliberator(jurors=jurors, moderator=moderator)
    phase1 = (
        (
            "jury-0",
            (
                Defect.from_payload(
                    {
                        "statement_refs": ["STEP-1"],
                        "what_went_wrong": "w",
                        "impact": "minor",
                        "evidence": "e",
                    }
                ),
            ),
        ),
    )

    verdict = deliberator.run(DEFECTIVE_TRACE, phase1=phase1)

    assert verdict.phase1_judgements == phase1
