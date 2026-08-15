"""`ReasoningJury`: Phase 1 fan-out with a minimum-juror threshold, then
delegates to whichever Phase 2 policy it's given."""

import pytest

from doom.jury import Defect, JuryVerdict, ReasoningJury
from tests.jury.fixtures import DEFECTIVE_TRACE


class _StubJuror:
    def __init__(self, juror_id, defects=(), fail=False):
        self.juror_id = juror_id
        self._defects = defects
        self._fail = fail

    def judge(self, trace):
        if self._fail:
            raise RuntimeError("stub juror failure")
        return self._defects


class _StubPhase2:
    def __init__(self):
        self.received_phase1 = None

    def run(self, trace, phase1):
        self.received_phase1 = phase1
        return JuryVerdict(negatives=())


def test_evaluate_passes_every_successful_jurors_defects_to_phase2():
    defect = Defect.from_payload(
        {"statement_refs": ["STEP-1"], "what_went_wrong": "w", "impact": "major", "evidence": "e"}
    )
    phase2 = _StubPhase2()
    jury = ReasoningJury(jurors=(_StubJuror("jury-0", defects=(defect,)),), phase2=phase2)

    jury.evaluate(DEFECTIVE_TRACE)

    assert phase2.received_phase1 == (("jury-0", (defect,)),)


def test_a_failed_juror_is_excluded_from_phase1_but_surfaced_in_the_verdict():
    phase2 = _StubPhase2()
    jury = ReasoningJury(
        jurors=(_StubJuror("jury-0", defects=()), _StubJuror("jury-1", fail=True)),
        phase2=phase2,
        min_jurors=1,
    )

    verdict = jury.evaluate(DEFECTIVE_TRACE)

    assert phase2.received_phase1 == (("jury-0", ()),)
    assert verdict.failed_jurors == ("jury-1",)


def test_falling_below_the_minimum_juror_threshold_raises():
    jury = ReasoningJury(
        jurors=(_StubJuror("jury-0", fail=True),), phase2=_StubPhase2(), min_jurors=1
    )

    with pytest.raises(RuntimeError, match="minimum is 1"):
        jury.evaluate(DEFECTIVE_TRACE)
