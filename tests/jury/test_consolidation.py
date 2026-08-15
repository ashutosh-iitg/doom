"""`Consolidator`: Phase 2 single-call consensus — mocked `emissary.call_tool`."""

from unittest.mock import patch

from emissary import CallResult

from doom.jury import Consolidator, Defect
from tests.jury.fixtures import DEFECTIVE_TRACE


def _result(payload):
    return CallResult(payload, "anthropic", "claude-opus-5", 10, 5, 0)


def _phase1():
    return (
        (
            "jury-0",
            (
                Defect.from_payload(
                    {
                        "statement_refs": ["STEP-2"],
                        "what_went_wrong": "2 * 3 computed as 9.",
                        "impact": "fatal",
                        "evidence": "e",
                    }
                ),
            ),
        ),
    )


def test_run_parses_the_consensus_negatives_with_vote_metadata():
    payload = {
        "negatives": [
            {
                "statement_refs": ["STEP-2"],
                "what_went_wrong": "2 * 3 computed as 9, should be 6.",
                "impact": "fatal",
                "evidence": "e",
                "vote_count": 1,
                "voters": ["jury-0"],
            }
        ]
    }

    with patch("emissary.call_tool", return_value=_result(payload)):
        verdict = Consolidator().run(DEFECTIVE_TRACE, _phase1())

    assert len(verdict.negatives) == 1
    assert verdict.negatives[0].vote_count == 1
    assert verdict.negatives[0].voters == ("jury-0",)
    assert verdict.phase1_judgements == _phase1()


def test_run_passes_the_panel_candidates_and_cache_marks_the_trace():
    with patch("emissary.call_tool", return_value=_result({"negatives": []})) as called:
        Consolidator().run(DEFECTIVE_TRACE, _phase1())

    blocks = called.call_args.kwargs["blocks"]
    assert blocks[0]["cache"] is True
    assert blocks[1]["cache"] is False
    assert "jury-0" in blocks[1]["text"]
    assert "2 * 3 computed as 9" in blocks[1]["text"]


def test_consolidation_has_no_deliberation_metadata():
    with patch("emissary.call_tool", return_value=_result({"negatives": []})):
        verdict = Consolidator().run(DEFECTIVE_TRACE, _phase1())

    assert verdict.deliberation_rounds == 0
    assert verdict.termination_reason is None
    assert verdict.transcript == ()
