"""`Defect.from_payload` — the one shared parser for Phase 1 and Phase 2
output payloads."""

from doom.jury import Defect


def test_from_payload_parses_the_required_fields():
    defect = Defect.from_payload(
        {
            "statement_refs": ["STEP-2"],
            "what_went_wrong": "2 * 3 was computed as 9.",
            "impact": "fatal",
            "evidence": "[STEP-2] '2 * 3 = $9'",
        }
    )

    assert defect.statement_refs == ("STEP-2",)
    assert defect.impact == "fatal"
    assert defect.correct_value is None
    assert defect.vote_count == 0
    assert defect.voters == ()


def test_from_payload_carries_optional_consensus_fields():
    defect = Defect.from_payload(
        {
            "statement_refs": ["STEP-2"],
            "what_went_wrong": "w",
            "impact": "major",
            "evidence": "e",
            "correct_value": "6",
            "vote_count": 2,
            "voters": ["jury-0", "jury-2"],
        }
    )

    assert defect.correct_value == "6"
    assert defect.vote_count == 2
    assert defect.voters == ("jury-0", "jury-2")
