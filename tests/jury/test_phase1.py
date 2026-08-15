"""`Juror`: Phase 1 independent judgement and Phase 2 deliberation turns —
mocked `emissary.call_tool`, same pattern as `test_judge.py`."""

from unittest.mock import patch

from emissary import CallResult

from doom.jury import Juror
from tests.jury.fixtures import DEFECTIVE_TRACE


def _result(payload):
    return CallResult(payload, "anthropic", "claude-opus-5", 10, 5, 0)


def test_judge_parses_a_list_of_defects_from_the_payload():
    payload = {
        "negatives": [
            {
                "statement_refs": ["STEP-2"],
                "what_went_wrong": "2 * 3 was computed as 9.",
                "impact": "fatal",
                "evidence": "[STEP-2] '2 * 3 = $9'",
            }
        ]
    }

    with patch("emissary.call_tool", return_value=_result(payload)):
        defects = Juror(juror_id="jury-0").judge(DEFECTIVE_TRACE)

    assert len(defects) == 1
    assert defects[0].impact == "fatal"


def test_judge_returns_no_defects_when_none_are_found():
    with patch("emissary.call_tool", return_value=_result({"negatives": []})):
        defects = Juror(juror_id="jury-0").judge(DEFECTIVE_TRACE)

    assert defects == ()


def test_judge_cache_marks_the_trace_and_uses_the_auditor_system_prompt():
    with patch("emissary.call_tool", return_value=_result({"negatives": []})) as called:
        Juror(juror_id="jury-0").judge(DEFECTIVE_TRACE)

    blocks = called.call_args.kwargs["blocks"]
    assert blocks[0]["cache"] is True
    assert "[STEP-1]" in blocks[0]["text"]
    assert "reasoning-trace auditor" in called.call_args.kwargs["system"]


def test_a_juror_defaults_to_the_configured_provider(monkeypatch):
    monkeypatch.delenv("DOOM_JURY_JUROR_PROVIDER", raising=False)

    assert str(Juror().spec) == "anthropic:claude-opus-5"

    monkeypatch.setenv("DOOM_JURY_JUROR_PROVIDER", "vllm:qwen3-8b")
    assert str(Juror().spec) == "vllm:qwen3-8b"


def test_contribute_returns_the_arguments_free_text():
    with patch("emissary.call_tool", return_value=_result({"argument": "I disagree with jury-1."})):
        argument = Juror(juror_id="jury-0").contribute(
            DEFECTIVE_TRACE, transcript=(), instruction="respond"
        )

    assert argument == "I disagree with jury-1."


def test_contribute_reuses_the_cache_marked_trace_render():
    with patch("emissary.call_tool", return_value=_result({"argument": "ok"})) as called:
        Juror(juror_id="jury-0").contribute(
            DEFECTIVE_TRACE, transcript=("[jury-1]: hi",), instruction="respond"
        )

    blocks = called.call_args.kwargs["blocks"]
    assert blocks[0]["cache"] is True
    assert blocks[1]["cache"] is False
    assert "[jury-1]: hi" in blocks[1]["text"]
