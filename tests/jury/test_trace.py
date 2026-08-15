"""`segment_trace` / `ReasoningTrace`: §2.1's regex-based step segmentation
and the Listing-3 input block."""

from doom.jury import ReasoningTrace, segment_trace


def test_segment_trace_splits_on_sentence_end_then_blank_line():
    raw = "First step ends here.\n\nSecond step ends here.\n\nThird."

    steps = segment_trace(raw)

    assert steps == ("First step ends here.", "Second step ends here.", "Third.")


def test_segment_trace_survives_a_single_newline_inside_a_step():
    """A soft line break inside a multi-line derivation must not be treated
    as a step boundary — only a blank line after sentence-ending punctuation
    does."""
    raw = "A multi-line derivation\nwith a soft break\nstill counts as one step.\n\nSecond step."

    steps = segment_trace(raw)

    assert len(steps) == 2
    assert "soft break" in steps[0]


def test_segment_trace_drops_empty_segments():
    raw = "First.\n\n\n\nSecond."

    steps = segment_trace(raw)

    assert steps == ("First.", "Second.")


def test_reasoning_trace_of_segments_the_raw_trace():
    trace = ReasoningTrace.of(problem="p", raw_trace="Step one.\n\nStep two.", final_solution="s")

    assert trace.steps == ("Step one.", "Step two.")


def test_render_labels_each_step_with_its_ordinal_position():
    trace = ReasoningTrace.of(problem="p", raw_trace="Alpha.\n\nBeta.", final_solution="s")

    rendered = trace.render()

    assert "[STEP-1] Alpha." in rendered
    assert "[STEP-2] Beta." in rendered
    assert "begin problem" in rendered
    assert "begin final response" in rendered
