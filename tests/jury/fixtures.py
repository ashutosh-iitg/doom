"""A small hand-written reasoning trace with a deliberately planted defect,
for exercising Reasoning Jury end to end — same illustrative role
`tests/fixtures/exchanges.py` plays for §3.
"""

from doom.jury import ReasoningTrace

DEFECTIVE_TRACE = ReasoningTrace.of(
    problem=(
        "A store sells apples at $2 each and oranges at $3 each. A customer buys 4 apples "
        "and 2 oranges. How much do they pay in total?"
    ),
    raw_trace=(
        "First, compute the cost of the apples: 4 apples at $2 each is 4 * 2 = $8.\n\n"
        "Next, compute the cost of the oranges: 2 oranges at $3 each is 2 * 3 = $9.\n\n"
        "Adding these together, the total is 8 + 9 = $17."
    ),
    final_solution="$17",
)
"""Step 2 miscalculates 2 * 3 as $9 instead of $6, which propagates into the
(also wrong) final total of $17 instead of the correct $14."""
