from dataclasses import dataclass


@dataclass(frozen=True)
class Verdict:
    """One judge's decision on one exchange.

    The shape any judge produces — prompted, fine-tuned, or (per the paper's
    §5) a linear probe reading model activations — so `Panel` never needs to
    know which kind of judge it's holding.
    """

    flagged: bool
    reasoning: str
    rule_ids: tuple[str, ...] = ()
