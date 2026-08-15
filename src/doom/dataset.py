"""`LabelledExchange`: the one labelled-exchange shape §4 calibration and §5
activation extraction both read, so the schema is defined once, not twice.
"""

import json
from dataclasses import dataclass
from typing import Literal

from .exchange import Exchange, Turn

Split = Literal["train", "holdout"]


@dataclass(frozen=True)
class LabelledExchange:
    """One exchange with a ground-truth label, for calibration or training —
    never for judging; `Verdict` is what a judge produces at run time."""

    exchange: Exchange
    flagged: bool
    rule_ids: tuple[str, ...] = ()
    source: str = ""
    split: Split = "train"


def _exchange_from_turns(turns: list[dict]) -> Exchange:
    return Exchange(tuple(Turn(t["role"], t["content"]) for t in turns))


def load_jsonl(path: str) -> tuple[LabelledExchange, ...]:
    """Read one `LabelledExchange` per line from a JSONL file.

    Each line: `{"turns": [{"role": ..., "content": ...}, ...], "flagged":
    bool, "rule_ids": [...] (optional), "source": str (optional), "split":
    "train" | "holdout"}`.
    """
    labelled = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            labelled.append(
                LabelledExchange(
                    exchange=_exchange_from_turns(row["turns"]),
                    flagged=bool(row["flagged"]),
                    rule_ids=tuple(row.get("rule_ids", ())),
                    source=row.get("source", ""),
                    split=row["split"],
                )
            )
    return tuple(labelled)
