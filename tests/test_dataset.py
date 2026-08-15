"""`LabelledExchange` and its JSONL loader — the shared schema §4 calibration
and §5 activation extraction both read.
"""

import json

from doom import Exchange, LabelledExchange
from doom.dataset import load_jsonl


def test_load_jsonl_roundtrips_a_labelled_exchange(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(
            {
                "turns": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                ],
                "flagged": True,
                "rule_ids": ["no-cbrn-uplift"],
                "source": "hand-written",
                "split": "train",
            }
        )
        + "\n"
    )

    loaded = load_jsonl(str(path))

    assert loaded == (
        LabelledExchange(
            exchange=Exchange.of(user="u", assistant="a"),
            flagged=True,
            rule_ids=("no-cbrn-uplift",),
            source="hand-written",
            split="train",
        ),
    )


def test_load_jsonl_defaults_rule_ids_and_source_when_omitted(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(
            {
                "turns": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                ],
                "flagged": False,
                "split": "holdout",
            }
        )
        + "\n"
    )

    loaded = load_jsonl(str(path))

    assert loaded[0].rule_ids == ()
    assert loaded[0].source == ""
    assert loaded[0].split == "holdout"


def test_load_jsonl_reads_every_line(tmp_path):
    path = tmp_path / "data.jsonl"
    lines = [
        json.dumps(
            {
                "turns": [
                    {"role": "user", "content": f"u{i}"},
                    {"role": "assistant", "content": "a"},
                ],
                "flagged": i % 2 == 0,
                "split": "train",
            }
        )
        for i in range(3)
    ]
    path.write_text("\n".join(lines) + "\n")

    loaded = load_jsonl(str(path))

    assert len(loaded) == 3
    assert [le.flagged for le in loaded] == [True, False, True]


def test_load_jsonl_preserves_a_system_turn(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(
            {
                "turns": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                ],
                "flagged": False,
                "split": "train",
            }
        )
        + "\n"
    )

    loaded = load_jsonl(str(path))

    assert loaded[0].exchange == Exchange.of(user="u", assistant="a", system="s")
