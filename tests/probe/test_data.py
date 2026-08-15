"""Activation cache I/O: hand-built parquet with an injected non-finite row
(mirrors bdsm's `training/train_head.py:load_cached`), plus the
train/holdout overlap guard `assert_no_overlap` ports from bdsm.
"""

import pytest

np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
torch = pytest.importorskip("torch")

from doom import Exchange, LabelledExchange
from doom.constitutions.cbrn_example import CBRN_EXAMPLE
from doom.probe.data import assert_no_overlap, extract_and_cache, load_cached


def _write(split_dir, exchange_ids, activations, flagged):
    split_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "exchange_id": pa.array(exchange_ids, pa.string()),
            "activation": pa.array(activations, pa.list_(pa.float32())),
            "flagged": pa.array(flagged, pa.bool_()),
        }
    )
    pq.write_table(table, split_dir / "part-000.parquet")


def test_load_cached_reads_every_row(tmp_path):
    split_dir = tmp_path / "train"
    _write(split_dir, ["a", "b"], [[1.0, 2.0], [3.0, 4.0]], [True, False])

    cached = load_cached(str(split_dir))

    assert cached["exchange_id"].tolist() == ["a", "b"]
    assert np.allclose(cached["x"], [[1.0, 2.0], [3.0, 4.0]])
    assert cached["y"].tolist() == [1.0, 0.0]


def test_load_cached_drops_non_finite_activation_rows(tmp_path):
    split_dir = tmp_path / "train"
    _write(
        split_dir,
        ["a", "b", "c"],
        [[1.0, 2.0], [float("nan"), 2.0], [3.0, 4.0]],
        [True, True, False],
    )

    cached = load_cached(str(split_dir))

    assert cached["exchange_id"].tolist() == ["a", "c"]
    assert len(cached["x"]) == 2


def test_load_cached_raises_when_the_split_directory_has_no_parquet(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cached(str(tmp_path / "missing"))


def test_assert_no_overlap_passes_for_disjoint_ids():
    train = {"exchange_id": np.array(["a", "b"])}
    holdout = {"exchange_id": np.array(["c", "d"])}

    assert_no_overlap(train, holdout)


def test_assert_no_overlap_catches_a_leaked_exchange_id():
    train = {"exchange_id": np.array(["a", "b"])}
    holdout = {"exchange_id": np.array(["b", "c"])}

    with pytest.raises(AssertionError, match="overlap"):
        assert_no_overlap(train, holdout)


class _StubBackbone:
    def __init__(self, vectors):
        self._vectors = iter(vectors)

    def activation(self, exchange, constitution):
        return next(self._vectors)


def test_extract_and_cache_routes_rows_by_split(tmp_path):
    exchanges = [Exchange.of(user=f"u{i}", assistant="a") for i in range(3)]
    labelled = [
        LabelledExchange(exchange=exchanges[0], flagged=True, split="train"),
        LabelledExchange(exchange=exchanges[1], flagged=False, split="train"),
        LabelledExchange(exchange=exchanges[2], flagged=True, split="holdout"),
    ]
    vectors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])]

    extract_and_cache(labelled, _StubBackbone(vectors), CBRN_EXAMPLE, str(tmp_path))

    train = load_cached(str(tmp_path / "train"))
    holdout = load_cached(str(tmp_path / "holdout"))
    assert len(train["x"]) == 2
    assert len(holdout["x"]) == 1


def test_extract_and_cache_gives_each_exchange_a_stable_deterministic_id(tmp_path):
    """The same exchange content must hash to the same `exchange_id` across
    separate extraction runs, or train/holdout overlap-checking (which
    compares ids, not content) would silently stop catching leaks."""
    exchange = Exchange.of(user="u", assistant="a")
    labelled = [LabelledExchange(exchange=exchange, flagged=True, split="train")]

    extract_and_cache(
        labelled, _StubBackbone([torch.tensor([1.0])]), CBRN_EXAMPLE, str(tmp_path / "run1")
    )
    extract_and_cache(
        labelled, _StubBackbone([torch.tensor([1.0])]), CBRN_EXAMPLE, str(tmp_path / "run2")
    )

    id1 = load_cached(str(tmp_path / "run1" / "train"))["exchange_id"][0]
    id2 = load_cached(str(tmp_path / "run2" / "train"))["exchange_id"][0]
    assert id1 == id2
