"""Activation cache: §5's frozen-backbone activations, extracted once per
labelled exchange and cached to parquet — mirrors bdsm's own cached-CLS
pattern (`bdsm/training/train_head.py:load_cached`), including its
drop-and-log non-finite-row behaviour at read time.
"""

import glob
import hashlib
import logging
import os
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ..constitution import Constitution
from ..dataset import LabelledExchange
from ..exchange import Exchange

log = logging.getLogger(__name__)


class ActivationSource(Protocol):
    def activation(self, exchange: Exchange, constitution: Constitution) -> torch.Tensor: ...


def _exchange_id(exchange: Exchange) -> str:
    """A stable id derived from content, not object identity — the same
    exchange hashes the same way across separate extraction runs, which is
    what lets `assert_no_overlap` compare ids across train/holdout caches."""
    return hashlib.sha256(exchange.render().encode()).hexdigest()[:16]


def extract_and_cache(
    labelled: Sequence[LabelledExchange],
    backbone: ActivationSource,
    constitution: Constitution,
    out_dir: str,
) -> None:
    """Run one frozen forward pass per exchange and write train/holdout
    parquet caches. This is the only place the backbone actually runs —
    training itself only ever reads these files."""
    by_split: dict[str, list[dict]] = {"train": [], "holdout": []}
    for le in labelled:
        vec = backbone.activation(le.exchange, constitution)
        by_split[le.split].append(
            {
                "exchange_id": _exchange_id(le.exchange),
                "activation": [float(v) for v in vec.tolist()],
                "flagged": le.flagged,
            }
        )

    os.makedirs(out_dir, exist_ok=True)
    for split, rows in by_split.items():
        if not rows:
            continue
        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        table = pa.table(
            {
                "exchange_id": pa.array([r["exchange_id"] for r in rows], pa.string()),
                "activation": pa.array([r["activation"] for r in rows], pa.list_(pa.float32())),
                "flagged": pa.array([r["flagged"] for r in rows], pa.bool_()),
            }
        )
        pq.write_table(table, os.path.join(split_dir, "part-000.parquet"), compression="zstd")


def load_cached(split_dir: str) -> dict:
    """Read a split's cached activations, dropping (and logging) any
    non-finite row rather than letting a NaN silently poison training."""
    files = sorted(glob.glob(os.path.join(split_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no cached activation parquet under {split_dir}")

    ids: list[str] = []
    xs: list[np.ndarray] = []
    ys: list[bool] = []
    n_nonfinite = 0
    for fpath in files:
        table = pq.read_table(fpath)
        x = np.asarray(table.column("activation").to_pylist(), dtype=np.float32)
        finite = np.isfinite(x).all(axis=1)
        n_nonfinite += int((~finite).sum())
        ids.extend(
            i
            for i, keep in zip(table.column("exchange_id").to_pylist(), finite, strict=True)
            if keep
        )
        xs.append(x[finite])
        ys.extend(
            y for y, keep in zip(table.column("flagged").to_pylist(), finite, strict=True) if keep
        )
    if n_nonfinite:
        log.info(f"{split_dir}: dropped {n_nonfinite} rows with non-finite activations")

    return {
        "exchange_id": np.asarray(ids),
        "x": np.concatenate(xs) if xs else np.zeros((0, 0), dtype=np.float32),
        "y": np.asarray(ys, dtype=np.float32),
    }


def assert_no_overlap(train: dict, holdout: dict) -> None:
    overlap = set(train["exchange_id"].tolist()) & set(holdout["exchange_id"].tolist())
    assert not overlap, f"train/holdout exchange_id overlap in caches: {len(overlap)}"
