"""A real, tiny gradient-descent test for §5's training loop.

bdsm's own `tests/test_train_head.py` never exercises its actual training
step — only shapes and data loading. This head is cheap enough on CPU to
actually train for real on synthetic, linearly-separable data and check it
converges; there's no excuse to skip that the way bdsm did.
"""

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
torch = pytest.importorskip("torch")

from doom.probe import train as train_probe
from doom.probe.data import load_cached
from doom.probe.head import ProbeHead
from doom.probe.metrics import average_precision


def _write_linearly_separable_cache(split_dir: Path, n: int, dim: int, seed: int) -> None:
    split_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim)).astype(np.float32)
    y = x[:, 0] > 0
    table = pa.table(
        {
            "exchange_id": pa.array([f"{split_dir.name}-{i}" for i in range(n)], pa.string()),
            "activation": pa.array(x.tolist(), pa.list_(pa.float32())),
            "flagged": pa.array(y.tolist(), pa.bool_()),
        }
    )
    pq.write_table(table, split_dir / "part-000.parquet")


def test_a_tiny_real_training_run_converges_and_writes_a_checkpoint(tmp_path):
    cls_dir = tmp_path / "cls"
    _write_linearly_separable_cache(cls_dir / "train", n=128, dim=4, seed=1)
    _write_linearly_separable_cache(cls_dir / "holdout", n=32, dim=4, seed=2)
    checkpoint_dir = tmp_path / "checkpoints"

    exit_code = train_probe.main(
        [
            "--cls-dir",
            str(cls_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--run-name",
            "test-run",
            "--steps",
            "600",
            "--batch-size",
            "16",
            "--peak-lr",
            "1e-2",
            "--warmup-steps",
            "20",
            "--eval-every",
            "300",
        ]
    )

    assert exit_code == 0
    run_dir = checkpoint_dir / "test-run"
    step_dir = Path((run_dir / "latest.txt").read_text().strip())
    assert (step_dir / "head.safetensors").exists()
    assert (step_dir / "MANIFEST.json").exists()
    assert (step_dir / "config.json").exists()

    config = json.loads((step_dir / "config.json").read_text())
    assert config["n_train"] == 128
    assert config["steps"] == 600

    head = ProbeHead.load(str(step_dir), emb_dim=4)
    holdout = load_cached(str(cls_dir / "holdout"))
    scores = head.score(torch.from_numpy(holdout["x"])).detach().numpy()
    ap = average_precision(holdout["y"], scores)

    assert ap > 0.9


def test_check_finite_raises_on_nan_loss():
    with pytest.raises(RuntimeError, match="non-finite"):
        train_probe._check_finite(float("nan"), 1.0, step=5)


def test_check_finite_raises_on_inf_grad_norm():
    with pytest.raises(RuntimeError, match="non-finite"):
        train_probe._check_finite(1.0, float("inf"), step=5)


def test_check_finite_allows_normal_values():
    train_probe._check_finite(0.5, 1.2, step=5)
