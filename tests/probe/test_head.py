"""`ProbeHead`: tiny deterministic tensors, no randomness — mirrors bdsm's
own shape/param-cap test style (`bdsm/tests/test_train_head.py`), not a
gradient-descent test (that's `test_train.py`).

Skips cleanly if the `probe` dependency group isn't installed — `torch` is
opt-in for doom, not a core dependency (see CLAUDE.md's §5 decision).
"""

import json

import pytest

torch = pytest.importorskip("torch")

from doom.probe.head import MAX_TRAINABLE_PARAMS, ProbeHead


def test_a_linear_head_has_exactly_emb_dim_plus_one_params():
    head = ProbeHead(emb_dim=4, hidden_dims=())

    n_trainable = sum(p.numel() for p in head.parameters())

    assert n_trainable == 4 * 1 + 1


def test_an_mlp_head_stacks_hidden_layers():
    head = ProbeHead(emb_dim=4, hidden_dims=(8,))

    n_trainable = sum(p.numel() for p in head.parameters())

    assert n_trainable == (4 * 8 + 8) + (8 * 1 + 1)


def test_a_head_over_the_param_cap_is_rejected():
    with pytest.raises(AssertionError):
        ProbeHead(emb_dim=MAX_TRAINABLE_PARAMS, hidden_dims=())


def test_score_is_a_probability_for_every_input_row():
    head = ProbeHead(emb_dim=4, hidden_dims=(8,))
    x = torch.zeros((3, 4))

    scores = head.score(x)

    assert scores.shape == (3,)
    assert torch.all((scores >= 0) & (scores <= 1))


def test_save_then_load_round_trips_identical_scores(tmp_path):
    head = ProbeHead(emb_dim=4, hidden_dims=(8,))
    x = torch.randn((2, 4))
    before = head.score(x)

    head.save(str(tmp_path))
    loaded = ProbeHead.load(str(tmp_path), emb_dim=4, hidden_dims=(8,))

    assert torch.allclose(before, loaded.score(x))


def test_load_rejects_a_manifest_with_a_missing_key(tmp_path):
    head = ProbeHead(emb_dim=4, hidden_dims=())
    head.save(str(tmp_path))
    manifest_path = tmp_path / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["params"][next(iter(manifest["params"]))]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="missing"):
        ProbeHead.load(str(tmp_path), emb_dim=4, hidden_dims=())


def test_load_rejects_a_shape_mismatch(tmp_path):
    head = ProbeHead(emb_dim=4, hidden_dims=())
    head.save(str(tmp_path))

    with pytest.raises(ValueError, match="emb_dim"):
        ProbeHead.load(str(tmp_path), emb_dim=8, hidden_dims=())
