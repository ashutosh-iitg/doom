"""§5's backbone: only the pooling logic is unit-tested, against a
hand-built `hidden_states` tuple — never a real HF model download in this
suite (that's a manual/e2e step, same status as the existing README's
live-model snippet for `ExchangeClassifier`).
"""

import pytest

torch = pytest.importorskip("torch")

from doom.probe.backbone import Backbone, _pool_last_token


def test_pool_last_token_reads_the_configured_layer():
    layer0 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (batch=1, seq=2, dim=2)
    layer1 = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
    hidden_states = (layer0, layer1)

    assert torch.equal(_pool_last_token(hidden_states, layer=0), torch.tensor([3.0, 4.0]))
    assert torch.equal(_pool_last_token(hidden_states, layer=1), torch.tensor([7.0, 8.0]))
    assert torch.equal(_pool_last_token(hidden_states, layer=-1), torch.tensor([7.0, 8.0]))


def test_backbone_requires_a_model_name_or_env_var(monkeypatch):
    monkeypatch.delenv("DOOM_PROBE_BACKBONE", raising=False)

    with pytest.raises(ValueError, match="DOOM_PROBE_BACKBONE"):
        Backbone()
