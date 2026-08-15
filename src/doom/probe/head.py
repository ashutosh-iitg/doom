"""`ProbeHead`: the trainable half of §5's linear probe over a frozen
backbone.

Kept deliberately small — a genuine probe, not a second model; the
`MAX_TRAINABLE_PARAMS` cap is the difference. `save`/`load` round-trip
through `safetensors` (no pickle) plus a `MANIFEST.json` checked before any
tensor data is trusted, mirroring the key-set-then-shape/dtype-then-
optional-hash discipline `bdsm/runtime/load_backbone.py` applies to its own
npz checkpoints.
"""

import hashlib
import json
import os
from collections.abc import Sequence

import torch
from safetensors.torch import load_file, save_file
from torch import nn

MAX_TRAINABLE_PARAMS = 2_000_000
"""A probe, not a model. Exceeding this means the head has stopped being a
lightweight readout of a frozen backbone."""


class ProbeHead(nn.Module):
    """Linear (`hidden_dims=()`) or shallow-MLP readout to one flagged logit."""

    def __init__(self, emb_dim: int, hidden_dims: Sequence[int] = ()):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dims = tuple(hidden_dims)

        dims = (emb_dim, *self.hidden_dims, 1)
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.layers = nn.Sequential(*layers)

        n_trainable = sum(p.numel() for p in self.parameters())
        assert (
            n_trainable <= MAX_TRAINABLE_PARAMS
        ), f"trainable params {n_trainable:,} exceeds the {MAX_TRAINABLE_PARAMS:,} probe cap"

    def logit(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logit(x))

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        tensors = {k: v.detach().contiguous().cpu() for k, v in self.state_dict().items()}
        save_file(tensors, os.path.join(out_dir, "head.safetensors"))

        manifest = {
            "emb_dim": self.emb_dim,
            "hidden_dims": list(self.hidden_dims),
            "params": {
                k: {
                    "shape": list(v.shape),
                    "dtype": str(v.dtype).removeprefix("torch."),
                    "sha256": hashlib.sha256(v.numpy().tobytes()).hexdigest(),
                }
                for k, v in tensors.items()
            },
        }
        with open(os.path.join(out_dir, "MANIFEST.json"), "w") as f:
            json.dump(manifest, f, indent=1)

    @classmethod
    def load(
        cls,
        in_dir: str,
        emb_dim: int,
        hidden_dims: Sequence[int] = (),
        verify_hashes: bool = False,
    ) -> "ProbeHead":
        with open(os.path.join(in_dir, "MANIFEST.json")) as f:
            manifest = json.load(f)
        if manifest["emb_dim"] != emb_dim or manifest["hidden_dims"] != list(hidden_dims):
            raise ValueError(
                f"checkpoint was saved with emb_dim={manifest['emb_dim']}, "
                f"hidden_dims={manifest['hidden_dims']}, but load() was called with "
                f"emb_dim={emb_dim}, hidden_dims={list(hidden_dims)}"
            )
        tensors = load_file(os.path.join(in_dir, "head.safetensors"))

        entries = manifest["params"]
        if set(entries) != set(tensors):
            missing = sorted(set(entries) - set(tensors))
            extra = sorted(set(tensors) - set(entries))
            raise ValueError(f"key mismatch vs manifest: missing={missing} extra={extra}")
        for key, tensor in tensors.items():
            meta = entries[key]
            dtype = str(tensor.dtype).removeprefix("torch.")
            if list(tensor.shape) != meta["shape"] or dtype != meta["dtype"]:
                raise ValueError(
                    f"{key}: got {list(tensor.shape)}/{dtype}, "
                    f"manifest says {meta['shape']}/{meta['dtype']}"
                )
            if verify_hashes:
                digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
                if digest != meta["sha256"]:
                    raise ValueError(f"{key}: sha256 mismatch")

        head = cls(emb_dim, hidden_dims)
        head.load_state_dict(tensors)
        return head
