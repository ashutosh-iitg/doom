"""`Backbone`: §5's frozen half of the linear probe — one forward pass over
a local HF `transformers` causal LM, never fine-tuned.

Loaded via PyTorch, not vLLM: extracting hidden states needs an in-process
forward pass over model weights, which is architecturally out of scope for
`emissary`'s HTTP-only wire adapters (see CLAUDE.md's §5 decision — emissary
has exactly two wire formats, both remote/local API calls, never a direct
model load). This module is `doom`'s only place that imports torch/
transformers; it is not imported from `doom/__init__.py`.
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..constitution import Constitution
from ..exchange import Exchange

DEFAULT_LAYER = -1
"""The final hidden layer, by default — a config knob (`layer=`), not a
hardcoded choice."""


def _pool_last_token(hidden_states: tuple, layer: int) -> torch.Tensor:
    """The last token's hidden state at `layer` — the simplest, most
    standard pooling for a causal LM (no CLS token to reach for instead)."""
    return hidden_states[layer][0, -1, :]


class Backbone:
    """A frozen local causal LM, read for its hidden states only."""

    def __init__(self, model_name: str | None = None, layer: int = DEFAULT_LAYER):
        self.model_name = model_name or os.environ.get("DOOM_PROBE_BACKBONE")
        if not self.model_name:
            raise ValueError(
                "no backbone model configured — pass model_name= or set DOOM_PROBE_BACKBONE"
            )
        self.layer = layer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.eval()

    @torch.no_grad()
    def activation(self, exchange: Exchange, constitution: Constitution) -> torch.Tensor:
        """The pooled hidden state for the constitution rendered ahead of
        the exchange — the same content `Screen`/`ExchangeClassifier` are
        shown, consumed as activations instead of text or logprobs."""
        text = (
            f"# Constitution: {constitution.name}\n\n{constitution.render()}\n\n"
            f"# Exchange to evaluate\n\n{exchange.render()}"
        )
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs, output_hidden_states=True)
        return _pool_last_token(outputs.hidden_states, self.layer)
