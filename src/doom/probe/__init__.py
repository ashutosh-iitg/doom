"""§5's linear probe: `exchange, constitution -> score in [0, 1]` read from a
frozen local backbone's hidden states, the same shape `Screen.score` has.

Deliberately **not** imported from `doom/__init__.py` — this subpackage is
doom's only dependency on torch/transformers/pyarrow/safetensors, all of
which are the optional `probe` dependency group, not core dependencies.
Import from here explicitly (`from doom.probe import LinearProbe`).
"""

from .backbone import Backbone
from .head import ProbeHead
from .probe import LinearProbe

__all__ = ["Backbone", "LinearProbe", "ProbeHead"]
