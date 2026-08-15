"""`LinearProbe`: §5's `exchange, constitution -> score in [0, 1]` — the
same shape `Screen.score` has, so it drops into `Cascade` as a screen on its
own, independent of §6's ensemble.
"""

from ..constitution import Constitution
from ..exchange import Exchange
from .backbone import Backbone
from .head import ProbeHead


class LinearProbe:
    def __init__(self, backbone: Backbone, head: ProbeHead):
        self.backbone = backbone
        self.head = head

    def score(self, exchange: Exchange, constitution: Constitution) -> float:
        activation = self.backbone.activation(exchange, constitution)
        return float(self.head.score(activation.unsqueeze(0)).item())
