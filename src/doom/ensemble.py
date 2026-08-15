"""§6's weighted ensemble: `z = alpha * z_probe + (1 - alpha) * z_screen`.

Both members already produce a real, logprob/activation-derived score in
`[0, 1]` — unlike `ExchangeClassifier`, which is prompted and tool-forced
and therefore has no calibrated probability to blend (see CLAUDE.md's §6
decision). `EnsembleScreen` implements the same `score(exchange,
constitution) -> float` shape `Screen`/`LinearProbe` already have, so it
drops into `Cascade` as its `screen` argument with no change to `Cascade`
itself.

Typed structurally against `ScreenLike` (the same protocol `calibration.py`
uses), not against `Screen`/`LinearProbe` by name — so this module carries
no dependency on `doom.probe` (and therefore no torch import), even though
its intended use is combining a `Screen` with a `LinearProbe`.
"""

from dataclasses import dataclass

from .calibration import ScreenLike
from .constitution import Constitution
from .exchange import Exchange


@dataclass(frozen=True)
class EnsembleScreen:
    screen: ScreenLike
    probe: ScreenLike
    alpha: float

    def score(self, exchange: Exchange, constitution: Constitution) -> float:
        return self.alpha * self.probe.score(exchange, constitution) + (
            1 - self.alpha
        ) * self.screen.score(exchange, constitution)
