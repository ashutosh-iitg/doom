"""§6's weighted ensemble: `z = alpha * z_probe + (1 - alpha) * z_screen`.

Per CLAUDE.md's §6 decision, the ensemble combines the probe with `Screen`,
not with the prompted `ExchangeClassifier` — the judge returns a boolean
`Verdict`, not a calibrated probability, so it has no continuous score to
blend. Both members here are stubs implementing the same duck-typed
`score(exchange, constitution) -> float` shape `Screen`/`LinearProbe` share;
no torch import needed for this test.
"""

from doom import Cascade, Exchange, Verdict
from doom.constitutions.cbrn_example import CBRN_EXAMPLE
from doom.ensemble import EnsembleScreen

EXCHANGE = Exchange.of(user="u", assistant="a")


class _StubScreen:
    def __init__(self, score: float):
        self._score = score

    def score(self, exchange, constitution) -> float:
        return self._score


class _StubJudge:
    def judge(self, exchange, constitution) -> Verdict:
        return Verdict(flagged=True, reasoning="stub")


def test_alpha_one_is_the_probe_alone():
    ensemble = EnsembleScreen(screen=_StubScreen(0.0), probe=_StubScreen(0.9), alpha=1.0)

    assert ensemble.score(EXCHANGE, CBRN_EXAMPLE) == 0.9


def test_alpha_zero_is_the_screen_alone():
    ensemble = EnsembleScreen(screen=_StubScreen(0.7), probe=_StubScreen(0.0), alpha=0.0)

    assert ensemble.score(EXCHANGE, CBRN_EXAMPLE) == 0.7


def test_alpha_half_averages_the_two_scores():
    ensemble = EnsembleScreen(screen=_StubScreen(0.2), probe=_StubScreen(0.8), alpha=0.5)

    assert ensemble.score(EXCHANGE, CBRN_EXAMPLE) == 0.5


def test_an_uneven_alpha_weights_the_probe_more():
    ensemble = EnsembleScreen(screen=_StubScreen(0.0), probe=_StubScreen(1.0), alpha=0.9)

    assert abs(ensemble.score(EXCHANGE, CBRN_EXAMPLE) - 0.9) < 1e-9


def test_an_ensemble_screen_satisfies_cascades_screen_interface():
    """`Cascade` only ever calls `.score(exchange, constitution)` — the
    ensemble must be usable as `Cascade`'s screen with zero changes to
    `Cascade` itself, exactly like a bare `Screen` or a bare `LinearProbe`."""
    ensemble = EnsembleScreen(screen=_StubScreen(0.6), probe=_StubScreen(0.6), alpha=0.5)
    cascade = Cascade(ensemble, _StubJudge(), CBRN_EXAMPLE, threshold=0.5)

    out = cascade.evaluate(EXCHANGE)

    assert out.escalated is True
