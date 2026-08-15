"""`LinearProbe`: the same `exchange, constitution -> score in [0, 1]` shape
`Screen.score` has, tested against a stub backbone — no real model."""

import pytest

torch = pytest.importorskip("torch")

from doom import Cascade, Exchange, Verdict
from doom.constitutions.cbrn_example import CBRN_EXAMPLE
from doom.probe.head import ProbeHead
from doom.probe.probe import LinearProbe


class _StubBackbone:
    def __init__(self, vector):
        self._vector = vector

    def activation(self, exchange, constitution):
        return self._vector


class _StubJudge:
    def judge(self, exchange, constitution):
        return Verdict(flagged=True, reasoning="stub")


def test_score_is_a_float_probability():
    head = ProbeHead(emb_dim=4, hidden_dims=())
    probe = LinearProbe(_StubBackbone(torch.zeros(4)), head)

    score = probe.score(Exchange.of(user="u", assistant="a"), CBRN_EXAMPLE)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_score_matches_the_head_applied_to_the_backbones_activation():
    head = ProbeHead(emb_dim=4, hidden_dims=())
    vector = torch.tensor([1.0, -2.0, 0.5, 3.0])
    probe = LinearProbe(_StubBackbone(vector), head)

    expected = float(head.score(vector.unsqueeze(0)).item())
    score = probe.score(Exchange.of(user="u", assistant="a"), CBRN_EXAMPLE)

    assert score == expected


def test_a_linear_probe_satisfies_cascades_screen_interface():
    """`Cascade` only ever calls `.score(exchange, constitution)` on its
    screen — a `LinearProbe` must be usable wherever a `Screen` is, with
    zero changes to `Cascade` itself."""
    head = ProbeHead(emb_dim=4, hidden_dims=())
    probe = LinearProbe(_StubBackbone(torch.zeros(4)), head)
    cascade = Cascade(probe, _StubJudge(), CBRN_EXAMPLE, threshold=0.0)

    out = cascade.evaluate(Exchange.of(user="u", assistant="a"))

    assert out.escalated is True
