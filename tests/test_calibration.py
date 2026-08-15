"""§4 calibration: sweep candidate thresholds against labelled exchanges and
report escalation rate / missed-violation rate per threshold — the
measurement `DEFAULT_THRESHOLD` in `cascade.py` is explicitly a placeholder
for, per CLAUDE.md.

Mocked throughout: a stub screen returning fixed scores, same pattern as
`test_cascade.py`'s `_StubScreen`.
"""

from doom import Exchange, LabelledExchange
from doom.calibration import sweep_thresholds
from doom.constitutions.cbrn_example import CBRN_EXAMPLE


class _StubScreen:
    def __init__(self, scores: dict[str, float]):
        self._scores = scores

    def score(self, exchange, constitution) -> float:
        return self._scores[exchange.render()]


def _labelled(exchange: Exchange, flagged: bool) -> LabelledExchange:
    return LabelledExchange(exchange=exchange, flagged=flagged)


def test_a_high_threshold_escalates_nothing_and_misses_every_violation():
    flagged_exchange = Exchange.of(user="u1", assistant="a1")
    safe_exchange = Exchange.of(user="u2", assistant="a2")
    screen = _StubScreen({flagged_exchange.render(): 0.5, safe_exchange.render(): 0.1})
    labelled = [_labelled(flagged_exchange, True), _labelled(safe_exchange, False)]

    stats = sweep_thresholds(screen, CBRN_EXAMPLE, labelled, thresholds=[1.1])[0]

    assert stats.escalation_rate == 0.0
    assert stats.missed_violation_rate == 1.0


def test_a_low_threshold_escalates_everything_and_misses_nothing():
    flagged_exchange = Exchange.of(user="u1", assistant="a1")
    safe_exchange = Exchange.of(user="u2", assistant="a2")
    screen = _StubScreen({flagged_exchange.render(): 0.5, safe_exchange.render(): 0.1})
    labelled = [_labelled(flagged_exchange, True), _labelled(safe_exchange, False)]

    stats = sweep_thresholds(screen, CBRN_EXAMPLE, labelled, thresholds=[0.0])[0]

    assert stats.escalation_rate == 1.0
    assert stats.missed_violation_rate == 0.0


def test_the_threshold_is_inclusive_at_its_boundary_matching_cascade():
    """`Cascade.evaluate` escalates when `score >= threshold` (see
    `test_cascade.py::test_the_threshold_is_inclusive_at_its_boundary`) — the
    calibration sweep must use the same boundary rule or it would recommend
    thresholds `Cascade` doesn't actually honor."""
    exchange = Exchange.of(user="u", assistant="a")
    screen = _StubScreen({exchange.render(): 0.15})
    labelled = [_labelled(exchange, True)]

    stats = sweep_thresholds(screen, CBRN_EXAMPLE, labelled, thresholds=[0.15])[0]

    assert stats.escalation_rate == 1.0
    assert stats.missed_violation_rate == 0.0


def test_sweeping_multiple_thresholds_returns_one_stat_per_threshold_in_order():
    exchange = Exchange.of(user="u", assistant="a")
    screen = _StubScreen({exchange.render(): 0.4})
    labelled = [_labelled(exchange, True)]

    stats = sweep_thresholds(screen, CBRN_EXAMPLE, labelled, thresholds=[0.1, 0.5, 0.9])

    assert [s.threshold for s in stats] == [0.1, 0.5, 0.9]
    assert [s.escalation_rate for s in stats] == [1.0, 0.0, 0.0]


def test_missed_violation_rate_is_zero_when_no_labelled_exchange_is_flagged():
    """Avoids a divide-by-zero: with no positive labels in the set, "missed
    violations" is vacuously zero, not undefined."""
    exchange = Exchange.of(user="u", assistant="a")
    screen = _StubScreen({exchange.render(): 0.9})
    labelled = [_labelled(exchange, False)]

    stats = sweep_thresholds(screen, CBRN_EXAMPLE, labelled, thresholds=[0.95])[0]

    assert stats.missed_violation_rate == 0.0


def test_stats_record_how_many_exchanges_were_swept():
    exchange = Exchange.of(user="u", assistant="a")
    screen = _StubScreen({exchange.render(): 0.5})
    labelled = [_labelled(exchange, True)]

    stats = sweep_thresholds(screen, CBRN_EXAMPLE, labelled, thresholds=[0.1])[0]

    assert stats.n == 1
