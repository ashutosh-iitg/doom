"""Hand-rolled average precision (`doom.probe.metrics`) — no scikit-learn
dependency for one function, same call bdsm's own `runtime/metrics.py`
makes. Checked against known small cases, not against another library.
"""

import pytest

np = pytest.importorskip("numpy")

from doom.probe.metrics import average_precision


def test_a_perfect_ranking_has_average_precision_one():
    y_true = np.array([1, 1, 0, 0], dtype=np.float32)
    y_score = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)

    assert average_precision(y_true, y_score) == 1.0


def test_the_worst_possible_ranking_scores_low():
    y_true = np.array([1, 1, 0, 0], dtype=np.float32)
    y_score = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)

    assert average_precision(y_true, y_score) < 0.7


def test_no_positive_labels_is_zero_not_undefined():
    y_true = np.array([0, 0, 0], dtype=np.float32)
    y_score = np.array([0.5, 0.2, 0.9], dtype=np.float32)

    assert average_precision(y_true, y_score) == 0.0


def test_a_known_three_point_case():
    """Ranking [pos, neg, pos]: precision at each positive is 1/1 and 2/3 —
    the textbook worked example for average precision."""
    y_true = np.array([1, 0, 1], dtype=np.float32)
    y_score = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    ap = average_precision(y_true, y_score)

    assert abs(ap - ((1 / 1 + 2 / 3) / 2)) < 1e-6
