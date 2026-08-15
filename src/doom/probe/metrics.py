"""Hand-rolled average precision — avoids pulling in scikit-learn for one
function, the same call bdsm's own `runtime/metrics.py` makes.
"""

import numpy as np


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact area under the precision-recall curve (no interpolation),
    computed by sweeping score thresholds from highest to lowest.

    Zero when there are no positive labels — vacuously, not undefined —
    rather than raising or returning NaN, since a labelled holdout split can
    legitimately contain zero flagged exchanges.
    """
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp_cumsum = np.cumsum(y_sorted)
    fp_cumsum = np.cumsum(1 - y_sorted)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / n_pos

    recall_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_prev) * precision))
