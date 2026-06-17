"""Split-conformal calibration of the uncertainty trigger threshold.

Unlike a plain empirical quantile, the split-conformal threshold uses the
finite-sample rank correction ``ceil((n+1)(1-alpha)) / n`` and carries a coverage
guarantee: on exchangeable in-distribution data, the probability that a fresh
calibration score exceeds the threshold is at most ``alpha``.  CURE-IL uses this to
size the recovery trigger so that the nominal (on-manifold) false-trigger rate is
controlled at a chosen level rather than hand-tuned.
"""
from __future__ import annotations

import numpy as np


def conformal_threshold(cal_scores: np.ndarray, alpha: float) -> float:
    """Split-conformal upper threshold at miscoverage level ``alpha``.

    Returns ``tau`` such that ``P(score_new > tau) <= alpha`` for an exchangeable
    fresh in-distribution score.  ``alpha`` near 0 gives a loose (high) threshold that
    rarely fires on clean rollouts; larger ``alpha`` gives a tighter (lower) one.
    """
    scores = np.sort(np.asarray(cal_scores, dtype=float))
    n = len(scores)
    if n == 0:
        raise ValueError("need at least one calibration score")
    alpha = float(np.clip(alpha, 1.0 / (n + 1), 1.0))
    # Rank of the conformal quantile (1-indexed), capped at n.
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    return float(scores[rank - 1])
