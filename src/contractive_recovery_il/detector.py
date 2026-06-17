"""Uncertainty detector: RND novelty score with conformal-calibrated thresholds.

The novelty signal is Random Network Distillation (``features.RNDUncertainty``).  The
trigger thresholds are sized by split-conformal calibration (``conformal``) so the
on-manifold false-trigger rate is controlled, with a hysteresis gap between the
``tau_on`` (enter recovery) and ``tau_off`` (leave recovery) levels.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .conformal import conformal_threshold
from .features import RNDUncertainty, fit_rnd


@dataclass
class RNDDetector:
    """RND uncertainty with conformal-calibrated on/off thresholds."""

    rnd: RNDUncertainty
    tau_on: float = 1.0
    tau_off: float = 0.7

    def score(self, state: np.ndarray) -> float:
        return self.rnd.score(state)

    def scores(self, states: np.ndarray) -> np.ndarray:
        return self.rnd.scores(states)


def calibrate_detector(
    train_states: np.ndarray,
    calibration_states: np.ndarray,
    *,
    quantile: float,
    tau_on_scale: float,
    tau_off_scale: float,
    seed: int = 0,
) -> RNDDetector:
    """Fit RND on the training manifold and conformally calibrate the trigger.

    ``quantile`` is read as the target coverage ``1 - alpha`` of the conformal base
    threshold; ``tau_on_scale`` / ``tau_off_scale`` set the enter/leave levels around
    that base (``tau_off <= tau_on``), giving switching hysteresis.
    """
    train_states = np.asarray(train_states, dtype=float)
    rnd = fit_rnd(train_states, seed=seed)
    cal_scores = rnd.scores(np.asarray(calibration_states, dtype=float))
    base = conformal_threshold(cal_scores, alpha=1.0 - quantile)
    base = max(base, 1e-6)  # nonzero floor so tiny-noise calibration does not chatter
    tau_on = base * tau_on_scale
    tau_off = min(tau_on, base * tau_off_scale)
    return RNDDetector(rnd=rnd, tau_on=tau_on, tau_off=tau_off)
