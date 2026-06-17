"""Latent encoder and Random Network Distillation (RND) uncertainty.

These are the two learned-from-demonstration components that the CURE-IL method
relies on:

- ``LatentEncoder`` is the encoder ``y = phi(s)``.  It is a linear (PCA-whitening)
  autoencoder fit on demonstration states, so the latent space is isotropic and the
  map is exactly invertible -- which lets the contractive recovery field defined in
  latent space be decoded back into a state-space action.
- ``RNDUncertainty`` is the Random Network Distillation novelty score used to detect
  when a rollout has left the demonstrated regime.  A fixed random *target* network
  produces an embedding of the state; a *predictor* is distilled to match the target
  on demonstration states only.  Off-manifold states are poorly predicted, so the
  prediction error is a calibrated novelty signal.

Both are deterministic given a seed and implemented in pure NumPy.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# --------------------------------------------------------------------------- #
# Latent encoder  y = phi(s)
# --------------------------------------------------------------------------- #
@dataclass
class LatentEncoder:
    """Invertible linear encoder fit on demonstration states (PCA whitening).

    ``encode`` maps a state to a whitened latent coordinate; ``decode`` is its exact
    inverse.  Because the map is linear, a latent velocity ``ydot`` corresponds to the
    state velocity ``W_inv @ ydot`` -- used to turn a latent contraction field into an
    executable action.
    """

    mean: np.ndarray
    W: np.ndarray        # whitening matrix:   y = W (s - mean)
    W_inv: np.ndarray    # un-whitening:        s = mean + W_inv y

    @property
    def latent_dim(self) -> int:
        return self.W.shape[0]

    def encode(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=float)
        return (s - self.mean) @ self.W.T

    def decode(self, latent: np.ndarray) -> np.ndarray:
        return self.mean + np.asarray(latent, dtype=float) @ self.W_inv.T

    def decode_velocity(self, latent_velocity: np.ndarray) -> np.ndarray:
        """Map a latent-space velocity back to a state-space velocity."""
        return np.asarray(latent_velocity, dtype=float) @ self.W_inv.T


def fit_latent_encoder(states: np.ndarray, *, eps: float = 1e-6) -> LatentEncoder:
    states = np.asarray(states, dtype=float)
    mean = states.mean(axis=0)
    centered = states - mean
    cov = np.cov(centered, rowvar=False)
    cov = np.atleast_2d(cov)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, eps, None)
    inv_sqrt = np.diag(1.0 / np.sqrt(evals))
    sqrt = np.diag(np.sqrt(evals))
    W = inv_sqrt @ evecs.T          # whitening
    W_inv = evecs @ sqrt            # exact inverse
    return LatentEncoder(mean=mean, W=W, W_inv=W_inv)


# --------------------------------------------------------------------------- #
# RND uncertainty
# --------------------------------------------------------------------------- #
def _random_fourier_features(states: np.ndarray, omega: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """cos(states @ omega + phase) random Fourier feature map."""
    states = np.atleast_2d(np.asarray(states, dtype=float))
    return np.sqrt(2.0 / omega.shape[1]) * np.cos(states @ omega + phase[None, :])


@dataclass
class RNDUncertainty:
    """Random Network Distillation novelty score.

    The *target* is a fixed random two-layer network ``g(s)``.  The *predictor* is a
    ridge-regression head on random Fourier features of the state, fit (distilled) to
    reproduce ``g(s)`` on the demonstration manifold.  ``score(s)`` is the squared
    prediction error, which is small on-manifold and grows off-manifold.
    """

    # target network params (random, fixed)
    t_w1: np.ndarray
    t_b1: np.ndarray
    t_w2: np.ndarray
    # predictor: ridge head on random Fourier features
    rff_omega: np.ndarray
    rff_phase: np.ndarray
    pred_head: np.ndarray   # (n_features, out_dim)
    # normalisation
    score_scale: float = 1.0

    def _target(self, states: np.ndarray) -> np.ndarray:
        states = np.atleast_2d(np.asarray(states, dtype=float))
        h = np.tanh(states @ self.t_w1 + self.t_b1[None, :])
        return h @ self.t_w2

    def _predict(self, states: np.ndarray) -> np.ndarray:
        feats = _random_fourier_features(states, self.rff_omega, self.rff_phase)
        return feats @ self.pred_head

    def score(self, state: np.ndarray) -> float:
        err = self._target(state) - self._predict(state)
        return float(np.sum(err ** 2)) * self.score_scale

    def scores(self, states: np.ndarray) -> np.ndarray:
        states = np.atleast_2d(np.asarray(states, dtype=float))
        err = self._target(states) - self._predict(states)
        return np.sum(err ** 2, axis=1) * self.score_scale


def fit_rnd(
    train_states: np.ndarray,
    *,
    seed: int = 0,
    embed_dim: int = 16,
    hidden_dim: int = 64,
    n_features: int = 256,
    ridge: float = 1e-2,
) -> RNDUncertainty:
    """Fit an RND novelty model on in-distribution demonstration states."""
    train_states = np.asarray(train_states, dtype=float)
    rng = np.random.default_rng(seed)
    d = train_states.shape[1]

    # Standardise inputs so the random networks see a well-scaled domain.
    mu = train_states.mean(axis=0)
    sd = train_states.std(axis=0) + 1e-8
    xs = (train_states - mu) / sd

    # --- fixed random target network g(s) ---
    t_w1 = rng.normal(0.0, 1.0, size=(d, hidden_dim))
    t_b1 = rng.normal(0.0, 1.0, size=(hidden_dim,))
    t_w2 = rng.normal(0.0, 1.0, size=(hidden_dim, embed_dim))

    # --- predictor: random Fourier features + ridge regression head ---
    # Bandwidth from the median pairwise distance keeps features informative.
    sample = xs[rng.choice(len(xs), size=min(len(xs), 400), replace=False)]
    pdist = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=2)
    med = np.median(pdist[pdist > 0]) if np.any(pdist > 0) else 1.0
    gamma = 1.0 / (2.0 * (med ** 2) + 1e-8)
    rff_omega = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(d, n_features))
    rff_phase = rng.uniform(0.0, 2.0 * np.pi, size=(n_features,))

    # Fold the input standardisation into the stored maps so callers pass raw states.
    t_w1_raw = (t_w1.T / sd).T
    t_b1_raw = t_b1 - (mu / sd) @ t_w1
    rff_omega_raw = (rff_omega.T / sd).T
    rff_phase_raw = rff_phase - (mu / sd) @ rff_omega

    model = RNDUncertainty(
        t_w1=t_w1_raw, t_b1=t_b1_raw, t_w2=t_w2,
        rff_omega=rff_omega_raw, rff_phase=rff_phase_raw,
        pred_head=np.zeros((n_features, embed_dim)),
    )

    feats = _random_fourier_features(train_states, model.rff_omega, model.rff_phase)
    targets = model._target(train_states)
    gram = feats.T @ feats + ridge * np.eye(n_features)
    model.pred_head = np.linalg.solve(gram, feats.T @ targets)

    # Normalise so a typical in-distribution score is O(1); keeps thresholds readable.
    train_scores = model.scores(train_states)
    med_score = float(np.median(train_scores)) + 1e-8
    model.score_scale = 1.0 / med_score
    return model
