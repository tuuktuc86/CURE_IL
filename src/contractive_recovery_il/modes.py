"""Behavior-mode separation and trajectory tubes in latent space.

Multimodal demonstrations are clustered into ``K`` behavior modes.  Each mode keeps
its own *tube* -- represented by a spine (the index-aligned mean latent trajectory of
the demonstrations assigned to that mode).  At recovery time CURE-IL projects the
current latent state onto a mode's spine to obtain the tangent direction and the
perpendicular error used by the contraction field, and compares a recover-cost against
a switch-cost to decide which mode to contract toward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .features import LatentEncoder


@dataclass(frozen=True)
class TubeProjection:
    point: np.ndarray      # nearest spine vertex (latent)
    tangent: np.ndarray    # unit tangent at that vertex (latent)
    e_perp: np.ndarray     # perpendicular error, orthogonal to tangent (latent)
    distance: float        # distance from latent state to the spine
    index: int
    length: int


def _kmeans(points: np.ndarray, k: int, *, seed: int, iters: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(points)
    if k <= 1 or n <= k:
        # one cluster (or one per point) -- nothing to separate.
        return np.zeros(n, dtype=int) if k <= 1 else np.arange(n) % k
    centers = points[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            if np.any(labels == j):
                centers[j] = points[labels == j].mean(axis=0)
    return labels


@dataclass
class ModeModel:
    """K behavior modes, each with a latent spine (trajectory tube)."""

    encoder: LatentEncoder
    spines: List[np.ndarray]            # one (L, d) latent spine per mode

    @property
    def K(self) -> int:
        return len(self.spines)

    def project(self, latent: np.ndarray, mode: int) -> TubeProjection:
        spine = self.spines[mode]
        d = np.linalg.norm(spine - latent[None, :], axis=1)
        idx = int(np.argmin(d))
        if idx <= 0:
            tangent = spine[1] - spine[0]
        elif idx >= len(spine) - 1:
            tangent = spine[-1] - spine[-2]
        else:
            tangent = spine[idx + 1] - spine[idx - 1]
        tangent = tangent / (np.linalg.norm(tangent) + 1e-12)
        e_total = latent - spine[idx]
        e_perp = e_total - float(e_total @ tangent) * tangent
        return TubeProjection(spine[idx].copy(), tangent, e_perp, float(d[idx]), idx, len(spine))

    def distance_to_mode(self, latent: np.ndarray, mode: int) -> float:
        spine = self.spines[mode]
        return float(np.min(np.linalg.norm(spine - latent[None, :], axis=1)))

    def assign(self, latent: np.ndarray) -> int:
        return int(np.argmin([self.distance_to_mode(latent, z) for z in range(self.K)]))

    def cost(self, latent: np.ndarray, mode: int, uncertainty: float, tau: float,
             *, lambda_d: float, lambda_u: float) -> float:
        """J(s) = lambda_d ||e_perp||^2 + lambda_u (U - tau)_+ for a given mode."""
        dist = self.distance_to_mode(latent, mode)
        return lambda_d * dist ** 2 + lambda_u * max(uncertainty - tau, 0.0)

    def recover_or_switch(self, latent: np.ndarray, current_mode: int, uncertainty: float,
                          tau: float, *, lambda_d: float, lambda_u: float,
                          switch_margin: float) -> Tuple[int, bool]:
        """Decide whether to recover to the current mode or switch to another one.

        Returns ``(chosen_mode, switched)``.  A switch happens only when some other
        mode's cost beats the current mode's cost by more than ``switch_margin`` --
        the hysteresis that keeps the policy from flip-flopping between tubes.
        """
        if self.K == 1:
            return current_mode, False
        j_recover = self.cost(latent, current_mode, uncertainty, tau,
                              lambda_d=lambda_d, lambda_u=lambda_u)
        best_other, j_switch = current_mode, np.inf
        for z in range(self.K):
            if z == current_mode:
                continue
            j = self.cost(latent, z, uncertainty, tau, lambda_d=lambda_d, lambda_u=lambda_u)
            if j < j_switch:
                best_other, j_switch = z, j
        if j_switch + switch_margin < j_recover:
            return best_other, True
        return current_mode, False


def fit_modes(traj_latent: List[np.ndarray], k: int, *, encoder: LatentEncoder, seed: int = 0) -> ModeModel:
    """Cluster latent demonstration trajectories into ``k`` modes and build spines.

    ``traj_latent`` is a list of (L, d) latent trajectories.  Trajectories are clustered
    by their latent endpoints; each mode's spine is the index-aligned mean of its
    members (all demonstrations share the rollout horizon, so the means are aligned).
    """
    endpoints = np.asarray([t[-1] for t in traj_latent])
    labels = _kmeans(endpoints, k, seed=seed)
    spines: List[np.ndarray] = []
    for z in range(max(1, k)):
        members = [traj_latent[i] for i in range(len(traj_latent)) if labels[i] == z]
        if not members:
            continue
        min_len = min(len(t) for t in members)
        stacked = np.stack([t[:min_len] for t in members], axis=0)
        spines.append(stacked.mean(axis=0))
    if not spines:  # degenerate fallback: single spine over all trajectories
        min_len = min(len(t) for t in traj_latent)
        spines = [np.stack([t[:min_len] for t in traj_latent], axis=0).mean(axis=0)]
    return ModeModel(encoder=encoder, spines=spines)
