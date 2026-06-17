from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .detector import RNDDetector
from .expert import Demonstrations
from .features import LatentEncoder
from .modes import ModeModel


class Policy:
    name = "policy"

    def reset(self) -> None:
        pass

    def action(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def mode(self) -> str:
        return "nominal"


@dataclass
class NearestNeighborBC(Policy):
    demos: Demonstrations
    k: int = 10
    name: str = "BC"

    def action(self, state: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(self.demos.states - state[None, :], axis=1)
        k = min(self.k, len(d))
        idx = np.argpartition(d, k - 1)[:k]
        w = np.exp(-(d[idx] ** 2) / 0.035)
        if float(w.sum()) <= 1e-12:
            return self.demos.actions[idx[0]].copy()
        return (self.demos.actions[idx] * (w / w.sum())[:, None]).sum(axis=0)


@dataclass
class ContractiveRecovery:
    """Latent-space contraction field toward a selected trajectory tube.

    The action realises ``g(s) = rho * v(s) - K_perp * e_perp(s)``: a forward term
    along the tube tangent ``v`` plus a perpendicular contraction of the deviation
    ``e_perp``.  Both terms are computed in the latent space of ``encoder`` and decoded
    back into a state-space velocity, so ``||e_perp||`` decays exponentially along the
    recovery -- the contraction guarantee, specialised to the *selected* tube.
    """

    encoder: LatentEncoder
    modes: ModeModel
    rho: float = 0.75          # forward speed along the tube
    k_perp: float = 7.0        # perpendicular contraction gain
    settle_window: int = 18    # damp forward drive near the tube end (goal)

    def action(self, state: np.ndarray, mode: int) -> np.ndarray:
        y = self.encoder.encode(state)
        proj = self.modes.project(y, mode)
        near_goal = proj.index >= proj.length - self.settle_window
        fwd = self.encoder.decode_velocity(proj.tangent)
        fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
        perp = self.encoder.decode_velocity(proj.e_perp)
        forward_gain = 0.0 if near_goal else self.rho
        return forward_gain * fwd - self.k_perp * perp


@dataclass
class ELCDPolicy(Policy):
    """Contractive baseline: contraction toward a single global trajectory.

    A genuine contractive dynamical system that, lacking CURE-IL's mode separation,
    contracts every state toward one global spine (``modes`` built with ``K = 1``).
    It recovers exponentially but cannot honour distinct behavior modes -- e.g. on a
    branching task it is pulled toward the average of the branches.
    """

    encoder: LatentEncoder
    modes: ModeModel              # single-mode (K=1) model
    field: ContractiveRecovery = None
    name: str = "ELCD"

    def __post_init__(self) -> None:
        if self.field is None:
            # Applied as the whole policy, so it must traverse the path at demo speed
            # (forward gain near the action cap), not just re-enter the tube.
            self.field = ContractiveRecovery(encoder=self.encoder, modes=self.modes,
                                             rho=1.25, k_perp=4.0, settle_window=3)

    def action(self, state: np.ndarray) -> np.ndarray:
        return self.field.action(state, mode=0)


@dataclass
class CureILPolicy(Policy):
    """CURE-IL: nominal BC + RND/conformal-triggered, mode-aware contractive recovery.

    When the RND uncertainty exceeds the conformal ``tau_on`` the policy enters a
    recovery mode (with ``tau_off`` hysteresis).  On entry it assigns the nearest
    behavior mode; each recovery step it compares a recover-cost against a switch-cost
    to decide whether to keep contracting toward the current tube or commit to another
    one, then applies the contraction field.  Expert-free: it never queries an oracle.
    """

    nominal: NearestNeighborBC
    recovery: ContractiveRecovery
    detector: RNDDetector
    encoder: LatentEncoder
    modes: ModeModel
    hold_steps: int = 2
    lambda_d: float = 1.0
    lambda_u: float = 0.2
    switch_margin: float = 0.05
    name: str = "CURE-IL"
    _in_recovery: bool = False
    _off_count: int = 0
    _mode: int = 0

    def reset(self) -> None:
        self._in_recovery = False
        self._off_count = 0
        self._mode = 0

    def action(self, state: np.ndarray) -> np.ndarray:
        score = self.detector.score(state)
        entering = False
        if score > self.detector.tau_on:
            if not self._in_recovery:
                entering = True
            self._in_recovery = True
            self._off_count = self.hold_steps
        elif self._in_recovery and score <= self.detector.tau_off:
            self._off_count -= 1
            if self._off_count <= 0:
                self._in_recovery = False
        if not self._in_recovery:
            return self.nominal.action(state)
        y = self.encoder.encode(state)
        if entering:
            self._mode = self.modes.assign(y)
        else:
            self._mode, _ = self.modes.recover_or_switch(
                y, self._mode, score, self.detector.tau_on,
                lambda_d=self.lambda_d, lambda_u=self.lambda_u,
                switch_margin=self.switch_margin,
            )
        return self.recovery.action(state, mode=self._mode)

    def mode(self) -> str:
        return "recovery" if self._in_recovery else "nominal"
