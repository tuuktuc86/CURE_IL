"""Policies for the robomimic recovery study (Lift / Can / Square).

Ports of the 2D ``contractive_recovery_il`` design to eef-position space:

- ``BCPolicy``                : trained robomimic BC network (nominal).
- ``RNDUncertaintyEEF``       : Random Network Distillation novelty in the JOINT
                                (eef, object) space (reuses ``features.fit_rnd``).
- ``DemoManifoldRecoveryEEF`` : project to nearest demo eef trajectory, drive to a
                                lookahead point -- the contraction-to-tube recovery.
- ``ELCDPolicy``              : ELCD-style contractive baseline (contraction to the
                                demo tube, no demo re-execution / re-grasp).
- ``SafeDAggerEEF``           : DAgger-style baseline -- on the uncertainty trigger it
                                executes the *nearest demonstrated expert action* and
                                counts one online query per corrective step.
- ``CureReplayPolicy``        : CURE-IL -- uncertainty-triggered demo re-execution
                                (mode/demo-aware, re-grasp capable, expert-free).

The uncertainty signal is RND (directive: RND-based) and the trigger thresholds are
conformally calibrated from clean rollouts (see ``rollout.calibrate_tau``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

from contractive_recovery_il.features import RNDUncertainty, fit_rnd

from .demo_bank import DemoBank
from .env_utils import OSC_POS_SCALE, policy_obs


# --------------------------------------------------------------------------- #
# Uncertainty detector: RND novelty in the JOINT (eef, object) space
# --------------------------------------------------------------------------- #
@dataclass
class RNDUncertaintyEEF:
    """RND novelty score in the JOINT (eef, object) space.

    On long multi-phase tasks (Can, Square) the eef-only manifold is space-filling
    — almost any eef position is near *some* demonstrated eef point — so an eef-only
    signal cannot detect a perturbation. The joint (eef, object) configuration leaves
    the demonstrated manifold whenever the object is displaced relative to the arm,
    which is exactly what a perturbation causes; RND is fit on those joint features.
    """
    rnd: RNDUncertainty
    cube_weight: float = 1.0
    tau_on: float = 1.0
    tau_off: float = 0.7

    def feat(self, eef: np.ndarray, cube: np.ndarray) -> np.ndarray:
        return np.concatenate([eef, self.cube_weight * cube])

    def score(self, eef: np.ndarray, cube: np.ndarray) -> float:
        return self.rnd.score(self.feat(eef, cube))


def build_detector(bank, *, cube_weight: float = 1.0, seed: int = 0) -> RNDUncertaintyEEF:
    feats = np.hstack([bank.eef_points, cube_weight * bank.cube_points])
    return RNDUncertaintyEEF(rnd=fit_rnd(feats, seed=seed), cube_weight=cube_weight)


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #
class Policy:
    name = "policy"

    def reset(self) -> None:
        pass

    def action(self, raw_obs: Dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def mode(self) -> str:
        return "nominal"


@dataclass
class BCPolicy(Policy):
    """Wraps a robomimic RolloutPolicy loaded from a checkpoint."""
    rollout_policy: object
    name: str = "BC"

    def reset(self) -> None:
        self.rollout_policy.start_episode()

    def action(self, raw_obs: Dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(self.rollout_policy(ob=policy_obs(raw_obs)), dtype=float)


@dataclass
class DemoManifoldRecoveryEEF(Policy):
    """Pull eef toward a lookahead point on the nearest demonstrated eef trajectory.

    Port of ``DemoManifoldRecovery``: nearest-trajectory projection + lookahead +
    proportional pull. Gripper command is copied from the nearest demo timestep so
    a grasped object is not dropped during recovery. Orientation is held (drot=0).
    """
    bank: DemoBank
    lookahead: int = 10
    pull_gain: float = 1.0
    beta: float = 0.5
    object_pos_key: str = "cube_pos"
    name: str = "demo_recovery"

    def action(self, raw_obs: Dict[str, np.ndarray]) -> np.ndarray:
        eef = np.asarray(raw_obs["robot0_eef_pos"], dtype=float)
        cube = np.asarray(raw_obs[self.object_pos_key], dtype=float)
        ti, idx, _ = self.bank.nearest_index(eef, cube)
        traj = self.bank.traj_eef[ti]
        here = traj[idx]
        ahead = traj[min(len(traj) - 1, idx + self.lookahead)]
        near_end = idx >= len(traj) - self.lookahead
        target = ahead if near_end else here + self.beta * (ahead - here)
        dpos = self.pull_gain * (target - eef) / OSC_POS_SCALE
        dpos = np.clip(dpos, -1.0, 1.0)
        gripper = float(self.bank.traj_act[ti][idx][-1])
        return np.array([dpos[0], dpos[1], dpos[2], 0.0, 0.0, 0.0, gripper])


@dataclass
class SafeDAggerEEF(Policy):
    """SafeDAgger baseline: query a demonstrated expert on the uncertainty trigger.

    When the RND novelty exceeds ``tau_on`` the policy executes the *recorded action of
    the nearest demonstration state* — the closest available stand-in for an online
    expert query in simulation — and counts one query per corrective step. Unlike
    CURE-IL it pays a (counted) supervision cost, and it issues single-step expert
    actions rather than latching a full demonstration re-execution.
    """
    nominal: BCPolicy
    bank: DemoBank
    detector: "RNDUncertaintyEEF"
    object_pos_key: str = "cube_pos"
    name: str = "SafeDAgger"
    query_count: int = 0
    _querying: bool = False

    def reset(self) -> None:
        self.query_count = 0
        self._querying = False
        self.nominal.reset()

    def action(self, raw_obs: Dict[str, np.ndarray]) -> np.ndarray:
        eef = np.asarray(raw_obs["robot0_eef_pos"], dtype=float)
        cube = np.asarray(raw_obs[self.object_pos_key], dtype=float)
        if self.detector.score(eef, cube) > self.detector.tau_on:
            self._querying = True
            self.query_count += 1
            ti, idx, _ = self.bank.nearest_index(eef, cube)
            return np.asarray(self.bank.traj_act[ti][idx], dtype=float)
        self._querying = False
        return self.nominal.action(raw_obs)

    def mode(self) -> str:
        return "expert_query" if self._querying else "nominal"


@dataclass
class SwitchingPolicy(Policy):
    """BC nominal + uncertainty-triggered contraction recovery with hysteresis.

    Used for the ELCD baseline: contraction toward the demonstrated eef tube via
    ``DemoManifoldRecoveryEEF``, with no demonstration re-execution / re-grasp.
    Expert-free.
    """
    nominal: BCPolicy
    recovery: Policy
    detector: RNDUncertaintyEEF
    hold_steps: int = 3
    object_pos_key: str = "cube_pos"
    name: str = "CURE-IL"
    _in_recovery: bool = False
    _off_count: int = 0
    query_count: int = 0  # always 0 (expert-free); kept for summary schema parity

    def reset(self) -> None:
        self._in_recovery = False
        self._off_count = 0
        self.query_count = 0
        self.nominal.reset()

    def action(self, raw_obs: Dict[str, np.ndarray]) -> np.ndarray:
        eef = np.asarray(raw_obs["robot0_eef_pos"], dtype=float)
        cube = np.asarray(raw_obs[self.object_pos_key], dtype=float)
        score = self.detector.score(eef, cube)
        if score > self.detector.tau_on:
            self._in_recovery = True
            self._off_count = self.hold_steps
        elif self._in_recovery and score <= self.detector.tau_off:
            self._off_count -= 1
            if self._off_count <= 0:
                self._in_recovery = False
        if self._in_recovery:
            return self.recovery.action(raw_obs)
        return self.nominal.action(raw_obs)

    def mode(self) -> str:
        return "recovery" if self._in_recovery else "nominal"


@dataclass
class CureReplayPolicy(Policy):
    """CURE-IL with demonstration *re-execution* recovery.

    The eef-only recovery (``SwitchingPolicy`` + ``DemoManifoldRecoveryEEF``) can
    return the arm to the demo tube but cannot re-establish a lost grasp. This
    variant, when the uncertainty trigger fires, latches onto the demonstration
    whose cube position best matches the current cube and *replays that demo's eef
    trajectory together with its recorded gripper schedule* — re-approaching,
    re-grasping, and lifting the (possibly displaced) cube. Still expert-free: it
    only uses offline demonstrations, never an online expert.
    """
    nominal: BCPolicy
    bank: DemoBank
    detector: RNDUncertaintyEEF
    speed: int = 1
    pull_gain: float = 1.0
    object_pos_key: str = "cube_pos"
    name: str = "CURE-IL"
    _replaying: bool = False
    _demo_id: int = 0
    _ptr: int = 0
    query_count: int = 0  # always 0 (expert-free); kept for summary schema parity

    def reset(self) -> None:
        self._replaying = False
        self._demo_id = 0
        self._ptr = 0
        self.query_count = 0
        self.nominal.reset()

    def action(self, raw_obs: Dict[str, np.ndarray]) -> np.ndarray:
        eef = np.asarray(raw_obs["robot0_eef_pos"], dtype=float)
        cube = np.asarray(raw_obs[self.object_pos_key], dtype=float)
        if not self._replaying and self.detector.score(eef, cube) > self.detector.tau_on:
            # Latch: pick the demo for this cube location and replay from approach.
            self._replaying = True
            self._demo_id = self.bank.nearest_demo_by_cube(cube)
            self._ptr = 0
        if not self._replaying:
            return self.nominal.action(raw_obs)

        traj = self.bank.traj_eef[self._demo_id]
        acts = self.bank.traj_act[self._demo_id]
        idx = min(self._ptr, len(traj) - 1)
        target = traj[idx]
        gripper = float(acts[idx][-1])
        dpos = np.clip(self.pull_gain * (target - eef) / OSC_POS_SCALE, -1.0, 1.0)
        # Advance the replay pointer only once the eef is near the current waypoint,
        # so re-approach/re-grasp are not skipped when the arm is still catching up.
        if float(np.linalg.norm(target - eef)) < 0.02 or idx >= len(traj) - 1:
            self._ptr += self.speed
        if self._ptr >= len(traj) - 1:
            self._replaying = False  # demo finished; hand back to BC
        return np.array([dpos[0], dpos[1], dpos[2], 0.0, 0.0, 0.0, gripper])

    def mode(self) -> str:
        return "recovery" if self._replaying else "nominal"
