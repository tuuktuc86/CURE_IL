"""Perturbation-recovery rollout harness for the robomimic Lift study.

Mirrors ``contractive_recovery_il.eval.rollout``: run a policy, apply a bounded
perturbation at a fixed step, and measure both task success and *windowed
recovery* (return to the demonstrated eef tube within a window, then succeed).

The perturbation is an external shove: for a few steps we override the policy's
translational action with a push in a fixed (seeded) direction, while keeping the
policy's gripper/orientation commands so a grasped cube is not simply dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from contractive_recovery_il.conformal import conformal_threshold

from .demo_bank import DemoBank
from .env_utils import policy_obs, render_frame
from .policies import Policy, build_detector


@dataclass(frozen=True)
class RobomimicEvalConfig:
    horizon: int = 250
    perturb_step: int = 32          # after grasp completes (demos grasp ~step 30-40)
    push_steps: int = 5
    tube_radius: float = 0.03
    recover_window: int = 120        # demo re-execution (re-approach+re-grasp+lift) needs time
    perturbations: tuple = (0.0, 0.3, 0.5, 0.7, 0.9)  # action-space shove scale in [0,1]
    n_rollouts: int = 25


@dataclass
class RobomimicRollout:
    method: str
    perturbation: float
    seed: int
    rollout_id: int
    task_success: bool
    recovery_success: bool
    expert_queries: int
    time_to_recover: Optional[int]
    mean_tube_distance: float
    final_distance: float
    task_name: str = "lift"
    frames: Optional[List[np.ndarray]] = None

    def row(self) -> dict:
        return {
            "task": self.task_name,
            "method": self.method,
            "perturbation": self.perturbation,
            "seed": self.seed,
            "rollout_id": self.rollout_id,
            "recovery_success": int(self.recovery_success),
            "goal_success": int(self.task_success),
            "expert_queries": self.expert_queries,
            "time_to_recover": "" if self.time_to_recover is None else self.time_to_recover,
            "mean_tube_distance": self.mean_tube_distance,
            "final_distance": self.final_distance,
        }


def calibrate_tau(env, bc, bank, detector, cfg, object_pos_key, *, n: int = 6,
                  on_q: float = 0.98, off_q: float = 0.90):
    """Conformally calibrate the switch thresholds from clean (unperturbed) BC rollouts.

    The clean-rollout RND scores are an in-distribution calibration set. ``tau_on`` is
    the split-conformal threshold at coverage ``on_q`` (so recovery fires on at most
    ``1 - on_q`` of on-manifold steps); ``tau_off`` at coverage ``off_q`` gives the
    hysteresis gap.
    """
    scores = []
    for rid in range(n):
        np.random.seed(90000 + rid)
        env.reset()
        bc.reset()
        obs = env._get_observations()
        for _ in range(cfg.horizon):
            obs, _, _, _ = env.step(bc.action(obs))
            eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
            cube = np.asarray(obs[object_pos_key], dtype=float)
            scores.append(detector.score(eef, cube))
            if env._check_success():
                break
    scores = np.asarray(scores)
    tau_on = conformal_threshold(scores, alpha=1.0 - on_q)
    tau_off = conformal_threshold(scores, alpha=1.0 - off_q)
    return float(tau_on), float(min(tau_off, tau_on))


def make_task_detector(bank, task, env, bc, cfg):
    """Build the uncertainty detector per the task's detector config (see TaskCfg):
    fixed thresholds when given, else auto-calibrated from clean rollouts."""
    det = build_detector(bank, cube_weight=task.det_cube_weight)
    if task.det_tau_on is not None:
        det.tau_on, det.tau_off = task.det_tau_on, task.det_tau_off
    else:
        det.tau_on, det.tau_off = calibrate_tau(env, bc, bank, det, cfg, task.object_pos_key)
    return det


def _perturb_direction(rng: np.random.Generator) -> np.ndarray:
    """Random horizontal-biased unit shove direction (xy strong, small z)."""
    v = rng.normal(size=3)
    v[2] *= 0.3
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def rollout(env, policy: Policy, bank: DemoBank, cfg: RobomimicEvalConfig, *,
            perturbation: float, seed: int, rollout_id: int, task_name: str = "lift",
            record_frames: bool = False, camera: str = "agentview",
            frame_hw: int = 256) -> RobomimicRollout:
    # Seed placement (robosuite samples object pose from global np.random) and the
    # perturbation direction so every method sees identical initial conditions.
    episode_seed = seed * 1000 + rollout_id
    np.random.seed(episode_seed)
    env.reset()
    policy.reset()
    rng = np.random.default_rng(episode_seed)
    pdir = _perturb_direction(rng)

    obs = env._get_observations()
    dists: List[float] = []
    frames: List[np.ndarray] = [] if record_frames else None
    recovered_at: Optional[int] = None
    task_success = False

    for t in range(cfg.horizon):
        action = policy.action(obs)
        in_push = perturbation > 0 and cfg.perturb_step <= t < cfg.perturb_step + cfg.push_steps
        if in_push:
            # External shove: override translation, keep policy gripper/orientation.
            action = np.array(action, dtype=float)
            action[0:3] = np.clip(perturbation * pdir, -1.0, 1.0)
        obs, _, _, _ = env.step(action)

        eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
        dist = bank.tube_distance(eef)
        dists.append(dist)
        if record_frames:
            frames.append(render_frame(env, camera, frame_hw, frame_hw))

        after_push = t > cfg.perturb_step + cfg.push_steps + 1
        if perturbation > 0 and recovered_at is None and after_push and dist <= cfg.tube_radius:
            recovered_at = t - cfg.perturb_step
        if env._check_success():
            task_success = True
            # keep stepping a few frames for video clarity, but success latches
            if not record_frames:
                # finish episode early once successful for speed
                break

    dists_arr = np.asarray(dists)
    if perturbation == 0:
        recovery_success = task_success
        recovered_at = 0 if task_success else None
    else:
        recovery_success = (recovered_at is not None and recovered_at <= cfg.recover_window
                            and task_success)
    queries = int(getattr(policy, "query_count", 0))
    return RobomimicRollout(
        method=policy.name,
        perturbation=perturbation,
        seed=seed,
        rollout_id=rollout_id,
        task_name=task_name,
        task_success=bool(task_success),
        recovery_success=bool(recovery_success),
        expert_queries=queries,
        time_to_recover=recovered_at,
        mean_tube_distance=float(dists_arr.mean()),
        final_distance=float(dists_arr[-1]),
        frames=frames,
    )
