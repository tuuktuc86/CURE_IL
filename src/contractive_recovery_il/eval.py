from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple
import csv
import json
from pathlib import Path
import numpy as np

from .baselines import SafeDAggerPolicy
from .config import ExperimentConfig
from .detector import calibrate_detector
from .envs import TaskSpec, all_tasks, goal_reached, perturbation_vector, step
from .expert import Demonstrations, ExpertOracle, generate_demonstrations
from .features import fit_latent_encoder
from .modes import fit_modes
from .policies import ContractiveRecovery, CureILPolicy, ELCDPolicy, NearestNeighborBC, Policy

METHOD_ORDER = ["BC", "SafeDAgger", "CURE-IL", "ELCD"]


@dataclass
class RolloutResult:
    task: str
    method: str
    perturbation: float
    seed: int
    rollout_id: int
    recovery_success: bool
    goal_success: bool
    expert_queries: int
    time_to_recover: int | None
    mean_tube_distance: float
    final_distance: float
    branch_preserved: bool | None
    states: np.ndarray
    modes: List[str]

    def row(self) -> dict:
        return {
            "task": self.task,
            "method": self.method,
            "perturbation": self.perturbation,
            "seed": self.seed,
            "rollout_id": self.rollout_id,
            "recovery_success": int(self.recovery_success),
            "goal_success": int(self.goal_success),
            "expert_queries": self.expert_queries,
            "time_to_recover": "" if self.time_to_recover is None else self.time_to_recover,
            "mean_tube_distance": self.mean_tube_distance,
            "final_distance": self.final_distance,
            "branch_preserved": "" if self.branch_preserved is None else int(self.branch_preserved),
        }


def split_demonstrations(demos: Demonstrations, calibration_fraction: float = 0.25, seed: int = 101) -> tuple[np.ndarray, np.ndarray]:
    """Return train/calibration state split using a persisted split seed."""
    n = len(demos.states)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cal_n = max(1, int(round(n * calibration_fraction)))
    cal_idx = idx[:cal_n]
    train_idx = idx[cal_n:]
    return demos.states[train_idx], demos.states[cal_idx]


def make_policies(task: TaskSpec, demos: Demonstrations, cfg: ExperimentConfig, preferred_path_id: str | None = None) -> Dict[str, Policy]:
    train_states, cal_states = split_demonstrations(demos, seed=cfg.calibration_seed)
    detector = calibrate_detector(train_states, cal_states, quantile=cfg.calibration_quantile,
                                  tau_on_scale=cfg.tau_on_scale, tau_off_scale=cfg.tau_off_scale,
                                  seed=cfg.calibration_seed)
    # Latent encoder phi and behavior-mode tubes, both fit on the demonstrations.
    encoder = fit_latent_encoder(demos.states)
    traj_latent = [encoder.encode(t) for t in demos.traj_states]
    n_modes = len(task.paths)
    modes = fit_modes(traj_latent, n_modes, encoder=encoder, seed=cfg.calibration_seed)
    global_mode = fit_modes(traj_latent, 1, encoder=encoder, seed=cfg.calibration_seed)

    bc = NearestNeighborBC(demos=demos)
    cure = CureILPolicy(nominal=NearestNeighborBC(demos=demos),
                        recovery=ContractiveRecovery(encoder=encoder, modes=modes),
                        detector=detector, encoder=encoder, modes=modes)
    oracle = ExpertOracle(task, cfg)
    safe = SafeDAggerPolicy(nominal=NearestNeighborBC(demos=demos), detector=detector, oracle=oracle, preferred_path_id=preferred_path_id)
    # ELCD: contraction toward a single global trajectory (no mode separation).
    elcd = ELCDPolicy(encoder=encoder, modes=global_mode)
    return {"BC": bc, "SafeDAgger": safe, "CURE-IL": cure, "ELCD": elcd}


def _initial_state(task: TaskSpec, rng: np.random.Generator, rollout_id: int) -> tuple[np.ndarray, str | None]:
    if task.name == "single":
        path = task.paths[0]
    else:
        path = task.paths[rollout_id % len(task.paths)]
    return path.start + rng.normal(0.0, 0.035, size=2), path.path_id


def rollout(task: TaskSpec, policy: Policy, cfg: ExperimentConfig, *, perturbation: float, seed: int, rollout_id: int) -> RolloutResult:
    rng = np.random.default_rng(seed * 1000 + rollout_id)
    state, branch = _initial_state(task, rng, rollout_id)
    policy.reset()
    states: List[np.ndarray] = []
    modes: List[str] = []
    recovered_at: int | None = None
    perturbed = False
    for t in range(cfg.horizon):
        if t == cfg.perturb_step and perturbation > 0:
            state = state + perturbation_vector(task, state, perturbation, rng)
            perturbed = True
        action = policy.action(state)
        states.append(state.copy())
        modes.append(policy.mode())
        dist = task.project(state).distance
        if perturbed and recovered_at is None and t > cfg.perturb_step + 2 and dist <= cfg.tube_radius:
            recovered_at = t - cfg.perturb_step
        state = step(state, action, dt=cfg.dt, max_action_norm=cfg.max_action_norm, noise=cfg.rollout_noise, rng=rng)
    arr = np.asarray(states)
    dists = np.asarray([task.project(s).distance for s in arr])
    final_goal = goal_reached(task, arr[-1], cfg.goal_radius)
    if perturbation == 0:
        recovery_success = final_goal and float(np.mean(dists[-15:])) <= cfg.tube_radius * 1.25
        recovered_at = 0 if recovery_success else None
    else:
        recovery_success = recovered_at is not None and recovered_at <= cfg.recover_window and final_goal
    if task.name == "multi":
        final_path = task.project(arr[-1]).path_id
        branch_preserved = branch == final_path
    else:
        branch_preserved = None
    queries = getattr(policy, "query_count", 0)
    return RolloutResult(
        task=task.name,
        method=policy.name,
        perturbation=perturbation,
        seed=seed,
        rollout_id=rollout_id,
        recovery_success=bool(recovery_success),
        goal_success=bool(final_goal),
        expert_queries=int(queries),
        time_to_recover=recovered_at,
        mean_tube_distance=float(dists.mean()),
        final_distance=float(dists[-1]),
        branch_preserved=branch_preserved,
        states=arr,
        modes=modes,
    )


def run_suite(cfg: ExperimentConfig, *, quick: bool = False, seed: int = 0) -> tuple[list[RolloutResult], dict[str, Demonstrations]]:
    results: list[RolloutResult] = []
    demos_by_task: dict[str, Demonstrations] = {}
    n_rollouts = cfg.quick_rollouts_per_seed if quick else cfg.rollouts_per_seed
    seeds = cfg.test_seeds if not quick else cfg.test_seeds[:3]
    for task in all_tasks():
        demos = generate_demonstrations(task, cfg, seed=cfg.train_seed + seed + (17 if task.name == "multi" else 0))
        demos_by_task[task.name] = demos
        policies = make_policies(task, demos, cfg, preferred_path_id=None)
        for perturb in cfg.perturbations:
            for test_seed in seeds:
                for rid in range(n_rollouts):
                    for method in METHOD_ORDER:
                        results.append(rollout(task, policies[method], cfg, perturbation=perturb, seed=test_seed, rollout_id=rid))
    return results, demos_by_task


def aggregate_rows(results: list[RolloutResult]) -> list[dict]:
    groups: dict[tuple, list[RolloutResult]] = {}
    for r in results:
        groups.setdefault((r.task, r.method, r.perturbation), []).append(r)
    rows = []
    for (task, method, perturb), vals in sorted(groups.items()):
        rec = np.asarray([v.recovery_success for v in vals], dtype=float)
        goal = np.asarray([v.goal_success for v in vals], dtype=float)
        queries = np.asarray([v.expert_queries for v in vals], dtype=float)
        dist = np.asarray([v.mean_tube_distance for v in vals], dtype=float)
        ttr_vals = np.asarray([v.time_to_recover for v in vals if v.time_to_recover is not None], dtype=float)
        branch_vals = [v.branch_preserved for v in vals if v.branch_preserved is not None]
        rows.append({
            "task": task,
            "method": method,
            "perturbation": perturb,
            "n": len(vals),
            "recovery_success_mean": float(rec.mean()),
            "recovery_success_se": float(rec.std(ddof=1) / np.sqrt(len(rec))) if len(rec) > 1 else 0.0,
            "goal_success_mean": float(goal.mean()),
            "expert_queries_mean": float(queries.mean()),
            "expert_queries_se": float(queries.std(ddof=1) / np.sqrt(len(queries))) if len(queries) > 1 else 0.0,
            "mean_tube_distance": float(dist.mean()),
            "time_to_recover_mean": "" if len(ttr_vals) == 0 else float(ttr_vals.mean()),
            "branch_preserved_mean": "" if not branch_vals else float(np.mean(branch_vals)),
        })
    return rows


def write_outputs(results: list[RolloutResult], cfg: ExperimentConfig, output_dir: str | Path) -> list[dict]:
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(metrics_dir / "config_snapshot.json")
    raw_path = metrics_dir / "rollouts.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].row().keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r.row())
    rows = aggregate_rows(results)
    summary_path = metrics_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (metrics_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows
