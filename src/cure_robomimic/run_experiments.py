#!/usr/bin/env python3
"""Run the robomimic Lift perturbation-recovery suite and write metric summaries.

Mirrors ``contractive_recovery_il.eval.run_suite`` / ``write_outputs``: evaluate
each expert-free method across perturbation magnitudes and aggregate per
(method, perturbation) into ``summary.csv`` (and per-rollout ``rollouts.csv``).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np

import robomimic.utils.file_utils as FileUtils

from cure_robomimic.demo_bank import load_demo_bank
from cure_robomimic.env_utils import build_env
from cure_robomimic.policies import (BCPolicy, CureReplayPolicy, DemoManifoldRecoveryEEF,
                                     SafeDAggerEEF, SwitchingPolicy)
from cure_robomimic.rollout import RobomimicEvalConfig, rollout, make_task_detector
from cure_robomimic.tasks import TASKS

METHOD_ORDER = ["BC", "CURE-IL", "ELCD", "SafeDAgger"]


def make_policies(ckpt: str, bank, env, cfg, task, *, device: str):
    object_pos_key = task.object_pos_key
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt, device=device, verbose=False)
    bc = BCPolicy(rollout_policy=policy, name="BC")
    # Detector per task config (eef-only fixed for Lift; joint + auto-calibrated for
    # the long multi-phase tasks). See cure_robomimic.tasks.TaskCfg.
    det = make_task_detector(bank, task, env, bc, cfg)
    print(f"detector: cube_weight={task.det_cube_weight} tau_on={det.tau_on:.4f} tau_off={det.tau_off:.4f}")
    # CURE-IL: uncertainty-triggered demonstration re-execution (re-grasp capable).
    cure = CureReplayPolicy(nominal=bc, bank=bank, detector=det,
                            object_pos_key=object_pos_key, name="CURE-IL")
    # ELCD: contractive baseline — the same trigger drives a contraction toward the
    # nearest demo eef tube (no demo re-execution / re-grasp), isolating the value of
    # CURE-IL's mode/demo-aware recovery. Expert-free.
    elcd = SwitchingPolicy(nominal=bc, recovery=DemoManifoldRecoveryEEF(bank=bank, object_pos_key=object_pos_key),
                           detector=det, object_pos_key=object_pos_key, name="ELCD")
    # SafeDAgger: on the uncertainty trigger it executes the nearest demonstrated
    # expert action (the closest stand-in for an online expert query in simulation)
    # and counts one query per corrective step — an independent, supervision-paying
    # baseline rather than a relabelling of CURE-IL.
    safe = SafeDAggerEEF(nominal=bc, bank=bank, detector=det,
                         object_pos_key=object_pos_key, name="SafeDAgger")
    return {"BC": bc, "CURE-IL": cure, "ELCD": elcd, "SafeDAgger": safe}


def _per_seed_rates(vals, attr):
    """Per-seed success rate, so std/se are computed *across seeds* (the standard
    way to report variability in RL/IL experiments)."""
    by_seed = {}
    for v in vals:
        by_seed.setdefault(v.seed, []).append(getattr(v, attr))
    return np.array([np.mean(x) for x in by_seed.values()], float)


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r.method, r.perturbation), []).append(r)
    rows = []
    for (method, perturb), vals in sorted(groups.items(), key=lambda kv: (METHOD_ORDER.index(kv[0][0]), kv[0][1])):
        rec = _per_seed_rates(vals, "recovery_success")
        goal = _per_seed_rates(vals, "task_success")
        n_seeds = len(rec)
        q = np.array([v.expert_queries for v in vals], float)
        dist = np.array([v.mean_tube_distance for v in vals], float)
        ttr = np.array([v.time_to_recover for v in vals if v.time_to_recover is not None], float)
        rows.append({
            "task": vals[0].task_name, "method": method, "perturbation": perturb,
            "n_seeds": n_seeds, "n_total": len(vals),
            "recovery_success_mean": float(rec.mean()),
            "recovery_success_std": float(rec.std(ddof=1)) if n_seeds > 1 else 0.0,
            "recovery_success_se": float(rec.std(ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0,
            "goal_success_mean": float(goal.mean()),
            "goal_success_std": float(goal.std(ddof=1)) if n_seeds > 1 else 0.0,
            "expert_queries_mean": float(q.mean()),
            "mean_tube_distance": float(dist.mean()),
            "time_to_recover_mean": "" if len(ttr) == 0 else float(ttr.mean()),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task", required=True, choices=list(TASKS.keys()))
    p.add_argument("--output-dir", default="outputs_robomimic")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--n-rollouts", type=int, default=15)
    p.add_argument("--cube-weight", type=float, default=1.0)
    p.add_argument("--point-stride", type=int, default=1)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    task = TASKS[args.task]
    bank = load_demo_bank(task.dataset, cube_weight=args.cube_weight, point_stride=args.point_stride)
    cfg = RobomimicEvalConfig(perturb_step=task.perturb_step, push_steps=task.push_steps,
                              horizon=task.horizon, recover_window=task.recover_window,
                              perturbations=task.perturbations, n_rollouts=args.n_rollouts)
    env = build_env(task.dataset, offscreen=False)
    policies = make_policies(args.ckpt, bank, env, cfg, task, device=args.device)

    results = []
    for perturb in cfg.perturbations:
        for seed in args.seeds:
            for rid in range(cfg.n_rollouts):
                for m in METHOD_ORDER:
                    results.append(rollout(env, policies[m], bank, cfg, perturbation=perturb,
                                           seed=seed, rollout_id=rid, task_name=task.short))
        done = (cfg.perturbations.index(perturb) + 1)
        print(f"[{done}/{len(cfg.perturbations)}] perturbation={perturb} done "
              f"({len(args.seeds)} seeds x {cfg.n_rollouts} rollouts)")
    env.close()

    out = Path(args.output_dir) / task.short / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    raw = [r.row() for r in results]
    with (out / "rollouts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0].keys())); w.writeheader(); w.writerows(raw)
    rows = aggregate(results)
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(results)} rollouts; summary -> {out/'summary.csv'}")


if __name__ == "__main__":
    main()
