#!/usr/bin/env python3
"""Render individual success / failure clips for each method under perturbation.

Default (``per-method``): for every method (BC, ELCD-analog, CURE-IL) find one
seed where it succeeds and one where it fails (independently — not a shared seed,
not contrastive) and render each as its own standalone, labelled clip. This is
robust even on tasks where CURE-IL only ties BC. ``sidebyside`` keeps the old
BC-vs-CURE comparison render.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import cv2
import imageio

import robomimic.utils.file_utils as FileUtils

from cure_robomimic.demo_bank import load_demo_bank
from cure_robomimic.env_utils import build_env
from cure_robomimic.policies import BCPolicy, CureReplayPolicy, DemoManifoldRecoveryEEF, SwitchingPolicy
from cure_robomimic.rollout import RobomimicEvalConfig, rollout, make_task_detector
from cure_robomimic.tasks import TASKS

GREEN, RED = (80, 255, 80), (80, 80, 255)


def label(frames, text, color):
    out = []
    for fr in frames:
        fr = np.ascontiguousarray(fr)
        cv2.rectangle(fr, (0, 0), (fr.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(fr, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        out.append(fr)
    return out


def compose(a, b):
    n = max(len(a), len(b))
    pad = lambda fs: fs + [fs[-1]] * (n - len(fs))
    a, b = pad(a), pad(b)
    sep = np.full((a[0].shape[0], 4, 3), 255, np.uint8)
    return [np.hstack([x, sep, y]) for x, y in zip(a, b)]


def build_methods(bank, task, env, cfg, ckpt, device):
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt, device=device, verbose=False)
    bc = BCPolicy(rollout_policy=policy, name="BC")
    det = make_task_detector(bank, task, env, bc, cfg)
    cure = CureReplayPolicy(nominal=bc, bank=bank, detector=det,
                            object_pos_key=task.object_pos_key, name="CURE-IL")
    elcd = SwitchingPolicy(nominal=bc, recovery=DemoManifoldRecoveryEEF(bank=bank, object_pos_key=task.object_pos_key),
                           detector=det, object_pos_key=task.object_pos_key, name="ELCD")
    return {"bc": bc, "elcd": elcd, "cure": cure}, det


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--output-dir", default="outputs_robomimic")
    p.add_argument("--perturbation", type=float, default=0.7)
    p.add_argument("--scan", type=int, default=40, help="max seeds to scan per method")
    p.add_argument("--point-stride", type=int, default=1)
    p.add_argument("--style", choices=["per-method", "sidebyside"], default="per-method")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    task = TASKS[args.task]
    bank = load_demo_bank(task.dataset, cube_weight=1.0, point_stride=args.point_stride)
    env = build_env(task.dataset, offscreen=True, cam_h=256, cam_w=256)
    cfg = RobomimicEvalConfig(perturb_step=task.perturb_step, push_steps=task.push_steps,
                              horizon=task.horizon, recover_window=task.recover_window)
    methods, _ = build_methods(bank, task, env, cfg, args.ckpt, args.device)
    outdir = Path(args.output_dir) / task.short / "videos"
    outdir.mkdir(parents=True, exist_ok=True)
    mag = args.perturbation

    if args.style == "sidebyside":
        bc, cure = methods["bc"], methods["cure"]
        for rid in range(args.scan):
            rb = rollout(env, bc, bank, cfg, perturbation=mag, seed=0, rollout_id=rid, record_frames=True)
            rc = rollout(env, cure, bank, cfg, perturbation=mag, seed=0, rollout_id=rid, record_frames=True)
            if rc.task_success and not rb.task_success:
                frames = compose(label(rb.frames, "BC  (fail)", RED),
                                 label(rc.frames, "CURE-IL  (recover)", GREEN))
                path = outdir / f"bc_vs_cure_mag{mag}_seed{rid}.mp4"
                imageio.mimsave(path, frames, fps=20); print(f"wrote {path}")
                break
        env.close(); return

    # per-method: one success + one failure clip for each method, found independently
    disp = {"bc": "BC", "elcd": "ELCD", "cure": "CURE-IL"}
    for key, pol in methods.items():
        succ = fail = None
        for rid in range(args.scan):
            if succ is None or fail is None:
                r = rollout(env, pol, bank, cfg, perturbation=mag, seed=0, rollout_id=rid,
                            record_frames=True)
                if r.task_success and succ is None:
                    succ = (rid, r)
                elif (not r.task_success) and fail is None:
                    fail = (rid, r)
            if succ and fail:
                break
        for outcome, clip, col in [("SUCCESS", succ, GREEN), ("FAILURE", fail, RED)]:
            if clip is None:
                print(f"{disp[key]}: no {outcome} clip found in {args.scan} seeds")
                continue
            rid, r = clip
            fn = outdir / f"{task.short}_{key}_{outcome}_seed{rid}.mp4"
            imageio.mimsave(fn, label(r.frames, f"{disp[key]}  -  {outcome}", col), fps=20)
            print(f"wrote {fn}")
    env.close()


if __name__ == "__main__":
    main()
