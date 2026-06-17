#!/usr/bin/env python3
"""Build a robomimic BC (low_dim) training config for the Lift feasibility study.

We keep the standard low_dim observation set and the default MLP policy, but
shorten the schedule so the feasibility gate finishes quickly while still
reaching a credible success rate on the (easy) Lift task.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from robomimic.config import config_factory


def build_config(dataset: str, output_dir: str, num_epochs: int, rollout_n: int,
                 filter_key: str | None = None) -> dict:
    config = config_factory(algo_name="bc")

    # --- data / output ---
    config.train.data = dataset
    config.train.output_dir = output_dir
    config.experiment.name = "bc_lift_lowdim"

    # Optional data-limited (brittle) regime: train on a subset filter key. This
    # exposes the covariate-shift weakness of BC that CURE-IL's recovery targets.
    if filter_key is not None:
        config.train.hdf5_filter_key = filter_key

    # --- observation modalities (standard robomimic low_dim set for Lift) ---
    config.observation.modalities.obs.low_dim = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object",
    ]
    config.observation.modalities.obs.rgb = []

    # --- shortened but real schedule ---
    config.train.num_epochs = num_epochs
    config.train.batch_size = 100
    config.train.num_data_workers = 0
    config.experiment.epoch_every_n_steps = 100

    # --- evaluation rollouts ---
    # Disabled on purpose: robomimic 0.3.0's EnvRobosuite hard-imports the legacy
    # `mujoco_py` binding, which is absent with robosuite 1.4 (new `mujoco` binding).
    # We instead evaluate with our own robosuite rollout harness (build_env in
    # rollout.py), which also lets us inject perturbations for the recovery study.
    config.experiment.rollout.enabled = False

    # --- checkpointing: save periodically + final so we can load for recovery rollouts ---
    config.experiment.save.enabled = True
    config.experiment.save.every_n_epochs = max(1, num_epochs // 3)
    config.experiment.save.on_best_rollout_success_rate = False
    config.experiment.save.on_best_rollout_return = False

    config.experiment.validate = False
    config.experiment.logging.terminal_output_to_txt = True

    return config.dump()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--out", required=True, help="path to write config json")
    p.add_argument("--num-epochs", type=int, default=300)
    p.add_argument("--rollout-n", type=int, default=20)
    p.add_argument("--filter-key", default=None)
    args = p.parse_args()

    cfg = build_config(args.dataset, args.output_dir, args.num_epochs, args.rollout_n,
                       filter_key=args.filter_key)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(cfg, encoding="utf-8")
    print(f"wrote config to {args.out}")


if __name__ == "__main__":
    main()
