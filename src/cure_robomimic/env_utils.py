"""Build the robosuite Lift environment directly from a dataset's env metadata.

We intentionally bypass robomimic's ``EnvRobosuite`` wrapper: the 0.3.0 PyPI
release hard-imports the legacy ``mujoco_py`` binding, which is incompatible with
robosuite 1.4 (new dm ``mujoco`` binding). Building the env directly with
``robosuite.make`` (verified working) also gives us full control of the step loop
so we can inject perturbations for the recovery study.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Tuple
import numpy as np
import h5py
import robosuite as suite

# Observation keys the BC policy was trained on (low_dim). robosuite exposes the
# concatenated object observation under "object-state"; the dataset stored it as
# "object", so we remap when assembling the policy input dict.
POLICY_OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]

# OSC_POSE position delta scale (output_max for x/y/z). action_dpos == 1.0 maps to
# this many metres of commanded eef displacement. Used to convert world-space
# recovery targets into the [-1, 1] action space.
OSC_POS_SCALE = 0.05


def read_env_args(hdf5_path: str) -> Dict[str, Any]:
    with h5py.File(hdf5_path, "r") as f:
        return json.loads(f["data"].attrs["env_args"])


def build_env(hdf5_path: str, *, offscreen: bool = False, camera: str = "agentview",
              cam_h: int = 256, cam_w: int = 256):
    """Create a robosuite env matching the dataset, optionally with offscreen render."""
    env_args = read_env_args(hdf5_path)
    kwargs = dict(env_args["env_kwargs"])
    kwargs["has_renderer"] = False
    kwargs["has_offscreen_renderer"] = offscreen
    kwargs["use_camera_obs"] = offscreen
    if offscreen:
        kwargs["camera_names"] = camera
        kwargs["camera_heights"] = cam_h
        kwargs["camera_widths"] = cam_w
    env = suite.make(env_name=env_args["env_name"], **kwargs)
    return env


def policy_obs(raw_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Map a robosuite observation dict to the key set the BC policy expects."""
    return {
        "robot0_eef_pos": np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32),
        "robot0_eef_quat": np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32),
        "object": np.asarray(raw_obs["object-state"], dtype=np.float32),
    }


def render_frame(env, camera: str = "agentview", h: int = 256, w: int = 256) -> np.ndarray:
    """Return an RGB frame (uint8, top-row-first) for video writing."""
    frame = env.sim.render(camera_name=camera, height=h, width=w)[::-1]
    return np.ascontiguousarray(frame)
