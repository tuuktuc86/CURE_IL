from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .config import ExperimentConfig
from .envs import ReferencePath, TaskSpec, make_task, step


class ExpertOracle:
    """Oracle used for demonstration generation and SafeDAgger only."""

    def __init__(self, task: TaskSpec, cfg: ExperimentConfig):
        self.task = task
        self.cfg = cfg
        self.calls = 0

    def action(self, state: np.ndarray, preferred_path_id: str | None = None, correction_gain: float = 2.05) -> np.ndarray:
        self.calls += 1
        if preferred_path_id is not None:
            path = self.task.path_by_id(preferred_path_id)
            proj = path.project(state)
            chosen = path
        else:
            proj = self.task.project(state)
            chosen = self.task.path_by_id(proj.path_id)
        target = chosen.point_ahead(proj.index, lookahead=10)
        near_goal = proj.index >= len(chosen.points) - 18
        tangent_drive = (0.0 if near_goal else 0.82) * proj.tangent
        normal_drive = correction_gain * (proj.point - state)
        progress_drive = (1.45 if near_goal else 0.70) * (target - state if near_goal else target - proj.point)
        return tangent_drive + normal_drive + progress_drive


@dataclass
class Demonstrations:
    states: np.ndarray
    actions: np.ndarray
    traj_states: List[np.ndarray]
    traj_actions: List[np.ndarray]
    path_ids: List[str]
    task_name: str


def generate_demonstrations(task: TaskSpec, cfg: ExperimentConfig, seed: int = 0) -> Demonstrations:
    rng = np.random.default_rng(seed)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    traj_states: list[np.ndarray] = []
    traj_actions: list[np.ndarray] = []
    path_ids: list[str] = []
    for path in task.paths:
        for _ in range(cfg.demo_trajectories_per_mode):
            oracle = ExpertOracle(task, cfg)
            s = path.start + rng.normal(0.0, cfg.demo_noise * 3.0, size=2)
            ts: list[np.ndarray] = []
            ta: list[np.ndarray] = []
            for _t in range(cfg.horizon):
                a = oracle.action(s, preferred_path_id=path.path_id, correction_gain=0.25)
                ts.append(s.copy())
                ta.append(a.copy())
                states.append(s.copy())
                actions.append(a.copy())
                s = step(s, a, dt=cfg.dt, max_action_norm=cfg.max_action_norm, noise=cfg.demo_noise, rng=rng)
            traj_states.append(np.asarray(ts))
            traj_actions.append(np.asarray(ta))
            path_ids.append(path.path_id)
    return Demonstrations(np.asarray(states), np.asarray(actions), traj_states, traj_actions, path_ids, task.name)
