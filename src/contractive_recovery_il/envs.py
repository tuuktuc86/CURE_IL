from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class Projection:
    point: Array
    tangent: Array
    distance: float
    index: int
    path_id: str
    progress: float


@dataclass(frozen=True)
class ReferencePath:
    path_id: str
    points: Array

    @property
    def start(self) -> Array:
        return self.points[0]

    @property
    def goal(self) -> Array:
        return self.points[-1]

    def project(self, state: Array) -> Projection:
        dists = np.linalg.norm(self.points - state[None, :], axis=1)
        idx = int(np.argmin(dists))
        if idx <= 0:
            tangent = self.points[1] - self.points[0]
        elif idx >= len(self.points) - 1:
            tangent = self.points[-1] - self.points[-2]
        else:
            tangent = self.points[idx + 1] - self.points[idx - 1]
        norm = np.linalg.norm(tangent) + 1e-12
        tangent = tangent / norm
        return Projection(self.points[idx].copy(), tangent, float(dists[idx]), idx, self.path_id, idx / (len(self.points) - 1))

    def point_ahead(self, index: int, lookahead: int = 8) -> Array:
        return self.points[min(len(self.points) - 1, max(0, index + lookahead))]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    paths: Tuple[ReferencePath, ...]

    def project(self, state: Array) -> Projection:
        projections = [p.project(state) for p in self.paths]
        return min(projections, key=lambda pr: pr.distance)

    def path_by_id(self, path_id: str) -> ReferencePath:
        for p in self.paths:
            if p.path_id == path_id:
                return p
        raise KeyError(path_id)

    @property
    def goal_points(self) -> Array:
        return np.stack([p.goal for p in self.paths], axis=0)


def _single_curve(num: int = 260) -> Array:
    x = np.linspace(0.0, 10.0, num)
    y = 0.72 * np.sin(0.72 * x - 0.25) + 0.13 * np.sin(1.65 * x)
    return np.stack([x, y], axis=1)


def _branch_curve(sign: float, num: int = 260) -> Array:
    x = np.linspace(0.0, 10.0, num)
    y = np.zeros_like(x)
    split = 3.85
    mask = x > split
    z = np.clip((x[mask] - split) / (10.0 - split), 0.0, 1.0)
    y[mask] = sign * (0.35 + 1.55 * (1.0 - np.cos(np.pi * z)) / 2.0)
    y += 0.05 * np.sin(1.3 * x)  # small shared curvature makes figures less synthetic
    return np.stack([x, y], axis=1)


def _arc_curve(num: int = 260) -> Array:
    """Open crescent/arc task: visually distinct from sine but still goal-directed."""
    theta = np.linspace(1.12 * np.pi, -0.18 * np.pi, num)
    x = 5.0 + 4.2 * np.cos(theta)
    y = -0.25 + 2.45 * np.sin(theta)
    return np.stack([x, y], axis=1)


def _spiral_curve(num: int = 260) -> Array:
    """Open spiral recovery task with continuously rotating tangent normals."""
    theta = np.linspace(0.25 * np.pi, 1.85 * np.pi, num)
    radius = np.linspace(0.55, 2.25, num)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    pts = np.stack([x, y], axis=1)
    # Normalize to the same rough plotting/rollout scale as the other tasks.
    pts[:, 0] = (pts[:, 0] - pts[:, 0].min()) / (np.ptp(pts[:, 0]) + 1e-12) * 8.8 + 0.6
    pts[:, 1] = (pts[:, 1] - pts[:, 1].mean()) * 0.82
    return pts


def _zigzag_curve(num: int = 260) -> Array:
    """Piecewise-linear switchback task testing sharp directional changes."""
    knots_x = np.array([0.0, 1.7, 3.2, 4.7, 6.2, 7.8, 10.0])
    knots_y = np.array([-1.2, 1.1, -0.95, 1.25, -0.8, 0.95, 0.0])
    x = np.linspace(0.0, 10.0, num)
    y = np.interp(x, knots_x, knots_y)
    # Light smoothing keeps the expert vector field stable while preserving corners.
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    y = np.convolve(np.pad(y, (2, 2), mode="edge"), kernel, mode="valid")
    return np.stack([x, y], axis=1)


def make_task(name: str) -> TaskSpec:
    if name == "single":
        return TaskSpec("single", (ReferencePath("single", _single_curve()),))
    if name == "multi":
        return TaskSpec(
            "multi",
            (
                ReferencePath("upper", _branch_curve(+1.0)),
                ReferencePath("lower", _branch_curve(-1.0)),
            ),
        )
    if name == "arc":
        return TaskSpec("arc", (ReferencePath("arc", _arc_curve()),))
    if name == "spiral":
        return TaskSpec("spiral", (ReferencePath("spiral", _spiral_curve()),))
    if name == "zigzag":
        return TaskSpec("zigzag", (ReferencePath("zigzag", _zigzag_curve()),))
    raise ValueError(f"unknown task {name!r}")


def all_tasks() -> Tuple[TaskSpec, ...]:
    return (
        make_task("single"),
        make_task("multi"),
        make_task("arc"),
        make_task("spiral"),
        make_task("zigzag"),
    )


def clip_action(action: Array, max_norm: float) -> Array:
    norm = float(np.linalg.norm(action))
    if norm > max_norm:
        return action / (norm + 1e-12) * max_norm
    return action


def step(state: Array, action: Array, *, dt: float, max_action_norm: float, noise: float, rng: np.random.Generator) -> Array:
    clipped = clip_action(np.asarray(action, dtype=float), max_action_norm)
    return np.asarray(state, dtype=float) + dt * clipped + rng.normal(0.0, noise, size=2)


def perturbation_vector(task: TaskSpec, state: Array, magnitude: float, rng: np.random.Generator) -> Array:
    if magnitude == 0:
        return np.zeros(2)
    proj = task.project(state)
    normal = np.array([-proj.tangent[1], proj.tangent[0]])
    # Deterministic-looking but seeded side choice prevents one-sided artifacts.
    side = -1.0 if rng.random() < 0.5 else 1.0
    return side * magnitude * normal


def distance_to_task(task: TaskSpec, state: Array) -> float:
    return task.project(state).distance


def goal_reached(task: TaskSpec, state: Array, radius: float) -> bool:
    return bool(np.min(np.linalg.norm(task.goal_points - state[None, :], axis=1)) <= radius)
