"""Extract the demonstration manifold (eef trajectories + gripper actions) from a
robomimic low_dim hdf5 dataset.

This is the robomimic analogue of ``contractive_recovery_il.expert.Demonstrations``:
instead of synthetic 2D trajectories it loads the recorded end-effector position
trajectories and per-step actions from real human demonstrations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import h5py


@dataclass
class DemoBank:
    """Flattened demonstration manifold in eef-position space.

    Attributes
    ----------
    eef_points : (N, 3) all demonstrated eef positions, concatenated.
    cube_points: (N, 3) all demonstrated cube positions (object obs[:3]).
    point_traj : (N,) trajectory id for each point.
    point_time : (N,) timestep index (within its trajectory) for each point.
    traj_eef   : list of (T_i, 3) per-demo eef position trajectories.
    traj_act   : list of (T_i, 7) per-demo recorded actions (OSC_POSE + gripper).

    ``cube_weight`` weights the cube channels relative to the eef channels when
    matching the nearest demonstrated state. Matching on (eef, cube) instead of eef
    alone resolves the trajectory-phase ambiguity that makes eef-only recovery pull
    toward the wrong task phase after a perturbation displaces the object.
    """

    eef_points: np.ndarray
    cube_points: np.ndarray
    point_traj: np.ndarray
    point_time: np.ndarray
    traj_eef: List[np.ndarray]
    traj_act: List[np.ndarray]
    traj_cube: List[np.ndarray] = field(default_factory=list)
    cube_weight: float = 1.0

    def nearest_demo_by_cube(self, cube: np.ndarray) -> int:
        """Demo id whose initial cube (x, y) is closest to ``cube`` (for re-grasp replay)."""
        starts = np.array([tc[0, :2] for tc in self.traj_cube])
        return int(np.argmin(np.sum((starts - cube[None, :2]) ** 2, axis=1)))

    @property
    def num_traj(self) -> int:
        return len(self.traj_eef)

    def tube_distance(self, eef: np.ndarray) -> float:
        """Distance from an eef position to the nearest demonstrated eef point."""
        d = np.linalg.norm(self.eef_points - eef[None, :], axis=1)
        return float(d.min())

    def nearest_index(self, eef: np.ndarray, cube: np.ndarray | None = None) -> Tuple[int, int, float]:
        """Return (traj_id, time_idx, distance) of the nearest demonstrated state.

        If ``cube`` is given, match on the weighted (eef, cube) feature so the
        recovered phase is consistent with the current object position.
        """
        d2 = np.sum((self.eef_points - eef[None, :]) ** 2, axis=1)
        if cube is not None:
            d2 = d2 + (self.cube_weight ** 2) * np.sum((self.cube_points - cube[None, :]) ** 2, axis=1)
        i = int(np.argmin(d2))
        return int(self.point_traj[i]), int(self.point_time[i]), float(np.sqrt(d2[i]))


def load_demo_bank(hdf5_path: str, max_demos: int | None = None, cube_weight: float = 1.0,
                   point_stride: int = 1) -> DemoBank:
    """Load the demo manifold.

    ``point_stride`` subsamples the flattened (eef, cube) point cloud used for
    nearest-neighbour / tube-distance / uncertainty queries (kept O(N) per step).
    Full per-trajectory arrays are retained for demo re-execution, so recovery is
    unaffected. Useful on long tasks (Square) where N ~ 30k makes per-step queries
    dominate wall-clock.
    """
    eef_points: list[np.ndarray] = []
    cube_points: list[np.ndarray] = []
    point_traj: list[int] = []
    point_time: list[int] = []
    traj_eef: list[np.ndarray] = []
    traj_act: list[np.ndarray] = []
    traj_cube: list[np.ndarray] = []

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        if max_demos is not None:
            demo_keys = demo_keys[:max_demos]
        for ti, key in enumerate(demo_keys):
            d = f["data"][key]
            eef = np.asarray(d["obs"]["robot0_eef_pos"][:], dtype=float)  # (T,3)
            cube = np.asarray(d["obs"]["object"][:, :3], dtype=float)     # (T,3) cube_pos
            act = np.asarray(d["actions"][:], dtype=float)                # (T,7)
            traj_eef.append(eef)
            traj_act.append(act)
            traj_cube.append(cube)
            for t in range(len(eef)):
                eef_points.append(eef[t])
                cube_points.append(cube[t])
                point_traj.append(ti)
                point_time.append(t)

    s = max(1, int(point_stride))
    return DemoBank(
        eef_points=np.asarray(eef_points)[::s],
        cube_points=np.asarray(cube_points)[::s],
        point_traj=np.asarray(point_traj, dtype=int)[::s],
        point_time=np.asarray(point_time, dtype=int)[::s],
        traj_eef=traj_eef,
        traj_act=traj_act,
        traj_cube=traj_cube,
        cube_weight=cube_weight,
    )


def demo_eef_states(bank: DemoBank) -> np.ndarray:
    """All demonstrated eef positions (for fitting the uncertainty detector)."""
    return bank.eef_points
