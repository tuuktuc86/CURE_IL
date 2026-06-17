"""Per-task configuration for the robomimic recovery study.

Each task differs in the object observation key, episode length, and the step at
which a post-grasp perturbation makes sense (just after the object is lifted in the
demonstrations). object[:, :3] of the low_dim ``object`` observation is the object
world position for all three tasks (verified), so the demo bank is task-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATASETS_ROOT = Path(
    "/home/user/anaconda3/envs/cure_robomimic/lib/python3.10/site-packages/datasets"
)


@dataclass(frozen=True)
class TaskCfg:
    short: str            # tag for paths/outputs
    env_name: str         # robomimic/robosuite env name
    object_pos_key: str   # raw obs key holding the object world position
    perturb_step: int     # apply the shove just after the object is grasped/lifted
    horizon: int
    recover_window: int
    push_steps: int = 5   # shove duration; longer for tasks with a denser eef manifold
    perturbations: tuple = (0.0, 0.3, 0.5, 0.7, 0.9)
    # Detector config. Lift's perturbation moves a grasped object *with* the arm, so
    # the (eef,object) joint state stays on-manifold — only an eef-only detector
    # (cube_weight=0) catches it. The long multi-phase tasks need the joint detector
    # with thresholds auto-calibrated from clean rollouts (det_tau_on=None).
    det_cube_weight: float = 1.0
    det_tau_on: float | None = None   # None => auto-calibrate from clean rollouts
    det_tau_off: float | None = None

    @property
    def dataset(self) -> str:
        return str(DATASETS_ROOT / self.short / "ph" / "low_dim_v141.hdf5")


TASKS = {
    "lift": TaskCfg("lift", "Lift", "cube_pos",
                    perturb_step=32, horizon=250, recover_window=120, push_steps=5,
                    det_cube_weight=0.0),  # RND scale => conformally auto-calibrated
    "can": TaskCfg("can", "PickPlaceCan", "Can_pos",
                   perturb_step=50, horizon=350, recover_window=250, push_steps=12),
    "square": TaskCfg("square", "NutAssemblySquare", "SquareNut_pos",
                      perturb_step=70, horizon=500, recover_window=350, push_steps=12),
}
