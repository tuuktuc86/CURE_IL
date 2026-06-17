from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List
import json


@dataclass(frozen=True)
class ExperimentConfig:
    """Central configuration recorded with every run."""

    dt: float = 0.09
    max_action_norm: float = 1.25
    horizon: int = 150
    perturb_step: int = 48
    tube_radius: float = 0.24
    goal_radius: float = 0.55
    recover_window: int = 18
    demo_trajectories_per_mode: int = 38
    demo_noise: float = 0.018
    rollout_noise: float = 0.006
    calibration_quantile: float = 0.97   # conformal coverage 1 - alpha for the trigger
    tau_on_scale: float = 1.0
    tau_off_scale: float = 0.7
    perturbations: List[float] = field(default_factory=lambda: [0.0, 0.15, 0.30, 0.45, 0.60])
    train_seed: int = 0
    calibration_seed: int = 101
    validation_seeds: List[int] = field(default_factory=lambda: [200, 201])
    test_seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    rollouts_per_seed: int = 18
    quick_rollouts_per_seed: int = 10
    output_dir: str = "outputs"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
