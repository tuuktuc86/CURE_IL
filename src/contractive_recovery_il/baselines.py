from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .detector import RNDDetector
from .envs import TaskSpec
from .expert import ExpertOracle
from .policies import Policy


@dataclass
class SafeDAggerPolicy(Policy):
    nominal: Policy
    detector: RNDDetector
    oracle: ExpertOracle
    preferred_path_id: str | None = None
    name: str = "SafeDAgger"
    query_count: int = 0
    _last_queried: bool = False

    def reset(self) -> None:
        self.query_count = 0
        self._last_queried = False
        self.nominal.reset()

    def action(self, state: np.ndarray) -> np.ndarray:
        if self.detector.score(state) > self.detector.tau_on:
            self.query_count += 1
            self._last_queried = True
            return self.oracle.action(state, preferred_path_id=self.preferred_path_id)
        self._last_queried = False
        return self.nominal.action(state)

    def mode(self) -> str:
        return "expert_query" if self._last_queried else "nominal"
