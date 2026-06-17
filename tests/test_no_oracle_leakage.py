import pytest

from contractive_recovery_il.config import ExperimentConfig
from contractive_recovery_il.envs import make_task
from contractive_recovery_il.eval import make_policies, rollout
from contractive_recovery_il.expert import ExpertOracle, generate_demonstrations


def test_proposed_rollout_does_not_call_oracle(monkeypatch):
    cfg = ExperimentConfig(demo_trajectories_per_mode=4, horizon=80, perturb_step=25)
    task = make_task("single")
    demos = generate_demonstrations(task, cfg, seed=0)

    def forbidden(*args, **kwargs):
        raise AssertionError("CURE-IL policy called the expert oracle")

    monkeypatch.setattr(ExpertOracle, "action", forbidden)
    policies = make_policies(task, demos, cfg)
    result = rollout(task, policies["CURE-IL"], cfg, perturbation=0.45, seed=0, rollout_id=0)
    assert result.expert_queries == 0


def test_safedagger_is_allowed_to_call_oracle():
    cfg = ExperimentConfig(demo_trajectories_per_mode=4, horizon=80, perturb_step=25)
    task = make_task("single")
    demos = generate_demonstrations(task, cfg, seed=0)
    policies = make_policies(task, demos, cfg)
    result = rollout(task, policies["SafeDAgger"], cfg, perturbation=0.60, seed=0, rollout_id=0)
    assert result.expert_queries > 0
