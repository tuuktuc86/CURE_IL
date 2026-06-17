import numpy as np

from contractive_recovery_il.config import ExperimentConfig
from contractive_recovery_il.envs import make_task
from contractive_recovery_il.eval import RolloutResult, aggregate_rows, make_policies, rollout
from contractive_recovery_il.expert import generate_demonstrations


def test_rollout_query_counts_and_aggregation():
    cfg = ExperimentConfig(demo_trajectories_per_mode=4, horizon=80, perturb_step=25)
    task = make_task("single")
    demos = generate_demonstrations(task, cfg, seed=0)
    policies = make_policies(task, demos, cfg)
    proposed = rollout(task, policies["CURE-IL"], cfg, perturbation=0.45, seed=0, rollout_id=0)
    safe = rollout(task, policies["SafeDAgger"], cfg, perturbation=0.60, seed=0, rollout_id=0)
    assert proposed.expert_queries == 0
    assert safe.expert_queries > 0
    rows = aggregate_rows([proposed, safe])
    assert {r["method"] for r in rows} == {"CURE-IL", "SafeDAgger"}


def test_recovery_success_requires_window_and_goal():
    states = np.zeros((5, 2))
    base = dict(task="single", method="X", perturbation=0.45, seed=0, rollout_id=0,
                expert_queries=0, mean_tube_distance=0.0, final_distance=0.0,
                branch_preserved=None, states=states, modes=[])
    good = RolloutResult(recovery_success=True, goal_success=True, time_to_recover=3, **base)
    missed_goal = RolloutResult(recovery_success=False, goal_success=False, time_to_recover=3, **base)
    late = RolloutResult(recovery_success=False, goal_success=True, time_to_recover=99, **base)
    rows = aggregate_rows([good, missed_goal, late])
    assert rows[0]["recovery_success_mean"] == 1 / 3
    assert rows[0]["goal_success_mean"] == 2 / 3
