import numpy as np

from contractive_recovery_il.config import ExperimentConfig
from contractive_recovery_il.envs import make_task, perturbation_vector, step


def test_projection_distance_lower_on_path_than_off_path():
    task = make_task("single")
    on = task.paths[0].points[80]
    off = on + np.array([0.0, 1.0])
    assert task.project(on).distance < 1e-9
    assert task.project(off).distance > 0.5


def test_step_clips_action_and_is_seeded():
    cfg = ExperimentConfig()
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(1)
    s = np.array([0.0, 0.0])
    a = np.array([100.0, 0.0])
    s1 = step(s, a, dt=cfg.dt, max_action_norm=cfg.max_action_norm, noise=0.0, rng=rng1)
    s2 = step(s, a, dt=cfg.dt, max_action_norm=cfg.max_action_norm, noise=0.0, rng=rng2)
    np.testing.assert_allclose(s1, s2)
    assert np.isclose(np.linalg.norm(s1 - s), cfg.dt * cfg.max_action_norm)


def test_perturbation_has_requested_magnitude():
    task = make_task("multi")
    rng = np.random.default_rng(2)
    state = task.paths[0].points[60]
    vec = perturbation_vector(task, state, 0.45, rng)
    assert np.isclose(np.linalg.norm(vec), 0.45)
