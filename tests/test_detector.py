import numpy as np

from contractive_recovery_il.config import ExperimentConfig
from contractive_recovery_il.detector import calibrate_detector
from contractive_recovery_il.envs import make_task
from contractive_recovery_il.expert import generate_demonstrations
from contractive_recovery_il.eval import split_demonstrations


def test_detector_scores_ood_above_in_distribution():
    cfg = ExperimentConfig(demo_trajectories_per_mode=4)
    demos = generate_demonstrations(make_task("single"), cfg, seed=0)
    train, cal = split_demonstrations(demos, seed=cfg.calibration_seed)
    det = calibrate_detector(train, cal, quantile=0.9, tau_on_scale=1.1, tau_off_scale=0.8)
    in_score = det.score(cal[0])
    ood_score = det.score(cal[0] + np.array([0.0, 2.0]))
    assert ood_score > in_score
    assert det.tau_on >= det.tau_off
    assert det.tau_on > 0
