"""CURE-IL ported to the robomimic / robosuite Lift manipulation benchmark.

This package mirrors the 2D study in ``contractive_recovery_il`` but operates on
a real MuJoCo manipulation task. The nominal policy is a trained robomimic BC
network; the recovery controller and the kNN uncertainty switch are direct ports
of the 2D ``DemoManifoldRecovery`` / ``KNNUncertainty`` / ``CureILPolicy`` design,
applied in end-effector (eef) position space (robosuite OSC_POSE: action = eef
delta pose + gripper).
"""
