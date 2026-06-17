#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

METHOD_ORDER = ["BC", "SafeDAgger", "CURE-IL", "ELCD"]
TASK_LABELS = {
    "single": "Sine path",
    "multi": "Y-branch",
    "arc": "Crescent arc",
    "spiral": "Open spiral",
    "zigzag": "Switchback",
}
TASK_ORDER = ["single", "multi", "arc", "spiral", "zigzag"]


def load_summary(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    out = Path(args.output_dir)
    rows = load_summary(out / "metrics" / "summary.csv")
    fig_dir = out / "figures"
    report_dir = out / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    present_tasks = {r["task"] for r in rows}
    tasks = [t for t in TASK_ORDER if t in present_tasks] + sorted(present_tasks - set(TASK_ORDER))

    def row(task, method, perturb):
        return next(r for r in rows if r["task"] == task and r["method"] == method and abs(float(r["perturbation"]) - perturb) < 1e-9)

    lines = [
        "# 2D Contractive Recovery IL — Experiment Report",
        "",
        "## Claim",
        "CURE-IL (Contractive Uncertainty-triggered Recovery for Expert-free Imitation Learning) is evaluated in 2D recoverable perturbation settings. It is designed to recover without online expert queries, improving windowed recovery over BC while exposing SafeDAgger's supervision cost.",
        "",
        "## Metric distinction",
        "",
        "- **Goal success** means the rollout eventually reaches the final goal region.",
        "- **Windowed recovery success** means the rollout re-enters the demonstration/expert tube within the configured recovery window and still reaches the goal.",
        "- Therefore BC can show high goal success but low recovery success when it reaches the goal only after a late or off-manifold correction.",
        "",
        "## Method",
        "",
        "### Benchmark and dynamics",
        "The benchmark uses deterministic 2D reference paths with stochastic point-mass rollouts. Each state is a 2D position and each policy outputs a 2D velocity-like action. The simulator clips actions to the configured maximum norm, advances with `s_{t+1} = s_t + dt * clip(a_t) + epsilon`, and adds Gaussian rollout noise. The evaluated tasks are a sine path, a two-branch Y path, a crescent arc, an open spiral, and a switchback/zigzag path. At the configured perturbation step, the state is displaced along the local normal of the nearest reference path by one of the perturbation magnitudes `{0.0, 0.15, 0.30, 0.45, 0.60}`.",
        "",
        "### Demonstrations and oracle",
        "Demonstrations are generated offline from an oracle controller that projects the current state onto the task path, then combines tangent progress, normal correction toward the path, and a lookahead target term. The oracle is used for demonstration generation and for the SafeDAgger baseline only. CURE-IL does not call the oracle during rollout; this invariant is covered by the test suite.",
        "",
        "### Policies",
        "- **BC** is a nearest-neighbor behavioral cloning policy. It averages the actions of the nearest demonstration states with an RBF-style distance weight.",
        "- **SafeDAgger** uses the same BC nominal policy and the same calibrated uncertainty detector as CURE-IL, but when the detector fires it queries the oracle online and records the query count.",
        "- **CURE-IL** combines BC with a demonstration-manifold recovery controller. A k-nearest-neighbor uncertainty score is calibrated on held-out demonstration states. If the score exceeds the on-threshold, the policy switches from BC to recovery; it switches back after the score stays below the off-threshold for the hold period. The recovery controller selects the nearest demonstration trajectory point, tracks a lookahead point on that trajectory, and adds a tangent drive, producing a contractive pull back toward the demonstrated manifold without online expert labels.",
        "- **ELCD** is a lightweight branch-agnostic contractive contrast. It bins demonstration states by x-coordinate, tracks averaged bin centers, and therefore serves as a simple contractive baseline rather than a full reproduction of the original ELCD method.",
        "",
        "### Metrics and protocol",
        "For each task, method, perturbation magnitude, seed, and rollout id, the suite records goal success, windowed recovery success, expert queries, time to recover, mean distance to the demonstration/expert tube, and branch preservation for the Y-branch task. A rollout is counted as recovered after perturbation only if it re-enters the tube within the configured recovery window and still reaches the goal. Aggregates report the mean and standard error over rollout records.",
        "",
        "### Reproducibility parameters",
        "The generated run used horizon `150`, perturbation step `48`, tube radius `0.24`, goal radius `0.55`, recovery window `18`, `38` demonstration trajectories per mode, train seed `0`, calibration seed `101`, test seeds `{0,1,2,3,4}`, and `18` rollouts per seed. These values are persisted in `outputs/metrics/config_snapshot.json`.",
        "",
        "## Main medium-perturbation results (magnitude 0.45)",
        "",
        "| Task | Method | Goal success | Recovery within window | Expert queries / rollout | Mean tube distance |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task in tasks:
        for method in METHOD_ORDER:
            r = row(task, method, 0.45)
            lines.append(
                f"| {TASK_LABELS.get(task, task)} | {method} | "
                f"{float(r['goal_success_mean']):.3f} | "
                f"{float(r['recovery_success_mean']):.3f} ± {float(r['recovery_success_se']):.3f} | "
                f"{float(r['expert_queries_mean']):.2f} | {float(r['mean_tube_distance']):.3f} |"
            )
    lines.extend([
        "",
        "## Visual artifacts",
        "",
        f"- Goal success curves: `{fig_dir / 'goal_success_curves.png'}`",
        f"- Windowed recovery curves: `{fig_dir / 'recovery_within_window_curves.png'}`",
        f"- Goal-vs-recovery comparison: `{fig_dir / 'goal_vs_recovery_curves.png'}`",
        f"- Time-to-recover curves: `{fig_dir / 'time_to_recover_curves.png'}`",
        f"- Expert query bars: `{fig_dir / 'expert_query_bars.png'}`",
        f"- 2D task shape gallery: `{fig_dir / 'task_shape_gallery.png'}`",
        f"- Sine trajectory overlay: `{fig_dir / 'trajectory_single_rich.png'}`",
        f"- Y-branch trajectory overlay: `{fig_dir / 'trajectory_multi_rich.png'}`",
        f"- Arc trajectory overlay: `{fig_dir / 'trajectory_arc_rich.png'}`",
        f"- Spiral trajectory overlay: `{fig_dir / 'trajectory_spiral_rich.png'}`",
        f"- Switchback trajectory overlay: `{fig_dir / 'trajectory_zigzag_rich.png'}`",
        f"- Paper-friendly one-method trajectory panels: `{fig_dir / 'trajectories_by_method'}`",
        f"- Combined dashboard: `{fig_dir / 'visual_results_dashboard.png'}`",
        f"- Separated dashboard panels: `{fig_dir / 'dashboard_recovery_window_panel.png'}`, `{fig_dir / 'dashboard_goal_success_panel.png'}`, `{fig_dir / 'dashboard_tube_distance_panel.png'}`, `{fig_dir / 'dashboard_expert_query_panel.png'}`",
        f"- Method schematic: `{fig_dir / 'method_schematic.png'}`",
        "",
        "## Integrity notes and limitations",
        "",
        "- Results are generated, not fabricated, from the included experiment harness.",
        "- BC, SafeDAgger, CURE-IL, and ELCD share the same environment, perturbation grid, and metric definitions per task.",
        "- CURE-IL rollout reports zero online expert queries and is tested not to call the oracle expert during rollout.",
        "- This is a 2D/toy-domain recovery study. It is not a full manipulation benchmark and does not prove trajectory-level contraction to an expert tube.",
        "- ELCD is an included lightweight branch-agnostic contractive contrast, not a full reproduction of every detail from the original ELCD literature.",
        "",
        "## Recommended figure usage",
        "Use `goal_success_curves.png` and `recovery_within_window_curves.png` together when explaining why BC can reach goals but fail fast recovery. Use `task_shape_gallery.png` plus the `trajectory_*_rich.png` overlays for qualitative evidence. Use either `visual_results_dashboard.png` or the separated dashboard panels depending on slide layout.",
    ])
    report = "\n".join(lines) + "\n"
    path = report_dir / "experiment_report.md"
    path.write_text(report, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
