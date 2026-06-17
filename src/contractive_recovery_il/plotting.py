from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .config import ExperimentConfig
from .envs import TaskSpec, all_tasks, make_task
from .eval import METHOD_ORDER, RolloutResult, aggregate_rows
from .expert import Demonstrations

CURE_NAME = "CURE-IL"
ELCD_NAME = "ELCD"
COLORS = {
    "BC": "#D55E00",
    "SafeDAgger": "#0072B2",
    CURE_NAME: "#009E73",
    ELCD_NAME: "#CC79A7",
}
ORDER = METHOD_ORDER
TASK_LABELS = {
    "single": "Sine path",
    "multi": "Y-branch",
    "arc": "Crescent arc",
    "spiral": "Open spiral",
    "zigzag": "Switchback",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.25,
    })


def _task_names(summary_rows: list[dict]) -> list[str]:
    present = {r["task"] for r in summary_rows}
    ordered = [task.name for task in all_tasks() if task.name in present]
    return ordered + sorted(present - set(ordered))


def _rows(summary_rows: list[dict], task: str, method: str) -> list[dict]:
    return sorted(
        [r for r in summary_rows if r["task"] == task and r["method"] == method],
        key=lambda r: float(r["perturbation"]),
    )


def _float_or_nan(value) -> float:
    if value == "" or value is None:
        return float("nan")
    return float(value)


def _grid(n: int, width: float = 5.1, height: float = 3.7):
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows), squeeze=False)
    return fig, axes.ravel()


def _plot_task_background(ax, task: TaskSpec, demos: Demonstrations, cfg: ExperimentConfig) -> None:
    for path in task.paths:
        pts = path.points
        # Dot tube works for arcs/spirals/switchbacks where fill_between would be invalid.
        tube_size = 900 * cfg.tube_radius
        ax.scatter(pts[::4, 0], pts[::4, 1], s=tube_size, color="gray", alpha=0.035, linewidths=0, zorder=0)
        ax.plot(pts[:, 0], pts[:, 1], color="black", lw=2.0, alpha=0.30, zorder=1)
    ds = demos.states[:: max(1, len(demos.states) // 1200)]
    ax.scatter(ds[:, 0], ds[:, 1], s=3, color="black", alpha=0.10, label="expert demos", zorder=0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)


def plot_metric_curves(summary_rows: list[dict], out: Path, *, metric: str, stderr: str | None, ylabel: str, title: str, filename: str) -> None:
    _style()
    tasks = _task_names(summary_rows)
    fig, axes = _grid(len(tasks))
    for ax, task in zip(axes, tasks):
        for method in ORDER:
            rows = _rows(summary_rows, task, method)
            if not rows:
                continue
            xs = np.asarray([float(r["perturbation"]) for r in rows], dtype=float)
            ys = np.asarray([float(r[metric]) for r in rows], dtype=float)
            ax.plot(xs, ys, marker="o", lw=2.4, color=COLORS[method], label=method)
            if stderr is not None and stderr in rows[0]:
                es = np.asarray([float(r[stderr]) for r in rows], dtype=float)
                ax.fill_between(xs, np.maximum(0, ys - es), np.minimum(1, ys + es), color=COLORS[method], alpha=0.12)
        ax.set_title(TASK_LABELS.get(task, task))
        ax.set_xlabel("Perturbation magnitude")
        ax.set_ylabel(ylabel)
        if "success" in metric:
            ax.set_ylim(-0.03, 1.03)
        ax.grid(True)
    for ax in axes[len(tasks):]:
        ax.axis("off")
    axes[0].legend(loc="lower left", frameon=True)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / filename, bbox_inches="tight")
    plt.close(fig)


def plot_success_curves(summary_rows: list[dict], out: Path) -> None:
    plot_metric_curves(
        summary_rows,
        out,
        metric="recovery_success_mean",
        stderr="recovery_success_se",
        ylabel="Recovery within window",
        title="Recovery success: re-enter the demonstration tube within the fixed recovery window",
        filename="recovery_success_curves.png",
    )
    # Explicit duplicate name for the user's requested interpretation.
    plot_metric_curves(
        summary_rows,
        out,
        metric="recovery_success_mean",
        stderr="recovery_success_se",
        ylabel="Recovery within window",
        title="Recovery within the configured step window",
        filename="recovery_within_window_curves.png",
    )


def plot_goal_success_curves(summary_rows: list[dict], out: Path) -> None:
    plot_metric_curves(
        summary_rows,
        out,
        metric="goal_success_mean",
        stderr=None,
        ylabel="Final goal success",
        title="Goal success: reach the final goal even if recovery was late",
        filename="goal_success_curves.png",
    )


def plot_time_to_recover_curves(summary_rows: list[dict], out: Path) -> None:
    _style()
    tasks = _task_names(summary_rows)
    fig, axes = _grid(len(tasks))
    for ax, task in zip(axes, tasks):
        for method in ORDER:
            rows = _rows(summary_rows, task, method)
            if not rows:
                continue
            xs = np.asarray([float(r["perturbation"]) for r in rows], dtype=float)
            ys = np.asarray([_float_or_nan(r["time_to_recover_mean"]) for r in rows], dtype=float)
            ax.plot(xs, ys, marker="o", lw=2.4, color=COLORS[method], label=method)
        ax.axhline(18, color="black", ls="--", lw=1.4, alpha=0.55, label="window=18" if task == tasks[0] else None)
        ax.set_title(TASK_LABELS.get(task, task))
        ax.set_xlabel("Perturbation magnitude")
        ax.set_ylabel("Mean steps to recover")
        ax.grid(True)
    for ax in axes[len(tasks):]:
        ax.axis("off")
    axes[0].legend(loc="upper left", frameon=True)
    fig.suptitle("How fast recovery happens after perturbation", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "time_to_recover_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_goal_vs_recovery_gap(summary_rows: list[dict], out: Path) -> None:
    _style()
    tasks = _task_names(summary_rows)
    fig, axes = _grid(len(tasks))
    for ax, task in zip(axes, tasks):
        for method in ["BC", CURE_NAME]:
            rows = _rows(summary_rows, task, method)
            if not rows:
                continue
            xs = np.asarray([float(r["perturbation"]) for r in rows], dtype=float)
            goal = np.asarray([float(r["goal_success_mean"]) for r in rows], dtype=float)
            recovery = np.asarray([float(r["recovery_success_mean"]) for r in rows], dtype=float)
            ax.plot(xs, goal, marker="o", lw=2.2, color=COLORS[method], ls="-", label=f"{method} goal")
            ax.plot(xs, recovery, marker="s", lw=2.2, color=COLORS[method], ls="--", label=f"{method} recovery")
        ax.set_title(TASK_LABELS.get(task, task))
        ax.set_xlabel("Perturbation magnitude")
        ax.set_ylabel("Success rate")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True)
    for ax in axes[len(tasks):]:
        ax.axis("off")
    axes[0].legend(loc="lower left", frameon=True, ncol=2)
    fig.suptitle("Goal success and windowed recovery are different outcomes", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "goal_vs_recovery_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_query_bars(summary_rows: list[dict], out: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(12.8, 5.2))
    perturb = 0.45
    tasks = _task_names(summary_rows)
    methods = ["BC", "SafeDAgger", CURE_NAME]
    width = 0.24
    x = np.arange(len(tasks))
    max_v = 1.0
    for j, method in enumerate(methods):
        values = []
        for task in tasks:
            cand = [r for r in summary_rows if r["task"] == task and r["method"] == method and abs(float(r["perturbation"]) - perturb) < 1e-9]
            values.append(float(cand[0]["expert_queries_mean"]) if cand else 0.0)
        max_v = max(max_v, max(values + [0.0]))
        ax.bar(x + (j - 1) * width, values, width=width, color=COLORS[method], alpha=0.88, label=method)
        for xi, v in zip(x + (j - 1) * width, values):
            ax.text(xi, v + max_v * 0.025, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [TASK_LABELS.get(t, t) for t in tasks], rotation=15, ha="right")
    ax.set_ylabel("Mean expert queries / rollout")
    ax.set_title("CURE-IL recovers without online expert queries", fontweight="bold")
    ax.grid(True, axis="y")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out / "expert_query_bars.png", bbox_inches="tight")
    plt.close(fig)


def _select_representatives(results: list[RolloutResult], task: str, perturbation: float) -> dict[str, RolloutResult]:
    reps = {}
    for method in ORDER:
        cand = [r for r in results if r.task == task and r.method == method and abs(r.perturbation - perturbation) < 1e-9]
        if method == "BC":
            cand = sorted(cand, key=lambda r: (r.recovery_success, r.goal_success, -r.mean_tube_distance))
        else:
            cand = sorted(cand, key=lambda r: (not r.recovery_success, -r.goal_success, r.mean_tube_distance))
        if cand:
            reps[method] = cand[0]
    return reps


def _draw_rollout(ax, task: TaskSpec, demos: Demonstrations, cfg: ExperimentConfig, method: str, r: RolloutResult, *, show_title: bool = True) -> None:
    _plot_task_background(ax, task, demos, cfg)
    xy = r.states
    ax.plot(xy[:, 0], xy[:, 1], color=COLORS[method], lw=2.9, alpha=0.96, label=method, zorder=3)
    ax.scatter(xy[0, 0], xy[0, 1], marker="o", s=48, color="white", edgecolor="black", zorder=4, label="start")
    ps = min(cfg.perturb_step, len(xy) - 1)
    ax.scatter(xy[ps, 0], xy[ps, 1], marker="X", s=96, color="#F0E442", edgecolor="black", zorder=5, label="perturb")
    ax.scatter(xy[-1, 0], xy[-1, 1], marker="D", s=46, color=COLORS[method], edgecolor="black", zorder=4, label="final")
    if method == "SafeDAgger":
        qidx = [i for i, m in enumerate(r.modes) if m == "expert_query"]
        if qidx:
            qxy = xy[qidx[:: max(1, len(qidx)//18)]]
            ax.scatter(qxy[:, 0], qxy[:, 1], marker="*", s=72, color="#0072B2", edgecolor="white", zorder=6, label="expert query")
    if method == CURE_NAME:
        ridx = [i for i, m in enumerate(r.modes) if m == "recovery"]
        if len(ridx) > 1:
            segs = np.stack([xy[ridx[:-1]], xy[ridx[1:]]], axis=1)
            ax.add_collection(LineCollection(segs, colors="#00CC96", linewidths=5.2, alpha=0.35, zorder=2, label="recovery mode"))
    if show_title:
        ax.set_title(f"{method}: recover={int(r.recovery_success)}, goal={int(r.goal_success)}, queries={r.expert_queries}")
    ax.legend(loc="upper left", frameon=True)


def plot_trajectories(results: list[RolloutResult], demos_by_task: Dict[str, Demonstrations], cfg: ExperimentConfig, out: Path) -> None:
    _style()
    for task_name in [task.name for task in all_tasks() if task.name in demos_by_task]:
        task = make_task(task_name)
        reps = _select_representatives(results, task_name, perturbation=0.45)
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), sharex=True, sharey=True)
        axes = axes.ravel()
        for ax, method in zip(axes, ORDER):
            if method not in reps:
                ax.set_title(f"{method} (missing)")
                continue
            _draw_rollout(ax, task, demos_by_task[task_name], cfg, method, reps[method])
        fig.suptitle(f"{TASK_LABELS.get(task_name, task_name)} — representative perturbed rollouts", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / f"trajectory_{task_name}_rich.png", bbox_inches="tight")
        plt.close(fig)


def plot_separated_trajectories(results: list[RolloutResult], demos_by_task: Dict[str, Demonstrations], cfg: ExperimentConfig, out: Path) -> None:
    """Save paper-friendly one-method-per-file trajectory panels."""
    _style()
    sep_dir = out / "trajectories_by_method"
    sep_dir.mkdir(parents=True, exist_ok=True)
    for task_name in [task.name for task in all_tasks() if task.name in demos_by_task]:
        task = make_task(task_name)
        reps = _select_representatives(results, task_name, perturbation=0.45)
        for method in ORDER:
            if method not in reps:
                continue
            fig, ax = plt.subplots(figsize=(6.2, 4.8))
            _draw_rollout(ax, task, demos_by_task[task_name], cfg, method, reps[method])
            fig.suptitle(f"{TASK_LABELS.get(task_name, task_name)} — {method}", fontweight="bold")
            fig.tight_layout()
            safe_method = method.lower().replace("-", "_")
            fig.savefig(sep_dir / f"trajectory_{task_name}_{safe_method}.png", bbox_inches="tight")
            plt.close(fig)


def plot_task_gallery(demos_by_task: Dict[str, Demonstrations], cfg: ExperimentConfig, out: Path) -> None:
    _style()
    task_names = [task.name for task in all_tasks() if task.name in demos_by_task]
    fig, axes = _grid(len(task_names), width=4.9, height=3.5)
    for ax, task_name in zip(axes, task_names):
        task = make_task(task_name)
        _plot_task_background(ax, task, demos_by_task[task_name], cfg)
        for path in task.paths:
            ax.scatter(path.start[0], path.start[1], s=52, color="white", edgecolor="black", zorder=4)
            ax.scatter(path.goal[0], path.goal[1], s=70, marker="*", color="#F0E442", edgecolor="black", zorder=4)
        ax.set_title(TASK_LABELS.get(task_name, task_name))
    for ax in axes[len(task_names):]:
        ax.axis("off")
    fig.suptitle("2D benchmark shape gallery", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "task_shape_gallery.png", bbox_inches="tight")
    plt.close(fig)


def _medium_rows(summary_rows: list[dict], perturb: float = 0.45) -> tuple[list[str], dict[tuple[str, str], dict]]:
    tasks = _task_names(summary_rows)
    rows = {(r["task"], r["method"]): r for r in summary_rows if abs(float(r["perturbation"]) - perturb) < 1e-9}
    return tasks, rows


def _plot_medium_bar_panel(ax, summary_rows: list[dict], *, metric: str, ylabel: str, title: str) -> None:
    tasks, rows = _medium_rows(summary_rows)
    x = np.arange(len(tasks))
    width = 0.19
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(ORDER))
    for off, method in zip(offsets, ORDER):
        vals = [float(rows[(t, method)][metric]) for t in tasks if (t, method) in rows]
        xpos = x[:len(vals)] + off
        ax.bar(xpos, vals, width=width, color=COLORS[method], alpha=0.9, label=method)
    ax.set_xticks(x, [TASK_LABELS.get(t, t) for t in tasks], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y")


def _plot_tube_distance_panel(ax, summary_rows: list[dict]) -> None:
    tasks, rows = _medium_rows(summary_rows)
    x = np.arange(len(tasks))
    width = 0.19
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(ORDER))
    for off, method in zip(offsets, ORDER):
        vals = [float(rows[(t, method)]["mean_tube_distance"]) for t in tasks if (t, method) in rows]
        ax.bar(x[:len(vals)] + off, vals, width=width, color=COLORS[method], alpha=0.9, label=method)
    ax.set_xticks(x, [TASK_LABELS.get(t, t) for t in tasks], rotation=15, ha="right")
    ax.set_ylabel("Mean tube distance")
    ax.set_title("Medium perturbation tube distance")
    ax.grid(True, axis="y")


def _plot_query_panel(ax, summary_rows: list[dict]) -> None:
    tasks, rows = _medium_rows(summary_rows)
    x = np.arange(len(tasks))
    width = 0.24
    for j, method in enumerate(["BC", "SafeDAgger", CURE_NAME]):
        vals = [float(rows[(t, method)]["expert_queries_mean"]) for t in tasks if (t, method) in rows]
        ax.bar(x[:len(vals)] + (j - 1) * width, vals, width=width, color=COLORS[method], alpha=0.9, label=method)
    ax.set_xticks(x, [TASK_LABELS.get(t, t) for t in tasks], rotation=15, ha="right")
    ax.set_ylabel("Expert queries / rollout")
    ax.set_title("Medium perturbation query cost")
    ax.grid(True, axis="y")


def plot_dashboard(summary_rows: list[dict], out: Path) -> None:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.4))
    _plot_medium_bar_panel(axes[0, 0], summary_rows, metric="recovery_success_mean", ylabel="Recovery within window", title="Windowed recovery success")
    _plot_medium_bar_panel(axes[0, 1], summary_rows, metric="goal_success_mean", ylabel="Final goal success", title="Goal success")
    _plot_tube_distance_panel(axes[1, 0], summary_rows)
    _plot_query_panel(axes[1, 1], summary_rows)
    for ax in axes.ravel():
        ax.legend(frameon=True, fontsize=7)
    fig.suptitle("Result dashboard: separate goal success, recovery speed, stability, and supervision cost", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "visual_results_dashboard.png", bbox_inches="tight")
    plt.close(fig)

    separated = [
        ("dashboard_recovery_window_panel.png", lambda ax: _plot_medium_bar_panel(ax, summary_rows, metric="recovery_success_mean", ylabel="Recovery within window", title="Windowed recovery success")),
        ("dashboard_goal_success_panel.png", lambda ax: _plot_medium_bar_panel(ax, summary_rows, metric="goal_success_mean", ylabel="Final goal success", title="Goal success")),
        ("dashboard_tube_distance_panel.png", lambda ax: _plot_tube_distance_panel(ax, summary_rows)),
        ("dashboard_expert_query_panel.png", lambda ax: _plot_query_panel(ax, summary_rows)),
    ]
    for filename, draw in separated:
        fig, ax = plt.subplots(figsize=(11.5, 5.0))
        draw(ax)
        ax.legend(frameon=True, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / filename, bbox_inches="tight")
        plt.close(fig)


def plot_method_schematic(out: Path) -> None:
    """Publication/slide-friendly visual summary of the hybrid controller."""
    _style()
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    ax.axis("off")
    boxes = [
        (0.04, 0.54, 0.18, 0.24, "State $s_t$", "2D point state"),
        (0.29, 0.68, 0.22, 0.20, "Nominal BC $\\pi_{nom}$", "task execution"),
        (0.29, 0.30, 0.22, 0.20, "Uncertainty $U(s_t)$", "kNN + conformal threshold"),
        (0.60, 0.30, 0.25, 0.20, "CURE-IL Recovery $\\pi_r$", "demo-manifold pullback"),
        (0.60, 0.68, 0.25, 0.20, "Action $a_t$", "nominal or recovery mode"),
    ]
    for x, y, w, h, title, sub in boxes:
        fc = "#E8F6F3" if "Recovery" in title else ("#EBF5FB" if "Nominal" in title else "#F8F9F9")
        ax.add_patch(plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=fc, edgecolor="#34495E", lw=1.8, zorder=2))
        ax.text(x+w/2, y+h*0.62, title, transform=ax.transAxes, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x+w/2, y+h*0.30, sub, transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#566573")
    def arrow(x1,y1,x2,y2,text=None,color="#34495E"):
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1), xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color=color, shrinkA=2, shrinkB=2))
        if text:
            ax.text((x1+x2)/2, (y1+y2)/2+0.035, text, transform=ax.transAxes, ha="center", fontsize=9, color=color, fontweight="bold")
    arrow(0.22,0.66,0.29,0.76)
    arrow(0.22,0.62,0.29,0.40)
    arrow(0.51,0.78,0.60,0.78,"if in-distribution", "#0072B2")
    arrow(0.51,0.40,0.60,0.40,"if $U(s_t)>\\tau$", "#D55E00")
    arrow(0.73,0.50,0.73,0.68,"return when stable", "#009E73")
    ax.text(0.5, 0.12, "CURE-IL: Contractive Uncertainty-triggered Recovery for Expert-free Imitation Learning", transform=ax.transAxes,
            ha="center", va="center", fontsize=12, fontweight="bold", color="#145A32")
    ax.text(0.5, 0.04, "Baselines: BC = nominal only; SafeDAgger = expert query on uncertain states; ELCD = branch-agnostic contraction contrast", transform=ax.transAxes,
            ha="center", va="center", fontsize=9, color="#566573")
    fig.tight_layout()
    fig.savefig(out / "method_schematic.png", bbox_inches="tight")
    plt.close(fig)


def create_all_figures(results: list[RolloutResult], demos_by_task: Dict[str, Demonstrations], cfg: ExperimentConfig, output_dir: str | Path) -> None:
    out = Path(output_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    summary_rows = aggregate_rows(results)
    plot_success_curves(summary_rows, out)
    plot_goal_success_curves(summary_rows, out)
    plot_goal_vs_recovery_gap(summary_rows, out)
    plot_time_to_recover_curves(summary_rows, out)
    plot_query_bars(summary_rows, out)
    plot_task_gallery(demos_by_task, cfg, out)
    plot_trajectories(results, demos_by_task, cfg, out)
    plot_separated_trajectories(results, demos_by_task, cfg, out)
    plot_dashboard(summary_rows, out)
    plot_method_schematic(out)
