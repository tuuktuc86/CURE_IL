# Figures

Images embedded in the top-level [`README.md`](../README.md), grouped by subfolder.

| Folder | Files | Used in |
|---|---|---|
| `DAggerConcept/` | `DAgger.png`, `human_intervention.jpg` | Figure 1 — how a human expert intervenes in DAgger. |
| `related_works_contraction/` | `CDP.png`, `NCDS.png`, `ELCD.png`, `SCDS.png` | Figure 2 — vector fields of existing contractive policies. |
| `2d_experiments_figures/` | `trajectory_single_rich.png` (sine), `trajectory_multi_rich.png` (Y-branch), `trajectory_arc_rich.png` (crescent arc), `trajectory_spiral_rich.png` (open spiral), `trajectory_zigzag_rich.png` (switchback) | Figure 3 — perturbed 2D rollouts. |
| `can_videos/` | `can_{bc,safedagger,elcd}_FAILURE.gif`, `can_cure_SUCCESS.gif` | Figure 4 — Robomimic Can rollouts. |
| `lift_videos/` | `lift_{bc,safedagger,elcd}_FAILURE.gif`, `lift_cure_SUCCESS.gif` | Figure 4 — Robomimic Lift rollouts. |

## Notes

- The 2D PNGs are produced by the experiment suite under `outputs/figures/`
  (`trajectory_*_rich.png`); see the repository's reproduction instructions.
- The `*_videos/` GIFs are rendered from Robomimic rollouts and animate inline on GitHub.
- Keep the file names and folder layout as above — the README links to them by path.
