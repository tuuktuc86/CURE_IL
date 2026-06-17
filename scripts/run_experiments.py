#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from contractive_recovery_il.config import ExperimentConfig
from contractive_recovery_il.eval import run_suite, write_outputs
from contractive_recovery_il.plotting import create_all_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2D contractive recovery IL experiments")
    parser.add_argument("--quick", action="store_true", help="run a smaller but complete experiment")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    cfg = ExperimentConfig(output_dir=args.output_dir)
    results, demos = run_suite(cfg, quick=args.quick, seed=args.seed)
    rows = write_outputs(results, cfg, args.output_dir)
    create_all_figures(results, demos, cfg, args.output_dir)
    print(f"wrote {len(results)} rollout records")
    print(f"wrote {len(rows)} aggregate rows")
    print(f"outputs: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
