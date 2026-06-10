#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tropicalgt.ablation import DEFAULT_TARGETS, write_bpb_ablation_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TropicalGT-I metrics against BPB and graph-BPB")
    parser.add_argument("reports", nargs="+", help="One or more train_report.json paths")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "ablation"))
    parser.add_argument("--baseline", default="", help="Baseline report path, filename, or run name. Defaults to the first report.")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS), help="Comma-separated target metrics")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-html", action="store_true")
    args = parser.parse_args()
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    paths = write_bpb_ablation_artifacts(
        args.reports,
        args.output_dir,
        targets=targets,
        baseline=args.baseline or None,
        top_k=args.top_k,
        render_html=not args.no_html,
    )
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
