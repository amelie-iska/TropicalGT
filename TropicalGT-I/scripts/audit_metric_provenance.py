#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tropicalgt.provenance import DEFAULT_EXCLUDED_PATH_SUFFIXES, write_provenance_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit TropicalGT-I proxy/surrogate/fallback metric provenance.")
    parser.add_argument(
        "--scan",
        nargs="+",
        default=[
            "TropicalGT-I/src/tropicalgt",
            "TropicalGT-I/scripts",
            "TropicalGT-I/README.md",
            "README.md",
            "planning/2026-06-11-got-visualization-audit.md",
        ],
        help="Files or directories to scan.",
    )
    parser.add_argument("--json-output", default="TropicalGT-I/outputs/metric_provenance_audit.json")
    parser.add_argument("--markdown-output", default="TropicalGT-I/outputs/metric_provenance_audit.md")
    parser.add_argument(
        "--exclude-suffix",
        nargs="*",
        default=list(DEFAULT_EXCLUDED_PATH_SUFFIXES),
        help="Path suffixes to exclude from scanning. Defaults exclude the provenance audit machinery itself.",
    )
    parser.add_argument(
        "--fail-on-uncovered",
        action="store_true",
        help="Exit nonzero if risk-word mentions are not covered by an explicit registry name.",
    )
    args = parser.parse_args()
    Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
    if args.markdown_output:
        Path(args.markdown_output).parent.mkdir(parents=True, exist_ok=True)
    report = write_provenance_audit(
        args.scan,
        args.json_output,
        args.markdown_output,
        excluded_suffixes=tuple(args.exclude_suffix or ()),
    )
    uncovered = int(report.get("uncovered_finding_count", 0))
    print(f"wrote {args.json_output}")
    if args.markdown_output:
        print(f"wrote {args.markdown_output}")
    print(
        "findings={finding_count} covered={covered_finding_count} uncovered={uncovered_finding_count}".format(
            **report
        )
    )
    if args.fail_on_uncovered and uncovered:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
