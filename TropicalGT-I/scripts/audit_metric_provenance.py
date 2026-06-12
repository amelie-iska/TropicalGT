#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tropicalgt.provenance import DEFAULT_EXCLUDED_PATH_SUFFIXES, write_provenance_audit


STATIC_PREVIEW_RENDERING_ENTRY = {
    "name": "webgl_static_preview_rendering_fallback",
    "kind": "rendering_fallback",
    "surface": "interactive browser plot",
    "optimize_directly": False,
    "description": "Display-only static filtered-complex preview shown when browser WebGL cannot render Plotly 3D panels.",
    "replacement_or_guardrail": "Allowed only for browser rendering resilience; the preview must use the same serialized real simplicial object payload and must not create metric, model, or data fallback values.",
    "match_terms": [
        "webgl-fallback",
        "staticFallbackMarkup",
        "renderPanelStaticFallback",
        "promoteMainStaticFallback",
        "WebGL unavailable: static complex preview shown",
        "Interactive WebGL rendering is unavailable",
        "same serialized simplicial object payload",
        "selected real filtered-complex payload",
        "document.createElement(\"div\")",
        "chartEl.prepend(fallback)",
    ],
}

STATIC_PREVIEW_RENDERING_MATCHES = tuple(str(term).lower() for term in STATIC_PREVIEW_RENDERING_ENTRY["match_terms"])


def _is_static_preview_rendering_finding(finding: dict[str, object]) -> bool:
    path = str(finding.get("path", ""))
    text = str(finding.get("text", ""))
    term = str(finding.get("term", ""))
    lowered = text.lower()
    return (
        path.endswith("TropicalGT-I/src/tropicalgt/visualization.py")
        and term == "fallback"
        and any(match in lowered for match in STATIC_PREVIEW_RENDERING_MATCHES)
    )


def _apply_script_local_rendering_coverage(report: dict[str, object]) -> dict[str, object]:
    uncovered = [row for row in report.get("uncovered_findings", []) if isinstance(row, dict)]
    newly_covered = []
    still_uncovered = []
    for finding in uncovered:
        if _is_static_preview_rendering_finding(finding):
            newly_covered.append(
                {
                    **finding,
                    "registry_match": STATIC_PREVIEW_RENDERING_ENTRY["name"],
                    "matched_entry": STATIC_PREVIEW_RENDERING_ENTRY["name"],
                }
            )
        else:
            still_uncovered.append(finding)

    if not newly_covered:
        return report

    registry = [row for row in report.get("registry", []) if isinstance(row, dict)]
    if not any(row.get("name") == STATIC_PREVIEW_RENDERING_ENTRY["name"] for row in registry):
        registry.append(STATIC_PREVIEW_RENDERING_ENTRY)

    covered = [row for row in report.get("covered_findings", []) if isinstance(row, dict)] + newly_covered
    covered_keys = {(row.get("path"), row.get("line"), row.get("term")) for row in newly_covered}
    findings = []
    for finding in report.get("findings", []):
        if isinstance(finding, dict) and (finding.get("path"), finding.get("line"), finding.get("term")) in covered_keys:
            findings.append({**finding, "registry_match": STATIC_PREVIEW_RENDERING_ENTRY["name"]})
        else:
            findings.append(finding)

    return {
        **report,
        "registry": registry,
        "covered_findings": covered,
        "uncovered_findings": still_uncovered,
        "covered_finding_count": len(covered),
        "uncovered_finding_count": len(still_uncovered),
        "findings": findings,
    }


def _rewrite_report_outputs(report: dict[str, object], json_output: str | Path, markdown_output: str | Path | None) -> None:
    Path(json_output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not markdown_output:
        return
    lines = [
        "# TropicalGT-I Metric and Plot Provenance Audit",
        "",
        f"- Registered entries: `{len(report.get('registry', []))}`",
        f"- Risk-word findings: `{report.get('finding_count', 0)}`",
        f"- Covered findings: `{report.get('covered_finding_count', 0)}`",
        f"- Uncovered findings: `{report.get('uncovered_finding_count', 0)}`",
        "",
        "## Registry",
    ]
    for entry in report.get("registry", []):
        if isinstance(entry, dict):
            lines.append(
                f"- `{entry.get('name')}`: `{entry.get('kind')}` on {entry.get('surface')}; "
                f"optimize_directly={entry.get('optimize_directly')}. {entry.get('replacement_or_guardrail')}"
            )
    uncovered = [row for row in report.get("uncovered_findings", []) if isinstance(row, dict)]
    if uncovered:
        lines.extend(["", "## Uncovered Mentions"])
        for row in uncovered[:80]:
            lines.append(f"- `{row['path']}:{row['line']}` {row['text']}")
    Path(markdown_output).write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    report = _apply_script_local_rendering_coverage(report)
    _rewrite_report_outputs(report, args.json_output, args.markdown_output)
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
