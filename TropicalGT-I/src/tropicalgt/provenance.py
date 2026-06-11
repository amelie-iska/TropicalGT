from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re
from typing import Iterable


RISK_WORDS = ("proxy", "surrogate", "synthetic", "fallback", "estimate", "approx", "optimistic")


@dataclass(frozen=True)
class ProvenanceEntry:
    name: str
    kind: str
    surface: str
    optimize_directly: bool
    description: str
    replacement_or_guardrail: str


PROVENANCE_REGISTRY: tuple[ProvenanceEntry, ...] = (
    ProvenanceEntry(
        name="bpb",
        kind="exact_metric",
        surface="training/eval metric",
        optimize_directly=True,
        description="Bits per target byte from token NLL bits divided by target byte count.",
        replacement_or_guardrail="Primary Parameter Golf metric when byte accounting is available.",
    ),
    ProvenanceEntry(
        name="bpb_proxy",
        kind="legacy_exact_alias",
        surface="training/eval metric",
        optimize_directly=False,
        description="Backward-compatible alias for bpb in older reports; despite the name it is not a proxy.",
        replacement_or_guardrail="Prefer bpb or bpb_exact in new dashboards and scripts.",
    ),
    ProvenanceEntry(
        name="graph_conditioned_bpb_no_side_cost",
        kind="optimistic_conditional_metric",
        surface="eval metric",
        optimize_directly=False,
        description="Graph-conditioned BPB that omits graph side-information bytes.",
        replacement_or_guardrail="Report next to graph_sideinfo_bpb and graph_bpb; do not treat as leaderboard BPB.",
    ),
    ProvenanceEntry(
        name="smooth_projected_nll_fitness_landscape",
        kind="visual_surrogate",
        surface="interactive plot",
        optimize_directly=False,
        description="Smooth projected energy landscape from sampled GoT NLL anchors and distance-to-support penalty.",
        replacement_or_guardrail="Displayed with exact anchor mesh and local_interpolating_nll_sheet residuals.",
    ),
    ProvenanceEntry(
        name="local_interpolating_nll_sheet",
        kind="sample_interpolant",
        surface="interactive plot",
        optimize_directly=False,
        description="Sample-supported IDW sheet through observed GoT state anchors plus rendered microstep anchors.",
        replacement_or_guardrail="Hover/meta report max_point_residual and duplicate-coordinate collapse diagnostics.",
    ),
    ProvenanceEntry(
        name="persistence_landscape",
        kind="fast_vectorized_topology",
        surface="metric/loss/retrieval/plot",
        optimize_directly=True,
        description="GUDHI Landscape vector lambda_k(t) from finite persistence intervals.",
        replacement_or_guardrail="Use as cached NumPy feature/reward/retrieval key unless replaced by torch-native differentiable layer.",
    ),
    ProvenanceEntry(
        name="persistence_image",
        kind="fast_vectorized_topology",
        surface="metric/retrieval/plot",
        optimize_directly=True,
        description="GUDHI PersistenceImage vector from finite persistence intervals.",
        replacement_or_guardrail="Treat as vectorized topology feature, not a barcode replacement in audit views.",
    ),
    ProvenanceEntry(
        name="multiparameter_free_resolution_proxy",
        kind="algebra_proxy",
        surface="topological algebra report",
        optimize_directly=False,
        description="Multigraded free-chain and boundary-monomial report over the displayed finite multi-filtered complex.",
        replacement_or_guardrail="Label as nonminimal proxy; minimal free resolutions require optional CAS/backend.",
    ),
    ProvenanceEntry(
        name="graphcg_direction_gram_condition_proxy",
        kind="spectral_diagnostic",
        surface="GraphCG metric",
        optimize_directly=False,
        description="Gram-condition diagnostic for GraphCG direction collapse.",
        replacement_or_guardrail="Pair with singular values, numerical rank, effective rank, and full-rank penalty.",
    ),
    ProvenanceEntry(
        name="graphcg_direction_svd_condition_proxy",
        kind="spectral_diagnostic",
        surface="GraphCG metric",
        optimize_directly=False,
        description="SVD max/min condition diagnostic for active GraphCG directions.",
        replacement_or_guardrail="Use full-rank loss and singular-min metrics for training decisions.",
    ),
    ProvenanceEntry(
        name="synthetic_h0_fallback",
        kind="visual_fallback",
        surface="persistence barcode plot",
        optimize_directly=False,
        description="Display-only H0 bars produced only when a topology payload has no finite intervals.",
        replacement_or_guardrail="Never log as training topology; validator favors real GUDHI representation payloads.",
    ),
    ProvenanceEntry(
        name="json_fallback_graph_trace",
        kind="data_fallback",
        surface="data/tokenization metric",
        optimize_directly=False,
        description="Fallback graph construction when graph_json is missing or invalid.",
        replacement_or_guardrail="Track graph_json_fallback_rate and invalid_graph_rate; investigate if nonzero.",
    ),
    ProvenanceEntry(
        name="parameter_golf_export_size_estimate",
        kind="estimate",
        surface="packaging report",
        optimize_directly=False,
        description="Estimated compressed export size for Parameter Golf stripped artifacts.",
        replacement_or_guardrail="Validate with actual archive bytes before final submission.",
    ),
)


def provenance_entries() -> list[dict[str, object]]:
    return [asdict(entry) for entry in PROVENANCE_REGISTRY]


def provenance_by_name() -> dict[str, dict[str, object]]:
    return {entry.name: asdict(entry) for entry in PROVENANCE_REGISTRY}


def scan_risky_terms(paths: Iterable[str | Path]) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            candidates = [
                row
                for row in path.rglob("*")
                if row.is_file()
                and "__pycache__" not in row.parts
                and row.suffix in {".py", ".md", ".json", ".tex"}
            ]
        else:
            candidates = [path] if path.exists() else []
        for candidate in candidates:
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                lowered = line.lower()
                matched = next((word for word in RISK_WORDS if word in lowered), "")
                if matched:
                    findings.append(
                        {
                            "path": str(candidate),
                            "line": line_no,
                            "term": matched,
                            "text": line.strip()[:240],
                        }
                    )
    return findings


def write_provenance_audit(paths: Iterable[str | Path], json_path: str | Path, markdown_path: str | Path | None = None) -> dict[str, object]:
    registry = provenance_by_name()
    findings = scan_risky_terms(paths)
    covered = []
    uncovered = []
    for finding in findings:
        text = str(finding.get("text", ""))
        if any(name in text for name in registry):
            covered.append(finding)
        else:
            uncovered.append(finding)
    report: dict[str, object] = {
        "registry": provenance_entries(),
        "risk_words": list(RISK_WORDS),
        "finding_count": len(findings),
        "covered_finding_count": len(covered),
        "uncovered_finding_count": len(uncovered),
        "uncovered_findings": uncovered,
        "findings": findings,
    }
    Path(json_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if markdown_path is not None:
        lines = [
            "# TropicalGT-I Metric and Plot Provenance Audit",
            "",
            f"- Registered entries: `{len(registry)}`",
            f"- Risk-word findings: `{len(findings)}`",
            f"- Covered findings: `{len(covered)}`",
            f"- Uncovered findings: `{len(uncovered)}`",
            "",
            "## Registry",
        ]
        for entry in PROVENANCE_REGISTRY:
            lines.append(
                f"- `{entry.name}`: `{entry.kind}` on {entry.surface}; optimize_directly={entry.optimize_directly}. {entry.replacement_or_guardrail}"
            )
        if uncovered:
            lines.extend(["", "## Uncovered Mentions"])
            for row in uncovered[:80]:
                lines.append(f"- `{row['path']}:{row['line']}` {row['text']}")
        Path(markdown_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
