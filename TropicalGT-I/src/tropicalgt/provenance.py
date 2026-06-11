from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Iterable


RISK_WORDS = ("proxy", "surrogate", "synthetic", "fallback", "estimate", "approx", "optimistic")
DEFAULT_EXCLUDED_PATH_SUFFIXES = (
    "TropicalGT-I/src/tropicalgt/provenance.py",
    "TropicalGT-I/scripts/audit_metric_provenance.py",
)


@dataclass(frozen=True)
class ProvenanceEntry:
    name: str
    kind: str
    surface: str
    optimize_directly: bool
    description: str
    replacement_or_guardrail: str
    match_terms: tuple[str, ...] = field(default_factory=tuple)


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
        match_terms=("bpb_proxy",),
    ),
    ProvenanceEntry(
        name="graph_conditioned_bpb_no_side_cost",
        kind="optimistic_conditional_metric",
        surface="eval metric",
        optimize_directly=False,
        description="Graph-conditioned BPB that omits graph side-information bytes.",
        replacement_or_guardrail="Report next to graph_sideinfo_bpb and graph_bpb; do not treat as leaderboard BPB.",
        match_terms=("graph_conditioned_bpb_no_side_cost", "optimistic graph-conditioned bpb", "optimistic graph-conditioned"),
    ),
    ProvenanceEntry(
        name="smooth_projected_nll_fitness_landscape",
        kind="visual_surrogate",
        surface="interactive plot",
        optimize_directly=False,
        description="Smooth projected energy landscape from sampled GoT NLL anchors and distance-to-support penalty.",
        replacement_or_guardrail="Displayed with exact anchor mesh and local_interpolating_nll_sheet residuals.",
        match_terms=(
            "smooth_projected_nll_fitness_landscape",
            "_nll_surrogate_landscape_trace",
            "surrogate_landscape_layer",
            "has_surrogate",
            "NLL surrogate layer",
            "NLL surrogate anchor residual",
            'surrogate.get("max_point_residual")',
            "surrogate term is zero",
            "surrogate_terms",
            "visual surrogates",
            "smooth projected nll/fitness surrogate",
            "smooth projected nll/fitness landscape",
            "smooth embedding surrogate",
            "projected nll/fitness landscape",
            "smoothed projected surrogate",
            "NLL surrogate rendered grid",
            "landscape_domain_covers_all_points",
            "anchor_grid_exact_residual",
        ),
    ),
    ProvenanceEntry(
        name="actual_sampled_nll_landscape",
        kind="sample_exact_visual_interpolant",
        surface="interactive plot",
        optimize_directly=False,
        description="Piecewise-linear mesh through sampled model-evaluated GoT states and their measured NLL values.",
        replacement_or_guardrail="Do not interpret as a dense latent-space model loss field; dense actual landscapes require additional forward evaluations on a defined perturbation family.",
        match_terms=(
            "actual_landscape_scope",
            "actual_landscape_layer",
            "actual sampled nll landscape",
            "exact sampled nll landscape",
            "exact piecewise-linear triangulation",
            "piecewise-linear interpolation through sampled model GoT states",
            "no synthetic z-values",
            "no synthetic anchor values",
        ),
    ),
    ProvenanceEntry(
        name="local_interpolating_nll_sheet",
        kind="sample_interpolant",
        surface="interactive plot",
        optimize_directly=False,
        description="Sample-supported IDW sheet through observed GoT state anchors plus rendered microstep anchors.",
        replacement_or_guardrail="Hover/meta report max_point_residual and duplicate-coordinate collapse diagnostics.",
        match_terms=("local_interpolating_nll_sheet", "local interpolating nll sheet", "_nll_local_interpolating_sheet_trace"),
    ),
    ProvenanceEntry(
        name="persistence_landscape",
        kind="fast_vectorized_topology",
        surface="metric/loss/retrieval/plot",
        optimize_directly=True,
        description="GUDHI Landscape vector lambda_k(t) from finite persistence intervals.",
        replacement_or_guardrail="Use as cached NumPy feature/reward/retrieval key unless replaced by torch-native differentiable layer.",
        match_terms=("persistence_landscape", "persistence landscapes", "gudhi persistence landscape", "lambda_k(t)"),
    ),
    ProvenanceEntry(
        name="persistence_image",
        kind="fast_vectorized_topology",
        surface="metric/retrieval/plot",
        optimize_directly=True,
        description="GUDHI PersistenceImage vector from finite persistence intervals.",
        replacement_or_guardrail="Treat as vectorized topology feature, not a barcode replacement in audit views.",
        match_terms=("persistence_image", "persistence images"),
    ),
    ProvenanceEntry(
        name="multiparameter_free_resolution_proxy",
        kind="algebra_proxy",
        surface="topological algebra report",
        optimize_directly=False,
        description="Multigraded free-chain and boundary-monomial report over the displayed finite multi-filtered complex.",
        replacement_or_guardrail="Label as nonminimal proxy; minimal free resolutions require optional CAS/backend.",
        match_terms=("multiparameter_free_resolution_proxy", "free-resolution proxy", "free_resolution_proxy"),
    ),
    ProvenanceEntry(
        name="graphcg_direction_gram_condition_proxy",
        kind="spectral_diagnostic",
        surface="GraphCG metric",
        optimize_directly=False,
        description="Gram-condition diagnostic for GraphCG direction collapse.",
        replacement_or_guardrail="Pair with singular values, numerical rank, effective rank, and full-rank penalty.",
        match_terms=("graphcg_direction_gram_condition_proxy", "direction_gram_condition_proxy"),
    ),
    ProvenanceEntry(
        name="graphcg_direction_svd_condition_proxy",
        kind="spectral_diagnostic",
        surface="GraphCG metric",
        optimize_directly=False,
        description="SVD max/min condition diagnostic for active GraphCG directions.",
        replacement_or_guardrail="Use full-rank loss and singular-min metrics for training decisions.",
        match_terms=("graphcg_direction_svd_condition_proxy", "svd_condition_proxy", "condition_proxy"),
    ),
    ProvenanceEntry(
        name="synthetic_h0_fallback",
        kind="visual_fallback",
        surface="persistence barcode plot",
        optimize_directly=False,
        description="Display-only H0 bars produced only when a topology payload has no finite intervals.",
        replacement_or_guardrail="Never log as training topology; validator favors real GUDHI representation payloads.",
        match_terms=("_intervals_or_synthetic_h0", "synthetic fallback", "synthetic_count", "module_beta0", 'interval.get("synthetic")'),
    ),
    ProvenanceEntry(
        name="json_fallback_graph_trace",
        kind="data_fallback",
        surface="data/tokenization metric",
        optimize_directly=False,
        description="Fallback graph construction when graph_json is missing or invalid.",
        replacement_or_guardrail="Track graph_json_fallback_rate and invalid_graph_rate; investigate if nonzero.",
        match_terms=("graph_json_fallback", "graph json fallback", "fallback_graph", "conservative fallback graphs"),
    ),
    ProvenanceEntry(
        name="parameter_golf_export_size_estimate",
        kind="estimate",
        surface="packaging report",
        optimize_directly=False,
        description="Estimated compressed export size for Parameter Golf stripped artifacts.",
        replacement_or_guardrail="Validate with actual archive bytes before final submission.",
        match_terms=(
            "estimate_parameter_golf_export_size",
            "estimated export",
            "estimated stripped competition export",
            "estimated stripped int8+zlib competition export",
            "estimate stripped parameter-golf export size",
        ),
    ),
    ProvenanceEntry(
        name="graph_structural_token_byte_estimate",
        kind="deterministic_accounting_estimate",
        surface="graph BPB metric",
        optimize_directly=True,
        description="Deterministic byte budget for TokenGT structural tokens and estimated token counts.",
        replacement_or_guardrail="Report together with explicit_graph_json_bytes and graph_bpb; validate against actual serialized graph bytes for final packaging.",
        match_terms=("estimated_tokens", "estimate a deterministic byte budget", "graph structural byte", "graph_token_structural_bytes"),
    ),
    ProvenanceEntry(
        name="config_default_fallback",
        kind="configuration_fallback",
        surface="runtime/config plumbing",
        optimize_directly=False,
        description="Fallback from a specific config key to a broader/default config key.",
        replacement_or_guardrail="Acceptable only for config compatibility; it is not a model metric or training signal.",
        match_terms=("fallback_shuffle", "fallback_root", "loader_shuffle", "config fallback"),
    ),
    ProvenanceEntry(
        name="action_selection_fallback",
        kind="control_flow_fallback",
        surface="GFlowNet/GoT sampler",
        optimize_directly=False,
        description="Control-flow fallback used when action sampling would otherwise return only stop/invalid actions.",
        replacement_or_guardrail="Track action entropy and sampled actions; do not treat fallback branches as learned policy quality.",
        match_terms=("fallback = next((row for row in action_probs", "audit_selection_score", "action_probs"),
    ),
    ProvenanceEntry(
        name="wandb_namespace_fallback",
        kind="logging_fallback",
        surface="W&B metric organization",
        optimize_directly=False,
        description="Default grouping for metrics that do not match a priority W&B namespace.",
        replacement_or_guardrail="Pure logging organization; no effect on training or evaluation.",
        match_terms=("_wandb_fallback_group", "fallback_group"),
    ),
    ProvenanceEntry(
        name="multipers_backend_approximation",
        kind="optional_backend_approximation",
        surface="multiparameter persistence backend note",
        optimize_directly=False,
        description="Optional multipers approximation/signed-measure backend recommendation for large multiparameter descriptors.",
        replacement_or_guardrail="The in-repo bounded finite-grid module remains the implemented path unless multipers is explicitly installed and audited.",
        match_terms=("module approximation/signed-measure", "multipers", "approximation/signed-measure"),
    ),
    ProvenanceEntry(
        name="gudhi_vectorizer_autograd_boundary",
        kind="autograd_boundary",
        surface="persistence vectorization train-time note",
        optimize_directly=False,
        description="GUDHI vector methods are NumPy/scikit-learn transforms in this implementation, not PyTorch autograd layers.",
        replacement_or_guardrail="Use as cached features/rewards/retrieval keys unless a torch-native differentiable replacement is introduced.",
        match_terms=("torch-native differentiable surrogate", "gudhi vectorizers are numpy", "not pytorch autograd losses"),
    ),
    ProvenanceEntry(
        name="simplex_tree_json_fallback",
        kind="serialization_fallback",
        surface="simplicial visualization payload",
        optimize_directly=False,
        description="JSON simplex-tree summary emitted when a real Gudhi SimplexTree object cannot be serialized into HTML payloads.",
        replacement_or_guardrail="Use only for rendering payloads; computation should use the actual topology report before serialization.",
        match_terms=("json-fallback", "simplex_tree", "domain_simplex_tree", "codomain_simplex_tree"),
    ),
    ProvenanceEntry(
        name="simplicial_projection_feature_fallback",
        kind="visual_projection_fallback",
        surface="filtered-complex plot",
        optimize_directly=False,
        description="Feature PCA/semantic-hash/circular coordinates used only when vertex embeddings are missing or degenerate.",
        replacement_or_guardrail="Payload reports fallback source, stress, and correlation; prefer embedding-derived coordinates when available.",
        match_terms=(
            "feature_pca3_from_vertex_embeddings_or_semantic_hash",
            "_feature_pca3_with_jitter",
            "circular_fallback",
            "fallback_stats",
            'stats.get("fallback")',
            "_vertex_pca_feature",
        ),
    ),
    ProvenanceEntry(
        name="provenance_audit_queue",
        kind="audit_process_metadata",
        surface="documentation/audit report",
        optimize_directly=False,
        description="Explicit prose describing the remaining proxy/surrogate/fallback audit queue.",
        replacement_or_guardrail="Use only in planning/docs; implementation claims must map to a more specific registry entry.",
        match_terms=(
            "Exact Versus Proxy/Surrogate Audit",
            "Explicit surrogates",
            "generic fallback/proxy/surrogate/estimate mentions",
            "uncovered prose/code mentions",
            "audit queue",
        ),
    ),
    ProvenanceEntry(
        name="embedding_coordinate_source_diagnostic",
        kind="plot_provenance_diagnostic",
        surface="interactive plot",
        optimize_directly=False,
        description="Audit statement that embedding maps use model graph_state PCA coordinates rather than tree layouts or synthetic coordinates.",
        replacement_or_guardrail="Keep PCA diagnostics, distance correlation, stress, and coordinate_source metadata in plot payloads.",
        match_terms=("no tree-layout or synthetic coordinates", "model graph_state pca only", "coordinate_source"),
    ),
)


def provenance_entries() -> list[dict[str, object]]:
    return [asdict(entry) for entry in PROVENANCE_REGISTRY]


def provenance_by_name() -> dict[str, dict[str, object]]:
    return {entry.name: asdict(entry) for entry in PROVENANCE_REGISTRY}


def _excluded(path: Path, excluded_suffixes: tuple[str, ...]) -> bool:
    rendered = path.as_posix()
    return any(rendered.endswith(suffix) for suffix in excluded_suffixes)


def _registry_match(text: str, registry: tuple[ProvenanceEntry, ...] = PROVENANCE_REGISTRY) -> str | None:
    lowered = text.lower()
    for entry in registry:
        terms = (entry.name, *entry.match_terms)
        if any(term and term.lower() in lowered for term in terms):
            return entry.name
    return None


def scan_risky_terms(
    paths: Iterable[str | Path],
    excluded_suffixes: tuple[str, ...] = DEFAULT_EXCLUDED_PATH_SUFFIXES,
) -> list[dict[str, object]]:
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
            if _excluded(candidate, excluded_suffixes):
                continue
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                lowered = line.lower()
                matched = next((word for word in RISK_WORDS if word in lowered), "")
                if matched:
                    registry_match = _registry_match(line)
                    findings.append(
                        {
                            "path": str(candidate),
                            "line": line_no,
                            "term": matched,
                            "registry_match": registry_match,
                            "text": line.strip()[:240],
                        }
                    )
    return findings


def write_provenance_audit(
    paths: Iterable[str | Path],
    json_path: str | Path,
    markdown_path: str | Path | None = None,
    excluded_suffixes: tuple[str, ...] = DEFAULT_EXCLUDED_PATH_SUFFIXES,
) -> dict[str, object]:
    registry = provenance_by_name()
    findings = scan_risky_terms(paths, excluded_suffixes=excluded_suffixes)
    covered = []
    uncovered = []
    for finding in findings:
        text = str(finding.get("text", ""))
        matched_entry = finding.get("registry_match") or _registry_match(text)
        if matched_entry:
            row = {**finding, "matched_entry": matched_entry}
            covered.append(row)
        else:
            uncovered.append(finding)
    report: dict[str, object] = {
        "registry": provenance_entries(),
        "risk_words": list(RISK_WORDS),
        "finding_count": len(findings),
        "covered_finding_count": len(covered),
        "uncovered_finding_count": len(uncovered),
        "covered_findings": covered,
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
