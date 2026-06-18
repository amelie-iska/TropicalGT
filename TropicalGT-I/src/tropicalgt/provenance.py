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
        name="bpb_exact",
        kind="legacy_exact_alias",
        surface="training/eval metric",
        optimize_directly=False,
        description="Backward-compatible alias for bpb in older reports; despite the name it is not a proxy.",
        replacement_or_guardrail="Prefer bpb or bpb_exact in new dashboards and scripts.",
        match_terms=("bpb_exact",),
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
            "retired global smooth_projected_nll_fitness_landscape entry",
            "retired broad NLL/fitness surrogate",
            "default retired broad NLL/fitness surrogate",
            "global surrogate NLL surfaces are disabled",
            "isinstance(surrogate, dict)",
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
            "surface-projected piecewise NLL mesh; no microstep or surrogate NLL anchors",
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
        name="persistence_vector_representation_similarity",
        kind="fast_vectorized_topology",
        surface="analogical retrieval metric",
        optimize_directly=True,
        description="Weighted vector-space comparison across real cached GUDHI Landscape, BettiCurve, Silhouette, Entropy, PersistenceLengths, TopologicalVector, and PersistenceImage features.",
        replacement_or_guardrail="Use only when both query and memory expose real GUDHI vector payloads; do not treat the NumPy vectorizer as a torch-native differentiable PH layer.",
        match_terms=("persistence_vector_representation_similarity", "persistence_vector_aggregate_similarity", "gudhi.representations.vector_methods"),
    ),
    ProvenanceEntry(
        name="legacy_multiparameter_free_resolution_alias_removed",
        kind="removed_legacy_alias",
        surface="topological algebra report",
        optimize_directly=False,
        description="Removed legacy alias for finite chain-presentation diagnostics; it must not be emitted by current topological-algebra reports.",
        replacement_or_guardrail="Use multiparameter_chain_presentation_diagnostics for exact finite-chain data and nested real_free_resolution certificate fields for CAS-certified resolutions.",
        match_terms=("multiparameter_free_resolution_legacy_alias", "free-resolution legacy alias", "free_resolution_legacy_alias"),
    ),
    ProvenanceEntry(
        name="buchsbaum_eisenbud_implied_rank_identity_diagnostic",
        kind="diagnostic_rank_identity_not_certificate",
        surface="CAS Buchsbaum-Eisenbud diagnostics",
        optimize_directly=False,
        description=(
            "Image-rank values in the Buchsbaum-Eisenbud rank-condition table are algebraic values "
            "implied by rank(F_i)=rank(d_i)+rank(d_{i+1}) under exactness, not independently certified image ranks."
        ),
        replacement_or_guardrail=(
            "Use image_rank_values_implied_by_exact_rank_identity in new reports; accept the retired "
            "image_rank_estimates_by_differential key only for cached-artifact compatibility and keep is_independent_certificate=false."
        ),
        match_terms=(
            "image_rank_values_implied_by_exact_rank_identity",
            "image_rank_estimates_by_differential",
            "implied_image_rank_values",
            "implied_rank=",
            "rank(F_i)=rank(d_i)+rank(d_{i+1})",
            "Buchsbaum-Eisenbud rank conditions require",
        ),
    ),
    ProvenanceEntry(
        name="determinantal_grade_depth_cas_unavailable",
        kind="cas_required_unavailable_state",
        surface="topological algebra report",
        optimize_directly=False,
        description="Grade/depth and regular-element checks for determinantal ideals are withheld unless Macaulay2, Singular, or Sage can certify them.",
        replacement_or_guardrail="Emit available=false for this subreport and attach no free-resolution certificate; do not substitute finite-rank diagnostics for CAS-certified grade/depth conditions.",
        match_terms=(
            "Ideal grade/depth and regular-element checks for determinantal ideals require Macaulay2/Singular/Sage",
            "no proxy is substituted",
            "grade/depth exactness conditions",
        ),
    ),
    ProvenanceEntry(
        name="graphcg_direction_gram_condition_number",
        kind="spectral_diagnostic",
        surface="GraphCG metric",
        optimize_directly=False,
        description="Gram-condition diagnostic for GraphCG direction collapse.",
        replacement_or_guardrail="Pair with singular values, numerical rank, effective rank, and full-rank penalty.",
        match_terms=("graphcg_direction_gram_condition_number", "direction_gram_condition_number"),
    ),
    ProvenanceEntry(
        name="graphcg_direction_svd_condition_number",
        kind="spectral_diagnostic",
        surface="GraphCG metric",
        optimize_directly=False,
        description="SVD max/min condition diagnostic for active GraphCG directions.",
        replacement_or_guardrail="Use full-rank loss and singular-min metrics for training decisions.",
        match_terms=("graphcg_direction_svd_condition_number", "svd_condition_number", "condition_number"),
    ),
    ProvenanceEntry(
        name="synthetic_h0_fallback",
        kind="retired_visual_fallback",
        surface="persistence barcode plot",
        optimize_directly=False,
        description="Retired display-only H0 bars that used to be produced when a topology payload had no finite intervals.",
        replacement_or_guardrail="Keep absent from visualization code; barcode plots should show real GUDHI intervals or no interval rows.",
        match_terms=("_intervals_or_synthetic_h0", "synthetic fallback", "synthetic_count", "module_beta0", 'interval.get("synthetic")'),
    ),
    ProvenanceEntry(
        name="json_fallback_graph_trace",
        kind="data_fallback",
        surface="data/tokenization metric",
        optimize_directly=False,
        description="Legacy audit label for prohibited graph fallback traces; text-only rows must be marked as derived text graphs, and invalid explicit graph_json must expose parse-unavailable metadata.",
        replacement_or_guardrail="Track graph_json_fallback_rate as a legacy must-remain-zero counter, graph_json_derived_text_graph_rate for required text-derived graph tokenization, and graph_json_parse_unavailable_rate for invalid explicit graph_json evidence.",
        match_terms=(
            "graph_json_fallback",
            "graph json fallback",
            "fallback_graph",
            "conservative fallback graphs",
            "graph_json_derived_from_text",
            "graph_json_parse_unavailable",
            "fallback_count == 0",
            "legacy graph-json substitution guardrail records",
        ),
    ),
    ProvenanceEntry(
        name="parameter_golf_token_id_fallback",
        kind="data_decode_fallback",
        surface="Parameter-Golf tokenizer decode",
        optimize_directly=False,
        description="Prohibited token-id decode fallback for Parameter-Golf shards when a tokenizer model is unavailable or corrupt.",
        replacement_or_guardrail="The loader rejects allow_token_id_fallback=true, missing tokenizer paths, corrupt tokenizer models, and unsupported tokenizer formats under the no_proxy_no_fallback policy.",
        match_terms=(
            "allow_token_id_fallback",
            "token id fallback",
            "token-id fallback",
            "Parameter Golf tokenizer fallback",
            "tokenizer fallback is disabled",
        ),
    ),
    ProvenanceEntry(
        name="training_data_budget_estimate",
        kind="deterministic_data_budget_estimate",
        surface="dataset budget audit",
        optimize_directly=False,
        description="Deterministic accounting estimate for configured and available training token slots from shard metadata.",
        replacement_or_guardrail="Use as a preflight coverage gate; it proves data-budget availability, not final model quality.",
        match_terms=(
            "Estimate available and scheduled training token slots",
            "available_token_slots",
            "configured_training_token_slots",
            "data_budget",
        ),
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
        description="Fallback from a specific config key to a broader/default config key; training data and tokenizer path fallbacks are prohibited.",
        replacement_or_guardrail="Training data roots and tokenizer paths must be explicit existing paths. Configured fallback_roots or tokenizer_fallback_paths fail closed under the no_proxy_no_fallback policy.",
        match_terms=(
            "fallback_shuffle",
            "fallback_roots",
            "loader_shuffle",
            "config fallback",
            "compatibility fallback",
            "tokenizer_fallback_paths",
            "_reject_config_path_fallbacks",
            "config path fallbacks are disabled",
        ),
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
        name="wandb_uncategorized_metric_namespace",
        kind="logging_namespace",
        surface="W&B metric organization",
        optimize_directly=False,
        description="Uncategorized grouping for scalar metrics that do not match a priority W&B namespace.",
        replacement_or_guardrail="Pure logging organization; no effect on training or evaluation. Retired fallback-era names remain audit terms only.",
        match_terms=("_wandb_uncategorized_metric_group", "wandb_uncategorized_metric_namespace", "_wandb_fallback_group", "fallback_group"),
    ),
    ProvenanceEntry(
        name="multipers_optional_signed_measure_backend",
        kind="optional_backend_diagnostic",
        surface="multiparameter persistence backend note",
        optimize_directly=False,
        description="Optional multipers signed-measure backend recommendation for large multiparameter descriptors.",
        replacement_or_guardrail="The in-repo bounded finite-grid module remains the implemented path unless multipers is explicitly installed and audited; signed-measure diagnostics are not substitutes for exact finite-grid or CAS-certified evidence.",
        match_terms=("optional signed-measure backend", "multipers_optional_signed_measure_backend", "module approximation/signed-measure", "multipers", "approximation/signed-measure"),
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
        name="simplex_tree_gudhi_unavailable_state",
        kind="certified_unavailable_state",
        surface="simplicial visualization payload",
        optimize_directly=False,
        description="Explicit unavailable marker emitted when a real Gudhi SimplexTree cannot be built or serialized for an HTML payload.",
        replacement_or_guardrail="Render an unavailable simplex-tree page with the exact GUDHI error; do not substitute raw JSON rows as a SimplexTree, trie, or face-coface poset.",
        match_terms=("unavailable_gudhi_simplex_tree", "simplex_tree_no_proxy_or_fallback", "safe_to_render_simplex_tree"),
    ),
    ProvenanceEntry(
        name="browser_same_data_static_preview_rendering",
        kind="same_data_rendering_contingency",
        surface="interactive browser plot",
        optimize_directly=False,
        description=(
            "Same-data static SVG/HTML preview displayed when the browser cannot create a WebGL context "
            "for a Plotly 3D panel."
        ),
        replacement_or_guardrail=(
            "Rendering-only contingency from the same serialized simplicial payload; never substitutes model "
            "probabilities, embeddings, losses, topology metrics, or training data."
        ),
        match_terms=(
            "webgl-static-preview",
            "main-static-preview",
            "staticPreviewMarkup",
            "renderPanelStaticPreview",
            "promoteMainStaticPreview",
            "WebGL unavailable: same-data static complex preview shown",
            "same-data preview below comes from the selected real filtered-complex payload",
            "Static SVG same-data preview from the same filtered-complex payload",
            "same serialized simplicial object payload",
            "Interactive WebGL rendering is unavailable",
            "webgl_failures",
            # Retired names remain covered so the audit catches stale fallback-era renderer code.
            "webgl-fallback",
            "main-fallback",
            "staticFallbackMarkup",
            "renderPanelStaticFallback",
            "promoteMainStaticFallback",
            "WebGL unavailable: static complex preview shown",
            "static preview below comes from the selected real filtered-complex payload",
            "Static SVG fallback preview from the same filtered-complex payload",
            "webgl fallback test",
        ),
    ),
    ProvenanceEntry(
        name="simplicial_projection_display_layout_evidence",
        kind="visual_display_layout_boundary",
        surface="filtered-complex plot",
        optimize_directly=False,
        description=(
            "Display coordinates for filtered-complex SVG panels may come from real vertex vectors or from "
            "display-only vertex metadata when vectors are unavailable."
        ),
        replacement_or_guardrail=(
            "Treat coordinates as metric/model evidence only when the rendered layout contract reports "
            "coordinate_evidence=real_vertex_vectors and safe_for_metric_claims=true; metadata-only layouts are visual arrangement only."
        ),
        match_terms=(
            "coordinate_evidence=display_only_vertex_metadata_layout",
            "coordinate_evidence=mixed_model_vectors_and_display_metadata_layout",
            "coordinate_evidence=real_vertex_vectors",
            "safe_for_metric_claims=false",
            "safe_for_metric_claims=true",
            "_vertex_display_layout_feature",
            "display_metadata_rows",
            "vector_rows",
            "_feature_pca3_without_synthetic_jitter",
            "feature_pca3_without_synthetic_jitter",
            # Retired names remain covered so the audit catches stale fallback-era code.
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
        match_terms=("no tree-layout or synthetic coordinates", "no synthetic duplicate points", "model graph_state pca only", "coordinate_source"),
    ),
    ProvenanceEntry(
        name="no_proxy_contract_flags",
        kind="guardrail_contract",
        surface="source contracts and unavailable diagnostics",
        optimize_directly=False,
        description="Explicit no-proxy/no-fallback flags and policy prose on visualization, CAS, memory, model, and training sidecars.",
        replacement_or_guardrail="These fields are admissible only as guardrails: unavailable evidence must stay unavailable, and downstream validators must reject substitute geometry, CAS certificates, topology, graph data, or metrics.",
        match_terms=(
            "no_proxy_or_fallback",
            "no_proxy_policy",
            "no_proxy_no_fallback",
            "no proxy/fallback",
            "no proxy artifact was substituted",
            "no proxy is substituted",
            "no proxies or fallbacks",
            "no-proxy policy",
            "no proxy diagnostics",
            "no proxy or fallback",
            "unavailable, not estimated or replaced by a fallback",
            "not estimated from chain diagnostics",
            "not a silent fallback",
            "explicit unavailable diagnostics rather than",
            "uses_global_trajectory_complex_as_proxy",
            "uses_embedding_trajectory_map_as_proxy",
            "uses_static_probability_complex_as_proxy",
            "all_step_simplex_tree_posets_no_proxy",
            "all_step_radius_sliders_no_proxy",
            "all_step_complex_source_contracts_no_proxy",
            "no trajectory-map proxy",
            "no trajectory/static proxy",
            "no proxy</span>",
            "no_proxy_resolution_claim",
            "no-proxy resolution boundary",
            "theorem-scope no-proxy boundaries",
            "no-proxy render contract",
            "no-proxy metric scope",
            "support-token proxies",
            "chart-bundle transport metadata is rendered by proxy",
            "chart_bundle_transport_contracts_missing_actual_data_only_or_no_proxy_flags",
            "no-proxy sidecars",
            "not a full persistence-module free resolution",
            "not a proxy comparison",
            "embedding-only proxy",
            "embedding-only proxy was used",
            "bounded_certified_cas_evidence_no_proxy",
            "no_proxies_or_fallbacks_for_unavailable_evidence",
            "vector-bundle paper sidecar fields",
            "not replaced by nll/fitness landscapes",
            "no-proxy analogical contract",
            "finite toric-ideal sidecar diagnostics rather than",
            "no resolution certificate was fabricated",
        ),
    ),
    ProvenanceEntry(
        name="safe_numeric_parse_default",
        kind="defensive_parser_default",
        surface="visualization count parsing",
        optimize_directly=False,
        description="Local helper default used only when coercing optional display-count metadata to an integer.",
        replacement_or_guardrail="This parser default must not create model metrics, topology, CAS evidence, graph data, or plotted scientific values; it only keeps optional count fields bounded.",
        match_terms=("_safe_int_count", "return int(fallback)"),
    ),
    ProvenanceEntry(
        name="analogical_assignment_solver_fallback",
        kind="algorithmic_solver_fallback_label",
        surface="analogical probability-vector assignment",
        optimize_directly=False,
        description="Greedy assignment label used when an optimal solver is unavailable for model-probability Jensen-Shannon matching.",
        replacement_or_guardrail="The assignment remains over real model probability vectors and must report solver identity, costs, preservation diagnostics, and unavailable states; it must not imply a certified simplicial map by itself.",
        match_terms=("greedy_fallback",),
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
