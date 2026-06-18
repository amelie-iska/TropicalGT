from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
import hashlib
import html
import json
import math
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from .data import encode_bytes
from .diagnostics import describe_graph_tokens, per_record_nll, record_diagnostics
from .memory import (
    persistence_landscape_vector_similarity as _memory_persistence_landscape_vector_similarity,
    persistence_vector_representation_similarity as _memory_persistence_vector_representation_similarity,
)
from .simplicial import build_embedding_radius_simplicial_object, build_reasoning_trajectory_complex
from .algebra import _bivariate_staircase_resolution_from_points


GRAPHCG_BROWSER_PRIORITY_METRICS: tuple[str, ...] = (
    "graphcg_loss",
    "graphcg_embedding_span_full_rank",
    "graphcg_direction_bank_clamped_to_embedding_dim",
    "graphcg_requested_num_directions",
    "graphcg_effective_num_directions",
    "graphcg_embedding_span_rank_target",
    "graphcg_num_directions",
    "graphcg_embedding_dim",
    "graphcg_active_directions",
    "graphcg_full_rank",
    "graphcg_active_full_rank",
    "graphcg_raw_full_rank",
    "graphcg_active_rank_fraction",
    "graphcg_raw_active_rank_fraction",
    "graphcg_active_full_rank_penalty",
    "graphcg_full_rank_penalty",
    "graphcg_raw_full_rank_penalty",
    "graphcg_full_rank_possible",
    "graphcg_effective_rank",
    "graphcg_numerical_rank",
    "graphcg_rank_target",
    "graphcg_raw_effective_rank",
    "graphcg_raw_numerical_rank",
    "graphcg_raw_singular_min",
    "graphcg_raw_singular_max",
    "graphcg_min_singular_value",
    "graphcg_max_singular_value",
    "graphcg_direction_norm_mean",
    "graphcg_direction_norm_min",
    "graphcg_direction_norm_max",
    "graphcg_direction_gram_offdiag_mean_abs",
    "graphcg_direction_gram_offdiag_max_abs",
    "graphcg_direction_covariance_mean_abs",
    "graphcg_direction_gram_condition_number",
    "graphcg_direction_effective_rank",
    "graphcg_direction_numerical_rank",
    "graphcg_direction_rank_target",
    "graphcg_direction_singular_min",
    "graphcg_direction_singular_max",
    "graphcg_direction_svd_condition_number",
)


def collect_states(
    model,
    dataset,
    tokenizer,
    seq_len: int,
    device: torch.device,
    limit: int = 8,
    audit_level: str = "none",
    ph_backend: str = "auto",
    audit_max_simplices: int = 1024,
):
    records = [dataset[i] for i in range(min(limit, len(dataset)))]
    xs, ys = zip(*(encode_bytes(r.text, seq_len) for r in records))
    graph_batch = tokenizer.batch_encode(records)
    with torch.no_grad():
        out = model(torch.stack(xs).to(device), graph_batch, torch.stack(ys).to(device))
    states = out["graph_state"].detach().cpu().numpy()
    logits_cpu = out["logits"].detach().cpu()
    predicted_ids = logits_cpu.argmax(dim=-1)
    nll_tensor, _ = per_record_nll(out["logits"].detach().cpu(), torch.stack(ys))
    nll = nll_tensor.numpy()
    graph_token_embeddings = out.get("graph_token_embeddings")
    graph_token_support_probabilities = out.get("graph_token_support_probabilities")
    graph_token_embeddings_cpu = graph_token_embeddings.detach().cpu() if torch.is_tensor(graph_token_embeddings) else None
    graph_token_support_probabilities_cpu = (
        graph_token_support_probabilities.detach().cpu()
        if torch.is_tensor(graph_token_support_probabilities)
        else None
    )
    filtered_objects = []
    embedding_filtered_objects = []
    for idx, record in enumerate(records):
        graph_token_count = int(graph_batch.graph_token_counts[idx].detach().cpu().item())
        descriptors = describe_graph_tokens(record, tokenizer)[:graph_token_count]
        embeddings = (
            graph_token_embeddings_cpu[idx, :graph_token_count]
            if graph_token_embeddings_cpu is not None and graph_token_embeddings_cpu.ndim >= 3
            else []
        )
        probabilities = (
            graph_token_support_probabilities_cpu[idx, :graph_token_count, :graph_token_count]
            if graph_token_support_probabilities_cpu is not None and graph_token_support_probabilities_cpu.ndim >= 3
            else None
        )
        filtered_objects.append(
            build_embedding_radius_simplicial_object(
                record,
                descriptors,
                embeddings,
                token_probabilities=probabilities,
                metric="jensen_shannon",
            )
        )
        embedding_filtered_objects.append(
            build_embedding_radius_simplicial_object(record, descriptors, embeddings, metric="euclidean")
        )
    diagnostics = []
    if (audit_level or "none").lower() != "none":
        diagnostics = record_diagnostics(
            records,
            graph_batch,
            {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in out.items()},
            tokenizer,
            target_ids=torch.stack(ys),
            max_records=len(records),
            max_trace_tokens=32,
            audit_level=audit_level,
            ph_backend=ph_backend,
            audit_max_simplices=audit_max_simplices,
        )
    hover = []
    base_hover = graph_batch.hover_payloads or [r.to_hover_html() for r in records]
    for idx, (html, obj) in enumerate(zip(base_hover, filtered_objects)):
        summary = obj["summary"]
        algebra = diagnostics[idx].get("topological_algebra", {}) if diagnostics else {}
        betti = algebra.get("chain_complex", {}).get("homology", {}).get("betti")
        hover.append(
            html
            + f"<br><b>per-record NLL</b>: {float(nll[idx]):.4f}"
            + f"<br><b>model input</b>: {_html_clip(records[idx].text, 900)}"
            + f"<br><b>model argmax output</b>: {_html_clip(_decode_shifted_bytes(predicted_ids[idx]), 900)}"
            + "<br><b>Filtered simplicial object</b>"
            + f"<br>0-simplices: {summary['num_vertices']}"
            + f"<br>1-simplices: {summary['num_edges']}"
            + f"<br>2-simplices: {summary['num_two_simplices']}"
            + f"<br>filtration thresholds: {summary['num_thresholds']}"
            + (f"<br><b>F2 Betti</b>: {betti}" if betti else "")
        )
    io_rows = [
        {
            "record_id": record.record_id,
            "input_text": record.text,
            "target_text": _decode_shifted_bytes(ys[idx]),
            "decoded_argmax": _decode_shifted_bytes(predicted_ids[idx]),
        }
        for idx, record in enumerate(records)
    ]
    return states, nll, hover, filtered_objects, embedding_filtered_objects, diagnostics, io_rows


def write_reasoning_visualizations(
    model,
    dataset,
    tokenizer,
    seq_len: int,
    device: torch.device,
    output_dir: str | Path,
    limit: int = 8,
    audit_level: str = "none",
    ph_backend: str = "auto",
    audit_max_simplices: int = 1024,
) -> dict[str, str]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    states, nll, hover, filtered_objects, embedding_filtered_objects, diagnostics, io_rows = collect_states(
        model,
        dataset,
        tokenizer,
        seq_len,
        device,
        limit,
        audit_level=audit_level,
        ph_backend=ph_backend,
        audit_max_simplices=audit_max_simplices,
    )
    source_state_count = int(states.shape[0])
    reasoning_visualization_diagnostics = {
        "source_state_count": source_state_count,
        "contrived_duplicate_for_pca": False,
        "single_state_degenerate_pca": source_state_count == 1,
        "pca_point_policy": (
            "no synthetic duplicate points; one-state inputs render as a degenerate single-anchor PCA/NLL view"
        ),
    }
    if source_state_count == 1:
        pca = np.zeros((1, 3), dtype=float)
    else:
        n_components = min(3, states.shape[0], states.shape[1])
        pca = PCA(n_components=n_components).fit_transform(states)
        if pca.shape[1] < 3:
            pca = np.pad(pca, ((0, 0), (0, 3 - pca.shape[1])), constant_values=0)
    surface_pc3, surface_meta = _nll_surface_trace(pca[:, 0], pca[:, 1], nll, z_values=pca[:, 2], mode="embedding_height", name="Interpolating embedding surface colored by NLL")
    fig3d = go.Figure()
    if surface_pc3 is not None:
        fig3d.add_trace(surface_pc3)
    fig3d.add_trace(_nll_anchor_trace(pca[:, 0], pca[:, 1], pca[:, 2], nll, name="Embedding surface anchors"))
    fig3d.add_trace(
        go.Scatter3d(
            x=pca[:, 0],
            y=pca[:, 1],
            z=pca[:, 2],
            mode="markers",
            marker=dict(size=6, color=nll, colorscale="Viridis", showscale=True, colorbar=dict(title="NLL")),
            text=hover,
            hoverinfo="text",
            name="validation graph state",
        )
    )
    node_indices = np.arange(len(pca), dtype=int)
    fig3d.data[-1].customdata = node_indices
    title_suffix = " (single-state degenerate PCA)" if source_state_count == 1 else ""
    fig3d.update_layout(title=f"TropicalGT-I validation graph-state PCA sample{title_suffix}", scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
    panel_items = _simplicial_panel_items(filtered_objects, hover)
    p3 = output_dir / "reasoning_trajectory_3d.html"; _write_plotly_dark_html(p3, fig3d, f"TropicalGT-I validation graph-state PCA sample{title_suffix}", panel_items, show_filtration_slider=True, show_selected_complex_panel=True)
    surface_nll, surface_nll_meta = _nll_surface_trace(pca[:, 0], pca[:, 1], nll, z_values=nll, mode="nll_height", name="Interpolating NLL surface through reasoning points")
    fig2 = go.Figure()
    if surface_nll is not None:
        fig2.add_trace(surface_nll)
    fig2.add_trace(_nll_anchor_trace(pca[:, 0], pca[:, 1], nll, nll, name="NLL surface anchors"))
    fig2.add_trace(
        go.Scatter3d(
            x=pca[:, 0],
            y=pca[:, 1],
            z=nll,
            mode="markers",
            marker=dict(size=6, color=nll, colorscale="Plasma", showscale=True, colorbar=dict(title="NLL")),
            text=hover,
            hoverinfo="text",
            name="validation graph state",
        )
    )
    fig2.data[-1].customdata = node_indices
    fig2.update_layout(title=f"TropicalGT-I validation PCA with NLL height{title_suffix}", scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="NLL"))
    p2 = output_dir / "reasoning_trajectory_pca_nll.html"; _write_plotly_dark_html(p2, fig2, f"TropicalGT-I validation PCA with NLL height{title_suffix}", panel_items, show_filtration_slider=True, show_selected_complex_panel=True)
    payload = output_dir / "reasoning_trajectory_payloads.json"
    points = []
    for idx, obj in enumerate(filtered_objects):
        points.append(
            {
                "index": idx,
                "record_id": obj.get("record_id"),
                "pca": {"pc1": float(pca[idx, 0]), "pc2": float(pca[idx, 1]), "pc3": float(pca[idx, 2])},
                "nll": float(nll[idx]),
                "filtered_summary": obj["summary"],
                "input_text": io_rows[idx]["input_text"] if idx < len(io_rows) else "",
                "target_text": io_rows[idx]["target_text"] if idx < len(io_rows) else "",
                "decoded_argmax": io_rows[idx]["decoded_argmax"] if idx < len(io_rows) else "",
            }
        )
    payload.write_text(
        json.dumps(
            {
                "hover": hover,
                "points": points,
                "filtered_simplicial_objects": filtered_objects,
                "embedding_filtered_simplicial_objects": embedding_filtered_objects,
                "nll_surface": {"embedding_height": surface_meta, "nll_height": surface_nll_meta},
                "model_io": io_rows,
                "reasoning_visualization_diagnostics": reasoning_visualization_diagnostics,
                **({"topological_algebra_diagnostics": diagnostics} if diagnostics else {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths = {"pca_3d": str(p3), "pca_nll": str(p2), "payloads": str(payload)}
    if diagnostics:
        diagnostics_path = output_dir / "reasoning_topological_algebra.json"
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        paths["topological_algebra"] = str(diagnostics_path)
    return paths


def write_inference_audit_artifacts(
    result: dict[str, object],
    output_dir: str | Path,
    render_html: bool = True,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    trajectory_growth = None

    audit_path = output_dir / "inference_audit.json"
    audit_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    paths["inference_audit"] = str(audit_path)

    topology = result.get("topological_algebra") if isinstance(result, dict) else None
    if isinstance(topology, dict):
        topology_path = output_dir / "inference_topology.json"
        topology_path.write_text(
            json.dumps(
                {
                    "chain_complex": topology.get("chain_complex"),
                    "persistence": topology.get("persistence"),
                    "persistence_module": topology.get("persistence_module"),
                    "graph_metrics": topology.get("graph_metrics"),
                    "derived_equivalence_signature": topology.get("derived_equivalence_signature"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        paths["topology"] = str(topology_path)
        algebra_path = output_dir / "inference_algebra.json"
        algebra_path.write_text(json.dumps(topology.get("commutative_algebra", {}), indent=2), encoding="utf-8")
        paths["algebra"] = str(algebra_path)

    scaling = result.get("inference_scaling") if isinstance(result, dict) else None
    if isinstance(scaling, dict):
        scaling_path = output_dir / "inference_scaling_tree.json"
        scaling_path.write_text(json.dumps(scaling, indent=2), encoding="utf-8")
        paths["scaling_tree_json"] = str(scaling_path)
        trajectory_topology = scaling.get("trajectory_topological_algebra")
        if isinstance(trajectory_topology, dict):
            trajectory_topology_path = output_dir / "trajectory_topological_algebra.json"
            trajectory_topology_path.write_text(json.dumps(trajectory_topology, indent=2), encoding="utf-8")
            paths["trajectory_topological_algebra"] = str(trajectory_topology_path)
        trajectory_growth = scaling.get("trajectory_growth")
        if isinstance(trajectory_growth, list):
            growth_path = output_dir / "trajectory_growth_topology.json"
            growth_path.write_text(json.dumps(trajectory_growth, indent=2), encoding="utf-8")
            paths["trajectory_growth"] = str(growth_path)
        level_radius = scaling.get("trajectory_level_radius_bifiltration")

        def _level_radius_report_needs_refresh(report: Any) -> bool:
            if not isinstance(report, dict):
                return True
            if not report.get("available"):
                return True
            if not isinstance(report.get("grid_fiber_provenance"), dict):
                return True
            fiber_rows = report.get("fiber_rank_profile")
            if not isinstance(fiber_rows, list) or not fiber_rows:
                return True
            return any(
                isinstance(row, dict)
                and (not isinstance(row.get("fiber_basis"), dict) or not isinstance(row.get("fiber_provenance"), dict))
                for row in fiber_rows
            )

        if isinstance(trajectory_growth, list) and _level_radius_report_needs_refresh(level_radius):
            try:
                from .algebra import compute_level_radius_bifiltration_report

                has_probability = all(
                    isinstance(row, dict) and _has_real_probability_filtration(row.get("probability_filtered_simplicial_object"))
                    for row in trajectory_growth
                )
                level_radius = compute_level_radius_bifiltration_report(
                    trajectory_growth,
                    object_key="probability_filtered_simplicial_object" if has_probability else "filtered_simplicial_object",
                    max_simplices=1024,
                )
                scaling["trajectory_level_radius_bifiltration"] = level_radius
            except Exception as exc:
                level_radius = {"available": False, "reason": f"level-radius bifiltration derivation failed: {type(exc).__name__}: {exc}"}
        if not isinstance(level_radius, dict):
            candidates = scaling.get("candidates") if isinstance(scaling, dict) else None
            has_nonempty_candidates = isinstance(candidates, list) and bool(candidates)
            reason = (
                "trajectory_level_radius_bifiltration_missing_for_nonempty_scaling_report_without_trajectory_growth"
                if has_nonempty_candidates
                else "trajectory growth unavailable only for empty or invalid trajectory"
            )
            level_radius = {
                "available": False,
                "num_parameters": 2,
                "parameters": [
                    {"name": "trajectory_level", "meaning": "reasoning growth level in the sampled graph-of-thought"},
                    {"name": "radius", "meaning": "scalar radius/filtration threshold"},
                ],
                "coefficient_ring": "F2[x_level,x_radius]",
                "fiber_rank_profile": [],
                "rank_invariant_samples": [],
                "structure_maps": [],
                "reason": reason,
                "object_key_policy": "unavailable: no trajectory_growth rows were present to compute an actual bifiltration",
                "object_key_selected": "unavailable",
                "grid_fiber_provenance": {"available": False, "reason": reason},
            }
            scaling["trajectory_level_radius_bifiltration"] = level_radius
        if isinstance(level_radius, dict):
            level_radius_path = output_dir / "trajectory_level_radius_bifiltration.json"
            level_radius_path.write_text(json.dumps(level_radius, indent=2), encoding="utf-8")
            paths["trajectory_level_radius_bifiltration"] = str(level_radius_path)

    memory = result.get("analogical_memory_retrieval") if isinstance(result, dict) else None
    if not isinstance(memory, dict):
        memory = {
            "available": False,
            "reason": "memory_retrieval_not_configured",
            "bank_path": "",
            "bank_size": 0,
            "records_added": 0,
            "top_k": 0,
            "retrieved": [],
        }
    memory_path = output_dir / "analogical_memory_retrieval.json"
    memory_path.write_text(json.dumps(memory, indent=2), encoding="utf-8")
    paths["analogical_memory_retrieval"] = str(memory_path)

    if render_html:
        if isinstance(scaling, dict):
            paths.update(write_got_trajectory_visualization(scaling, output_dir))
            paths.update(write_graphcg_trajectory_visualization(scaling, output_dir))
        paths.update(write_tropical_support_heatmap(result, output_dir))
        paths.update(write_tropical_fan_diagnostics(result, output_dir))
        paths.update(write_toric_embedding_sidecar(result, output_dir))
        paths.update(write_chart_bundle_transport_sidecar(result, output_dir))
        if isinstance(topology, dict):
            paths.update(write_persistence_visualizations(topology, output_dir))
        trajectory_topology = scaling.get("trajectory_topological_algebra") if isinstance(scaling, dict) else None
        if isinstance(trajectory_topology, dict):
            paths.update(
                {
                    f"trajectory_{key}": value
                    for key, value in write_persistence_visualizations(
                        trajectory_topology,
                        output_dir / "trajectory_persistence",
                        growth=trajectory_growth if isinstance(trajectory_growth, list) else None,
                        title_prefix="Trajectory ",
                    ).items()
                }
            )
        level_radius = scaling.get("trajectory_level_radius_bifiltration") if isinstance(scaling, dict) else None
        if isinstance(level_radius, dict):
            bifiltration_html = Path(write_two_parameter_bifiltration_visualization(
                output_dir / "trajectory_persistence" / "two_parameter_bifiltration.html",
                level_radius,
                title="Trajectory 2-parameter persistence over F2[x_level,x_radius]",
            ))
            paths["trajectory_level_radius_bifiltration"] = str(bifiltration_html)
            sidecar = bifiltration_html.with_suffix(".json")
            if sidecar.exists():
                paths["trajectory_level_radius_bifiltration_visual_payload"] = str(sidecar)
        if isinstance(memory, dict):
            query_context = {}
            if isinstance(scaling, dict):
                best = scaling.get("best") if isinstance(scaling.get("best"), dict) else {}
                query_context = {
                    "label": "query trajectory",
                    "embedding": best.get("embedding", []) if isinstance(best, dict) else [],
                    "filtered_simplicial_object": scaling.get("trajectory_filtered_simplicial_object")
                    or (best.get("filtered_simplicial_object") if isinstance(best, dict) else {}),
                    "trajectory_probability_filtered_simplicial_object": scaling.get("trajectory_probability_filtered_simplicial_object")
                    or (best.get("probability_filtered_simplicial_object") if isinstance(best, dict) else {}),
                    "topological_algebra": scaling.get("trajectory_probability_topological_algebra")
                    or scaling.get("trajectory_topological_algebra")
                    or (best.get("topological_algebra") if isinstance(best, dict) else {}),
                }
            paths.update(write_analogical_memory_visualization(memory, output_dir, query_context=query_context))
        paths["dashboard"] = str(_write_inference_dashboard(paths, output_dir))
    return paths


def write_got_trajectory_visualization(scaling_report: dict[str, object], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [row for row in scaling_report.get("candidates", []) if isinstance(row, dict) and row.get("embedding") is not None]
    path = output_dir / "got_trajectory_pca_3d.html"
    payload_path = output_dir / "got_trajectory_payloads.json"
    if not candidates:
        _write_dark_empty(path, "No graph-of-thought candidate embeddings available.")
        payload_path.write_text(json.dumps({"nodes": [], "edges": []}, indent=2), encoding="utf-8")
        return {"got_trajectory_3d": str(path), "got_payloads": str(payload_path)}

    embeddings = np.asarray([row["embedding"] for row in candidates], dtype=float)
    pca, pca_report = _pca3_with_report(embeddings)
    pca_multiplicity = _coordinate_multiplicities(pca)
    ids = [str(row.get("record_id", idx)) for idx, row in enumerate(candidates)]
    id_to_idx = {rid: idx for idx, rid in enumerate(ids)}
    inferred_levels = _infer_candidate_levels(candidates, ids)
    scores = [float(row.get("score", 0.0)) for row in candidates]
    nll_rows: list[float] = []
    missing_nll_ids: list[str] = []
    for idx, row in enumerate(candidates):
        try:
            value = float(row.get("nll"))
        except (TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value):
            missing_nll_ids.append(ids[idx])
        nll_rows.append(value)
    if missing_nll_ids:
        reason = (
            "Graph-of-thought NLL outputs unavailable for "
            f"{len(missing_nll_ids)}/{len(candidates)} model-evaluated states; GoT observed-NLL PCA view was not rendered."
        )
        missing = set(missing_nll_ids)
        _write_dark_empty(path, reason)
        payload_path.write_text(
            json.dumps(
                {
                    "available": False,
                    "reason": "missing_model_nll_outputs",
                    "missing_nll_record_ids": missing_nll_ids,
                    "nodes": [
                        {
                            "record_id": ids[idx],
                            "parent": candidates[idx].get("parent"),
                            "path": candidates[idx].get("path", []),
                            "embedding": candidates[idx].get("embedding"),
                            "nll_available": ids[idx] not in missing,
                        }
                        for idx in range(len(candidates))
                    ],
                    "edges": [],
                    "nll_surface": {"available": False, "reason": "missing_model_nll_outputs"},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"got_trajectory_3d": str(path), "got_payloads": str(payload_path)}
    nll_values = np.asarray(nll_rows, dtype=float)
    nll_center = float(np.nanmedian(nll_values)) if nll_values.size else 0.0
    nll_plot_scale = _nll_visual_scale(nll_values)
    raw_nll_plot_z = (nll_values - nll_center) * nll_plot_scale
    nll_plot_z, surface_projection_meta = _project_points_to_nll_surface_z(pca[:, 0], pca[:, 1], raw_nll_plot_z)
    step_page_links = [
        {
            "reasoning_step_index": idx,
            "step_complex_href": f"reasoning_step_complex_maps/reasoning_step_{idx:03d}.html",
            "step_simplex_tree_href": f"reasoning_step_complex_maps/reasoning_step_{idx:03d}_simplex_tree.html",
        }
        for idx in range(len(candidates))
    ]
    hover = [_candidate_hover(row) for row in candidates]
    nll_progress = _trajectory_nll_progress_diagnostics(candidates, ids, id_to_idx, nll_values, inferred_levels)
    candidate_objects = []
    for idx, row in enumerate(candidates):
        obj = row.get("filtered_simplicial_object") if isinstance(row.get("filtered_simplicial_object"), dict) else {}
        obj = dict(obj)
        obj.setdefault("record_id", ids[idx])
        obj.update(step_page_links[idx])
        candidate_objects.append(obj)
    panel_objects = list(candidate_objects)
    panel_hover = list(hover)
    microstep_entries: list[dict[str, object]] = []
    microsteps_by_candidate: dict[int, list[dict[str, object]]] = {}
    fig = go.Figure()
    energy_anchor_coords = pca[:, :3]
    trajectory_plot_z = pca[:, 2]
    density_cloud, density_cloud_meta = _gaussian_nll_density_cloud(energy_anchor_coords, nll_values, max_samples=4200)
    density_cloud_meta.update(
        {
            "coordinate_space": "actual 3D PCA coordinates of model graph-state embeddings (PC1, PC2, PC3); raw NLL is color/density metadata",
            "separate_true_3d_pca_density_page": "got_nll_density_cloud_pca_3d.html",
            "support_sample_trace_visibility": "legendonly",
            "support_samples_hidden_as_model_states": True,
            "sample_points_are_model_states": False,
            "visible_density_layers": ["density_volume", "anchor_gaussian_neighborhoods", "actual_model_anchor_markers"],
        }
    )
    if density_cloud is not None:
        volume_trace, volume_meta = _gaussian_density_volume_trace(
            energy_anchor_coords,
            nll_values,
            float(density_cloud_meta.get("sigma", 0.0) or 0.0),
            name="translucent NLL density field",
        )
        density_cloud_meta["density_volume"] = volume_meta
        if volume_trace is not None:
            fig.add_trace(volume_trace)
        density_points = density_cloud["points"]
        fig.add_trace(
            go.Scatter3d(
                x=density_points[:, 0],
                y=density_points[:, 1],
                z=density_points[:, 2],
                mode="markers",
                marker=dict(
                    size=2.0,
                    color=density_cloud["local_nll"],
                    colorscale="Plasma",
                    opacity=0.08,
                    showscale=False,
                ),
                customdata=np.column_stack(
                    [
                        density_cloud["local_nll"],
                        density_cloud["density"],
                        density_cloud["nearest_distance"],
                        density_cloud["nearest"],
                    ]
                ),
                hovertemplate=(
                    "NLL density sample around actual GoT embedding anchor<br>"
                    "PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<br>"
                    "kernel local raw NLL=%{customdata[0]:.6f}<br>"
                    "Gaussian density mass=%{customdata[1]:.4g}<br>"
                    "nearest actual state distance=%{customdata[2]:.4g}<br>"
                    "nearest actual state index=%{customdata[3]:.0f}<br>"
                    "not a model state; visualization-only Gaussian neighborhood around actual embedding anchors<extra></extra>"
                ),
                name="audit samples from the Gaussian NLL field (hidden by default)",
                visible="legendonly",
                showlegend=True,
            )
        )
    else:
        density_cloud_meta = {"available": False, "reason": "no finite PCA/NLL anchors"}
    nll_surface, nll_surface_meta = _nll_triangulated_surface_trace(
        pca[:, 0],
        pca[:, 1],
        nll_plot_z,
        nll_values,
        name="Sparse observed-state anchor mesh diagnostic",
    )
    nll_surface_meta.update(
        {
            "z_axis": "projected_nll_fitness_energy",
            "z_axis_center_raw_nll": nll_center,
            "z_axis_scale": nll_plot_scale,
            "z_axis_label": f"observed-state NLL anchor z; raw centered NLL x {nll_plot_scale:g} retained separately",
            "raw_nll_range": float(np.nanmax(nll_values) - np.nanmin(nll_values)) if nll_values.size else 0.0,
            "exact_anchor_scope": "observed model-evaluated GoT states only",
            "exact_anchor_layer": True,
            "actual_landscape_scope": "unavailable: this artifact shows only a sparse observed-state anchor mesh, not a dense model-evaluated landscape",
            "sparse_observed_anchor_layer": True,
            "dense_model_evaluated_field": False,
            "truthfulness_warning": "The surface is only a piecewise-linear mesh through sampled model GoT states; it must not be read as a dense latent-space NLL field.",
            "local_interpolation_anchor_scope": "observed model-evaluated GoT states only",
            "model_state_anchor_count": int(len(candidates)),
            "rendered_microsteps_are_nll_surface_anchors": False,
            "rendered_microsteps_policy": "disabled; only model-evaluated GoT states are plotted as trajectory points",
            "surface_contact_contract": "disabled for the main trajectory page: rendered GoT state marker and edge endpoint z values are actual PC3 coordinates; raw NLL is encoded by color/hover and Gaussian density metadata",
            "trajectory_point_surface_residual_max": 0.0,
            "raw_nll_z_residual_max": float(np.nanmax(np.abs(nll_plot_z - raw_nll_plot_z))) if nll_plot_z.size else 0.0,
            "surface_projection": surface_projection_meta,
            "surface_projected_z_by_record_id": {ids[i]: float(nll_plot_z[i]) for i in range(len(ids))},
        }
    )
    nll_surface_meta["local_interpolating_sheet"] = {
        "available": False,
        "reason": "disabled_to_preserve_exact_reasoning_point_surface_contact",
    }
    nll_surface_meta["surrogate_landscape_layer"] = {
        "available": False,
        "reason": "disabled_by_default_not_model_evaluated",
    }
    if nll_surface is not None:
        nll_surface.visible = "legendonly"
        nll_surface.opacity = 0.18
        fig.add_trace(nll_surface)
    density_blob_count = _add_gaussian_nll_blob_meshes(fig, energy_anchor_coords, nll_values, float(density_cloud_meta.get("sigma", 0.0) or 0.0))
    density_cloud_meta["nll_gaussian_blob_count"] = int(density_blob_count)
    fig.add_trace(_nll_anchor_trace(pca[:, 0], pca[:, 1], trajectory_plot_z, nll_values, name="model-evaluated GoT NLL anchors"))
    main_density_camera_eye = dict(x=1.52, y=-1.72, z=1.18)
    main_density_render_contract = _nll_density_render_contract(
        anchor_count=len(candidates),
        support_sample_count=int(density_cloud_meta.get("sample_count", 0) or 0),
        sigma=float(density_cloud_meta.get("sigma", 0.0) or 0.0),
        page="main_trajectory",
        camera_eye=main_density_camera_eye,
        anchor_trace_name="model-evaluated GoT NLL anchors",
        support_sample_trace_name="audit samples from the Gaussian NLL field (hidden by default)",
    )
    density_cloud_meta["visual_layer_contract"] = main_density_render_contract
    for idx, row in enumerate(candidates):
        parent = row.get("parent")
        if isinstance(parent, str) and parent in id_to_idx:
            j = id_to_idx[parent]
            action = _edge_action_label(row)
            raw_delta = float(nll_values[idx] - nll_values[j])
            improvement_label = "improved" if raw_delta < 0 else ("flat" if abs(raw_delta) <= 1e-12 else "regressed")
            chain = [
                {"x": float(pca[j, 0]), "y": float(pca[j, 1]), "z": float(trajectory_plot_z[j]), "label": str(parent)},
                *microsteps_by_candidate.get(idx, []),
                {"x": float(pca[idx, 0]), "y": float(pca[idx, 1]), "z": float(trajectory_plot_z[idx]), "label": ids[idx]},
            ]
            fig.add_trace(
                go.Scatter3d(
                    x=[float(point["x"]) for point in chain],
                    y=[float(point["y"]) for point in chain],
                    z=[float(point["z"]) for point in chain],
                    mode="lines",
                    line=dict(color=_action_color(action), width=4),
                    showlegend=False,
                    hovertext=(
                        f"{parent} -> {ids[idx]}<br>"
                        f"action={html.escape(action)}<br>"
                        f"PCA-edge=endpoint z values are actual PC3 graph-state coordinates; NLL is color/hover metadata<br>"
                        f"parent raw NLL={float(nll_values[j]):.6f}<br>"
                        f"child raw NLL={float(nll_values[idx]):.6f}<br>"
                        f"delta child-parent={raw_delta:+.6g} ({improvement_label})"
                    ),
                    hoverinfo="text",
                )
            )
    if microstep_entries:
        fig.add_trace(
            go.Scatter3d(
                x=[float(entry["x"]) for entry in microstep_entries],
                y=[float(entry["y"]) for entry in microstep_entries],
                z=[float(entry["z"]) for entry in microstep_entries],
                mode="markers",
                marker=dict(
                    size=5,
                    color=[float(entry.get("filtration", 0.0)) for entry in microstep_entries],
                    colorscale="Tealgrn",
                    showscale=False,
                    symbol="square",
                    line=dict(width=1.0, color="#f8fafc"),
                ),
                text=[str(entry["label"]) for entry in microstep_entries],
                textposition="middle right",
                hovertext=[str(entry["hover"]) for entry in microstep_entries],
                hoverinfo="text",
                customdata=[int(entry["panel_index"]) for entry in microstep_entries],
                name="reasoning microstep",
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=pca[:, 0],
            y=pca[:, 1],
            z=trajectory_plot_z,
            mode="markers+text",
            marker=dict(
                size=[8.0 + min(12.0, 2.6 * math.log1p(float(count))) for count in pca_multiplicity],
                color=nll_values,
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(title="raw NLL", x=1.035, y=0.48, len=0.62, thickness=16),
                line=dict(width=1.5, color="#e8eef8"),
            ),
            text=_sparse_got_state_labels(candidates, ids, inferred_levels, nll_values),
            textposition="top center",
            hovertext=[
                text
                + f"<br><b>embedding/PCA multiplicity</b>: {int(pca_multiplicity[idx])} candidate(s) at this rounded coordinate"
                + f"<br><b>raw embedding unique ratio</b>: {float(pca_report.get('unique_embedding_ratio_rounded8', 1.0)):.3f}"
                + f"<br><b>step complex page</b>: {html.escape(step_page_links[idx]['step_complex_href'])}"
                + f"<br><b>step simplex tree</b>: {html.escape(step_page_links[idx]['step_simplex_tree_href'])}"
                for idx, text in enumerate(hover)
            ],
            hoverinfo="text",
            customdata=np.arange(len(candidates), dtype=int),
            name="GoT state",
            showlegend=False,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        meta={"nll_density_render_contract": main_density_render_contract},
        title="Graph-of-thought branching trajectory with Gaussian NLL density cloud",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3(graph_state embedding)",
            aspectmode="cube",
            camera=dict(eye=main_density_camera_eye),
        ),
    )
    unique_ratio = float(pca_report.get("unique_embedding_ratio_rounded8", 1.0))
    if unique_ratio < 0.8:
        fig.add_annotation(
            text=(
                "embedding-state collapse diagnostic: "
                f"{int(pca_report.get('unique_embeddings_rounded8', len(candidates)))}/{len(candidates)} unique graph_state vectors; "
                f"max multiplicity={int(pca_report.get('max_embedding_multiplicity_rounded8', 1))}<br>"
                "coordinates are actual PCA; duplicate states are retained as original model-evaluated anchors"
            ),
            x=0,
            y=1.035,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            font=dict(size=12, color="#fbbf24"),
            bgcolor="rgba(15,23,42,0.92)",
            bordercolor="rgba(251,191,36,0.45)",
            borderwidth=1,
        )
        fig.update_layout(margin=dict(t=122))
    raw_nll_range = float(np.nanmax(nll_values) - np.nanmin(nll_values)) if nll_values.size else 0.0
    if raw_nll_range < 0.01:
        fig.add_annotation(
            text=(
                "NLL field diagnostic: raw range="
                f"{raw_nll_range:.6g}; z-axis is actual PC3, not NLL height.<br>"
                "Visible layers: Gaussian density around actual GoT embeddings plus exact model-evaluated anchors; sparse NLL anchor mesh is legend-only diagnostic."
            ),
            x=0,
            y=0.985,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            font=dict(size=12, color="#bae6fd"),
            bgcolor="rgba(15,23,42,0.88)",
            bordercolor="rgba(125,211,252,0.42)",
            borderwidth=1,
        )
        current_margin = fig.layout.margin.to_plotly_json() if fig.layout.margin else {}
        fig.update_layout(margin=dict(t=max(int(current_margin.get("t", 82)), 138)))
    fig.add_annotation(
        text=(
            "GoT NLL progress: "
            f"{100.0 * float(nll_progress.get('improving_edge_fraction', 0.0)):.1f}% improving edges; "
            f"mean edge delta={float(nll_progress.get('mean_edge_delta', 0.0)):+.3g}; "
            f"best terminal improvement={float(nll_progress.get('best_terminal_improvement_from_root', 0.0)):+.3g}"
        ),
        x=1,
        y=1.035,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="right",
        font=dict(size=12, color="#d9f99d"),
        bgcolor="rgba(15,23,42,0.88)",
        bordercolor="rgba(132,204,22,0.38)",
        borderwidth=1,
    )
    current_margin = fig.layout.margin.to_plotly_json() if fig.layout.margin else {}
    fig.update_layout(margin=dict(t=max(int(current_margin.get("t", 82)), 138)))
    panel_items = _simplicial_panel_items(panel_objects, panel_hover)
    _write_plotly_dark_html(path, fig, "Graph-of-thought trajectory PCA in embedding space", panel_items, show_filtration_slider=True, show_selected_complex_panel=True)
    payload = {
        "embedding_pca_diagnostics": pca_report,
        "nodes": [
            {
                "record_id": ids[idx],
                "parent": candidates[idx].get("parent"),
                "path": candidates[idx].get("path", []),
                "embedding": candidates[idx].get("embedding"),
                "pca": {"pc1": float(pca[idx, 0]), "pc2": float(pca[idx, 1]), "pc3": float(pca[idx, 2])},
                "embedding_pca": {"pc1": float(pca[idx, 0]), "pc2": float(pca[idx, 1]), "pc3": float(pca[idx, 2])},
                "plot": {
                    "x": float(pca[idx, 0]),
                    "y": float(pca[idx, 1]),
                    "z": float(trajectory_plot_z[idx]),
                    "z_surface": None,
                    "z_centered_scaled_nll": float(nll_plot_z[idx]),
                    "raw_centered_scaled_nll": float(raw_nll_plot_z[idx]),
                    "raw_nll": float(nll_values[idx]),
                    "touches_nll_surface": False,
                },
                "reasoning_step_index": int(step_page_links[idx]["reasoning_step_index"]),
                "step_complex_href": step_page_links[idx]["step_complex_href"],
                "step_simplex_tree_href": step_page_links[idx]["step_simplex_tree_href"],
                "step_complex_contract": "this GoT state maps to a per-step interactive filtered simplicial complex page and simplex-tree page",
                "score": scores[idx],
                "nll": float(nll_values[idx]),
                "level": int(inferred_levels[idx]),
                "input_text": candidates[idx].get("input_text", ""),
                "target_text": candidates[idx].get("target_text", ""),
                "decoded_argmax": candidates[idx].get("decoded_argmax", ""),
                "graph_json_summary": candidates[idx].get("graph_json_summary", {}),
                "filtered_simplicial_object": candidates[idx].get("filtered_simplicial_object"),
                "topological_algebra": candidates[idx].get("topological_algebra"),
            }
            for idx in range(len(candidates))
        ],
        "edges": [
            {"source": row.get("parent"), "target": ids[idx], "action_path": row.get("path", []), "action": _edge_action_label(row)}
            for idx, row in enumerate(candidates)
            if row.get("parent") is not None
        ],
        "microstep_nodes": [
            {
                "candidate_record_id": str(entry.get("candidate_record_id", "")),
                "parent_record_id": str(entry.get("parent_record_id", "")),
                "simplex_label": str(entry.get("simplex_label", "")),
                "type": str(entry.get("type", "")),
                "plot": {"x": float(entry["x"]), "y": float(entry["y"]), "z_centered_scaled_nll": float(entry["z"])},
                "filtration": float(entry.get("filtration", 0.0)),
                "filtered_simplicial_object": entry.get("filtered_simplicial_object"),
            }
            for entry in microstep_entries
        ],
        "filtered_simplicial_objects": [
            candidates[idx].get("filtered_simplicial_object")
            if isinstance(candidates[idx].get("filtered_simplicial_object"), dict)
            else {}
            for idx in range(len(candidates))
        ],
        "nll_density_cloud": density_cloud_meta,
        "nll_surface": nll_surface_meta,
        "nll_progress": nll_progress,
    }
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    extra_paths = {}
    extra_paths.update(
        _write_got_embedding_map(
            scaling_report,
            output_dir,
            candidates,
            ids,
            id_to_idx,
            pca,
            nll_values,
            hover,
            inferred_levels,
            pca_report,
        )
    )
    extra_paths.update(
        _write_got_nll_density_cloud_map(
            output_dir,
            candidates,
            ids,
            id_to_idx,
            pca,
            nll_values,
            hover,
            inferred_levels,
            pca_report,
        )
    )
    extra_paths.update(_write_full_trajectory_complex_map(scaling_report, output_dir))
    extra_paths.update(_write_reasoning_step_complex_maps(candidates, output_dir))
    return {"got_trajectory_3d": str(path), "got_payloads": str(payload_path), **extra_paths}


def _write_got_embedding_map(
    scaling_report: dict[str, object],
    output_dir: Path,
    candidates: list[dict[str, object]],
    ids: list[str],
    id_to_idx: dict[str, int],
    pca: np.ndarray,
    nll_values: np.ndarray,
    hover: list[str],
    inferred_levels: np.ndarray,
    pca_report: dict[str, object],
) -> dict[str, str]:
    path = output_dir / "got_embedding_map_3d.html"
    payload_path = output_dir / "got_embedding_map_payloads.json"
    pca_multiplicity = _coordinate_multiplicities(pca)
    candidate_objects = [
        row.get("filtered_simplicial_object") if isinstance(row.get("filtered_simplicial_object"), dict) else {}
        for row in candidates
    ]
    panel_items = _simplicial_panel_items(candidate_objects, hover)
    fig = go.Figure()
    for idx, row in enumerate(candidates):
        parent = row.get("parent")
        if isinstance(parent, str) and parent in id_to_idx:
            j = id_to_idx[parent]
            action = _edge_action_label(row)
            fig.add_trace(
                go.Scatter3d(
                    x=[float(pca[j, 0]), float(pca[idx, 0])],
                    y=[float(pca[j, 1]), float(pca[idx, 1])],
                    z=[float(pca[j, 2]), float(pca[idx, 2])],
                    mode="lines",
                    line=dict(color=_action_color(action), width=4),
                    hovertext=f"{html.escape(parent)} -> {html.escape(ids[idx])}<br>action={html.escape(action)}",
                    hoverinfo="text",
                    showlegend=False,
                    name=f"edge:{action}",
                )
            )
    fig.add_trace(
        go.Scatter3d(
            x=pca[:, 0],
            y=pca[:, 1],
            z=pca[:, 2],
            mode="markers+text",
            marker=dict(
                size=8,
                color=nll_values,
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(title="NLL"),
                line=dict(width=1.2, color="#e8eef8"),
            ),
            text=[_state_plot_label(row, idx, int(inferred_levels[idx])) for idx, row in enumerate(candidates)],
            textposition="top center",
            hovertext=[
                text
                + f"<br><b>embedding PCA source</b>: model graph_state"
                + f"<br><b>PC coords</b>: ({pca[idx,0]:.4g}, {pca[idx,1]:.4g}, {pca[idx,2]:.4g})"
                + f"<br><b>coordinate multiplicity</b>: {int(pca_multiplicity[idx])}"
                for idx, text in enumerate(hover)
            ],
            hoverinfo="text",
            customdata=np.arange(len(panel_items), dtype=int),
            name="GoT state: actual graph_state PCA",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title=(
            "Graph-of-thought embedding-space trajectory map "
            f"(actual graph_state PCA; distance corr={float(pca_report.get('pairwise_distance_correlation', 0.0)):.3f}, "
            f"stress={float(pca_report.get('normalized_stress', 0.0)):.3f})"
        ),
        scene=dict(xaxis_title="PC1(graph_state)", yaxis_title="PC2(graph_state)", zaxis_title="PC3(graph_state)"),
        annotations=[
            dict(
                text=html.escape(str(pca_report)),
                x=0,
                y=-0.12,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
                font=dict(size=10, color="#9fb3c8"),
            )
        ],
    )
    _write_plotly_dark_html(path, fig, "Graph-of-thought embedding-space trajectory map", panel_items, show_filtration_slider=True)
    payload = {
        "coordinate_source": "PCA of model graph_state embeddings; no level/tree layout coordinates are used",
        "sampling": {
            "stochastic_actions": bool(scaling_report.get("stochastic_actions", False)),
            "temperature": scaling_report.get("sampling_temperature"),
            "exploration": scaling_report.get("sampling_exploration"),
            "seed": scaling_report.get("sampling_seed"),
        },
        "embedding_pca_diagnostics": pca_report,
        "nodes": [
            {
                "record_id": ids[idx],
                "parent": candidates[idx].get("parent"),
                "path": candidates[idx].get("path", []),
                "level": int(inferred_levels[idx]),
                "nll": float(nll_values[idx]),
                "embedding_pca": {"pc1": float(pca[idx, 0]), "pc2": float(pca[idx, 1]), "pc3": float(pca[idx, 2])},
                "embedding": candidates[idx].get("embedding"),
            }
            for idx in range(len(candidates))
        ],
        "filtered_simplicial_objects": candidate_objects,
        "edges": [
            {"source": row.get("parent"), "target": ids[idx], "action": _edge_action_label(row)}
            for idx, row in enumerate(candidates)
            if row.get("parent") is not None
        ],
    }
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"got_embedding_map_3d": str(path), "got_embedding_map_payloads": str(payload_path)}


def _stable_density_seed(pca: np.ndarray, nll_values: np.ndarray) -> int:
    rounded = np.round(np.asarray(pca, dtype=float), 6)
    rounded_nll = np.round(np.asarray(nll_values, dtype=float), 6)
    digest = hashlib.sha256(rounded.tobytes() + rounded_nll.tobytes()).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32 - 1)


def _gaussian_nll_density_cloud(
    pca: np.ndarray,
    nll_values: np.ndarray,
    *,
    max_samples: int = 2600,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    points = np.asarray(pca, dtype=float)
    nll = np.asarray(nll_values, dtype=float).reshape(-1)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(nll)
    points = points[finite]
    nll = nll[finite]
    if points.shape[0] == 0:
        return None, {"available": False, "reason": "no finite PCA/NLL anchors"}
    if points.shape[0] == 1:
        sigma = 0.08
    else:
        diffs = points[:, None, :] - points[None, :, :]
        distances = np.linalg.norm(diffs, axis=-1)
        positive = distances[distances > 1e-12]
        span = float(np.max(np.ptp(points, axis=0))) if points.size else 1.0
        sigma = float(max(np.quantile(positive, 0.25) * 0.38 if positive.size else 0.0, span * 0.035, 1e-5))
    sample_count = int(min(max_samples, max(360, 110 * points.shape[0])))
    rng = np.random.default_rng(_stable_density_seed(points, nll))
    anchor_idx = rng.integers(0, points.shape[0], size=sample_count)
    noise = rng.normal(loc=0.0, scale=sigma, size=(sample_count, 3))
    cloud = points[anchor_idx] + noise
    d2 = np.sum((cloud[:, None, :] - points[None, :, :]) ** 2, axis=-1)
    weights = np.exp(-0.5 * d2 / max(sigma * sigma, 1e-12))
    denom = np.sum(weights, axis=1)
    denom = np.where(denom <= 1e-12, 1.0, denom)
    local_nll = (weights @ nll) / denom
    density = denom / float(points.shape[0])
    nearest = np.argmin(d2, axis=1)
    nearest_distance = np.sqrt(np.min(d2, axis=1))
    return {
        "points": cloud,
        "local_nll": local_nll,
        "density": density,
        "nearest": nearest.astype(int),
        "nearest_distance": nearest_distance,
        "anchor_idx": anchor_idx.astype(int),
    }, {
        "available": True,
        "source": "actual model-evaluated graph_state PCA anchors and measured raw NLL values",
        "support_samples_are_not_model_states": True,
        "exact_anchor_layer": True,
        "exact_anchor_scope": "large labeled anchors are observed model-evaluated GoT states only",
        "sample_count": sample_count,
        "anchor_count": int(points.shape[0]),
        "sigma": sigma,
        "kernel": "isotropic Gaussian in 3D PCA coordinates",
        "local_nll_rule": "kernel-weighted mean of measured NLL at actual GoT states",
        "render_contract": "Gaussian cloud points are not model states; they visualize local NLL density around actual embedding vectors, while only large labeled markers are model states",
        "nll_min": float(np.min(nll)),
        "nll_max": float(np.max(nll)),
    }


def _nll_blob_color(value: float, vmin: float, vmax: float) -> str:
    """Small Plasma-like ramp without adding a dependency to renderer hot paths."""
    if not math.isfinite(value):
        return "rgba(148,163,184,0.20)"
    t = 0.5 if vmax <= vmin else (float(value) - float(vmin)) / max(float(vmax) - float(vmin), 1e-12)
    t = max(0.0, min(1.0, t))
    stops = [
        (0.00, (49, 16, 129)),
        (0.28, (126, 34, 206)),
        (0.52, (236, 72, 153)),
        (0.75, (245, 158, 11)),
        (1.00, (254, 240, 138)),
    ]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            rgb = tuple(int(round(c0[i] + u * (c1[i] - c0[i]))) for i in range(3))
            return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.34)"
    r, g, b = stops[-1][1]
    return f"rgba({r},{g},{b},0.34)"


def _add_gaussian_nll_blob_meshes(fig: go.Figure, points: np.ndarray, nll: np.ndarray, sigma: float) -> int:
    """Add translucent ellipsoid blobs centered at actual model states only."""
    points = np.asarray(points, dtype=float)
    nll = np.asarray(nll, dtype=float).reshape(-1)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(nll)
    points = points[finite]
    nll = nll[finite]
    if points.shape[0] == 0:
        return 0
    # Ellipsoid radii come from the displayed PCA spread and the same Gaussian
    # bandwidth used for the density computation.  These are visual kernels around
    # real states, not additional states or model samples.
    spread = np.maximum(np.std(points, axis=0), 1e-5)
    radius = np.maximum(spread * 0.09, float(sigma) * 1.15)
    u = np.linspace(0, 2 * np.pi, 18)
    v = np.linspace(0, np.pi, 10)
    uu, vv = np.meshgrid(u, v)
    unit = np.column_stack([
        np.cos(uu).ravel() * np.sin(vv).ravel(),
        np.sin(uu).ravel() * np.sin(vv).ravel(),
        np.cos(vv).ravel(),
    ])
    faces_i: list[int] = []
    faces_j: list[int] = []
    faces_k: list[int] = []
    nu = len(u)
    nv = len(v)
    for a in range(nv - 1):
        for b in range(nu - 1):
            p00 = a * nu + b
            p01 = a * nu + b + 1
            p10 = (a + 1) * nu + b
            p11 = (a + 1) * nu + b + 1
            faces_i.extend([p00, p01])
            faces_j.extend([p10, p10])
            faces_k.extend([p11, p11])
    vmin = float(np.min(nll))
    vmax = float(np.max(nll))
    added = 0
    for idx, center in enumerate(points):
        coords = center[None, :] + unit * radius[None, :]
        fig.add_trace(
            go.Mesh3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                i=faces_i,
                j=faces_j,
                k=faces_k,
                color=_nll_blob_color(float(nll[idx]), vmin, vmax),
                opacity=0.30,
                name="NLL Gaussian neighborhood" if idx == 0 else "NLL Gaussian neighborhood",
                showlegend=(idx == 0),
                hovertemplate=(
                    "NLL Gaussian neighborhood around actual state<br>"
                    f"state index={idx}<br>"
                    f"raw NLL={float(nll[idx]):.6f}<br>"
                    "kernel is visual support only; center is the model state<extra></extra>"
                ),
            )
        )
        added += 1
    return added


def _gaussian_density_volume_trace(
    anchor_points: np.ndarray,
    nll_values: np.ndarray,
    sigma: float,
    *,
    name: str,
    grid_size: int = 28,
) -> tuple[go.Volume | None, dict[str, object]]:
    points = np.asarray(anchor_points, dtype=float)
    nll = np.asarray(nll_values, dtype=float).reshape(-1)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(nll)
    points = points[finite]
    nll = nll[finite]
    if points.shape[0] == 0 or not math.isfinite(float(sigma)) or float(sigma) <= 0:
        return None, {"available": False, "reason": "no finite anchors or invalid sigma"}
    if points.shape[0] < 2:
        return None, {"available": False, "reason": "at least two anchors required for a meaningful density volume"}
    span = np.ptp(points, axis=0)
    pad = np.maximum(span * 0.12, float(sigma) * 2.4)
    mins = np.min(points, axis=0) - pad
    maxs = np.max(points, axis=0) + pad
    axes = [np.linspace(float(mins[d]), float(maxs[d]), int(grid_size)) for d in range(3)]
    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    grid = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    d2 = np.sum((grid[:, None, :] - points[None, :, :]) ** 2, axis=-1)
    weights = np.exp(-0.5 * d2 / max(float(sigma) * float(sigma), 1e-12))
    density = np.sum(weights, axis=1) / float(points.shape[0])
    density_max = float(np.max(density)) if density.size else 0.0
    if density_max <= 1e-12:
        return None, {"available": False, "reason": "zero density field"}
    density_norm = density / density_max
    local_nll = (weights @ nll) / np.maximum(np.sum(weights, axis=1), 1e-12)
    positive = density_norm[density_norm > 1e-8]
    # Render the visible volume only inside a genuine Gaussian support around the
    # actual model anchors.  The scalar shown by the volume is kernel-smoothed NLL,
    # so the object reads as an NLL density/fitness cloud rather than a triangulated
    # simplex sheet or a scatter of fake states.
    density_floor = float(np.quantile(positive, 0.24)) if positive.size else 0.04
    density_floor = max(min(density_floor, 0.26), 0.012)
    trace = go.Volume(
        x=grid[:, 0],
        y=grid[:, 1],
        z=grid[:, 2],
        value=density_norm,
        isomin=density_floor,
        isomax=1.0,
        opacity=0.34,
        surface_count=26,
        colorscale=[[0.0, "#0f172a"], [0.30, "#1d4ed8"], [0.62, "#22d3ee"], [1.0, "#facc15"]],
        showscale=False,
        customdata=local_nll,
        opacityscale=[[0.0, 0.0], [density_floor, 0.045], [0.35, 0.14], [0.70, 0.30], [1.0, 0.46]],
        hovertemplate=(
            "Gaussian support density around actual anchors<br>"
            "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>"
            "normalized Gaussian support=%{value:.4f}<br>"
            "kernel local raw NLL=%{customdata:.6f}<br>"
            "continuous visualization field; not a model state<extra></extra>"
        ),
        name=name,
        showlegend=True,
    )
    return trace, {
        "available": True,
        "grid_size": int(grid_size),
        "grid_points": int(grid.shape[0]),
        "sigma": float(sigma),
        "density_floor": density_floor,
        "value": "normalized Gaussian density support around actual anchors",
        "color": "density shell color; raw NLL is shown by translucent anchor blobs and hover",
        "support_samples_are_not_model_states": True,
    }


def _nll_density_render_contract(
    *,
    anchor_count: int,
    support_sample_count: int,
    sigma: float,
    page: str,
    camera_eye: Mapping[str, float],
    anchor_trace_name: str,
    support_sample_trace_name: str,
) -> dict[str, object]:
    return {
        "schema_version": "tropicalgt.nll_density_render.v1",
        "page": str(page),
        "coordinate_space": "actual 3D PCA coordinates of model graph-state embeddings",
        "z_axis_policy": "z is PC3(graph_state embedding); raw NLL is encoded by color, hover, and density metadata",
        "actual_model_anchor_count": int(anchor_count),
        "support_sample_count": int(support_sample_count),
        "kernel": "isotropic Gaussian in 3D PCA coordinates",
        "kernel_bandwidth": float(sigma),
        "actual_anchor_trace_name": str(anchor_trace_name),
        "actual_anchor_layer_visible_by_default": True,
        "support_sample_trace_name": str(support_sample_trace_name),
        "support_sample_trace_visibility": "legendonly",
        "support_samples_are_model_states": False,
        "support_samples_hidden_as_model_states": True,
        "visible_density_layers": ["density_volume", "anchor_gaussian_neighborhoods", "actual_model_anchor_markers"],
        "audit_sample_layer_role": "legend-only Gaussian support samples for local NLL density inspection",
        "camera_eye": {key: float(value) for key, value in dict(camera_eye).items()},
        "no_proxy_or_fallback": True,
    }


def _write_got_nll_density_cloud_map(
    output_dir: Path,
    candidates: list[dict[str, object]],
    ids: list[str],
    id_to_idx: dict[str, int],
    pca: np.ndarray,
    nll_values: np.ndarray,
    hover: list[str],
    inferred_levels: np.ndarray,
    pca_report: dict[str, object],
) -> dict[str, str]:
    path = output_dir / "got_nll_density_cloud_pca_3d.html"
    payload_path = output_dir / "got_nll_density_cloud_payload.json"
    cloud, cloud_meta = _gaussian_nll_density_cloud(pca, nll_values)
    if cloud is None:
        _write_dark_empty(path, "No finite model GoT embeddings/NLL anchors available for the 3D PCA NLL density cloud.")
        payload_path.write_text(json.dumps({"available": False, "density_cloud": cloud_meta}, indent=2), encoding="utf-8")
        return {"got_nll_density_cloud_pca_3d": str(path), "got_nll_density_cloud_payload": str(payload_path)}

    fig = go.Figure()
    volume_trace, volume_meta = _gaussian_density_volume_trace(
        pca,
        nll_values,
        float(cloud_meta.get("sigma", 0.0) or 0.0),
        name="translucent 3D Gaussian density field",
    )
    cloud_meta["density_volume"] = volume_meta
    if volume_trace is not None:
        fig.add_trace(volume_trace)
    cloud_meta["nll_gaussian_blob_count"] = _add_gaussian_nll_blob_meshes(
        fig, pca, nll_values, float(cloud_meta.get("sigma", 0.0) or 0.0)
    )
    cloud_points = cloud["points"]
    nearest = cloud["nearest"]
    fig.add_trace(
        go.Scatter3d(
            x=cloud_points[:, 0],
            y=cloud_points[:, 1],
            z=cloud_points[:, 2],
            mode="markers",
            marker=dict(
                size=1.55,
                color=cloud["local_nll"],
                colorscale="Plasma",
                opacity=0.10,
                showscale=False,
            ),
            customdata=np.column_stack([cloud["local_nll"], cloud["density"], cloud["nearest_distance"], nearest]),
            hovertemplate=(
                "NLL density cloud sample<br>"
                "PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<br>"
                "kernel local raw NLL=%{customdata[0]:.6f}<br>"
                "Gaussian density mass=%{customdata[1]:.4g}<br>"
                "nearest actual state distance=%{customdata[2]:.4g}<br>"
                "nearest actual state index=%{customdata[3]:.0f}<br>"
                "not a model state; density around actual embeddings<extra></extra>"
            ),
            name="audit samples from the Gaussian NLL field (hidden by default)",
            visible="legendonly",
        )
    )
    for idx, row in enumerate(candidates):
        parent = row.get("parent")
        if isinstance(parent, str) and parent in id_to_idx:
            j = id_to_idx[parent]
            action = _edge_action_label(row)
            raw_delta = float(nll_values[idx] - nll_values[j])
            fig.add_trace(
                go.Scatter3d(
                    x=[float(pca[j, 0]), float(pca[idx, 0])],
                    y=[float(pca[j, 1]), float(pca[idx, 1])],
                    z=[float(pca[j, 2]), float(pca[idx, 2])],
                    mode="lines",
                    line=dict(color=_action_color(action), width=5),
                    hovertext=(
                        f"{html.escape(parent)} -> {html.escape(ids[idx])}<br>"
                        f"action={html.escape(action)}<br>"
                        f"parent raw NLL={float(nll_values[j]):.6f}<br>"
                        f"child raw NLL={float(nll_values[idx]):.6f}<br>"
                        f"delta child-parent={raw_delta:+.6g}"
                    ),
                    hoverinfo="text",
                    showlegend=False,
                    name=f"GoT edge:{action}",
                )
            )
    fig.add_trace(
        go.Scatter3d(
            x=pca[:, 0],
            y=pca[:, 1],
            z=pca[:, 2],
            mode="markers+text",
            marker=dict(
                size=10,
                color=nll_values,
                colorscale="Plasma",
                showscale=False,
                line=dict(width=1.4, color="#f8fafc"),
            ),
            text=[_state_plot_label(row, idx, int(inferred_levels[idx])) for idx, row in enumerate(candidates)],
            textposition="top center",
            hovertext=[
                text
                + f"<br><b>density-cloud role</b>: actual model GoT state anchor"
                + f"<br><b>raw NLL</b>: {float(nll_values[idx]):.6f}"
                + f"<br><b>PC coords</b>: ({pca[idx,0]:.4g}, {pca[idx,1]:.4g}, {pca[idx,2]:.4g})"
                for idx, text in enumerate(hover)
            ],
            hoverinfo="text",
            customdata=np.arange(len(candidates), dtype=int),
            name="actual model GoT state anchors",
        )
    )
    density_camera_eye = dict(x=1.55, y=-1.65, z=1.08)
    visual_layer_contract = _nll_density_render_contract(
        anchor_count=len(candidates),
        support_sample_count=int(cloud_meta.get("sample_count", int(cloud_points.shape[0])) or int(cloud_points.shape[0])),
        sigma=float(cloud_meta.get("sigma", 0.0) or 0.0),
        page="standalone_density_cloud",
        camera_eye=density_camera_eye,
        anchor_trace_name="actual model GoT state anchors",
        support_sample_trace_name="audit samples from the Gaussian NLL field (hidden by default)",
    )
    cloud_meta["visual_layer_contract"] = visual_layer_contract
    fig.update_layout(
        template="plotly_dark",
        meta={"nll_density_render_contract": visual_layer_contract},
        title=dict(
            text=(
                "3D PCA NLL density around actual GoT embeddings"
                "<br><sup>Gaussian neighborhoods are rendered around actual model states only; audit samples stay hidden.</sup>"
            ),
            x=0.02,
            y=0.975,
            xanchor="left",
            yanchor="top",
        ),
        scene=dict(
            xaxis_title="PC1(graph_state)",
            yaxis_title="PC2(graph_state)",
            zaxis_title="PC3(graph_state)",
            aspectmode="cube",
            camera=dict(eye=density_camera_eye),
        ),
        margin=dict(t=178, r=150, b=42, l=42),
        legend=dict(orientation="v", x=0.015, y=0.91, xanchor="left", yanchor="top", bgcolor="rgba(2,6,23,0.76)", bordercolor="rgba(125,211,252,0.25)", borderwidth=1, font=dict(size=11)),
    )
    fig.add_annotation(
        text=(
            "actual PCA distance corr="
            f"{float(pca_report.get('pairwise_distance_correlation', 0.0)):.3f}; "
            f"stress={float(pca_report.get('normalized_stress', 0.0)):.3f}; "
            f"density sigma={float(cloud_meta.get('sigma', 0.0)):.4g}"
        ),
        x=0,
        y=1.01,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        font=dict(size=12, color="#bae6fd"),
        bgcolor="rgba(15,23,42,0.88)",
        bordercolor="rgba(125,211,252,0.42)",
        borderwidth=1,
    )
    _write_plotly_dark_html(path, fig, "3D PCA NLL density cloud around actual GoT embeddings")
    node_payload = [
        {
            "record_id": ids[idx],
            "parent": candidates[idx].get("parent"),
            "level": int(inferred_levels[idx]),
            "path": candidates[idx].get("path", []),
            "pca": {"pc1": float(pca[idx, 0]), "pc2": float(pca[idx, 1]), "pc3": float(pca[idx, 2])},
            "nll": float(nll_values[idx]),
            "density_cloud_role": "actual_model_evaluated_graph_state_anchor",
        }
        for idx in range(len(candidates))
    ]
    edge_payload = [
        {
            "source": row.get("parent"),
            "target": ids[idx],
            "action": _edge_action_label(row),
            "source_nll": float(nll_values[id_to_idx[row.get("parent")]]),
            "target_nll": float(nll_values[idx]),
            "nll_delta": float(nll_values[idx] - nll_values[id_to_idx[row.get("parent")]]),
            "improves": bool(float(nll_values[idx] - nll_values[id_to_idx[row.get("parent")]]) < 0.0),
        }
        for idx, row in enumerate(candidates)
        if isinstance(row.get("parent"), str) and row.get("parent") in id_to_idx
    ]
    nll_progress = _trajectory_nll_progress_diagnostics(candidates, ids, id_to_idx, nll_values, inferred_levels)
    support_sample_count = int(cloud_meta.get("sample_count", int(cloud_points.shape[0])) or int(cloud_points.shape[0]))
    anchor_count = int(cloud_meta.get("anchor_count", len(node_payload)) or len(node_payload))
    nll_min = float(np.nanmin(nll_values)) if len(nll_values) else None
    nll_max = float(np.nanmax(nll_values)) if len(nll_values) else None
    pca_diagnostics = {
        key: float(value)
        for key, value in pca_report.items()
        if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))
    }
    density_contract = {
        "actual_model_anchor_layer": True,
        "actual_model_anchor_layer_description": "large labeled markers are actual model-evaluated GoT states in graph_state PCA coordinates",
        "support_sample_layer": "legendonly Gaussian density samples around actual anchors",
        "support_samples_hidden_as_model_states": True,
        "sample_points_are_model_states": False,
        "support_sample_count": support_sample_count,
        "actual_model_anchor_count": anchor_count,
        "kernel": str(cloud_meta.get("kernel", "isotropic Gaussian in 3D PCA coordinates")),
        "kernel_bandwidth": float(cloud_meta.get("sigma", 0.0) or 0.0),
        "local_nll_rule": str(cloud_meta.get("local_nll_rule", "kernel-weighted mean of measured NLL at actual GoT states")),
        "edge_delta_rule": "target raw NLL minus source raw NLL over actual GoT tree edges",
        "support_sample_trace_visibility": "legendonly",
        "visible_density_layers": visual_layer_contract["visible_density_layers"],
        "z_axis_policy": visual_layer_contract["z_axis_policy"],
    }
    support_samples = {
        "available": True,
        "count": support_sample_count,
        "visible_by_default": False,
        "visible_as_model_states": False,
        "rendered_trace_visibility": "legendonly",
        "role": "continuous Gaussian support field around actual graph-state anchors",
        "local_nll_summary": _numeric_summary([float(value) for value in cloud["local_nll"]]),
        "density_weight_summary": _numeric_summary([float(value) for value in cloud["density"]]),
        "nearest_anchor_distance_summary": _numeric_summary([float(value) for value in cloud["nearest_distance"]]),
    }
    payload = {
        "available": True,
        "render_contract": str(cloud_meta.get("render_contract", "")),
        "density_contract": density_contract,
        "anchor_count": anchor_count,
        "actual_model_anchor_count": anchor_count,
        "support_sample_count": support_sample_count,
        "support_samples_hidden_as_model_states": True,
        "sample_points_are_model_states": False,
        "kernel_bandwidth": float(cloud_meta.get("sigma", 0.0) or 0.0),
        "nll_range": {
            "min": nll_min,
            "max": nll_max,
            "span": float(nll_max - nll_min) if nll_min is not None and nll_max is not None else None,
        },
        "local_nll_summary": support_samples["local_nll_summary"],
        "density_weight_summary": support_samples["density_weight_summary"],
        "nearest_anchor_distance_summary": support_samples["nearest_anchor_distance_summary"],
        "edge_nll_delta_summary": _numeric_summary([float(edge["nll_delta"]) for edge in edge_payload]),
        "terminal_nll_progress": {
            "root_mean_nll": nll_progress.get("root_mean_nll"),
            "terminal_count": nll_progress.get("terminal_count"),
            "terminal_mean_nll": nll_progress.get("terminal_mean_nll"),
            "terminal_min_nll": nll_progress.get("terminal_min_nll"),
            "terminal_mean_improvement_from_root": nll_progress.get("terminal_mean_improvement_from_root"),
            "best_terminal_improvement_from_root": nll_progress.get("best_terminal_improvement_from_root"),
            "improving_edge_fraction": nll_progress.get("improving_edge_fraction"),
        },
        "pca_diagnostics": pca_diagnostics,
        "density_volume": cloud_meta.get("density_volume", {}),
        "visual_layer_contract": visual_layer_contract,
        "density_cloud": cloud_meta,
        "anchors": node_payload,
        "support_samples": support_samples,
        "nodes": node_payload,
        "edges": edge_payload,
    }
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"got_nll_density_cloud_pca_3d": str(path), "got_nll_density_cloud_payload": str(payload_path)}


def _trajectory_complex_overlay_view_contract(
    obj: object,
    *,
    view: str,
    source: str,
    expected_metric: str,
    unavailable_reason: str | None = None,
) -> dict[str, object]:
    if not isinstance(obj, Mapping) or obj.get("available") is False:
        return {
            "schema_version": "tropicalgt.trajectory_complex_overlay_view_contract.v1",
            "view": view,
            "available": False,
            "reason": unavailable_reason or (str(obj.get("reason")) if isinstance(obj, Mapping) else "complex_unavailable"),
            "source": source,
            "expected_distance_metric": expected_metric,
            "actual_data_only": True,
            "no_proxy_or_fallback": True,
            "safe_to_render_overlay_semantics": False,
        }
    simplices = [row for row in obj.get("simplices", []) if isinstance(row, Mapping)]
    vertices = [row for row in simplices if _simplex_dimension(row) == 0]
    edges = [row for row in simplices if _simplex_dimension(row) == 1]
    faces = [row for row in simplices if _simplex_dimension(row) == 2]
    summary = obj.get("summary", {}) if isinstance(obj.get("summary"), Mapping) else {}
    tree = obj.get("simplex_tree", {}) if isinstance(obj.get("simplex_tree"), Mapping) else {}
    trajectory_overlay = obj.get("trajectory_overlay", {}) if isinstance(obj.get("trajectory_overlay"), Mapping) else {}
    decoding_overlay = obj.get("decoding_causal_overlay", {}) if isinstance(obj.get("decoding_causal_overlay"), Mapping) else {}
    trajectory_edges = trajectory_overlay.get("edges", []) if isinstance(trajectory_overlay.get("edges"), list) else []
    decoding_edges = decoding_overlay.get("edges", []) if isinstance(decoding_overlay.get("edges"), list) else []
    contract = {
        "schema_version": "tropicalgt.trajectory_complex_overlay_view_contract.v1",
        "view": view,
        "available": True,
        "source": source,
        "actual_data_only": True,
        "no_proxy_or_fallback": True,
        "distance_metric": str(summary.get("embedding_metric") or trajectory_overlay.get("distance_metric") or expected_metric),
        "expected_distance_metric": expected_metric,
        "filtration_model": summary.get("filtration_model"),
        "radius_filtration": summary.get("radius_filtration") is True,
        "solid_lines_semantics": "radius-filtered 1-simplices only",
        "filled_faces_semantics": "radius-gated 2-simplices only",
        "dotted_lines_semantics": "trajectory and decoding/order overlays only",
        "solid_edges_from_radius_simplices": True,
        "filled_faces_from_radius_simplices": True,
        "dotted_edges_reserved_for_overlays": True,
        "vertex_count": int(len(vertices)),
        "radius_edge_count": int(len(edges)),
        "radius_face_count": int(len(faces)),
        "summary_vertex_count": int(summary.get("num_vertices", len(vertices)) or 0),
        "summary_edge_count": int(summary.get("num_edges", len(edges)) or 0),
        "summary_two_simplex_count": int(summary.get("num_two_simplices", len(faces)) or 0),
        "simplex_tree_backend": tree.get("backend", "missing"),
        "simplex_tree_available": tree.get("available") is not False,
        "trajectory_overlay_source": trajectory_overlay.get("source"),
        "trajectory_overlay_distance_metric": trajectory_overlay.get("distance_metric"),
        "trajectory_overlay_edge_count": int(len(trajectory_edges)),
        "decoding_overlay_source": decoding_overlay.get("source"),
        "decoding_overlay_distance_metric": decoding_overlay.get("distance_metric"),
        "decoding_overlay_edge_count": int(len(decoding_edges)),
        "decoding_overlay_edges_are_dotted": bool(decoding_edges) and all(row.get("style") == "dotted" for row in decoding_edges if isinstance(row, Mapping)),
        "decoding_overlay_edges_are_directed": bool(decoding_edges) and all(row.get("directed") is True for row in decoding_edges if isinstance(row, Mapping)),
        "trajectory_overlay_does_not_change_radius_filtration": True,
    }
    contract["source_counts_match_summary"] = (
        contract["vertex_count"] == contract["summary_vertex_count"]
        and contract["radius_edge_count"] == contract["summary_edge_count"]
        and contract["radius_face_count"] == contract["summary_two_simplex_count"]
    )
    contract["safe_to_render_overlay_semantics"] = all(
        [
            contract["available"],
            contract["actual_data_only"],
            contract["no_proxy_or_fallback"],
            contract["radius_filtration"],
            contract["distance_metric"] == expected_metric,
            contract["vertex_count"] > 0,
            contract["source_counts_match_summary"],
            contract["simplex_tree_backend"] == "gudhi.SimplexTree",
            contract["trajectory_overlay_source"] == "graph_of_thought_parent_edges",
            contract["trajectory_overlay_distance_metric"] == expected_metric,
            contract["decoding_overlay_source"] == "graph_of_thought_parent_decoding_order",
            contract["decoding_overlay_distance_metric"] == expected_metric,
            contract["decoding_overlay_edges_are_dotted"] if contract["decoding_overlay_edge_count"] else True,
            contract["decoding_overlay_edges_are_directed"] if contract["decoding_overlay_edge_count"] else True,
            contract["dotted_edges_reserved_for_overlays"],
            contract["solid_edges_from_radius_simplices"],
        ]
    )
    return contract


def _trajectory_complex_overlay_contract(full_obj: object, probability_obj: object) -> dict[str, object]:
    full_contract = _trajectory_complex_overlay_view_contract(
        full_obj,
        view="embedding_radius_trajectory_complex",
        source="trajectory_filtered_simplicial_object",
        expected_metric="euclidean",
    )
    probability_available = isinstance(probability_obj, Mapping) and probability_obj.get("available") is not False
    probability_contract = _trajectory_complex_overlay_view_contract(
        probability_obj,
        view="probability_jensen_shannon_trajectory_complex",
        source="trajectory_probability_filtered_simplicial_object",
        expected_metric="jensen_shannon",
        unavailable_reason=None if probability_available else "missing_model_probability_vectors",
    )
    contract = {
        "schema_version": "tropicalgt.trajectory_complex_overlay_contract.v1",
        "actual_data_only": True,
        "no_proxy_or_fallback": True,
        "solid_lines_reserved_for_radius_simplices": True,
        "filled_faces_reserved_for_radius_simplices": True,
        "dotted_lines_reserved_for_trajectory_decoding_order_overlays": True,
        "embedding_view": full_contract,
        "probability_view": probability_contract,
        "probability_view_available": bool(probability_available),
        "safe_to_render_embedding_view": full_contract.get("safe_to_render_overlay_semantics") is True,
        "safe_to_render_probability_view": probability_contract.get("safe_to_render_overlay_semantics") is True if probability_available else False,
        "render_contract": "Full trajectory complex pages reserve solid lines/faces for radius-filtered simplices and dotted directed lines for GoT trajectory/decoding-order overlays. The Jensen-Shannon page is rendered only from model candidate probability vectors; unavailable probability views are explicit and are not substituted by embedding or static probability proxies.",
    }
    contract["safe_to_render_available_views"] = contract["safe_to_render_embedding_view"] and (
        contract["safe_to_render_probability_view"] if probability_available else True
    )
    return contract


def _write_full_trajectory_complex_map(scaling_report: dict[str, object], output_dir: Path) -> dict[str, str]:
    candidates = [row for row in scaling_report.get("candidates", []) if isinstance(row, dict)]
    obj = scaling_report.get("trajectory_filtered_simplicial_object")
    if candidates and (not isinstance(obj, dict) or not _complex_has_model_io(obj)):
        obj = build_reasoning_trajectory_complex(candidates)
    if not isinstance(obj, dict):
        return {}
    obj = _gudhi_canonical_complex(obj)
    obj = _attach_trajectory_overlay(obj, candidates)
    obj = _attach_trajectory_decoding_order_overlay(obj, candidates, distance_metric="euclidean")
    probability_obj = scaling_report.get("trajectory_probability_filtered_simplicial_object")
    if candidates and not isinstance(probability_obj, dict):
        probability_obj = build_reasoning_trajectory_complex(candidates, metric="jensen_shannon")
    probability_available = _has_real_probability_filtration(probability_obj)
    probability_obj = _gudhi_canonical_complex(probability_obj) if probability_available and isinstance(probability_obj, dict) else None
    if isinstance(probability_obj, dict):
        probability_obj = _attach_trajectory_overlay(probability_obj, candidates, distance_metric="jensen_shannon")
        probability_obj = _attach_trajectory_decoding_order_overlay(probability_obj, candidates, distance_metric="jensen_shannon")
    path = output_dir / "got_full_trajectory_complex.html"
    tree_path = output_dir / "got_full_trajectory_simplex_tree_3d.html"
    probability_path = output_dir / "got_full_trajectory_complex_jensen_shannon.html"
    probability_tree_path = output_dir / "got_full_trajectory_simplex_tree_3d_jensen_shannon.html"
    payload_path = output_dir / "got_full_trajectory_complex_payload.json"
    title = "Full graph-of-thought trajectory filtered simplicial complex"
    _write_complex_slider_map(
        path,
        obj,
        title=title,
        subtitle=(
            "Full trajectory complex; slider filters solid radius/simplicial edges induced from the same embeddings "
            "while the GoT overlay shows parent-child trajectory edges."
        ),
    )
    _write_simplex_tree_3d_map(
        tree_path,
        obj,
        title="Full graph-of-thought trajectory GUDHI SimplexTree face-coface poset",
        subtitle="3D face-coface poset view computed from the canonical GUDHI SimplexTree; hover reveals simplex, filtration, and source metadata.",
    )
    result = {
        "got_full_trajectory_complex": str(path),
        "got_full_trajectory_simplex_tree_3d": str(tree_path),
        "got_full_trajectory_complex_payload": str(payload_path),
    }
    payload = {"filtered_simplicial_object": obj}
    if isinstance(probability_obj, dict):
        _write_complex_slider_map(
            probability_path,
            probability_obj,
            title="Full graph-of-thought trajectory probability filtered simplicial complex",
            subtitle="Vietoris-Rips complex using Jensen-Shannon distance on model candidate probability vectors.",
        )
        _write_simplex_tree_3d_map(
            probability_tree_path,
            probability_obj,
            title="Full graph-of-thought trajectory Jensen-Shannon probability SimplexTree face-coface poset",
            subtitle="Face-coface poset view of the canonical SimplexTree for the Jensen-Shannon probability filtration.",
        )
        payload["probability_filtered_simplicial_object"] = probability_obj
    else:
        unavailable = _unavailable_complex("missing_model_probability_vectors")
        message = (
            "Full graph-of-thought trajectory probability filtered simplicial complex unavailable: "
            "model candidate probability vectors were not present, so no Jensen-Shannon radius complex or simplex tree was rendered."
        )
        _write_dark_empty(probability_path, message)
        _write_simplex_tree_poset_contract(
            probability_tree_path,
            _unavailable_simplex_tree_poset_contract(
                probability_tree_path,
                title="Full graph-of-thought trajectory Jensen-Shannon probability SimplexTree face-coface poset",
                reason="missing_model_probability_vectors",
            ),
        )
        _write_dark_empty(probability_tree_path, message)
        payload["probability_filtered_simplicial_object"] = unavailable
    result["got_full_trajectory_complex_jensen_shannon"] = str(probability_path)
    result["got_full_trajectory_simplex_tree_3d_jensen_shannon"] = str(probability_tree_path)
    payload["trajectory_complex_overlay_contract"] = _trajectory_complex_overlay_contract(
        obj,
        payload.get("probability_filtered_simplicial_object"),
    )
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return result


def _attach_trajectory_overlay(
    obj: dict[str, object],
    candidates: list[dict[str, object]],
    *,
    distance_metric: str = "euclidean",
) -> dict[str, object]:
    ids = [str(row.get("record_id", idx)) for idx, row in enumerate(candidates)]
    id_set = set(ids)
    row_by_id = {rid: row for rid, row in zip(ids, candidates)}
    edges = []
    for target_id, row in row_by_id.items():
        parent = row.get("parent")
        if not isinstance(parent, str) or parent not in id_set:
            continue
        parent_row = row_by_id.get(parent, {})
        edges.append(
            {
                "source": parent,
                "target": target_id,
                "action": _edge_action_label(row),
                "source_level": int(parent_row.get("level", 0) or 0),
                "target_level": int(row.get("level", 0) or 0),
                "source_nll": _safe_float(parent_row.get("nll")),
                "target_nll": _safe_float(row.get("nll")),
                "source_path": parent_row.get("path", []),
                "target_path": row.get("path", []),
            }
        )
    updated = dict(obj)
    updated["trajectory_overlay"] = {
        "source": "graph_of_thought_parent_edges",
        "semantic_note": (
            "Radius topology is induced from these trajectory embeddings; overlay highlights the observed "
            "GoT parent-child path over the same vertices without changing the radius/simplex filtration."
        ),
        "distance_metric": distance_metric,
        "edge_count": len(edges),
        "edges": edges,
    }
    return updated


def _attach_trajectory_decoding_order_overlay(
    obj: dict[str, object],
    candidates: list[dict[str, object]],
    *,
    distance_metric: str = "euclidean",
) -> dict[str, object]:
    """Attach dotted GoT parent-child generation order to a trajectory-state complex.

    Full trajectory complexes have GoT state vertices, not graph-token vertices.  The
    dotted overlay here is therefore the actual sampled parent-child decoding/search
    order of the reasoning trajectory.  Graph-node causal and token decoding edges are
    rendered on each per-state TokenGT complex, where those vertices exist.
    """
    simplices = obj.get("simplices", []) if isinstance(obj, dict) else []
    vertex_filtration: dict[str, float] = {}
    for simplex in simplices if isinstance(simplices, list) else []:
        if not isinstance(simplex, dict) or int(simplex.get("dimension", -1)) != 0:
            continue
        raw = simplex.get("simplex") if isinstance(simplex.get("simplex"), list) else []
        if not raw:
            continue
        label = str(raw[0])
        try:
            filt = float(simplex.get("filtration", 0.0) or 0.0)
        except (TypeError, ValueError):
            filt = 0.0
        vertex_filtration[label] = filt if math.isfinite(filt) else 0.0
    visible = set(vertex_filtration)
    row_by_id = {str(row.get("record_id", idx)): row for idx, row in enumerate(candidates) if isinstance(row, dict)}
    edges: list[dict[str, object]] = []
    for target_id, row in row_by_id.items():
        parent = row.get("parent")
        if not isinstance(parent, str) or parent not in row_by_id:
            continue
        if parent not in visible or target_id not in visible:
            continue
        target_level = int(row.get("level", 0) or 0)
        try:
            decoding_step = int(row.get("reasoning_step_index", len(edges) + 1) or len(edges) + 1)
        except (TypeError, ValueError):
            decoding_step = len(edges) + 1
        action = _edge_action_label(row)
        filtration = max(vertex_filtration.get(parent, 0.0), vertex_filtration.get(target_id, 0.0))
        edges.append(
            {
                "source": parent,
                "target": target_id,
                "source_node_id": parent,
                "target_node_id": target_id,
                "role": "got_parent_child_decoding_order",
                "edge_type": f"got_{action}_transition",
                "action": action,
                "decoding_step": decoding_step,
                "reasoning_level": target_level,
                "filtration": filtration,
                "style": "dotted",
                "color": _action_color(action),
                "directed": True,
                "causal": False,
                "distance_metric": distance_metric,
                "source_nll": _safe_float(row_by_id.get(parent, {}).get("nll")),
                "target_nll": _safe_float(row.get("nll")),
                "source_path": row_by_id.get(parent, {}).get("path", []),
                "target_path": row.get("path", []),
                "gate": {
                    "radius": filtration,
                    "reasoning_step": target_level,
                    "decoding_step": decoding_step,
                },
            }
        )
    updated = dict(obj)
    updated["decoding_causal_overlay"] = {
        "source": "graph_of_thought_parent_decoding_order",
        "semantic_note": (
            "Dotted directed edges on the full trajectory complex show the observed sampled GoT parent-child "
            "generation order between trajectory-state vertices. They are not graph-node causal edges; those are "
            "rendered on per-reasoning-step TokenGT complexes where graph-token vertices are present."
        ),
        "distance_metric": distance_metric,
        "decoding_order_kind": "graph_of_thought_parent_child_order",
        "decoding_reverse_order_kind": "not_applicable_on_state_complex",
        "decoding_is_dag": True,
        "edge_count": len(edges),
        "edges": edges,
    }
    return updated


def _coerce_token_index(value: object) -> int | None:
    try:
        out = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out


def _attach_graph_token_direction_overlay(obj: dict[str, object], row: dict[str, object]) -> dict[str, object]:
    trace = row.get("graph_token_trace") if isinstance(row, dict) else None
    tokens = trace.get("tokens", []) if isinstance(trace, dict) else []
    if not isinstance(tokens, list):
        tokens = []
    label_by_token_index: dict[int, str] = {}
    node_label_by_id: dict[str, str] = {}
    for simplex in obj.get("simplices", []) if isinstance(obj.get("simplices"), list) else []:
        if not isinstance(simplex, dict) or int(simplex.get("dimension", -1)) != 0:
            continue
        simplex_vertices = simplex.get("simplex") if isinstance(simplex.get("simplex"), list) else []
        if not simplex_vertices:
            continue
        label = str(simplex_vertices[0])
        token_index = _coerce_token_index(simplex.get("token_index"))
        if token_index is not None:
            label_by_token_index[token_index] = label
        node_id = simplex.get("node_id")
        if isinstance(node_id, str) and node_id:
            node_label_by_id[node_id] = label
    for token in tokens:
        if not isinstance(token, dict):
            continue
        token_index = _coerce_token_index(token.get("index"))
        if str(token.get("kind", "")) == "node":
            node_id = token.get("node_id")
            if token_index is not None and isinstance(node_id, str) and node_id in node_label_by_id:
                label_by_token_index.setdefault(token_index, node_label_by_id[node_id])
    direction_edges = []
    for token in tokens:
        if not isinstance(token, dict) or str(token.get("kind", "")) != "edge":
            continue
        token_index = _coerce_token_index(token.get("index"))
        if token_index is None or token_index not in label_by_token_index:
            continue
        edge_label = label_by_token_index[token_index]
        source_node = token.get("source")
        target_node = token.get("target")
        source_label = node_label_by_id.get(str(source_node)) if source_node is not None else None
        target_label = node_label_by_id.get(str(target_node)) if target_node is not None else None
        base = {
            "edge_token": edge_label,
            "edge_token_index": token_index,
            "edge_label": str(token.get("label", "")),
            "edge_type": str(token.get("edge_type", "graph_edge")),
            "source_node_id": str(source_node) if source_node is not None else "",
            "target_node_id": str(target_node) if target_node is not None else "",
            "active_support_index": token.get("active_support_index"),
            "active_support_label": token.get("active_support_label"),
            "margin": _safe_float(token.get("margin")),
        }
        if source_label:
            direction_edges.append({**base, "source": source_label, "target": edge_label, "role": "source-node-to-edge-token"})
        if target_label:
            direction_edges.append({**base, "source": edge_label, "target": target_label, "role": "edge-token-to-target-node"})
    updated = dict(obj)
    updated["graph_token_direction_overlay"] = {
        "source": "graph_token_trace_directed_edges",
        "semantic_note": (
            "Directed overlay is read from the same model graph-token trace as the vertices: each graph edge token is "
            "shown as source-node -> edge-token -> target-node on top of the radius filtered vector complex."
        ),
        "edge_count": len(direction_edges),
        "edges": direction_edges,
    }
    return updated



def _vertex_label_maps(obj: dict[str, object]) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    node_label_by_id: dict[str, str] = {}
    vertex_by_label: dict[str, dict[str, object]] = {}
    for simplex in obj.get("simplices", []) if isinstance(obj.get("simplices"), list) else []:
        if not isinstance(simplex, dict) or int(simplex.get("dimension", -1)) != 0:
            continue
        simplex_vertices = simplex.get("simplex") if isinstance(simplex.get("simplex"), list) else []
        if not simplex_vertices:
            continue
        label = str(simplex_vertices[0])
        vertex_by_label[label] = simplex
        # Some complexes store graph-node ids directly as 0-simplex labels; model-derived
        # TokenGT complexes may instead carry them as node_id metadata. Accept both, but
        # never invent correspondences beyond labels already present in the complex.
        node_label_by_id.setdefault(label, label)
        node_id = simplex.get("node_id")
        if isinstance(node_id, str) and node_id:
            node_label_by_id[node_id] = label
    return node_label_by_id, vertex_by_label


def _attach_decoding_causal_overlay(obj: dict[str, object], row: dict[str, object]) -> dict[str, object]:
    report = row.get("decoding_order_report") if isinstance(row, dict) else None
    if not isinstance(report, dict):
        updated = dict(obj)
        updated["decoding_causal_overlay"] = {
            "source": "unavailable_decoding_order_report",
            "edge_count": 0,
            "edges": [],
            "semantic_note": "No decoding-order report was emitted for this model row, so no causal/decoding overlay is rendered.",
        }
        return updated
    node_label_by_id, vertex_by_label = _vertex_label_maps(obj)

    def vertex_filtration(label: str) -> float:
        row_obj = vertex_by_label.get(label, {})
        try:
            value = float(row_obj.get("filtration", 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        return value if math.isfinite(value) else 0.0

    edges: list[dict[str, object]] = []

    def add_edge(raw: object, role: str, color: str, style: str = "dotted") -> None:
        if not isinstance(raw, dict):
            return
        source_node = str(raw.get("source", ""))
        target_node = str(raw.get("target", ""))
        source = node_label_by_id.get(source_node)
        target = node_label_by_id.get(target_node)
        if not source or not target or source == target:
            return
        try:
            decoding_step = int(raw.get("decoding_step", raw.get("edge_index", len(edges) + 1)) or len(edges) + 1)
        except (TypeError, ValueError):
            decoding_step = len(edges) + 1
        filtration = max(vertex_filtration(source), vertex_filtration(target))
        edges.append(
            {
                "source": source,
                "target": target,
                "source_node_id": source_node,
                "target_node_id": target_node,
                "role": role,
                "edge_type": str(raw.get("edge_type", role)),
                "decoding_step": decoding_step,
                "reasoning_level": int(row.get("level", 0) or 0),
                "filtration": filtration,
                "style": style,
                "color": color,
                "directed": True,
                "causal": bool(raw.get("causal", False)),
                "gate": {
                    "radius": filtration,
                    "reasoning_step": int(row.get("level", 0) or 0),
                    "decoding_step": decoding_step,
                },
            }
        )

    for raw in report.get("causal_edges", []) if isinstance(report.get("causal_edges"), list) else []:
        add_edge(raw, "causal_graph_edge", "rgba(250,204,21,0.86)")
    for raw in report.get("forward_decoding_edges", []) if isinstance(report.get("forward_decoding_edges"), list) else []:
        add_edge(raw, "forward_decoding_order", "rgba(94,234,212,0.88)")
    for raw in report.get("reverse_decoding_edges", []) if isinstance(report.get("reverse_decoding_edges"), list) else []:
        add_edge(raw, "reverse_decoding_order", "rgba(244,114,182,0.82)")

    updated = dict(obj)
    updated["decoding_causal_overlay"] = {
        "source": "GraphRecord.metadata.graph_decoding_order",
        "semantic_note": (
            "Dotted directed edges are model-record decoding/causal order overlays. They are not simplices: "
            "the radius slider reveals them only when their endpoint vertices have entered the displayed complex. "
            "DAG rows show causal forward and reverse order; cyclic/noncausal rows show ROAR/random-order autoregressive order."
        ),
        "decoding_order_kind": str(report.get("decoding_order_kind", "unknown")),
        "decoding_reverse_order_kind": str(report.get("decoding_reverse_order_kind", "unknown")),
        "decoding_is_dag": bool(report.get("decoding_is_dag")),
        "edge_count": len(edges),
        "edges": edges,
    }
    return updated

def _safe_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _complex_has_model_io(obj: dict[str, object]) -> bool:
    for simplex in obj.get("simplices", []) if isinstance(obj, dict) else []:
        if not isinstance(simplex, dict) or int(simplex.get("dimension", -1)) != 0:
            continue
        if simplex.get("input_text") or simplex.get("decoded_argmax") or simplex.get("target_text"):
            return True
    return False


def _has_real_probability_filtration(obj: object) -> bool:
    if not isinstance(obj, dict) or obj.get("available") is False:
        return False
    summary = obj.get("summary", {}) if isinstance(obj.get("summary"), dict) else {}
    allowed_models = {
        "model_candidate_probability_jensen_shannon_vietoris_rips_2_skeleton",
        "model_tropical_support_probability_jensen_shannon_vietoris_rips_2_skeleton",
    }
    if summary.get("filtration_model") not in allowed_models:
        return False
    if summary.get("radius_filtration") is not True:
        return False
    vertices = [
        simplex
        for simplex in obj.get("simplices", [])
        if isinstance(simplex, dict) and int(simplex.get("dimension", -1)) == 0
    ]
    return bool(vertices) and all(_probability_feature_vector(simplex) is not None for simplex in vertices)


def _unavailable_complex(reason: str, *, source: str = "model_probability_jensen_shannon") -> dict[str, object]:
    return {
        "available": False,
        "reason": reason,
        "source": source,
        "summary": {
            "num_vertices": 0,
            "num_edges": 0,
            "num_two_simplices": 0,
            "num_thresholds": 0,
            "filtration_model": "unavailable_model_probability_jensen_shannon",
        },
        "thresholds": [],
        "simplices": [],
    }


def _simplex_dimension(row: Mapping[str, object]) -> int:
    raw = row.get("dimension", row.get("dim"))
    try:
        dim = int(raw)
    except (TypeError, ValueError):
        dim = -1
    if dim >= 0:
        return dim
    simplex = row.get("simplex")
    if isinstance(simplex, list) and simplex:
        return len(simplex) - 1
    return dim


def _reasoning_step_complex_fingerprint(obj_summary: dict[str, object]) -> tuple[str, dict[str, object]]:
    simplices = [row for row in obj_summary.get("simplices", []) if isinstance(row, dict)] if isinstance(obj_summary, dict) else []
    canonical_simplices = sorted(
        [
            {
                "simplex": [str(value) for value in (row.get("simplex", []) if isinstance(row.get("simplex"), list) else [])],
                "dimension": _simplex_dimension(row),
                "filtration": float(row.get("filtration", 0.0) or 0.0),
                "type": str(row.get("type", "")),
                "probability_source": str(row.get("probability_source", row.get("model_probability_source", ""))),
                "has_probability_vector": bool(_probability_feature_vector(row) is not None),
                "has_embedding": bool(isinstance(row.get("embedding"), list) and len(row.get("embedding", [])) > 0),
            }
            for row in simplices
        ],
        key=lambda row: (row["dimension"], row["filtration"], row["simplex"], row["type"]),
    )
    tree = obj_summary.get("simplex_tree", {}) if isinstance(obj_summary.get("simplex_tree"), dict) else {}
    basis = {
        "schema_version": "tropicalgt.reasoning_step_complex_fingerprint_basis.v1",
        "hash_algorithm": "sha256_canonical_json",
        "source": "gudhi_canonical_complex(filtered_simplicial_object)",
        "summary": obj_summary.get("summary", {}) if isinstance(obj_summary.get("summary"), dict) else {},
        "thresholds": [float(value) for value in obj_summary.get("thresholds", []) if isinstance(value, (int, float)) and math.isfinite(float(value))],
        "simplices": canonical_simplices,
        "simplex_tree": {
            "backend": tree.get("backend", "missing"),
            "available": tree.get("available") is not False,
            "dimension": tree.get("dimension"),
            "num_simplices": tree.get("num_simplices"),
            "num_vertices": tree.get("num_vertices"),
        },
        "no_record_id_or_path_in_hash": True,
    }
    return _stable_artifact_hash(basis), basis


def _slider_summary_int(value: object, default: int = 0) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return int(numeric)


def _reasoning_step_slider_contract_summary(
    contract: Mapping[str, object],
    *,
    contract_file: str,
    html_file: str,
) -> dict[str, object]:
    first_vertices = _slider_summary_int(contract.get("first_frame_vertex_count"), 0)
    first_edges = _slider_summary_int(contract.get("first_frame_solid_edge_count"), -1)
    first_faces = _slider_summary_int(contract.get("first_frame_filled_face_count"), -1)
    first_dotted = _slider_summary_int(contract.get("first_frame_dotted_overlay_count"), -1)
    semantic_checks = {
        "solid_lines_semantics_ok": contract.get("solid_lines_semantics") == "radius-filtered 1-simplices only",
        "filled_faces_semantics_ok": contract.get("filled_faces_semantics") == "radius-gated 2-simplices only",
        "dotted_lines_semantics_ok": contract.get("dotted_lines_semantics") == "causal_decoding_or_direction_overlay_only_and_radius_gated",
    }
    summary = {
        "schema_version": "tropicalgt.reasoning_step_radius_slider_summary.v1",
        "source": f"reasoning_step_complex_maps/{contract_file}",
        "contract_file": contract_file,
        "html_file": html_file,
        "contract_schema_version": contract.get("schema_version"),
        "contract_source": contract.get("source"),
        "actual_data_only": contract.get("actual_data_only") is True,
        "no_proxy_or_fallback": contract.get("no_proxy_or_fallback") is True,
        "radius_filtration": contract.get("radius_filtration") is True,
        "threshold_order": contract.get("threshold_order"),
        "thresholds_ascending": contract.get("thresholds_ascending") is True,
        "threshold_count": _slider_summary_int(contract.get("threshold_count"), 0),
        "frame_count": _slider_summary_int(contract.get("frame_count"), 0),
        "first_frame_vertex_count": first_vertices,
        "first_frame_solid_edge_count": first_edges,
        "first_frame_filled_face_count": first_faces,
        "first_frame_dotted_overlay_count": first_dotted,
        "initial_radius_frame_hides_dotted_overlays": contract.get("initial_radius_frame_hides_dotted_overlays") is True,
        "initial_radius_frame_hides_solid_edges_and_faces": contract.get("initial_radius_frame_hides_solid_edges_and_faces") is True,
        "first_frame_disjoint_vertices_only": contract.get("first_frame_disjoint_vertices_only") is True,
        "monotone_visible_counts": contract.get("monotone_visible_counts") is True,
        "monotone_solid_radius_edges": contract.get("monotone_solid_radius_edges") is True,
        "monotone_filled_radius_faces": contract.get("monotone_filled_radius_faces") is True,
        **semantic_checks,
    }
    summary["safe_to_render_radius_filtration"] = all(
        [
            summary["contract_schema_version"] == "tropicalgt.radius_filtration_slider_contract.v1",
            summary["contract_source"] == "canonical_gudhi_filtered_complex_simplices",
            summary["actual_data_only"],
            summary["no_proxy_or_fallback"],
            summary["radius_filtration"],
            summary["threshold_order"] == "ascending_min_to_max",
            summary["thresholds_ascending"],
            summary["frame_count"] == summary["threshold_count"],
            summary["frame_count"] > 0,
            summary["first_frame_disjoint_vertices_only"],
            first_vertices > 0,
            first_edges == 0,
            first_faces == 0,
            first_dotted == 0,
            summary["initial_radius_frame_hides_dotted_overlays"],
            summary["initial_radius_frame_hides_solid_edges_and_faces"],
            summary["monotone_visible_counts"],
            summary["monotone_solid_radius_edges"],
            summary["monotone_filled_radius_faces"],
            all(semantic_checks.values()),
        ]
    )
    return summary


def _reasoning_step_complex_source_contract(
    row: Mapping[str, object],
    obj_summary: Mapping[str, object],
    step_complex_fingerprint: str,
) -> dict[str, object]:
    simplices = [item for item in obj_summary.get("simplices", []) if isinstance(item, dict)]
    vertices = [item for item in simplices if _simplex_dimension(item) == 0]
    edges = [item for item in simplices if _simplex_dimension(item) == 1]
    faces = [item for item in simplices if _simplex_dimension(item) == 2]
    tree = obj_summary.get("simplex_tree", {}) if isinstance(obj_summary.get("simplex_tree"), Mapping) else {}
    vertex_labels = [
        str((vertex.get("simplex") or [f"v{idx}"])[0]) if isinstance(vertex.get("simplex"), list) else f"v{idx}"
        for idx, vertex in enumerate(vertices)
    ]
    probability_vertex_count = int(sum(1 for vertex in vertices if _probability_feature_vector(vertex) is not None))
    embedding_vertex_count = int(sum(1 for vertex in vertices if isinstance(vertex.get("embedding"), list) and len(vertex.get("embedding", [])) > 0))
    summary = obj_summary.get("summary", {}) if isinstance(obj_summary.get("summary"), Mapping) else {}
    contract = {
        "schema_version": "tropicalgt.reasoning_step_complex_source_contract.v1",
        "source": "candidate.filtered_simplicial_object",
        "actual_data_only": True,
        "no_proxy_or_fallback": True,
        "candidate_record_id": str(row.get("record_id", "")),
        "candidate_level": int(row.get("level", 0) or 0),
        "candidate_path": row.get("path", []),
        "step_complex_fingerprint": step_complex_fingerprint,
        "uses_global_trajectory_complex_as_proxy": False,
        "uses_embedding_trajectory_map_as_proxy": False,
        "uses_static_probability_complex_as_proxy": False,
        "displayed_vertex_count": int(len(vertices)),
        "displayed_edge_count": int(len(edges)),
        "displayed_face_count": int(len(faces)),
        "summary_vertex_count": int(summary.get("num_vertices", len(vertices)) or 0),
        "summary_edge_count": int(summary.get("num_edges", len(edges)) or 0),
        "summary_two_simplex_count": int(summary.get("num_two_simplices", len(faces)) or 0),
        "displayed_probability_vector_vertex_count": probability_vertex_count,
        "displayed_embedding_vertex_count": embedding_vertex_count,
        "displayed_vertex_labels_sample": vertex_labels[:32],
        "simplex_tree_backend": tree.get("backend", "missing"),
        "simplex_tree_available": tree.get("available") is not False,
        "simplex_tree_num_vertices": tree.get("num_vertices"),
        "simplex_tree_num_simplices": tree.get("num_simplices"),
        "graph_token_direction_overlay_source": "candidate_trace_overlay_when_present",
        "decoding_causal_overlay_source": "candidate_decoding_or_causal_metadata_when_present",
    }
    contract["source_counts_match_canonical_summary"] = (
        contract["displayed_vertex_count"] == contract["summary_vertex_count"]
        and contract["displayed_edge_count"] == contract["summary_edge_count"]
        and contract["displayed_face_count"] == contract["summary_two_simplex_count"]
    )
    contract["safe_to_render_as_step_complex"] = all(
        [
            bool(contract["candidate_record_id"]),
            bool(step_complex_fingerprint),
            contract["displayed_vertex_count"] > 0,
            contract["source_counts_match_canonical_summary"],
            contract["actual_data_only"],
            contract["no_proxy_or_fallback"],
            contract["uses_global_trajectory_complex_as_proxy"] is False,
            contract["uses_embedding_trajectory_map_as_proxy"] is False,
            contract["uses_static_probability_complex_as_proxy"] is False,
        ]
    )
    return contract


def _write_reasoning_step_complex_maps(candidates: list[dict[str, object]], output_dir: Path) -> dict[str, str]:
    directory = output_dir / "reasoning_step_complex_maps"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, row in enumerate(candidates):
        obj = row.get("filtered_simplicial_object")
        if not isinstance(obj, dict):
            continue
        obj = _attach_graph_token_direction_overlay(obj, row)
        obj = _attach_decoding_causal_overlay(obj, row)
        obj_summary = _gudhi_canonical_complex(obj)
        step_complex_fingerprint, fingerprint_basis = _reasoning_step_complex_fingerprint(obj_summary)
        source_contract = _reasoning_step_complex_source_contract(row, obj_summary, step_complex_fingerprint)
        record_id = str(row.get("record_id", f"step-{idx}"))
        file_name = f"reasoning_step_{idx:03d}.html"
        path = directory / file_name
        tree_file_name = f"reasoning_step_{idx:03d}_simplex_tree.html"
        tree_path = directory / tree_file_name
        _write_complex_slider_map(
            path,
            obj,
            title=f"Reasoning step filtered simplicial complex map: q{idx}",
            subtitle=f"record_id={record_id}; level={row.get('level')}; path={row.get('path', [])}",
        )
        slider_contract_path = _complex_slider_contract_path(path)
        if slider_contract_path.exists():
            try:
                slider_contract_payload = json.loads(slider_contract_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                slider_contract_payload = {
                    "schema_version": "unreadable",
                    "source": "reasoning_step_slider_contract_json_decode_error",
                    "actual_data_only": False,
                    "no_proxy_or_fallback": False,
                    "radius_filtration": False,
                }
        else:
            slider_contract_payload = {
                "schema_version": "missing",
                "source": "reasoning_step_slider_contract_missing",
                "actual_data_only": False,
                "no_proxy_or_fallback": False,
                "radius_filtration": False,
            }
        radius_slider_contract = _reasoning_step_slider_contract_summary(
            slider_contract_payload if isinstance(slider_contract_payload, Mapping) else {},
            contract_file=slider_contract_path.name,
            html_file=file_name,
        )
        _write_simplex_tree_3d_map(
            tree_path,
            obj,
            title=f"Reasoning step GUDHI SimplexTree face-coface poset: q{idx}",
            subtitle=f"record_id={record_id}; level={row.get('level')}; path={row.get('path', [])}",
        )
        tree_poset_contract_path = _simplex_tree_poset_contract_path(tree_path)
        if tree_poset_contract_path.exists():
            try:
                tree_poset_contract = json.loads(tree_poset_contract_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                tree_poset_contract = {
                    "schema_version": "unreadable",
                    "available": False,
                    "reason": "simplex_tree_poset_contract_json_decode_error",
                    "actual_data_only": False,
                    "no_proxy_or_fallback": False,
                    "safe_to_render_simplex_tree": False,
                }
        else:
            tree_poset_contract = {
                "schema_version": "missing",
                "available": False,
                "reason": "simplex_tree_poset_contract_missing",
                "actual_data_only": False,
                "no_proxy_or_fallback": False,
                "safe_to_render_simplex_tree": False,
            }
        rows.append(
            {
                "index": idx,
                "record_id": record_id,
                "level": int(row.get("level", 0) or 0),
                "path": row.get("path", []),
                "file": file_name,
                "simplex_tree_file": tree_file_name,
                "simplex_tree_poset_contract_file": tree_poset_contract_path.name,
                "simplex_tree_poset_contract": tree_poset_contract,
                "slider_contract_file": slider_contract_path.name,
                "radius_slider_contract": radius_slider_contract,
                "step_complex_source_contract": source_contract,
                "summary": obj_summary.get("summary", {}),
                "step_complex_fingerprint": step_complex_fingerprint,
                "step_complex_fingerprint_basis": fingerprint_basis,
                "simplex_tree": obj_summary.get("simplex_tree", {}),
                "simplex_tree_backend": (obj_summary.get("simplex_tree", {}) if isinstance(obj_summary.get("simplex_tree"), dict) else {}).get("backend", "missing"),
                "simplex_tree_available": (obj_summary.get("simplex_tree", {}) if isinstance(obj_summary.get("simplex_tree"), dict) else {}).get("available") is not False,
                "complex_render_contract": "actual per-step radius-filtered complex from this model-evaluated reasoning state; no trajectory-map proxy",
                "simplex_tree_render_contract": "actual GUDHI SimplexTree face-coface poset when available; unavailable is explicit and not substituted",
                "graph_token_direction_overlay": obj.get("graph_token_direction_overlay", {}),
                "decoding_causal_overlay": obj.get("decoding_causal_overlay", {}),
            }
        )
    index_path = directory / "index.html"
    contract = _reasoning_step_complex_manifest_contract(rows)
    _write_reasoning_step_complex_index(index_path, rows, contract=contract)
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps({"contract": contract, "steps": rows}, indent=2), encoding="utf-8")
    return {
        "got_reasoning_step_complex_index": str(index_path),
        "got_reasoning_step_complex_manifest": str(manifest_path),
    }


def _write_complex_slider_map(path: Path, obj: dict[str, object], title: str, subtitle: str = "") -> None:
    obj = _gudhi_canonical_complex(obj)
    vertices = [s for s in obj.get("simplices", []) if isinstance(s, dict) and int(s.get("dimension", -1)) == 0]
    edges = [s for s in obj.get("simplices", []) if isinstance(s, dict) and int(s.get("dimension", -1)) == 1]
    triangles = [s for s in obj.get("simplices", []) if isinstance(s, dict) and int(s.get("dimension", -1)) == 2]
    labels = [str((row.get("simplex") or [f"v{idx}"])[0]) for idx, row in enumerate(vertices)]
    if not labels:
        _write_dark_empty(path, f"{title}: no vertices available.")
        return
    coords3, _projected, layout_kind = _simplicial_pca3_radius_layout(labels, vertices, edges, width=760, height=560)
    panel_objects = [obj]
    panel_hovers = [f"<b>{html.escape(title)}</b><br>{html.escape(subtitle)}<br>{_summary_line(obj)}"]
    panel_index_by_label: dict[str, int] = {}
    for vertex in vertices:
        local_obj = vertex.get("filtered_simplicial_object")
        if isinstance(local_obj, dict):
            label = str((vertex.get("simplex") or [""])[0])
            panel_index_by_label[label] = len(panel_objects)
            panel_objects.append(_gudhi_canonical_complex(local_obj))
            panel_hovers.append(_vertex_readable_summary(vertex, include_output=True))
    thresholds = _display_thresholds(obj)
    initial = thresholds[0] if thresholds else float("inf")
    base_traces = _complex_slider_traces(obj, coords3, threshold=initial, panel_index_by_label=panel_index_by_label)
    fig = go.Figure(data=base_traces)
    frames = []
    frame_thresholds = thresholds
    for threshold in frame_thresholds:
        frames.append(
            go.Frame(
                name=f"{threshold:.6f}",
                data=_complex_slider_traces(obj, coords3, threshold=threshold, panel_index_by_label=panel_index_by_label),
                traces=list(range(len(base_traces))),
            )
        )
    fig.frames = frames
    if frames:
        steps = [
            {
                "args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                "label": f"{float(frame.name):.3f}",
                "method": "animate",
            }
            for frame in frames
        ]
        active_frame = 0
        fig.update_layout(
            sliders=[
                {
                    "active": active_frame,
                    "currentvalue": {"prefix": "radius/filtration <= ", "font": {"color": "#dbeafe"}},
                    "pad": {"t": 44},
                    "x": 0.06,
                    "len": 0.88,
                    "steps": steps,
                }
            ],
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "x": 0.02,
                    "y": 0,
                    "xanchor": "left",
                    "yanchor": "top",
                    "buttons": [
                        {
                            "label": "play filtration min-to-max",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 220, "redraw": True}, "fromcurrent": False, "transition": {"duration": 0}}],
                        }
                    ],
                }
            ],
        )
    slider_contract = _complex_slider_frame_contract(obj, thresholds=thresholds)
    _complex_slider_contract_path(path).write_text(
        json.dumps(
            {
                **slider_contract,
                "html_file": path.name,
                "title": title,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = obj.get("summary", {})
    simplex_tree = obj.get("simplex_tree", {}) if isinstance(obj.get("simplex_tree"), dict) else {}
    backend_label = simplex_tree.get("backend", "json")
    fig.update_layout(
        template="plotly_dark",
        meta={"radius_filtration_slider_contract": slider_contract},
        title=(
            f"{title}<br><sup>{html.escape(subtitle)} | filtration backend={html.escape(str(backend_label))} "
            f"| {html.escape(layout_kind)} | V={summary.get('num_vertices', len(vertices))}, "
            f"E={summary.get('num_edges', len(edges))}, T={summary.get('num_two_simplices', len(triangles))} "
            "| first radius frame is vertex-only | solid radius edges and filled radius-gated 2-simplices "
            "| dotted causal and decoding overlays</sup>"
        ),
        scene=dict(
            xaxis_title="projected coordinate 1",
            yaxis_title="projected coordinate 2",
            zaxis_title="projected coordinate 3",
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=1.25, z=0.9)),
        ),
        legend=dict(
            itemsizing="constant",
            orientation="h",
            x=0.02,
            y=0.96,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(7,11,18,0.72)",
            bordercolor="rgba(148,163,184,0.28)",
            borderwidth=1,
        ),
    )
    _write_plotly_dark_html(
        path,
        fig,
        title,
        _simplicial_panel_items(panel_objects, panel_hovers),
        show_filtration_slider=True,
    )


def _complex_slider_contract_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_slider_contract.json")


def _simplex_tree_poset_contract_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_simplex_tree_poset_contract.json")


def _unavailable_simplex_tree_poset_contract(
    path: Path,
    *,
    title: str,
    reason: str,
    backend: str = "unavailable_gudhi_simplex_tree",
) -> dict[str, object]:
    return {
        "schema_version": "tropicalgt.simplex_tree_poset.v1",
        "available": False,
        "html_file": path.name,
        "title": title,
        "source": "gudhi_canonical_complex(filtered_simplicial_object).simplex_tree",
        "backend": backend,
        "safe_to_render_simplex_tree": False,
        "reason": reason,
        "actual_data_only": True,
        "no_proxy_or_fallback": True,
        "safe_unavailable_render": True,
        "layout": "unavailable_no_simplex_tree_poset_rendered",
        "primary_edges": "unavailable",
        "not_disconnected_simplex_columns": True,
        "empty_simplex_root_present": False,
        "displayed_simplex_count": 0,
        "source_simplex_count": 0,
        "actual_face_to_coface_cover_edges": 0,
        "empty_root_vertex_cover_edges": 0,
        "optional_sorted_label_trie_prefix_edges": 0,
    }


def _write_simplex_tree_poset_contract(path: Path, contract: Mapping[str, object]) -> None:
    _simplex_tree_poset_contract_path(path).write_text(
        json.dumps(dict(contract), indent=2),
        encoding="utf-8",
    )


def _write_simplex_tree_3d_map(path: Path, obj: dict[str, object], title: str, subtitle: str = "") -> None:
    obj = _gudhi_canonical_complex(obj)
    tree_status = obj.get("simplex_tree", {}) if isinstance(obj.get("simplex_tree"), dict) else {}
    if tree_status and tree_status.get("available") is False:
        reason = str(tree_status.get("error") or tree_status.get("reason") or "gudhi_simplex_tree_unavailable")
        _write_simplex_tree_poset_contract(
            path,
            _unavailable_simplex_tree_poset_contract(
                path,
                title=title,
                reason=reason,
                backend=str(tree_status.get("backend", "unavailable_gudhi_simplex_tree")),
            ),
        )
        _write_dark_empty(
            path,
            (
                f"{title}: unavailable_gudhi_simplex_tree; {reason}. "
                "No simplex-tree/trie or face-coface poset is rendered without a real GUDHI SimplexTree."
            ),
        )
        return
    simplices = [row for row in obj.get("simplices", []) if isinstance(row, dict) and row.get("simplex")]
    if not simplices:
        _write_simplex_tree_poset_contract(
            path,
            _unavailable_simplex_tree_poset_contract(
                path,
                title=title,
                reason="no_simplices_available",
                backend=str(tree_status.get("backend", "missing")),
            ),
        )
        _write_dark_empty(path, f"{title}: no simplices available.")
        return
    max_nodes = 1600
    truncated = len(simplices) > max_nodes
    if truncated:
        simplices = sorted(
            simplices,
            key=lambda row: (
                int(row.get("dimension", 99)),
                float(row.get("filtration", 0.0) or 0.0),
                _simplex_key(row.get("simplex", [])),
            ),
        )[:max_nodes]
    simplex_rows: dict[tuple[str, ...], dict[str, object]] = {
        tuple(sorted(str(vertex) for vertex in (row.get("simplex") or []))): row for row in simplices
    }
    if not simplex_rows:
        _write_simplex_tree_poset_contract(
            path,
            _unavailable_simplex_tree_poset_contract(
                path,
                title=title,
                reason="no_canonical_simplices_available",
                backend=str(tree_status.get("backend", "missing")),
            ),
        )
        _write_dark_empty(path, f"{title}: no canonical simplices available.")
        return

    vertex_rows = [row for key, row in simplex_rows.items() if len(key) == 1]
    edge_rows = [row for key, row in simplex_rows.items() if len(key) == 2]
    vertex_labels = sorted(str((row.get("simplex") or [""])[0]) for row in vertex_rows)
    coords3, _projected2, layout_report = _simplicial_pca3_radius_layout(vertex_labels, vertex_rows, edge_rows, 900, 720)
    if not coords3:
        coords3 = {label: (0.5, 0.5, 0.5) for label in vertex_labels}

    filtrations = [float(row.get("filtration", 0.0) or 0.0) for row in simplex_rows.values()]
    f_min = min(filtrations) if filtrations else 0.0
    f_span = max(max(filtrations) - f_min, 1e-9) if filtrations else 1.0

    root_key: tuple[str, ...] = tuple()
    positions: dict[tuple[str, ...], tuple[float, float, float]] = {}
    for key, row in simplex_rows.items():
        dim = int(row.get("dimension", len(key) - 1))
        base = [coords3[v] for v in key if v in coords3]
        if base:
            arr = np.asarray(base, dtype=float)
            bary = arr.mean(axis=0)
        else:
            # Invalid/truncated simplex: expose missing-coordinate issue in hover instead of fabricating vertices.
            bary = np.asarray([0.5, 0.5, 0.5], dtype=float)
        filt = float(row.get("filtration", 0.0) or 0.0)
        f_unit = (filt - f_min) / f_span
        positions[key] = (
            float(bary[0]),
            float(bary[1] + 0.105 * dim),
            float(bary[2] + 0.155 * dim + 0.075 * f_unit),
        )
    if vertex_labels:
        center = np.asarray([coords3[v] for v in vertex_labels if v in coords3], dtype=float).mean(axis=0)
        positions[root_key] = (float(center[0]), float(center[1] - 0.18), float(center[2] - 0.18))
    else:
        positions[root_key] = (0.5, 0.32, 0.32)

    def _edge_trace_data(edge_pairs: list[tuple[tuple[str, ...], tuple[str, ...], str]]) -> tuple[list[float | None], list[float | None], list[float | None], list[str | None]]:
        ex: list[float | None] = []
        ey: list[float | None] = []
        ez: list[float | None] = []
        eh: list[str | None] = []
        for source, target, label in edge_pairs:
            if source not in positions or target not in positions:
                continue
            ax, ay, az = positions[source]
            bx, by, bz = positions[target]
            ex.extend([ax, bx, None])
            ey.extend([ay, by, None])
            ez.extend([az, bz, None])
            eh.extend([label, label, None])
        return ex, ey, ez, eh

    hasse_pairs: list[tuple[tuple[str, ...], tuple[str, ...], str]] = []
    for key, row in simplex_rows.items():
        dim = int(row.get("dimension", len(key) - 1))
        if len(key) == 1:
            hasse_pairs.append(
                (
                    root_key,
                    key,
                    f"<b>empty-simplex cover</b><br>vertex={html.escape(_simplex_key(key))}"
                    f"<br>filtration={float(row.get('filtration', 0.0) or 0.0):.6g}",
                )
            )
            continue
        for face in combinations(key, len(key) - 1):
            face_key = tuple(sorted(face))
            if face_key not in positions:
                continue
            hasse_pairs.append(
                (
                    face_key,
                    key,
                    f"<b>actual face-to-coface cover</b>"
                    f"<br>face={html.escape(_simplex_key(face_key))}"
                    f"<br>coface={html.escape(_simplex_key(key))}"
                    f"<br>coface dimension={dim}"
                    f"<br>coface filtration={float(row.get('filtration', 0.0) or 0.0):.6g}",
                )
            )
    hasse_x, hasse_y, hasse_z, hasse_hover = _edge_trace_data(hasse_pairs)

    prefix_pairs: list[tuple[tuple[str, ...], tuple[str, ...], str]] = []
    for key, row in simplex_rows.items():
        if len(key) <= 1:
            parent = root_key
        else:
            parent = tuple(key[:-1])
            if parent not in positions:
                parent = root_key
        prefix_pairs.append(
            (
                parent,
                key,
                f"<b>SimplexTree trie prefix link</b><br>parent={html.escape(_simplex_key(parent) if parent else '{}')}"
                f"<br>child={html.escape(_simplex_key(key))}"
                f"<br>child filtration={float(row.get('filtration', 0.0) or 0.0):.6g}",
            )
        )
    prefix_x, prefix_y, prefix_z, prefix_hover = _edge_trace_data(prefix_pairs)

    node_keys = sorted(simplex_rows, key=lambda key: (len(key), float(simplex_rows[key].get("filtration", 0.0) or 0.0), _simplex_key(key)))
    node_x = [positions[key][0] for key in node_keys]
    node_y = [positions[key][1] for key in node_keys]
    node_z = [positions[key][2] for key in node_keys]
    node_dim = [int(simplex_rows[key].get("dimension", len(key) - 1)) for key in node_keys]
    node_filtration = [float(simplex_rows[key].get("filtration", 0.0) or 0.0) for key in node_keys]
    node_hover: list[str] = []
    for key in node_keys:
        row = simplex_rows[key]
        faces = []
        if len(key) > 1:
            faces = [_simplex_key(tuple(sorted(face))) for face in combinations(key, len(key) - 1)]
        node_hover.append(
            f"<b>{html.escape(_simplex_key(key))}</b>"
            f"<br>dimension={int(row.get('dimension', len(key) - 1))}"
            f"<br>filtration={float(row.get('filtration', 0.0) or 0.0):.6g}"
            f"<br>poset covers from faces={html.escape(', '.join(faces[:6])) if faces else '{}'}"
            f"<br>type={html.escape(str(row.get('type', 'simplex')))}"
            f"<br>source={html.escape(str(row.get('filtration_source', 'gudhi.SimplexTree')))}"
            + (f"<br>embedding distance={float(row.get('embedding_distance')):.6g}" if isinstance(row.get("embedding_distance"), (int, float)) else "")
            + (f"<br>reasoning transition={bool(row.get('reasoning_transition'))}" if row.get("reasoning_transition") is not None else "")
            + (f"<br>{_vertex_readable_summary(row, include_output=True)}" if int(row.get("dimension", -1)) == 0 else "")
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=hasse_x,
            y=hasse_y,
            z=hasse_z,
            mode="lines",
            line=dict(width=1.65, color="rgba(94,234,212,0.46)"),
            hovertext=hasse_hover,
            hoverinfo="text",
            name=f"actual face-to-coface covers ({len(hasse_pairs)})",
        )
    )
    if prefix_pairs:
        fig.add_trace(
            go.Scatter3d(
                x=prefix_x,
                y=prefix_y,
                z=prefix_z,
                mode="lines",
                line=dict(width=1.0, color="rgba(251,191,36,0.20)", dash="dot"),
                hovertext=prefix_hover,
                hoverinfo="text",
                name=f"optional sorted-label trie prefix links ({len(prefix_pairs)})",
                visible="legendonly",
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=[positions[root_key][0]],
            y=[positions[root_key][1]],
            z=[positions[root_key][2]],
            mode="markers+text",
            marker=dict(size=10, color="#f472b6", line=dict(color="#fff7ed", width=1.1), opacity=0.78),
            text=["empty"],
            textposition="top center",
            hovertext=["<b>empty simplex</b><br>mathematical root of the simplex tree; covers every 0-simplex"],
            hoverinfo="text",
            name="empty simplex",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers",
            marker=dict(
                size=[10 if dim == 0 else 7 if dim == 1 else 5 for dim in node_dim],
                color=node_dim,
                colorscale=[[0, "#5eead4"], [0.5, "#60a5fa"], [1.0, "#facc15"]],
                showscale=True,
                colorbar=dict(title="dim", x=1.04, y=0.74, len=0.42, thickness=14),
                line=dict(color="#e8eef8", width=0.75),
                opacity=0.92,
            ),
            customdata=np.asarray(node_filtration, dtype=float),
            hovertext=node_hover,
            hoverinfo="text",
            name="simplices",
        )
    )
    summary = obj.get("summary", {}) if isinstance(obj.get("summary"), dict) else {}
    tree = obj.get("simplex_tree", {}) if isinstance(obj.get("simplex_tree"), dict) else {}
    dimension_counts: dict[str, int] = {}
    for dim in node_dim:
        key = f"dim_{int(dim)}"
        dimension_counts[key] = int(dimension_counts.get(key, 0) + 1)
    simplex_tree_poset_contract = {
        "schema_version": "tropicalgt.simplex_tree_poset.v1",
        "available": True,
        "html_file": path.name,
        "title": title,
        "source": "gudhi_canonical_complex(filtered_simplicial_object).simplex_tree",
        "backend": str(tree.get("backend", "json")),
        "safe_to_render_simplex_tree": tree.get("available") is not False,
        "actual_data_only": True,
        "no_proxy_or_fallback": True,
        "layout": "model_embedding_barycentric_face_coface_poset",
        "not_disconnected_simplex_columns": True,
        "empty_simplex_root_present": root_key in positions,
        "displayed_simplex_count": int(len(node_keys)),
        "source_simplex_count": int(tree.get("num_simplices", len(simplex_rows)) or len(simplex_rows)),
        "truncated": bool(truncated),
        "max_nodes": int(max_nodes),
        "dimension_counts": dimension_counts,
        "actual_face_to_coface_cover_edges": int(len(hasse_pairs)),
        "empty_root_vertex_cover_edges": int(sum(1 for source, target, _label in hasse_pairs if source == root_key and len(target) == 1)),
        "optional_sorted_label_trie_prefix_edges": int(len(prefix_pairs)),
        "primary_edges": "actual_face_to_coface_covers",
        "optional_prefix_links_visible": "legendonly",
        "position_source": "model_embedding_barycenters_with_dimension_and_filtration_lift",
    }
    simplex_tree_poset_contract["all_non_vertex_simplices_have_face_cover_edges"] = all(
        any(target == key and source != root_key for source, target, _label in hasse_pairs)
        for key in node_keys
        if len(key) > 1
    )
    simplex_tree_poset_contract["safe_to_render_simplex_tree"] = all(
        [
            simplex_tree_poset_contract["available"],
            simplex_tree_poset_contract["actual_data_only"],
            simplex_tree_poset_contract["no_proxy_or_fallback"],
            simplex_tree_poset_contract["backend"] == "gudhi.SimplexTree",
            simplex_tree_poset_contract["not_disconnected_simplex_columns"],
            simplex_tree_poset_contract["empty_simplex_root_present"],
            int(simplex_tree_poset_contract["displayed_simplex_count"]) > 0,
            simplex_tree_poset_contract["primary_edges"] == "actual_face_to_coface_covers",
            simplex_tree_poset_contract["all_non_vertex_simplices_have_face_cover_edges"],
        ]
    )
    _write_simplex_tree_poset_contract(path, simplex_tree_poset_contract)
    fig.update_layout(
        template="plotly_dark",
        meta={"simplex_tree_poset_contract": simplex_tree_poset_contract},
        title=(
            f"{title}<br><sup>{html.escape(subtitle)} | backend={html.escape(str(tree.get('backend', 'json')))} "
            f"| displayed={len(node_keys)}/{int(tree.get('num_simplices', len(simplex_rows)) or len(simplex_rows))} "
            f"| layout=model-embedding barycentric face/coface poset, not disconnected simplex columns; "
            f"actual cover edges are primary and optional sorted-label trie links are legend-only; {html.escape(layout_report)} "
            f"| V={summary.get('num_vertices', 0)}, E={summary.get('num_edges', 0)}, T={summary.get('num_two_simplices', 0)}"
            + (" | truncated for browser performance" if truncated else "")
            + "</sup>"
        ),
        scene=dict(
            xaxis_title="embedding barycenter PC/MDS-1",
            yaxis_title="embedding barycenter PC/MDS-2 + dimension lift",
            zaxis_title="embedding barycenter PC/MDS-3 + filtration/dim lift",
            aspectmode="cube",
            camera=dict(eye=dict(x=1.62, y=-1.7, z=1.24)),
        ),
        legend=dict(orientation="v", x=0.01, y=0.91, xanchor="left", yanchor="top", font=dict(size=10), bgcolor="rgba(2,6,23,0.72)", bordercolor="rgba(125,211,252,0.22)", borderwidth=1),
        margin=dict(t=176, l=0, r=112, b=28),
    )
    _write_plotly_dark_html(path, fig, title)

def _simplex_key(simplex: object) -> str:
    if isinstance(simplex, tuple):
        return "{" + ",".join(str(vertex) for vertex in simplex) + "}"
    if isinstance(simplex, list):
        return "{" + ",".join(str(vertex) for vertex in simplex) + "}"
    return "{" + str(simplex) + "}"


def _is_radius_filtration_complex(obj: dict[str, object]) -> bool:
    if not isinstance(obj, dict):
        return False
    summary = obj.get("summary", {}) if isinstance(obj.get("summary"), dict) else {}
    model = str(summary.get("filtration_model", "")).lower()
    if "unavailable" in model or "non_radius" in model:
        return False
    return bool(summary.get("radius_filtration")) or "vietoris_rips" in model or "radius" in model


def _initial_radius_frame_threshold(obj: dict[str, object]) -> float | None:
    if not _is_radius_filtration_complex(obj):
        return None
    thresholds = _display_thresholds(obj)
    if not thresholds:
        return None
    return float(thresholds[0])


def _is_initial_radius_frame(obj: dict[str, object], threshold: float) -> bool:
    initial = _initial_radius_frame_threshold(obj)
    if initial is None:
        return False
    try:
        value = float(threshold)
    except (TypeError, ValueError):
        return False
    return abs(value - initial) <= 1e-12


def _complex_slider_frame_contract(obj: dict[str, object], thresholds: list[float] | None = None) -> dict[str, object]:
    """Machine-readable contract for the radius slider frames.

    For true radius filtrations the first frame is a disjoint 0-simplex cloud: no
    solid edges, filled faces, or dotted order overlays are rendered there.
    Subsequent frames are monotone in visible simplices and overlays.
    """

    thresholds = list(thresholds) if thresholds is not None else _display_thresholds(obj)
    frames: list[dict[str, object]] = []
    previous = {"vertices": 0, "solid_edges": 0, "filled_faces": 0, "dotted_overlays": 0}
    monotone = True
    thresholds_ascending = all(float(a) <= float(b) + 1e-12 for a, b in zip(thresholds, thresholds[1:]))
    radius_filtration = _is_radius_filtration_complex(obj)
    for threshold in thresholds:
        initial_radius_frame = _is_initial_radius_frame(obj, float(threshold))
        vertices = [
            s for s in obj.get("simplices", [])
            if isinstance(s, dict)
            and int(s.get("dimension", -1)) == 0
            and (radius_filtration or float(s.get("filtration", 0.0) or 0.0) <= float(threshold) + 1e-12)
        ]
        visible = {str((row.get("simplex") or [""])[0]) for row in vertices}
        solid_edges = [] if initial_radius_frame else [
            s for s in obj.get("simplices", [])
            if isinstance(s, dict)
            and int(s.get("dimension", -1)) == 1
            and float(s.get("filtration", 0.0) or 0.0) <= float(threshold) + 1e-12
            and all(str(v) in visible for v in (s.get("simplex") or [])[:2])
        ]
        filled_faces = [] if initial_radius_frame else [
            s for s in obj.get("simplices", [])
            if isinstance(s, dict)
            and int(s.get("dimension", -1)) == 2
            and float(s.get("filtration", 0.0) or 0.0) <= float(threshold) + 1e-12
            and all(str(v) in visible for v in (s.get("simplex") or [])[:3])
        ]

        def overlay_visible(edge: object) -> bool:
            if initial_radius_frame or not isinstance(edge, dict):
                return False
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            if source not in visible or target not in visible:
                return False
            try:
                filt = float(edge.get("filtration", edge.get("radius", 0.0)) or 0.0)
            except (TypeError, ValueError):
                filt = 0.0
            return filt <= float(threshold) + 1e-12

        trajectory_overlay = obj.get("trajectory_overlay", {}) if isinstance(obj.get("trajectory_overlay"), dict) else {}
        direction_overlay = obj.get("graph_token_direction_overlay", {}) if isinstance(obj.get("graph_token_direction_overlay"), dict) else {}
        decoding_overlay = obj.get("decoding_causal_overlay", {}) if isinstance(obj.get("decoding_causal_overlay"), dict) else {}
        trajectory_count = sum(1 for edge in trajectory_overlay.get("edges", []) if overlay_visible(edge)) if isinstance(trajectory_overlay.get("edges"), list) else 0
        direction_count = sum(1 for edge in direction_overlay.get("edges", []) if overlay_visible(edge)) if isinstance(direction_overlay.get("edges"), list) else 0
        decoding_count = sum(1 for edge in decoding_overlay.get("edges", []) if overlay_visible(edge)) if isinstance(decoding_overlay.get("edges"), list) else 0
        dotted_total = trajectory_count + direction_count + decoding_count
        row = {
            "threshold": float(threshold),
            "initial_radius_frame": bool(initial_radius_frame),
            "vertices": len(vertices),
            "solid_edges": len(solid_edges),
            "filled_faces": len(filled_faces),
            "dotted_trajectory_overlays": int(trajectory_count),
            "dotted_direction_overlays": int(direction_count),
            "dotted_decoding_overlays": int(decoding_count),
            "dotted_overlays": int(dotted_total),
        }
        for key in previous:
            if row[key] < previous[key]:
                monotone = False
        previous = {key: int(row[key]) for key in previous}
        frames.append(row)
    first_frame = frames[0] if frames else {}
    last_frame = frames[-1] if frames else {}
    return {
        "schema_version": "tropicalgt.radius_filtration_slider_contract.v1",
        "source": "canonical_gudhi_filtered_complex_simplices",
        "actual_data_only": True,
        "no_proxy_or_fallback": True,
        "radius_filtration": radius_filtration,
        "threshold_order": "ascending_min_to_max",
        "thresholds_ascending": bool(thresholds_ascending),
        "threshold_count": int(len(thresholds)),
        "first_threshold": float(thresholds[0]) if thresholds else None,
        "last_threshold": float(thresholds[-1]) if thresholds else None,
        "frame_count": int(len(frames)),
        "first_frame_vertex_count": int(first_frame.get("vertices", 0) or 0),
        "first_frame_solid_edge_count": int(first_frame.get("solid_edges", 0) or 0),
        "first_frame_filled_face_count": int(first_frame.get("filled_faces", 0) or 0),
        "first_frame_dotted_overlay_count": int(first_frame.get("dotted_overlays", 0) or 0),
        "last_frame_solid_edge_count": int(last_frame.get("solid_edges", 0) or 0),
        "last_frame_filled_face_count": int(last_frame.get("filled_faces", 0) or 0),
        "solid_lines_semantics": "radius-filtered 1-simplices only",
        "filled_faces_semantics": "radius-gated 2-simplices only",
        "dotted_lines_semantics": "causal_decoding_or_direction_overlay_only_and_radius_gated",
        "initial_radius_frame_hides_dotted_overlays": True,
        "initial_radius_frame_hides_solid_edges_and_faces": True,
        "first_frame_disjoint_vertices_only": (
            bool(frames)
            and bool(frames[0].get("initial_radius_frame"))
            and int(frames[0].get("solid_edges", 0)) == 0
            and int(frames[0].get("filled_faces", 0)) == 0
            and int(frames[0].get("dotted_overlays", 0)) == 0
        ),
        "monotone_visible_counts": bool(monotone),
        "monotone_solid_radius_edges": bool(monotone),
        "monotone_filled_radius_faces": bool(monotone),
        "frames": frames,
    }


def _gudhi_canonical_complex(obj: dict[str, object]) -> dict[str, object]:
    """Canonicalize a JSON filtered complex through GUDHI SimplexTree when available."""
    if not isinstance(obj, dict):
        return obj
    raw_simplices = [row for row in obj.get("simplices", []) if isinstance(row, dict) and row.get("simplex")]
    if not raw_simplices:
        return obj
    try:
        import gudhi  # type: ignore

        labels = sorted({str(vertex) for row in raw_simplices for vertex in (row.get("simplex") or [])})
        label_to_int = {label: idx for idx, label in enumerate(labels)}
        int_to_label = {idx: label for label, idx in label_to_int.items()}
        tree = gudhi.SimplexTree()
        metadata: dict[tuple[str, ...], dict[str, object]] = {}
        for row in raw_simplices:
            simplex = [str(vertex) for vertex in (row.get("simplex") or [])]
            if not simplex:
                continue
            filt = float(row.get("filtration", 0.0) or 0.0)
            tree.insert([label_to_int[label] for label in simplex], filtration=filt)
            metadata[tuple(sorted(simplex))] = row
        tree.make_filtration_non_decreasing()
        canonical = []
        thresholds: set[float] = set()
        closure_inserted_count = 0
        for simplex_ints, filtration in tree.get_filtration():
            simplex = [int_to_label[int(vertex)] for vertex in simplex_ints]
            key = tuple(sorted(simplex))
            closure_inserted = key not in metadata
            if closure_inserted:
                closure_inserted_count += 1
            base = dict(metadata.get(key, {}))
            base.update(
                {
                    "simplex": simplex,
                    "dimension": len(simplex) - 1,
                    "filtration": float(filtration),
                    "gudhi_simplex_tree": True,
                    "gudhi_closure_inserted": bool(closure_inserted),
                    "type": base.get("type", f"gudhi_closure_dim_{len(simplex) - 1}" if closure_inserted else f"gudhi_dim_{len(simplex) - 1}"),
                    "filtration_source": base.get("filtration_source", "gudhi_simplex_tree_closure" if closure_inserted else "gudhi_simplex_tree"),
                }
            )
            canonical.append(base)
            thresholds.add(float(filtration))
        summary = dict(obj.get("summary", {})) if isinstance(obj.get("summary"), dict) else {}
        summary.update(
            {
                "num_vertices": sum(1 for row in canonical if int(row.get("dimension", -1)) == 0),
                "num_edges": sum(1 for row in canonical if int(row.get("dimension", -1)) == 1),
                "num_two_simplices": sum(1 for row in canonical if int(row.get("dimension", -1)) == 2),
                "num_thresholds": len(thresholds),
                "simplex_tree_backend": "gudhi.SimplexTree",
                "simplex_tree_available": True,
                "simplex_tree_closure_inserted_simplices": int(closure_inserted_count),
            }
        )
        return {
            **obj,
            "summary": summary,
            "thresholds": sorted(thresholds),
            "simplices": canonical,
            "simplex_tree": {
                "backend": "gudhi.SimplexTree",
                "available": True,
                "num_vertices": int(tree.num_vertices()),
                "num_simplices": int(tree.num_simplices()),
                "dimension": int(tree.dimension()),
                "filtration_non_decreasing": True,
                "closure_inserted_simplices": int(closure_inserted_count),
            },
        }
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        summary = dict(obj.get("summary", {})) if isinstance(obj.get("summary"), dict) else {}
        summary.update(
            {
                "simplex_tree_backend": "unavailable_gudhi_simplex_tree",
                "simplex_tree_available": False,
                "simplex_tree_unavailable_reason": error,
                "simplex_tree_no_proxy_or_fallback": True,
            }
        )
        serialized_obj = dict(obj)
        serialized_obj["summary"] = summary
        serialized_obj["simplex_tree"] = {
            "backend": "unavailable_gudhi_simplex_tree",
            "available": False,
            "reason": "gudhi_simplex_tree_unavailable",
            "error": error,
            "safe_to_render_simplex_tree": False,
            "no_proxy_or_fallback": True,
        }
        return serialized_obj


def _complex_slider_traces(
    obj: dict[str, object],
    coords3: dict[str, tuple[float, float, float]],
    threshold: float,
    panel_index_by_label: dict[str, int] | None = None,
) -> list[go.Scatter3d | go.Mesh3d]:
    radius_vertices_enter_at_zero = _is_radius_filtration_complex(obj)
    vertices = [
        s for s in obj.get("simplices", [])
        if isinstance(s, dict)
        and int(s.get("dimension", -1)) == 0
        and (radius_vertices_enter_at_zero or float(s.get("filtration", 0.0) or 0.0) <= threshold + 1e-12)
    ]
    visible = {str((row.get("simplex") or [""])[0]) for row in vertices}
    initial_radius_frame = _is_initial_radius_frame(obj, threshold)
    edges = [] if initial_radius_frame else [
        s for s in obj.get("simplices", [])
        if isinstance(s, dict)
        and int(s.get("dimension", -1)) == 1
        and float(s.get("filtration", 0.0) or 0.0) <= threshold + 1e-12
        and all(str(v) in visible for v in (s.get("simplex") or [])[:2])
    ]
    triangles = [] if initial_radius_frame else [
        s for s in obj.get("simplices", [])
        if isinstance(s, dict)
        and int(s.get("dimension", -1)) == 2
        and float(s.get("filtration", 0.0) or 0.0) <= threshold + 1e-12
        and all(str(v) in visible for v in (s.get("simplex") or [])[:3])
    ]
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    edge_hover: list[str | None] = []
    for edge in edges:
        simplex = edge.get("simplex") or []
        if len(simplex) < 2:
            continue
        a, b = str(simplex[0]), str(simplex[1])
        if a not in coords3 or b not in coords3:
            continue
        ax, ay, az = coords3[a]
        bx, by, bz = coords3[b]
        label = f"{html.escape(a)} -> {html.escape(b)}<br>filtration={float(edge.get('filtration', 0.0) or 0.0):.4f}<br>type={html.escape(str(edge.get('type', 'edge')))}"
        edge_x.extend([ax, bx, None])
        edge_y.extend([ay, by, None])
        edge_z.extend([az, bz, None])
        edge_hover.extend([label, label, None])
    overlay = obj.get("trajectory_overlay", {}) if isinstance(obj.get("trajectory_overlay"), dict) else {}
    overlay_x: list[float | None] = []
    overlay_y: list[float | None] = []
    overlay_z: list[float | None] = []
    overlay_hover: list[str | None] = []
    for overlay_edge in ([] if initial_radius_frame else (overlay.get("edges", []) if isinstance(overlay.get("edges"), list) else [])):
        if not isinstance(overlay_edge, dict):
            continue
        source = str(overlay_edge.get("source", ""))
        target = str(overlay_edge.get("target", ""))
        if source not in coords3 or target not in coords3 or source not in visible or target not in visible:
            continue
        try:
            overlay_filtration = float(overlay_edge.get("filtration", overlay_edge.get("radius", 0.0)) or 0.0)
        except (TypeError, ValueError):
            overlay_filtration = 0.0
        if _is_radius_filtration_complex(obj) and overlay_filtration > threshold + 1e-12:
            continue
        sx, sy, sz = coords3[source]
        tx, ty, tz = coords3[target]
        source_nll = overlay_edge.get("source_nll")
        target_nll = overlay_edge.get("target_nll")
        nll_line = ""
        if isinstance(source_nll, (int, float)) and isinstance(target_nll, (int, float)):
            nll_line = f"<br>NLL delta={float(target_nll) - float(source_nll):+.6g}"
        label = (
            f"GoT parent-child trajectory edge<br>{html.escape(source)} -> {html.escape(target)}"
            f"<br>action={html.escape(str(overlay_edge.get('action', 'transition')))}"
            f"<br>metric={html.escape(str(overlay.get('distance_metric', 'euclidean')))}"
            f"{nll_line}<br>overlay source={html.escape(str(overlay.get('source', 'graph_of_thought_parent_edges')))}"
        )
        overlay_x.extend([sx, tx, None])
        overlay_y.extend([sy, ty, None])
        overlay_z.extend([sz, tz, None])
        overlay_hover.extend([label, label, None])
    direction_overlay = obj.get("graph_token_direction_overlay", {}) if isinstance(obj.get("graph_token_direction_overlay"), dict) else {}
    direction_x: list[float | None] = []
    direction_y: list[float | None] = []
    direction_z: list[float | None] = []
    direction_hover: list[str | None] = []
    for directed_edge in ([] if initial_radius_frame else (direction_overlay.get("edges", []) if isinstance(direction_overlay.get("edges"), list) else [])):
        if not isinstance(directed_edge, dict):
            continue
        source = str(directed_edge.get("source", ""))
        target = str(directed_edge.get("target", ""))
        if source not in coords3 or target not in coords3 or source not in visible or target not in visible:
            continue
        sx, sy, sz = coords3[source]
        tx, ty, tz = coords3[target]
        margin = directed_edge.get("margin")
        margin_line = f"<br>model margin={float(margin):.6g}" if isinstance(margin, (int, float)) else ""
        label = (
            f"directed GoT graph-token edge<br>{html.escape(source)} -> {html.escape(target)}"
            f"<br>role={html.escape(str(directed_edge.get('role', 'graph-token-direction')))}"
            f"<br>edge token={html.escape(str(directed_edge.get('edge_token', '')))}"
            f"<br>edge type={html.escape(str(directed_edge.get('edge_type', 'graph_edge')))}"
            f"<br>source node={html.escape(str(directed_edge.get('source_node_id', '')))}"
            f"<br>target node={html.escape(str(directed_edge.get('target_node_id', '')))}"
            f"{margin_line}<br>overlay source={html.escape(str(direction_overlay.get('source', 'graph_token_trace_directed_edges')))}"
        )
        direction_x.extend([sx, tx, None])
        direction_y.extend([sy, ty, None])
        direction_z.extend([sz, tz, None])
        direction_hover.extend([label, label, None])

    def append_dotted_segment(
        xs: list[float | None],
        ys: list[float | None],
        zs: list[float | None],
        hovers: list[str | None],
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        label: str,
        segments: int = 10,
        duty: float = 0.46,
    ) -> None:
        ax, ay, az = start
        bx, by, bz = end
        for segment in range(max(segments, 1)):
            t0 = segment / max(segments, 1)
            t1 = min((segment + duty) / max(segments, 1), 1.0)
            xs.extend([ax + (bx - ax) * t0, ax + (bx - ax) * t1, None])
            ys.extend([ay + (by - ay) * t0, ay + (by - ay) * t1, None])
            zs.extend([az + (bz - az) * t0, az + (bz - az) * t1, None])
            hovers.extend([label, label, None])

    decoding_overlay = obj.get("decoding_causal_overlay", {}) if isinstance(obj.get("decoding_causal_overlay"), dict) else {}
    decoding_x: list[float | None] = []
    decoding_y: list[float | None] = []
    decoding_z: list[float | None] = []
    decoding_hover: list[str | None] = []
    decoding_marker_x: list[float] = []
    decoding_marker_y: list[float] = []
    decoding_marker_z: list[float] = []
    decoding_marker_hover: list[str] = []
    decoding_marker_color: list[str] = []
    for directed_edge in ([] if initial_radius_frame else (decoding_overlay.get("edges", []) if isinstance(decoding_overlay.get("edges"), list) else [])):
        if not isinstance(directed_edge, dict):
            continue
        source = str(directed_edge.get("source", ""))
        target = str(directed_edge.get("target", ""))
        try:
            edge_filtration = float(directed_edge.get("filtration", 0.0) or 0.0)
        except (TypeError, ValueError):
            edge_filtration = 0.0
        if edge_filtration > threshold + 1e-12:
            continue
        if source not in coords3 or target not in coords3 or source not in visible or target not in visible:
            continue
        sx, sy, sz = coords3[source]
        tx, ty, tz = coords3[target]
        color = str(directed_edge.get("color", "rgba(94,234,212,0.78)"))
        label = (
            f"dotted causal/decoding overlay<br>{html.escape(source)} -> {html.escape(target)}"
            f"<br>role={html.escape(str(directed_edge.get('role', 'decoding_order')))}"
            f"<br>edge type={html.escape(str(directed_edge.get('edge_type', 'order_edge')))}"
            f"<br>source node={html.escape(str(directed_edge.get('source_node_id', '')))}"
            f"<br>target node={html.escape(str(directed_edge.get('target_node_id', '')))}"
            f"<br>decoding step={html.escape(str(directed_edge.get('decoding_step', '')))}"
            f"<br>reasoning level={html.escape(str(directed_edge.get('reasoning_level', '')))}"
            f"<br>gate radius={edge_filtration:.6g}"
            f"<br>overlay source={html.escape(str(decoding_overlay.get('source', 'GraphRecord.metadata.graph_decoding_order')))}"
        )
        append_dotted_segment(decoding_x, decoding_y, decoding_z, decoding_hover, (sx, sy, sz), (tx, ty, tz), label)
        decoding_marker_x.append(tx)
        decoding_marker_y.append(ty)
        decoding_marker_z.append(tz)
        decoding_marker_hover.append(label)
        decoding_marker_color.append(color)
    labels = [str((row.get("simplex") or [""])[0]) for row in vertices]
    vertex_x = [coords3[label][0] for label in labels if label in coords3]
    vertex_y = [coords3[label][1] for label in labels if label in coords3]
    vertex_z = [coords3[label][2] for label in labels if label in coords3]
    vertex_labels = [label for label in labels if label in coords3]
    vertex_by_label = {str((row.get("simplex") or [""])[0]): row for row in vertices}
    vertex_hover = []
    for label in vertex_labels:
        vertex_row = vertex_by_label.get(label, {})
        vertex_hover.append(_vertex_readable_summary(vertex_row, include_output=True))
    show_text = len(vertex_labels) <= 18
    traces: list[go.Scatter3d | go.Mesh3d] = [
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(width=2.2, color="rgba(125,211,252,0.42)"),
            hovertext=edge_hover,
            hoverinfo="text",
            name="solid radius/simplicial edges induced from the same embeddings",
        ),
        go.Scatter3d(
            x=overlay_x,
            y=overlay_y,
            z=overlay_z,
            mode="lines",
            line=dict(width=2.2, color="rgba(250,204,21,0.30)"),
            hovertext=overlay_hover,
            hoverinfo="text",
            name="faint GoT parent-child trajectory overlay",
        ),
        go.Scatter3d(
            x=direction_x,
            y=direction_y,
            z=direction_z,
            mode="lines",
            line=dict(width=1.8, color="rgba(244,114,182,0.28)"),
            hovertext=direction_hover,
            hoverinfo="text",
            name="faint directed graph-token overlay",
        ),
        go.Scatter3d(
            x=decoding_x,
            y=decoding_y,
            z=decoding_z,
            mode="lines",
            line=dict(width=3.0, color="rgba(255,255,255,0.68)"),
            hovertext=decoding_hover,
            hoverinfo="text",
            name="dotted causal/decoding order overlay",
        ),
        go.Scatter3d(
            x=decoding_marker_x,
            y=decoding_marker_y,
            z=decoding_marker_z,
            mode="markers",
            marker=dict(size=4.5, color=decoding_marker_color, symbol="diamond", line=dict(color="#0f172a", width=0.6)),
            hovertext=decoding_marker_hover,
            hoverinfo="text",
            name="causal/decoding arrowheads",
        ),
        go.Scatter3d(
            x=vertex_x,
            y=vertex_y,
            z=vertex_z,
            mode="markers+text",
            marker=dict(
                size=7,
                color=[float(vertex_by_label.get(label, {}).get("filtration", 0.0) or 0.0) for label in vertex_labels],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="filtration", x=1.03, y=0.48, len=0.62, thickness=16),
                line=dict(color="#e8eef8", width=1),
            ),
            text=[
                _short_complex_vertex_label(label, vertex_by_label.get(label, {}), pos) if show_text else ""
                for pos, label in enumerate(vertex_labels)
            ],
            textposition="top center",
            textfont=dict(size=9, color="#dbeafe"),
            hovertext=vertex_hover,
            hoverinfo="text",
            customdata=[int((panel_index_by_label or {}).get(label, 0)) for label in vertex_labels],
            name="0-simplices",
        ),
    ]
    mesh_vertices: dict[str, int] = {}
    mesh_x: list[float] = []
    mesh_y: list[float] = []
    mesh_z: list[float] = []
    tri_i: list[int] = []
    tri_j: list[int] = []
    tri_k: list[int] = []
    for tri in triangles[:400]:
        simplex = [str(v) for v in (tri.get("simplex") or [])[:3]]
        if len(simplex) < 3 or any(label not in coords3 for label in simplex):
            continue
        indices = []
        for label in simplex:
            if label not in mesh_vertices:
                mesh_vertices[label] = len(mesh_x)
                x, y, z = coords3[label]
                mesh_x.append(x)
                mesh_y.append(y)
                mesh_z.append(z)
            indices.append(mesh_vertices[label])
        tri_i.append(indices[0]); tri_j.append(indices[1]); tri_k.append(indices[2])
    traces.append(
        go.Mesh3d(
            x=mesh_x,
            y=mesh_y,
            z=mesh_z,
            i=tri_i,
            j=tri_j,
            k=tri_k,
            color="rgba(94,234,212,0.14)",
            opacity=0.18,
            name="filled 2-simplices gated by radius slider",
            hoverinfo="skip",
            showscale=False,
        )
    )
    return traces


def _short_complex_vertex_label(label: str, vertex: dict[str, object] | None = None, idx: int = 0) -> str:
    vertex = vertex or {}
    vertex_type = str(vertex.get("type", "") or "").lower()
    action = vertex.get("action") or vertex.get("step_action") or vertex.get("operator")
    level = vertex.get("level")
    if action:
        prefix = str(action).replace("_", "-")[:10]
    elif "problem" in vertex_type:
        prefix = "problem"
    elif "graph" in vertex_type:
        prefix = "graph"
    elif "reason" in vertex_type or "step" in vertex_type:
        prefix = "step"
    elif "seq" in vertex_type:
        prefix = "seq"
    else:
        prefix = "v"
    if isinstance(level, (int, float)) and math.isfinite(float(level)):
        return f"{prefix} L{int(level)}"
    if label.startswith("seq_") or label.startswith("step_") or label.startswith("expand_"):
        return label[:18]
    return f"{prefix}{idx:02d}"


def _display_thresholds(obj: dict[str, object], max_steps: int = 32) -> list[float]:
    raw = obj.get("thresholds", []) if isinstance(obj, dict) else []
    values = sorted({float(v) for v in raw if isinstance(v, (int, float)) and math.isfinite(float(v))})
    if not values:
        values = sorted(
            {
                float(row.get("filtration", 0.0) or 0.0)
                for row in obj.get("simplices", [])
                if isinstance(row, dict) and isinstance(row.get("filtration", 0.0), (int, float))
            }
        )
    if not values:
        return [1.0]
    if len(values) <= max_steps:
        return values
    keep = np.linspace(0, len(values) - 1, max_steps, dtype=int)
    return [values[int(idx)] for idx in keep]


def _reasoning_step_complex_manifest_contract(rows: list[dict[str, object]]) -> dict[str, object]:
    simplex_tree_available = [row for row in rows if bool(row.get("simplex_tree_available"))]
    slider_rows = [row.get("radius_slider_contract") for row in rows if isinstance(row.get("radius_slider_contract"), dict)]
    poset_rows = [row.get("simplex_tree_poset_contract") for row in rows if isinstance(row.get("simplex_tree_poset_contract"), dict)]
    source_rows = [row.get("step_complex_source_contract") for row in rows if isinstance(row.get("step_complex_source_contract"), dict)]
    source_unavailable = [
        {
            "index": int(row.get("index", 0) or 0),
            "record_id": str(row.get("record_id", "")),
            "reason": "missing_or_unsafe_step_complex_source_contract",
        }
        for row in rows
        if not (isinstance(row.get("step_complex_source_contract"), dict) and row.get("step_complex_source_contract", {}).get("safe_to_render_as_step_complex") is True)
    ]
    slider_unavailable = [
        {
            "index": int(row.get("index", 0) or 0),
            "record_id": str(row.get("record_id", "")),
            "reason": "missing_or_unsafe_radius_slider_contract",
            "slider_contract_file": str(row.get("slider_contract_file", "")),
        }
        for row in rows
        if not (isinstance(row.get("radius_slider_contract"), dict) and row.get("radius_slider_contract", {}).get("safe_to_render_radius_filtration") is True)
    ]
    poset_unavailable = [
        {
            "index": int(row.get("index", 0) or 0),
            "record_id": str(row.get("record_id", "")),
            "reason": str((row.get("simplex_tree_poset_contract") if isinstance(row.get("simplex_tree_poset_contract"), dict) else {}).get("reason", "missing_or_unsafe_simplex_tree_poset_contract")),
            "simplex_tree_poset_contract_file": str(row.get("simplex_tree_poset_contract_file", "")),
        }
        for row in rows
        if not (isinstance(row.get("simplex_tree_poset_contract"), dict) and row.get("simplex_tree_poset_contract", {}).get("safe_to_render_simplex_tree") is True)
    ]
    fingerprint_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        fingerprint = str(row.get("step_complex_fingerprint", ""))
        if fingerprint:
            fingerprint_groups[fingerprint].append(row)
    duplicate_fingerprint_groups = [
        {
            "fingerprint": fingerprint,
            "step_indices": [int(row.get("index", 0) or 0) for row in group],
            "record_ids": [str(row.get("record_id", "")) for row in group],
            "allowed_only_if_canonical_filtered_complex_payload_identical": True,
        }
        for fingerprint, group in sorted(fingerprint_groups.items())
        if len(group) > 1
    ]
    unavailable = []
    for row in rows:
        if bool(row.get("simplex_tree_available")):
            continue
        tree = row.get("simplex_tree") if isinstance(row.get("simplex_tree"), dict) else {}
        unavailable.append(
            {
                "index": int(row.get("index", 0) or 0),
                "record_id": str(row.get("record_id", "")),
                "backend": str(row.get("simplex_tree_backend", tree.get("backend", "missing"))),
                "reason": str(tree.get("reason") or tree.get("error") or "simplex_tree_unavailable"),
            }
        )
    return {
        "schema_version": "tropicalgt.reasoning_step_complex_maps.v1",
        "available": bool(rows),
        "no_proxy_or_fallback": True,
        "actual_data_only": True,
        "one_page_per_model_evaluated_reasoning_step": True,
        "complex_source": "filtered_simplicial_object on each observed model-evaluated GoT state",
        "complex_filtration": "radius/filtration values from the serialized filtered simplicial object",
        "simplex_tree_source": "serialized GUDHI SimplexTree attached to each filtered simplicial object when available",
        "embedding_trajectory_map_is_not_a_step_complex": True,
        "step_count": int(len(rows)),
        "rendered_complex_pages": int(len([row for row in rows if row.get("file")])),
        "rendered_simplex_tree_pages": int(len([row for row in rows if row.get("simplex_tree_file")])),
        "simplex_tree_poset_contract_schema_version": "tropicalgt.simplex_tree_poset.v1",
        "simplex_tree_poset_contract_source": "per-step reasoning_step_*_simplex_tree_poset_contract.json sidecars summarized into this manifest",
        "rendered_simplex_tree_poset_contracts": int(len(poset_rows)),
        "all_steps_have_simplex_tree_poset_contracts": bool(rows) and len(poset_rows) == len(rows),
        "all_step_simplex_tree_posets_no_proxy": bool(rows)
        and all(row.get("actual_data_only") is True and row.get("no_proxy_or_fallback") is True for row in poset_rows),
        "all_step_simplex_tree_posets_use_gudhi": bool(rows)
        and all(row.get("backend") == "gudhi.SimplexTree" for row in poset_rows),
        "all_step_simplex_tree_posets_face_coface_primary": bool(rows)
        and all(row.get("primary_edges") == "actual_face_to_coface_covers" for row in poset_rows),
        "all_step_simplex_tree_posets_safe_to_render": bool(rows)
        and len(poset_unavailable) == 0
        and all(row.get("safe_to_render_simplex_tree") is True for row in poset_rows),
        "simplex_tree_poset_unavailable_count": int(len(poset_unavailable)),
        "simplex_tree_poset_unavailable_steps": poset_unavailable,
        "slider_contract_schema_version": "tropicalgt.reasoning_step_radius_slider_summary.v1",
        "radius_slider_contract_source": "per-step reasoning_step_*_slider_contract.json sidecars summarized into this manifest",
        "rendered_slider_contracts": int(len(slider_rows)),
        "all_steps_have_radius_slider_contracts": bool(rows) and len(slider_rows) == len(rows),
        "all_step_radius_sliders_start_disjoint_vertices": bool(rows)
        and all(row.get("first_frame_disjoint_vertices_only") is True for row in slider_rows),
        "all_step_radius_sliders_monotone": bool(rows)
        and all(
            row.get("monotone_visible_counts") is True
            and row.get("monotone_solid_radius_edges") is True
            and row.get("monotone_filled_radius_faces") is True
            for row in slider_rows
        ),
        "all_step_radius_sliders_no_proxy": bool(rows)
        and all(row.get("actual_data_only") is True and row.get("no_proxy_or_fallback") is True for row in slider_rows),
        "all_step_radius_sliders_safe_to_render": bool(rows)
        and len(slider_unavailable) == 0
        and all(row.get("safe_to_render_radius_filtration") is True for row in slider_rows),
        "radius_slider_unavailable_count": int(len(slider_unavailable)),
        "radius_slider_unavailable_steps": slider_unavailable,
        "source_contract_schema_version": "tropicalgt.reasoning_step_complex_source_contract.v1",
        "source_contract_source": "candidate.filtered_simplicial_object on each manifest row",
        "rendered_source_contracts": int(len(source_rows)),
        "all_steps_have_source_contracts": bool(rows) and len(source_rows) == len(rows),
        "all_step_complexes_use_candidate_filtered_object_source": bool(rows)
        and all(row.get("source") == "candidate.filtered_simplicial_object" for row in source_rows),
        "all_step_complex_source_contracts_no_proxy": bool(rows)
        and all(
            row.get("actual_data_only") is True
            and row.get("no_proxy_or_fallback") is True
            and row.get("uses_global_trajectory_complex_as_proxy") is False
            and row.get("uses_embedding_trajectory_map_as_proxy") is False
            and row.get("uses_static_probability_complex_as_proxy") is False
            for row in source_rows
        ),
        "all_step_complex_source_contracts_safe": bool(rows)
        and len(source_unavailable) == 0
        and all(row.get("safe_to_render_as_step_complex") is True for row in source_rows),
        "all_step_complex_source_counts_match_summary": bool(rows)
        and all(row.get("source_counts_match_canonical_summary") is True for row in source_rows),
        "all_step_complexes_have_vertices": bool(rows) and all(int(row.get("displayed_vertex_count", 0) or 0) > 0 for row in source_rows),
        "source_contract_unavailable_count": int(len(source_unavailable)),
        "source_contract_unavailable_steps": source_unavailable,
        "fingerprint_source": "sha256 canonical JSON over per-step gudhi_canonical_complex(filtered_simplicial_object); record id/path excluded",
        "all_step_complex_fingerprints_present": bool(rows) and all(bool(row.get("step_complex_fingerprint")) for row in rows),
        "unique_step_complex_fingerprint_count": int(len(fingerprint_groups)),
        "all_step_complex_fingerprints_unique": bool(len(fingerprint_groups) == len(rows)),
        "duplicate_step_complex_fingerprint_groups": duplicate_fingerprint_groups,
        "gudhi_simplex_tree_step_count": int(len(simplex_tree_available)),
        "simplex_tree_unavailable_count": int(len(unavailable)),
        "simplex_tree_unavailable_steps": unavailable,
        "claim": "Each listed step opens its own radius-filtered complex and SimplexTree/explicit-unavailable page; no global trajectory PCA surface is used as a substitute.",
    }


def _reasoning_step_contract_panel(contract: Mapping[str, object] | None) -> str:
    if not isinstance(contract, Mapping):
        return ""
    return (
        "<aside class='contract'>"
        "<div><span class='badge'>no proxy</span><span class='badge'>per-step complex</span><span class='badge'>radius slider verified</span><span class='badge'>simplex-tree explicit</span></div>"
        f"<p><strong>Status:</strong> rendered {int(contract.get('rendered_complex_pages', 0) or 0)} complex pages, "
        f"{int(contract.get('rendered_slider_contracts', 0) or 0)} radius-slider contract summaries, and "
        f"{int(contract.get('rendered_simplex_tree_pages', 0) or 0)} simplex-tree pages, and "
        f"{int(contract.get('rendered_simplex_tree_poset_contracts', 0) or 0)} simplex-tree poset contracts for "
        f"{int(contract.get('step_count', 0) or 0)} model-evaluated reasoning states.</p>"
        f"<p><strong>Contract:</strong> {html.escape(str(contract.get('claim', '')))} "
        f"GUDHI SimplexTree available for {int(contract.get('gudhi_simplex_tree_step_count', 0) or 0)} step(s); "
        f"explicitly unavailable for {int(contract.get('simplex_tree_unavailable_count', 0) or 0)}. "
        f"Auditable poset sidecars safe={html.escape(str(contract.get('all_step_simplex_tree_posets_safe_to_render', False)))}.</p>"
        f"<p><strong>Step source:</strong> candidate-owned filtered objects={html.escape(str(contract.get('all_step_complexes_use_candidate_filtered_object_source', False)))}; "
        f"no trajectory/static proxy={html.escape(str(contract.get('all_step_complex_source_contracts_no_proxy', False)))}; "
        f"safe source contracts={html.escape(str(contract.get('all_step_complex_source_contracts_safe', False)))}.</p>"
        f"<p><strong>Radius sliders:</strong> first frame disjoint vertices={html.escape(str(contract.get('all_step_radius_sliders_start_disjoint_vertices', False)))}; "
        f"monotone growth={html.escape(str(contract.get('all_step_radius_sliders_monotone', False)))}; "
        f"safe summaries={html.escape(str(contract.get('all_step_radius_sliders_safe_to_render', False)))}.</p>"
        f"<p><strong>Per-step fingerprints:</strong> {int(contract.get('unique_step_complex_fingerprint_count', 0) or 0)} unique canonical complex payload hash(es); "
        f"duplicates are listed only when the canonical filtered-complex payload is byte-identical after normalization.</p>"
        "</aside>"
    )


def _write_reasoning_step_complex_index(path: Path, rows: list[dict[str, object]], *, contract: Mapping[str, object] | None = None) -> None:
    contract_panel = _reasoning_step_contract_panel(contract)
    body = "\n".join(
        "<tr>"
        f"<td>{int(row['index'])}</td>"
        f"<td><a href='{html.escape(str(row['file']))}'>{html.escape(str(row['record_id']))}</a></td>"
        f"<td><a href='{html.escape(str(row.get('simplex_tree_file', '')))}'>simplex tree</a></td>"
        f"<td>{int(row.get('level', 0))}</td>"
        f"<td>{html.escape(_json_clip(row.get('path', []), 96))}</td>"
        f"<td>{html.escape(_json_clip(row.get('summary', {}), 140))}<br><small>fingerprint={html.escape(str(row.get('step_complex_fingerprint', 'missing'))[:16])}</small></td>"
        "</tr>"
        for row in rows
    ) or "<tr><td colspan='6'>No reasoning-step complexes were generated.</td></tr>"
    path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reasoning step simplicial complex maps</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin: 0; background: #090b12; color: #e8eef8; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px; }}
    table {{ width: 100%; border-collapse: collapse; background: #101623; border: 1px solid rgba(148,163,184,0.25); }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid rgba(148,163,184,0.15); text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ color: #99f6e4; }}
    a {{ color: #7dd3fc; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    p {{ color: #9fb3c8; line-height: 1.55; }}
    .contract {{ border: 1px solid rgba(125,211,252,0.28); background: #0d1626; padding: 14px 16px; margin: 18px 0; }}
    .contract p {{ margin: 8px 0 0; }}
    .badge {{ display: inline-block; margin: 0 8px 8px 0; padding: 3px 8px; border: 1px solid rgba(153,246,228,0.35); color: #99f6e4; font-size: 11px; text-transform: uppercase; letter-spacing: 0; }}
  </style>
</head>
<body>
  <main>
    <h1>Reasoning step filtered simplicial complex maps</h1>
    <p>Each row opens a separate 3D PCoA/MDS radius-filtered complex for one observed model-evaluated graph-of-thought state. These are deliberately separate from the embedding-space trajectory map and are not reconstructed from the global trajectory PCA surface.</p>
    {contract_panel}
    <table>
      <thead><tr><th>index</th><th>complex map</th><th>simplex tree</th><th>level</th><th>path</th><th>summary</th></tr></thead>
      <tbody>{body}</tbody>
    </table>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def _got_microstep_entries(
    candidates: list[dict[str, object]],
    ids: list[str],
    id_to_idx: dict[str, int],
    pca: np.ndarray,
    nll_values: np.ndarray,
    candidate_objects: list[dict[str, object]],
    panel_objects: list[dict[str, object]],
    panel_hover: list[str],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    x_span = float(np.ptp(pca[:, 0])) if pca.size else 0.0
    y_span = float(np.ptp(pca[:, 1])) if pca.size else 0.0
    offset_scale = max(x_span, y_span, 1e-3) * 0.075
    for idx, row in enumerate(candidates):
        parent = row.get("parent")
        if not isinstance(parent, str) or parent not in id_to_idx:
            continue
        parent_idx = id_to_idx[parent]
        new_vertices = _new_reasoning_vertices(candidate_objects[idx], candidate_objects[parent_idx])
        if not new_vertices:
            continue
        parent_point = np.asarray([pca[parent_idx, 0], pca[parent_idx, 1], nll_values[parent_idx]], dtype=float)
        child_point = np.asarray([pca[idx, 0], pca[idx, 1], nll_values[idx]], dtype=float)
        direction = child_point - parent_point
        normal = np.asarray([-direction[1], direction[0], 0.0], dtype=float)
        norm = float(np.linalg.norm(normal[:2]))
        normal = normal / norm if norm > 1e-12 else np.asarray([0.0, offset_scale, 0.0], dtype=float)
        displayed_vertices = new_vertices[:8]
        for step_idx, vertex in enumerate(displayed_vertices):
            t = float(step_idx + 1) / float(len(displayed_vertices) + 1)
            point = parent_point * (1.0 - t) + child_point * t + normal * (offset_scale * math.sin(math.pi * t))
            label = _microstep_plot_label(vertex)
            local_complex = _local_filtered_subcomplex(candidate_objects[idx], str(vertex["label"]))
            hover = (
                f"<b>{html.escape(label)}</b>"
                f"<br>candidate={html.escape(ids[idx])}"
                f"<br>type={html.escape(str(vertex.get('type', '')))}"
                f"<br>filtration={float(vertex.get('filtration', 0.0)):.4f}"
                f"<br>{_summary_line(local_complex)}"
                + (f"<br>{_html_clip(vertex.get('text', ''), 500)}" if vertex.get("text") else "")
            )
            panel_idx = len(panel_objects)
            panel_objects.append(local_complex)
            panel_hover.append(hover)
            entries.append(
                {
                    "candidate_index": idx,
                    "candidate_record_id": ids[idx],
                    "parent_record_id": parent,
                    "simplex_label": vertex["label"],
                    "type": vertex.get("type", ""),
                    "label": label,
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                    "filtration": float(vertex.get("filtration", 0.0)),
                    "hover": hover,
                    "panel_index": panel_idx,
                    "filtered_simplicial_object": local_complex,
                }
            )
    return entries


def _new_reasoning_vertices(candidate_obj: dict[str, object], parent_obj: dict[str, object]) -> list[dict[str, object]]:
    parent_labels = {str(row["label"]) for row in _complex_vertex_records(parent_obj)}
    vertices = [row for row in _complex_vertex_records(candidate_obj) if str(row["label"]) not in parent_labels]
    preferred = [row for row in vertices if _is_reasoning_vertex_type(str(row.get("type", "")))]
    return preferred or vertices


def _is_reasoning_vertex_type(kind: str) -> bool:
    kind = kind.lower()
    return any(
        key in kind
        for key in (
            "reasoning",
            "verification",
            "refinement",
            "merged_state",
            "retrieved_evidence",
            "compressed_state",
            "rejected_branch",
        )
    )


def _microstep_plot_label(vertex: dict[str, object]) -> str:
    kind = str(vertex.get("type", "step"))
    for prefix in (
        "reasoning_step_",
        "verification_",
        "refinement_",
        "merged_state_",
        "retrieved_evidence_",
        "compressed_state_",
        "rejected_branch_",
    ):
        if kind.startswith(prefix):
            return kind[len(prefix):].replace("_", " ")
    return kind.replace("_", " ")[:18]


def _state_plot_label(row: dict[str, object], idx: int, level: int) -> str:
    if level <= 0:
        return "q0"
    return f"q{idx}"


def _sparse_got_state_labels(
    candidates: list[dict[str, object]],
    ids: list[str],
    inferred_levels: list[int],
    nll_values: np.ndarray,
    max_labels: int = 12,
) -> list[str]:
    if not candidates:
        return []
    children: dict[str, int] = defaultdict(int)
    for row in candidates:
        parent = row.get("parent")
        if isinstance(parent, str):
            children[parent] += 1
    keep: set[int] = set()
    for idx, row in enumerate(candidates):
        level = int(inferred_levels[idx]) if idx < len(inferred_levels) else int(row.get("level", 0) or 0)
        if level <= 0 or children.get(ids[idx], 0) > 1:
            keep.add(idx)
    leaves = [idx for idx, rid in enumerate(ids) if children.get(rid, 0) == 0]
    finite_order = [idx for idx in range(len(candidates)) if idx < len(nll_values) and math.isfinite(float(nll_values[idx]))]
    if finite_order:
        keep.add(min(finite_order, key=lambda idx: float(nll_values[idx])))
    for idx in sorted(leaves, key=lambda idx: float(nll_values[idx]) if idx < len(nll_values) and math.isfinite(float(nll_values[idx])) else math.inf)[:4]:
        keep.add(idx)
    if len(keep) < min(max_labels, len(candidates)):
        stride = max(1, int(math.ceil(len(candidates) / max(1, max_labels))))
        keep.update(range(0, len(candidates), stride))
    if len(keep) > max_labels:
        priority = sorted(
            keep,
            key=lambda idx: (
                0 if int(inferred_levels[idx]) <= 0 else 1,
                float(nll_values[idx]) if idx < len(nll_values) and math.isfinite(float(nll_values[idx])) else math.inf,
                idx,
            ),
        )[:max_labels]
        keep = set(priority)
    return [_state_plot_label(candidates[idx], idx, int(inferred_levels[idx])) if idx in keep else "" for idx in range(len(candidates))]


def _local_filtered_subcomplex(obj: dict[str, object], focus_label: str) -> dict[str, object]:
    focus = str(focus_label)
    labels = {focus}
    for a, b in _complex_edge_pairs(obj):
        if a == focus or b == focus:
            labels.update([a, b])
    for simplex in _complex_simplices(obj, 2):
        if focus in simplex:
            labels.update(simplex)
    simplices = []
    for simplex in obj.get("simplices", []) if isinstance(obj, dict) else []:
        if not isinstance(simplex, dict):
            continue
        raw = simplex.get("simplex", [])
        if not isinstance(raw, list) or not raw:
            continue
        raw_labels = {str(v) for v in raw}
        if raw_labels <= labels:
            simplices.append(simplex)
    if not simplices:
        for vertex in _complex_vertex_records(obj):
            if vertex["label"] == focus:
                simplices.append({"simplex": [focus], "dimension": 0, "filtration": vertex.get("filtration", 0.0), "type": vertex.get("type", "vertex"), "text": vertex.get("text", "")})
                break
    thresholds = sorted({float(row.get("filtration", 0.0) or 0.0) for row in simplices if isinstance(row.get("filtration", 0.0), (int, float))})
    return {
        "record_id": f"{obj.get('record_id', 'got')}|local:{focus}" if isinstance(obj, dict) else f"local:{focus}",
        "summary": {
            "num_vertices": sum(1 for row in simplices if int(row.get("dimension", -1)) == 0),
            "num_edges": sum(1 for row in simplices if int(row.get("dimension", -1)) == 1),
            "num_two_simplices": sum(1 for row in simplices if int(row.get("dimension", -1)) == 2),
            "num_thresholds": len(thresholds),
        },
        "thresholds": thresholds,
        "simplices": simplices,
    }


def write_tropical_support_heatmap(result: dict[str, object], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "tropical_support_heatmap.html"
    payload_path = output_dir / "tropical_support_payload.json"
    trace = result.get("graph_token_trace", {}) if isinstance(result, dict) else {}
    if (not isinstance(trace, dict) or not trace.get("tokens")) and isinstance(result, dict):
        scaling = result.get("inference_scaling")
        if isinstance(scaling, dict):
            best = scaling.get("best")
            if isinstance(best, dict):
                trace = best.get("graph_token_trace", {})
            if (not isinstance(trace, dict) or not trace.get("tokens")):
                candidates = scaling.get("candidates", [])
                if isinstance(candidates, list):
                    for candidate in candidates:
                        if isinstance(candidate, dict) and isinstance(candidate.get("graph_token_trace"), dict):
                            trace = candidate["graph_token_trace"]
                            break
    tokens = trace.get("tokens", []) if isinstance(trace, dict) else []
    if not tokens:
        _write_dark_empty(path, "No graph-token trace available.")
        payload_path.write_text(json.dumps({"tokens": [], "supports": [], "metrics": {"available": False}}, indent=2), encoding="utf-8")
        return {"tropical_support_heatmap": str(path), "tropical_support_payload": str(payload_path)}
    n = len(tokens)

    def active_support_index(token: dict[str, object]) -> int:
        try:
            return int(token.get("active_support_index", -1))
        except (TypeError, ValueError):
            return -1

    def finite_token_float(token: dict[str, object], key: str) -> float | None:
        try:
            value = float(token.get(key))
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def finite_object_float(obj: object, key: str) -> float | None:
        if not isinstance(obj, dict):
            return None
        try:
            value = float(obj.get(key))
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    result_metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
    trace_metrics = trace.get("metrics", {}) if isinstance(trace, dict) else {}
    def first_finite_float(default: float, *values: float | None) -> float:
        for value in values:
            if value is not None:
                return float(value)
        return float(default)

    wall_threshold = first_finite_float(
        1.0e-3,
        finite_object_float(trace, "wall_margin_threshold"),
        finite_object_float(trace_metrics, "wall_margin_threshold"),
        finite_object_float(result, "wall_margin_threshold"),
        finite_object_float(result_metrics, "wall_margin_threshold"),
    )
    near_wall_threshold = first_finite_float(
        max(float(wall_threshold) * 10.0, float(wall_threshold)),
        finite_object_float(trace, "near_wall_margin_threshold"),
        finite_object_float(trace_metrics, "near_wall_margin_threshold"),
        finite_object_float(result, "near_wall_margin_threshold"),
        finite_object_float(result_metrics, "near_wall_margin_threshold"),
    )
    wall_threshold = max(float(wall_threshold), 0.0)
    near_wall_threshold = max(float(near_wall_threshold), wall_threshold)

    def wall_margin_bucket(value: float | None) -> str:
        if value is None:
            return "unavailable"
        if value <= wall_threshold:
            return "strict_wall"
        if value <= near_wall_threshold:
            return "near_wall"
        return "interior"

    support_indices = []
    for token in tokens:
        active = active_support_index(token) if isinstance(token, dict) else -1
        if 0 <= active < n and active not in support_indices:
            support_indices.append(active)
    support_indices = sorted(support_indices, key=lambda idx: (-sum(1 for token in tokens if isinstance(token, dict) and active_support_index(token) == idx), idx))
    if not support_indices:
        reason = "no_valid_model_active_support_indices"
        _write_dark_empty(path, "No valid model active-support indices were available; tropical support was not rendered.")
        payload_path.write_text(
            json.dumps(
                {
                    "tokens": tokens,
                    "supports": [],
                    "support_flow_edges": [],
                    "metrics": {
                        "available": False,
                        "reason": reason,
                        "token_count": int(n),
                        "invalid_support_count": int(n),
                        "interpretation": "No support index was fabricated; rerun inference with graph_token_trace active_support_index values from the model.",
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"tropical_support_heatmap": str(path), "tropical_support_payload": str(payload_path)}
    active_indices = [active_support_index(token) if isinstance(token, dict) else -1 for token in tokens]
    invalid_support_rows = [
        {
            "query_index": int(idx),
            "query_label": _support_token_label(idx, token if isinstance(token, dict) else {}),
            "active_support_index": int(active),
            "reason": "active_support_index_out_of_range",
        }
        for idx, (token, active) in enumerate(zip(tokens, active_indices))
        if not (0 <= int(active) < n)
    ]
    z = np.zeros((n, len(support_indices)), dtype=float)
    assignment_z = np.zeros((n, len(support_indices)), dtype=float)
    selected_margin_matrix = np.full((n, len(support_indices)), np.nan, dtype=float)
    hover_grid: list[list[str]] = []
    query_labels = []
    for row_idx, token in enumerate(tokens):
        active = active_support_index(token) if isinstance(token, dict) else -1
        active_valid = 0 <= active < n
        margin = float(token.get("margin", 0.0) or 0.0)
        query_labels.append(_support_token_label(row_idx, token))
        hover_row = []
        for col_idx, support_idx in enumerate(support_indices):
            support = tokens[support_idx] if 0 <= support_idx < n else {}
            selected = active == support_idx
            assignment_status = (
                "selected_observed_support"
                if selected
                else ("unselected_observed_support_column" if active_valid else "invalid_active_support_index")
            )
            if selected:
                assignment_z[row_idx, col_idx] = 1.0
                z[row_idx, col_idx] = margin
                selected_margin_matrix[row_idx, col_idx] = margin
            margin_text = f"{margin:.5f}" if active_valid else "unavailable (invalid active_support_index; no support column fabricated)"
            hover_row.append(
                f"query={html.escape(_support_token_label(row_idx, token, long=True))}<br>"
                f"candidate support={html.escape(_support_token_label(support_idx, support, long=True))}<br>"
                f"selected={str(selected).lower()}<br>"
                f"assignment status={assignment_status}<br>"
                f"selected-support margin={margin_text}" + ("" if selected else " (shown only on the selected support column)") + "<br>"
                f"active support probability={finite_token_float(token, 'active_support_probability') if finite_token_float(token, 'active_support_probability') is not None else 'unavailable'}<br>"
                f"support probability entropy={finite_token_float(token, 'support_probability_entropy_bits') if finite_token_float(token, 'support_probability_entropy_bits') is not None else 'unavailable'} bits<br>"
                f"wall margin bucket={wall_margin_bucket(margin)} (strict <= {wall_threshold:.4g}; near <= {near_wall_threshold:.4g})<br>"
                f"query text={_html_clip(token.get('text', ''), 360)}<br>"
                f"support text={_html_clip(support.get('text', ''), 360)}"
            )
        hover_grid.append(hover_row)
    support_labels = [_support_token_label(idx, tokens[idx]) for idx in support_indices]
    counts = np.asarray([sum(1 for token in tokens if isinstance(token, dict) and active_support_index(token) == idx) for idx in support_indices], dtype=float)
    mean_margins = []
    for idx in support_indices:
        vals = [float(token.get("margin", 0.0) or 0.0) for token in tokens if isinstance(token, dict) and active_support_index(token) == idx]
        mean_margins.append(float(np.mean(vals)) if vals else 0.0)

    def token_group_label(token: dict[str, object]) -> str:
        kind = str(token.get("kind", "?") or "?")
        semantic = str(
            token.get("node_type")
            or token.get("edge_type")
            or token.get("active_support_kind")
            or token.get("label")
            or kind
        )
        return f"{kind}:{semantic}"

    def grouped_token_summary(indices: Iterable[int]) -> list[dict[str, object]]:
        groups: dict[str, dict[str, object]] = {}
        for idx in indices:
            if idx < 0 or idx >= n:
                continue
            token = tokens[idx]
            if not isinstance(token, dict):
                continue
            group = token_group_label(token)
            entry = groups.setdefault(
                group,
                {
                    "group": group,
                    "kind": str(token.get("kind", "?")),
                    "semantic_type": group.split(":", 1)[1] if ":" in group else group,
                    "count": 0,
                    "indices": [],
                    "labels": [],
                },
            )
            entry["count"] = int(entry["count"]) + 1
            entry["indices"].append(int(idx))  # type: ignore[union-attr]
            entry["labels"].append(_support_token_label(idx, token))  # type: ignore[union-attr]
        return sorted(groups.values(), key=lambda row: (-int(row["count"]), str(row["group"])))

    query_token_group_summary = grouped_token_summary(range(n))
    support_token_group_summary = grouped_token_summary(support_indices)
    query_token_groups = [
        token_group_label(token) if isinstance(token, dict) else "unknown:unknown"
        for token in tokens
    ]
    query_group_counts = {group: query_token_groups.count(group) for group in sorted(set(query_token_groups))}
    query_group_order = sorted(query_group_counts, key=lambda group: (-int(query_group_counts[group]), str(group)))
    query_group_id_by_name = {group: idx for idx, group in enumerate(query_group_order)}
    query_group_ids = [int(query_group_id_by_name[group]) for group in query_token_groups]
    query_group_tick_text = [_short_label(group, 24) for group in query_group_order]
    query_token_category_strip = []
    for idx, group in enumerate(query_token_groups):
        token = tokens[idx] if isinstance(tokens[idx], dict) else {}
        semantic_type = group.split(":", 1)[1] if ":" in group else group
        query_token_category_strip.append(
            {
                "query_index": int(idx),
                "query_label": _support_token_label(idx, token if isinstance(token, dict) else {}),
                "token_group": str(group),
                "token_category_id": int(query_group_id_by_name[group]),
                "kind": str(token.get("kind", "?")) if isinstance(token, dict) else "?",
                "semantic_type": str(semantic_type),
                "source_fields": ["kind", "node_type", "edge_type", "active_support_kind", "label"],
                "no_proxy_or_fallback": True,
            }
        )
    top_support_position = int(np.argmax(counts)) if counts.size else -1
    top_support_index = int(support_indices[top_support_position]) if top_support_position >= 0 else -1
    top_support_token = tokens[top_support_index] if 0 <= top_support_index < n and isinstance(tokens[top_support_index], dict) else {}
    top_support_selected = [token for token in tokens if isinstance(token, dict) and active_support_index(token) == top_support_index]
    top_support_probabilities = [finite_token_float(token, "active_support_probability") for token in top_support_selected]
    top_support_probabilities = [value for value in top_support_probabilities if value is not None]
    top_support_summary = {
        "available": bool(top_support_position >= 0),
        "support_index": int(top_support_index),
        "support_label": _support_token_label(top_support_index, top_support_token) if top_support_position >= 0 else "unavailable",
        "support_group": token_group_label(top_support_token) if top_support_position >= 0 and isinstance(top_support_token, dict) else "unavailable",
        "selected_query_count": int(counts[top_support_position]) if top_support_position >= 0 else 0,
        "capture_rate": float(counts[top_support_position] / max(float(n), 1.0)) if top_support_position >= 0 else 0.0,
        "mean_selected_margin": float(mean_margins[top_support_position]) if top_support_position >= 0 else None,
        "active_support_probability_mean": float(np.mean(top_support_probabilities)) if top_support_probabilities else None,
    }
    grouped_label_summary = ", ".join(f"{row['group']}={row['count']}" for row in query_token_group_summary[:8])
    margin_values = np.asarray([float(token.get("margin", 0.0) or 0.0) for token in tokens], dtype=float)
    finite_margins = margin_values[np.isfinite(margin_values)]

    active_probability_values = [
        value
        for token in tokens
        if isinstance(token, dict)
        for value in [finite_token_float(token, "active_support_probability")]
        if value is not None
    ]
    entropy_values = [
        value
        for token in tokens
        if isinstance(token, dict)
        for value in [finite_token_float(token, "support_probability_entropy_bits")]
        if value is not None
    ]

    def value_summary(values: list[float]) -> dict[str, object]:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return {
            "available": bool(arr.size),
            "count": int(arr.size),
            "min": float(np.min(arr)) if arr.size else None,
            "max": float(np.max(arr)) if arr.size else None,
            "mean": float(np.mean(arr)) if arr.size else None,
            "p05": float(np.quantile(arr, 0.05)) if arr.size else None,
            "p50": float(np.quantile(arr, 0.50)) if arr.size else None,
            "p95": float(np.quantile(arr, 0.95)) if arr.size else None,
        }

    margin_summary = {
        "min": float(np.min(finite_margins)) if finite_margins.size else 0.0,
        "max": float(np.max(finite_margins)) if finite_margins.size else 0.0,
        "mean": float(np.mean(finite_margins)) if finite_margins.size else 0.0,
        "std": float(np.std(finite_margins)) if finite_margins.size else 0.0,
        "p05": float(np.quantile(finite_margins, 0.05)) if finite_margins.size else 0.0,
        "p50": float(np.quantile(finite_margins, 0.50)) if finite_margins.size else 0.0,
        "p95": float(np.quantile(finite_margins, 0.95)) if finite_margins.size else 0.0,
    }
    active_probability_summary = value_summary(active_probability_values)
    support_probability_entropy_summary = value_summary(entropy_values)
    strict_wall_flags = [bool(np.isfinite(value) and value <= wall_threshold) for value in margin_values.tolist()]
    near_wall_flags = [bool(np.isfinite(value) and value <= near_wall_threshold) for value in margin_values.tolist()]
    near_wall_only_flags = [bool(near and not strict) for strict, near in zip(strict_wall_flags, near_wall_flags)]
    strict_wall_hit_count = int(sum(strict_wall_flags))
    near_wall_hit_count = int(sum(near_wall_flags))
    near_wall_only_count = int(sum(near_wall_only_flags))
    if not finite_margins.size:
        wall_interpretation_status = "metric_issue_no_finite_margins"
        wall_interpretation = "No finite tropical margins were present, so strict and near-wall rates are an unavailable metric state rather than evidence about chamber crossings."
        wall_metric_issue = True
    elif strict_wall_hit_count > 0:
        wall_interpretation_status = "strict_wall_margin_events_observed"
        wall_interpretation = "At least one selected-support margin is within the configured strict threshold; inspect the per-token wall buckets for the actual observed margin events."
        wall_metric_issue = False
    elif near_wall_only_count > 0:
        wall_interpretation_status = "low_strict_expected_near_wall_ambiguity"
        wall_interpretation = "The strict rate is low because no selected-support margin entered the strict band, but near-wall-only events show ambiguity close to the chamber boundary."
        wall_metric_issue = False
    else:
        wall_interpretation_status = "low_strict_expected_interior_margins"
        wall_interpretation = "The strict rate is low because all finite selected-support margins are above the near-wall threshold, consistent with interior chamber behavior for this margin audit."
        wall_metric_issue = False
    wall_margin_audit = {
        "wall_margin_threshold": float(wall_threshold),
        "near_wall_margin_threshold": float(near_wall_threshold),
        "strict_wall_hit_count": strict_wall_hit_count,
        "near_wall_hit_count": near_wall_hit_count,
        "near_wall_only_count": near_wall_only_count,
        "strict_wall_hit_rate": float(strict_wall_hit_count / max(n, 1)),
        "near_wall_hit_rate": float(near_wall_hit_count / max(n, 1)),
        "near_wall_only_rate": float(near_wall_only_count / max(n, 1)),
        "metric_scope": "margin_threshold_audit_not_certified_normal_fan_wall_crossing",
        "strict_definition": "strict_wall_hit_rate counts finite selected-support margins with margin <= wall_margin_threshold.",
        "near_wall_definition": "near_wall_hit_rate counts finite selected-support margins with margin <= near_wall_margin_threshold; near_wall_only counts near-wall hits outside the strict band.",
        "low_strict_wall_interpretation_status": wall_interpretation_status,
        "low_strict_wall_interpretation": wall_interpretation,
        "metric_issue": bool(wall_metric_issue),
        "definition": "strict_wall_hit_rate counts margin <= wall_margin_threshold; near_wall_hit_rate counts margin <= near_wall_margin_threshold so low strict wall-hit rate can still reveal near-wall ambiguity. This is a model tropical-margin threshold audit, not a certified normal-fan wall-crossing count.",
    }
    support_flow_edges = []
    support_assignment_status_by_token = []
    for row_idx, token in enumerate(tokens):
        active = active_support_index(token) if isinstance(token, dict) else -1
        active_valid = 0 <= active < n
        support = tokens[active] if active_valid else {}
        assignment_status = "selected_observed_support" if active_valid else "invalid_active_support_index"
        rendered_as_assignment_cell = bool(active in support_indices)
        support_assignment_status_by_token.append(
            {
                "query_index": int(row_idx),
                "query_label": _support_token_label(row_idx, token),
                "active_support_index": int(active),
                "status": assignment_status,
                "rendered_as_assignment_cell": rendered_as_assignment_cell,
                "reason": None if active_valid else "active_support_index_out_of_range",
            }
        )
        support_flow_edges.append(
            {
                "query_index": int(row_idx),
                "query_label": _support_token_label(row_idx, token),
                "support_index": int(active),
                "support_label": _support_token_label(active, support) if active_valid else "invalid",
                "margin": float(token.get("margin", 0.0) or 0.0),
                "active_support_probability": finite_token_float(token, "active_support_probability"),
                "support_probability_entropy_bits": finite_token_float(token, "support_probability_entropy_bits"),
                "top_model_support_probabilities": token.get("top_model_support_probabilities", []) if isinstance(token.get("top_model_support_probabilities"), list) else [],
                "support_probability_source": token.get("support_probability_source"),
                "strict_wall_hit": bool(strict_wall_flags[row_idx]) if row_idx < len(strict_wall_flags) else False,
                "near_wall_hit": bool(near_wall_flags[row_idx]) if row_idx < len(near_wall_flags) else False,
                "wall_margin_bucket": wall_margin_bucket(finite_token_float(token, "margin")),
                "wall_margin_threshold": float(wall_threshold),
                "near_wall_margin_threshold": float(near_wall_threshold),
                "query_kind": str(token.get("kind", "?")),
                "support_kind": str(support.get("kind", "?")) if isinstance(support, dict) else "?",
                "support_assignment_status": assignment_status,
                "rendered_as_assignment_cell": rendered_as_assignment_cell,
                "no_proxy_or_fallback": True,
            }
        )
    collapse_rate = float(counts.max() / max(float(n), 1.0)) if counts.size else 0.0
    support_probs = counts / max(float(counts.sum()), 1.0)
    support_entropy = float(-np.sum(support_probs * np.log2(np.maximum(support_probs, 1e-12)))) if counts.size else 0.0
    effective_supports = float(2.0 ** support_entropy) if counts.size else 0.0
    collapse_like_layout = bool(len(support_indices) == 1 or collapse_rate >= 0.70)
    support_panel_roles = [
        "observed_support_assignment_matrix",
        "selected_margin_profile",
        "wall_margin_threshold_overlays",
        "token_group_summary",
        "top_support_collapse_diagnostic",
    ]
    if collapse_like_layout:
        support_panel_roles.extend(["margin_distribution", "collapse_metrics_table"])
    else:
        support_panel_roles.extend(["support_frequency", "mean_selected_margin_by_support", "query_token_category_strip"])
    support_readability_contract = {
        "schema_version": "tropicalgt.tropical_support_readability.v1",
        "source_trace": "graph_token_trace.tokens",
        "layout_mode": "collapse_diagnostic" if collapse_like_layout else "observed_support_matrix",
        "no_proxy_or_fallback": True,
        "panels_are_separate": True,
        "panel_roles": support_panel_roles,
        "required_panel_roles": [
            "observed_support_assignment_matrix",
            "selected_margin_profile",
            "wall_margin_threshold_overlays",
            "token_group_summary",
        ],
        "observed_layout_required_panel_roles": [
            "support_frequency",
            "mean_selected_margin_by_support",
            "query_token_category_strip",
        ],
        "assignment_and_margin_panels_separated": True,
        "support_strip_split_from_margin_profile": True,
        "support_frequency_and_mean_margin_split": bool(not collapse_like_layout),
        "query_token_category_strip_visible": bool(not collapse_like_layout),
        "model_probability_summaries_separate_from_assignment_matrix": True,
        "compact_tick_labels": True,
        "full_token_text_preserved_in_hover_and_payload": True,
        "exact_token_indices_preserved_in_payload": True,
        "group_summaries_from_trace_fields": True,
        "collapse_diagnostic_visible": bool(collapse_like_layout),
        "collapse_diagnostic_available": bool(collapse_like_layout),
        "invalid_active_support_indices_not_fabricated": True,
        "normal_fan_wall_crossing_certified": False,
    }
    support_render_contract = {
        "schema_version": "tropicalgt.tropical_support_render.v1",
        "source_trace": "graph_token_trace.tokens",
        "support_index_source": "token.active_support_index emitted by the model tropical attention trace",
        "support_columns_policy": "observed_valid_active_support_indices_only",
        "assignment_matrix_semantics": "binary argmax support-selection mask; zeros are unselected cells, not zero margins",
        "assignment_matrix_binary": True,
        "selected_margin_matrix_policy": "finite model tropical margins are stored only on selected observed support cells; unselected cells are null",
        "legacy_margin_matrix_policy": "zero-filled display matrix retained for backward compatibility; use selected_margin_matrix for margin evidence",
        "support_flow_edge_policy": "one edge per query token; invalid active_support_index rows are marked invalid and no support column is fabricated",
        "normal_fan_wall_crossing_certified": False,
        "wall_margin_metric_scope": "margin_threshold_audit_not_certified_normal_fan_wall_crossing",
        "support_probability_source": "model_tropical_support_probabilities" if active_probability_values or entropy_values else "unavailable_in_trace",
        "observed_support_count": int(len(support_indices)),
        "token_count": int(n),
        "valid_support_assignment_count": int(n - len(invalid_support_rows)),
        "invalid_support_count": int(len(invalid_support_rows)),
        "invalid_support_rows": invalid_support_rows,
        "assignment_matrix_shape": [int(n), int(len(support_indices))],
        "no_proxy_or_fallback": True,
    }
    support_metrics = {
        "available": True,
        "token_count": int(n),
        "unique_support_count": int(len(support_indices)),
        "effective_supports": effective_supports,
        "support_entropy_bits": support_entropy,
        "top_support_collapse_rate": collapse_rate,
        "support_indices": [int(idx) for idx in support_indices],
        "support_labels": support_labels,
        "support_counts": [int(c) for c in counts.tolist()],
        "mean_margins": [float(v) for v in mean_margins],
        "valid_support_assignment_count": int(n - len(invalid_support_rows)),
        "invalid_support_count": int(len(invalid_support_rows)),
        "invalid_support_rows": invalid_support_rows,
        "support_columns_policy": "observed_valid_active_support_indices_only",
        "query_token_group_summary": query_token_group_summary,
        "support_token_group_summary": support_token_group_summary,
        "query_token_category_count": int(len(query_group_order)),
        "query_token_category_labels": query_group_order,
        "query_token_category_strip_available": bool(query_token_category_strip),
        "top_support_summary": top_support_summary,
        "grouped_token_label_policy": "query/support labels are grouped by model graph-token kind plus node_type/edge_type/semantic label; no labels are fabricated beyond graph_token_trace fields",
        "margin_summary": margin_summary,
        "wall_margin_audit": wall_margin_audit,
        "strict_wall_hit_rate": wall_margin_audit['strict_wall_hit_rate'],
        "near_wall_hit_rate": wall_margin_audit['near_wall_hit_rate'],
        "near_wall_only_rate": wall_margin_audit["near_wall_only_rate"],
        "wall_margin_threshold": float(wall_threshold),
        "near_wall_margin_threshold": float(near_wall_threshold),
        "active_support_probability_summary": active_probability_summary,
        "support_probability_entropy_bits_summary": support_probability_entropy_summary,
        "support_probability_source": "model_tropical_support_probabilities" if active_probability_values or entropy_values else "unavailable_in_trace",
        "layout_mode": "collapse_diagnostic" if collapse_like_layout else "observed_support_matrix",
        "raw_token_labels_truncated": True,
        "render_contract_schema_version": support_render_contract["schema_version"],
        "readability_contract_schema_version": support_readability_contract["schema_version"],
        "readability_panel_roles": support_panel_roles,
        "normal_fan_wall_crossing_certified": False,
        "no_proxy_or_fallback": True,
        "render_contract": "assignment_matrix is binary model argmax support; selected_margin_matrix is model tropical margin only on selected cells; probability summaries come from model_tropical_support_probabilities and are not fabricated scores",
        "interpretation": "The heatmap is an assignment matrix: yellow cells mean the model selected that support token. Confidence lives in the separate selected-margin profile, distribution, and model support-probability summaries. No support-token proxies are introduced for invalid active_support_index rows.",
    }
    payload_path.write_text(
        json.dumps(
            {
                "metrics": support_metrics,
                "tropical_support_render_contract": support_render_contract,
                "tropical_support_readability_contract": support_readability_contract,
                "query_labels": query_labels,
                "support_labels": support_labels,
                "assignment_matrix": assignment_z.tolist(),
                "selected_margin_matrix": [[None if not np.isfinite(value) else float(value) for value in row] for row in selected_margin_matrix.tolist()],
                "margin_matrix": z.tolist(),
                "tokens": tokens,
                "query_token_category_strip": query_token_category_strip,
                "support_assignment_status_by_token": support_assignment_status_by_token,
                "support_flow_edges": support_flow_edges,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if collapse_like_layout:
        margin_profile = [float(token.get("margin", 0.0) or 0.0) for token in tokens]
        row_numbers = list(range(n))
        token_kinds = [str(token.get("kind", "?")) for token in tokens]
        top_support_label = support_labels[0] if support_labels else "n/a"
        table_rows = [
            ("tokens", str(int(n))),
            ("observed supports", f"{len(support_indices)}/{n}"),
            ("effective supports", f"{effective_supports:.3f}"),
            ("entropy", f"{support_entropy:.3f} bits"),
            ("collapse rate", f"{collapse_rate:.3f}"),
            ("top support", top_support_label),
            ("top support group", str(top_support_summary.get("support_group", "unavailable"))),
            ("Grouped token labels", grouped_label_summary or "unavailable"),
            ("token groups", grouped_label_summary or "unavailable"),
            ("mean margin", f"{margin_summary['mean']:.4f}"),
            ("margin range", f"{margin_summary['min']:.4f} to {margin_summary['max']:.4f}"),
            ("margin std", f"{margin_summary['std']:.4f}"),
            ("strict wall hits", f"{strict_wall_hit_count}/{n} <= {wall_threshold:.4g}"),
            ("near-wall hits", f"{near_wall_hit_count}/{n} <= {near_wall_threshold:.4g}"),
            ("near-wall only", f"{near_wall_only_count}/{n}"),
            ("wall audit scope", str(wall_margin_audit.get("metric_scope", "unavailable"))),
            ("low strict interpretation", str(wall_margin_audit.get("low_strict_wall_interpretation_status", "unavailable"))),
        ]
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{"type": "heatmap"}, {"type": "scatter"}], [{"type": "histogram"}, {"type": "table"}]],
            column_widths=[0.48, 0.52],
            row_heights=[0.62, 0.38],
            horizontal_spacing=0.18,
            vertical_spacing=0.20,
            subplot_titles=(
                "Support assignments",
                "Margin profile",
                "Margin distribution",
                "Collapse metrics",
            ),
        )
        fig.add_trace(
            go.Heatmap(
                z=assignment_z,
                x=support_labels,
                y=query_labels,
                colorscale=[[0.0, "#082f49"], [0.49, "#082f49"], [0.5, "#facc15"], [1.0, "#facc15"]],
                showscale=True,
                colorbar=dict(title="selected", len=0.44, thickness=12),
                customdata=hover_grid,
                hovertemplate="%{customdata}<extra></extra>",
                zmin=0.0,
                zmax=1.0,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=row_numbers,
                y=margin_profile,
                mode="lines+markers",
                line=dict(color="#5eead4", width=2),
                marker=dict(
                    size=7,
                    color=margin_profile,
                    colorscale="Cividis",
                    cmin=0.0,
                    cmax=max(float(np.nanmax(finite_margins)), 1.0) if finite_margins.size else 1.0,
                    line=dict(color="#e8eef8", width=0.7),
                ),
                customdata=[
                    f"token={html.escape(query_labels[i])}<br>kind={html.escape(token_kinds[i])}<br>margin={float(margin_profile[i]):.5f}<br>{hover_grid[i][0]}"
                    for i in range(n)
                ],
                hovertemplate="%{customdata}<extra></extra>",
                name="margin profile",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        for threshold_value, threshold_name, threshold_color, threshold_dash in (
            (wall_threshold, "strict wall threshold", "#f87171", "dash"),
            (near_wall_threshold, "near-wall threshold", "#fbbf24", "dot"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=row_numbers,
                    y=[float(threshold_value)] * len(row_numbers),
                    mode="lines",
                    line=dict(color=threshold_color, width=1.5, dash=threshold_dash),
                    hovertemplate=f"{threshold_name}={float(threshold_value):.5g}<extra></extra>",
                    name=threshold_name,
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
        fig.add_trace(
            go.Histogram(
                x=margin_profile,
                nbinsx=min(18, max(5, int(math.sqrt(max(n, 1))) + 2)),
                marker=dict(color="#38bdf8", line=dict(color="#e0f2fe", width=0.7)),
                opacity=0.82,
                hovertemplate="margin bin=%{x}<br>count=%{y}<extra></extra>",
                name="margin distribution",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Table(
                header=dict(values=["metric", "value"], fill_color="#111827", font=dict(color="#e8eef8", size=12), align="left"),
                cells=dict(
                    values=[[row[0] for row in table_rows], [row[1] for row in table_rows]],
                    fill_color="#0f172a",
                    font=dict(color="#dbeafe", size=11),
                    align="left",
                    height=28,
                ),
            ),
            row=2,
            col=2,
        )
        fig.update_layout(
            template="plotly_dark",
            title=(
                "Tropical active-support collapse diagnostic: observed supports only"
                f"<br><sup>top-support collapse rate={collapse_rate:.3f}; top support {html.escape(top_support_label)} captures {100.0 * collapse_rate:.1f}% of tokens. Strict wall-hit rate={wall_margin_audit['strict_wall_hit_rate']:.3f}; near-wall hit rate={wall_margin_audit['near_wall_hit_rate']:.3f}. Wall audit scope: {html.escape(str(wall_margin_audit.get('metric_scope', 'unavailable')))}; interpretation: {html.escape(str(wall_margin_audit.get('low_strict_wall_interpretation_status', 'unavailable')))}. Yellow cells are selected-support assignments; margins are plotted separately. No support-token proxies.</sup>"
            ),
            meta={"tropical_support_render_contract": support_render_contract, "tropical_support_readability_contract": support_readability_contract},
            margin=dict(t=126, l=88, r=48, b=96),
            height=max(960, min(1420, 620 + 10 * n)),
        )
        fig.update_xaxes(title_text="", tickangle=45, row=1, col=1)
        fig.update_yaxes(title_text="query token", row=1, col=1)
        fig.update_xaxes(title_text="graph-token index", row=1, col=2)
        fig.update_yaxes(title_text="active-support margin", row=1, col=2)
        fig.update_xaxes(title_text="active-support margin", row=2, col=1)
        fig.update_yaxes(title_text="token count", row=2, col=1)
        _write_plotly_dark_html(path, fig, "Tropical active-support collapse diagnostic")
        return {"tropical_support_heatmap": str(path), "tropical_support_payload": str(payload_path)}
    row_numbers = list(range(n))
    token_kinds = [str(token.get("kind", "?")) for token in tokens]
    token_margins = [float(value) for value in margin_values.tolist()]
    fig = make_subplots(
        rows=5,
        cols=1,
        specs=[[{"type": "heatmap"}], [{"type": "bar"}], [{"type": "bar"}], [{"type": "heatmap"}], [{"type": "scatter"}]],
        row_heights=[0.38, 0.16, 0.16, 0.10, 0.20],
        vertical_spacing=0.055,
        subplot_titles=(
            "Active-support assignment matrix",
            "Support frequency by observed support",
            "Mean selected margin by observed support",
            "Query token categories from graph-token trace",
            "Margin profile by graph-token order",
        ),
    )
    fig.add_trace(
        go.Heatmap(
            z=assignment_z,
            x=support_labels,
            y=query_labels,
            colorscale=[[0.0, "#082f49"], [0.49, "#082f49"], [0.5, "#facc15"], [1.0, "#facc15"]],
            colorbar=dict(title="selected", x=1.08, y=0.84, len=0.32, thickness=14),
            customdata=hover_grid,
            hovertemplate="%{customdata}<extra></extra>",
            zmin=0.0,
            zmax=1.0,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=support_labels,
            y=counts.tolist(),
            marker=dict(color="#38bdf8", line=dict(color="#e8eef8", width=0.8)),
            text=[f"n={int(c)}" for c in counts.tolist()],
            textposition="outside",
            hovertext=[f"support={html.escape(label)}<br>selected query count={int(c)}" for label, c in zip(support_labels, counts.tolist())],
            hoverinfo="text",
            name="support frequency",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=support_labels,
            y=mean_margins,
            marker=dict(color=mean_margins, colorscale="Turbo", line=dict(color="#e8eef8", width=0.8)),
            text=[f"m={m:.3f}" for m in mean_margins],
            textposition="outside",
            hovertext=[f"support={html.escape(label)}<br>mean selected margin={m:.4f}<br>selected query count={int(c)}" for label, m, c in zip(support_labels, mean_margins, counts.tolist())],
            hoverinfo="text",
            name="mean selected margin",
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=[query_group_ids],
            x=row_numbers,
            y=["token category"],
            colorscale="Turbo",
            zmin=0,
            zmax=max(len(query_group_order) - 1, 1),
            colorbar=dict(
                title="category",
                x=1.08,
                y=0.33,
                len=0.16,
                thickness=10,
                tickmode="array",
                tickvals=list(range(len(query_group_order))),
                ticktext=query_group_tick_text,
            ),
            customdata=[[
                f"token={html.escape(query_labels[i])}<br>category={html.escape(query_token_groups[i])}<br>category id={query_group_ids[i]}<br>text={_html_clip(tokens[i].get('text', '') if isinstance(tokens[i], dict) else '', 280)}"
                for i in range(n)
            ]],
            hovertemplate="%{customdata}<extra></extra>",
            name="query token category strip",
            showscale=True,
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=row_numbers,
            y=token_margins,
            mode="lines+markers",
            line=dict(color="#5eead4", width=2),
            marker=dict(
                size=7,
                color=token_margins,
                colorscale="Cividis",
                cmin=0.0,
                cmax=max(float(np.nanmax(margin_values)), 1.0) if margin_values.size and np.isfinite(margin_values).any() else 1.0,
                line=dict(color="#e8eef8", width=0.7),
            ),
            customdata=[
                f"token={html.escape(query_labels[i])}<br>kind={html.escape(token_kinds[i])}<br>margin={float(token_margins[i]):.5f}<br>active support={html.escape(support_flow_edges[i]['support_label'])}"
                for i in range(n)
            ],
            hovertemplate="%{customdata}<extra></extra>",
            name="margin profile",
        ),
        row=5,
        col=1,
    )
    for threshold_value, threshold_name, threshold_color, threshold_dash in (
        (wall_threshold, "strict wall threshold", "#f87171", "dash"),
        (near_wall_threshold, "near-wall threshold", "#fbbf24", "dot"),
    ):
        fig.add_trace(
            go.Scatter(
                x=row_numbers,
                y=[float(threshold_value)] * len(row_numbers),
                mode="lines",
                line=dict(color=threshold_color, width=1.5, dash=threshold_dash),
                hovertemplate=f"{threshold_name}={float(threshold_value):.5g}<extra></extra>",
                name=threshold_name,
                showlegend=False,
            ),
            row=5,
            col=1,
        )
    fig.update_layout(
        template="plotly_dark",
        title=(
            "Tropical active-support audit: observed supports only"
            f"<br><sup>observed supports={len(support_indices)}/{n}; top-support collapse rate={collapse_rate:.3f}; effective={effective_supports:.2f}; entropy={support_entropy:.3f} bits. Grouped token labels: {html.escape(grouped_label_summary or 'unavailable')}; top support group={html.escape(str(top_support_summary.get('support_group', 'unavailable')))}. Strict wall-hit rate={wall_margin_audit['strict_wall_hit_rate']:.3f}; near-wall hit rate={wall_margin_audit['near_wall_hit_rate']:.3f}. Wall audit scope: {html.escape(str(wall_margin_audit.get('metric_scope', 'unavailable')))}; interpretation: {html.escape(str(wall_margin_audit.get('low_strict_wall_interpretation_status', 'unavailable')))}. Yellow cells mark selected support assignments only. No support-token proxies.</sup>"
        ),
        meta={"tropical_support_render_contract": support_render_contract, "tropical_support_readability_contract": support_readability_contract},
        height=max(1240, min(1980, 860 + 14 * n)),
        margin=dict(t=166, l=112, r=220, b=128),
        showlegend=False,
    )
    if collapse_rate >= 0.95:
        fig.add_annotation(
            text=(
                "active-support collapse: nearly every query token selects the same support<br>"
                "assignment collapse is a model-trace diagnostic; margin confidence is shown in the profile below"
            ),
            x=0.02,
            y=1.08,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            font=dict(size=12, color="#fbbf24"),
            bgcolor="rgba(15,23,42,0.92)",
            bordercolor="rgba(251,191,36,0.45)",
            borderwidth=1,
        )
    fig.update_xaxes(title_text="active support token", tickangle=38, automargin=True, row=1, col=1)
    fig.update_yaxes(title_text="query token", automargin=True, row=1, col=1)
    fig.update_xaxes(title_text="support token", tickangle=28, automargin=True, row=2, col=1)
    fig.update_yaxes(title_text="selected query count", automargin=True, row=2, col=1)
    fig.update_xaxes(title_text="support token", tickangle=28, automargin=True, row=3, col=1)
    fig.update_yaxes(title_text="mean selected margin", automargin=True, row=3, col=1)
    fig.update_xaxes(title_text="graph-token index", automargin=True, row=4, col=1)
    fig.update_yaxes(title_text="trace category", automargin=True, row=4, col=1)
    fig.update_xaxes(title_text="graph-token index", automargin=True, row=5, col=1)
    fig.update_yaxes(title_text="active-support margin", automargin=True, row=5, col=1)
    _write_plotly_dark_html(path, fig, "Tropical active-support heatmap")
    return {"tropical_support_heatmap": str(path), "tropical_support_payload": str(payload_path)}


_TROPICAL_IDEAL_KEYS = (
    "model_derived_tropical_ideal",
    "tropical_ideal",
    "tropical_fan_ideal",
    "tropical_variety_ideal",
)


def _find_explicit_tropical_ideal_spec(result: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]] | None:
    containers: list[tuple[str, Any]] = [("result", result)]
    trace = result.get("graph_token_trace") if isinstance(result, Mapping) else None
    if isinstance(trace, Mapping):
        containers.append(("result.graph_token_trace", trace))
    metrics = result.get("metrics") if isinstance(result, Mapping) else None
    if isinstance(metrics, Mapping):
        containers.append(("result.metrics", metrics))
    scaling = result.get("inference_scaling") if isinstance(result, Mapping) else None
    if isinstance(scaling, Mapping):
        containers.append(("result.inference_scaling", scaling))
        best = scaling.get("best")
        if isinstance(best, Mapping):
            containers.append(("result.inference_scaling.best", best))
        candidates = scaling.get("candidates")
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
            for idx, candidate in enumerate(candidates[:8]):
                if isinstance(candidate, Mapping):
                    containers.append((f"result.inference_scaling.candidates[{idx}]", candidate))
    for container_path, container in containers:
        if not isinstance(container, Mapping):
            continue
        for key in _TROPICAL_IDEAL_KEYS:
            value = container.get(key)
            if isinstance(value, Mapping) and isinstance(value.get("variables"), Sequence) and isinstance(value.get("generators"), Sequence):
                return f"{container_path}.{key}", value
    return None


def _precomputed_tropical_fan_report(result: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]] | None:
    for key in ("tropical_fan_diagnostics", "macaulay2_tropical_fan_diagnostics"):
        value = result.get(key) if isinstance(result, Mapping) else None
        if isinstance(value, Mapping) and value.get("schema_version") == "tropicalgt.cas_tropical_fan.v1":
            return f"result.{key}", value
    return None


def _unavailable_tropical_fan_payload(reason: str, *, source_path: str = "unavailable") -> dict[str, Any]:
    diagnostics = {
        "schema_version": "tropicalgt.cas_tropical_fan.v1",
        "available": False,
        "status": "unavailable_no_model_derived_tropical_ideal",
        "reason": reason,
        "backend": "Macaulay2",
        "certificate_attached": False,
        "tropical_cycle_certified": False,
        "fan_diagnostics_certified": False,
        "safe_to_render_as_tropical_fan": False,
        "cas_artifacts": {},
        "certificate_contract": {
            "certificate_source": "Macaulay2 Tropical tropicalVariety on an explicit model-derived QQ ideal",
            "sage_scope": (
                "Sage tropical polynomial/variety diagnostics are not a replacement for the Macaulay2 "
                "ideal-to-tropical-cycle fan certificate."
            ),
            "no_proxy_policy": (
                "No support-token, chain-presentation, rank-sample, embedding-only, or visualization diagnostic may substitute for the real certificate."
            ),
        },
    }
    return {
        "schema_version": "tropicalgt.tropical_fan_visual_audit.v1",
        "available": False,
        "source_path": source_path,
        "ideal_spec": None,
        "diagnostics": diagnostics,
        "safe_to_render_as_tropical_fan": False,
        "render_contract": "Tropical fan diagnostics render one dimensional cones only from explicit model-derived ideal specs and real Macaulay2 Tropical certificates; unavailable states are not substituted by support-token proxies.",
    }


def _tropical_fan_ray_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rays = summary.get("rays") if isinstance(summary, Mapping) else []
    if not isinstance(rays, Sequence) or isinstance(rays, (str, bytes)) or not rays:
        return []
    rows = [list(row) for row in rays if isinstance(row, Sequence) and not isinstance(row, (str, bytes))]
    if not rows:
        return []
    ray_count = max((len(row) for row in rows), default=0)
    out: list[dict[str, Any]] = []
    for ray_idx in range(ray_count):
        coords = []
        for row in rows:
            try:
                coords.append(int(row[ray_idx]))
            except Exception:
                coords.append(0)
        out.append({"ray_index": int(ray_idx), "coordinates": coords})
    return out


def _write_tropical_fan_html(path: Path, payload: Mapping[str, Any]) -> None:
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), Mapping) else {}
    summary = diagnostics.get("fan_summary") if isinstance(diagnostics.get("fan_summary"), Mapping) else {}
    ray_rows = _tropical_fan_ray_rows(summary)
    available = bool(payload.get("available") and diagnostics.get("safe_to_render_as_tropical_fan") is True and ray_rows)
    if available:
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "xy"}, {"type": "table"}]],
            column_widths=[0.56, 0.44],
            horizontal_spacing=0.14,
            subplot_titles=("one dimensional cones from certified rays", "Macaulay2 Tropical certificate"),
        )
        for row in ray_rows:
            coords = row["coordinates"]
            x = float(coords[0]) if coords else 0.0
            y = float(coords[1]) if len(coords) > 1 else 0.0
            label = f"rho_{row['ray_index']} = ({', '.join(str(v) for v in coords)})"
            fig.add_trace(
                go.Scatter(
                    x=[0.0, x],
                    y=[0.0, y],
                    mode="lines+markers+text",
                    text=["", label],
                    textposition="top center",
                    line=dict(width=3),
                    marker=dict(size=[5, 10]),
                    hovertemplate=f"{html.escape(label)}<br>one dimensional cone / ray<extra></extra>",
                    name=label,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
        max_cones = summary.get("max_cones") if isinstance(summary.get("max_cones"), list) else []
        multiplicities = summary.get("multiplicities") if isinstance(summary.get("multiplicities"), list) else []
        basis_check = diagnostics.get("tropical_basis_check") if isinstance(diagnostics.get("tropical_basis_check"), Mapping) else {}
        prevariety = diagnostics.get("tropical_prevariety_summary") if isinstance(diagnostics.get("tropical_prevariety_summary"), Mapping) else {}
        contract = diagnostics.get("certificate_contract") if isinstance(diagnostics.get("certificate_contract"), Mapping) else {}
        table_rows = [
            ("status", diagnostics.get("status", "certified")),
            ("backend", diagnostics.get("backend", "Macaulay2")),
            ("certificate source", contract.get("certificate_source", "Macaulay2 Tropical tropicalVariety certificate required")),
            ("tool scope", contract.get("ordinary_to_laurent_torus_scope", "explicit ideal-to-tropical-cycle fan diagnostics only")),
            ("source", payload.get("source_path", "unknown")),
            ("ray count", summary.get("ray_count", len(ray_rows))),
            ("ambient dimension", summary.get("ambient_dimension", "")),
            ("max cones", _json_clip(max_cones, 220)),
            ("multiplicities", _json_clip(multiplicities, 160)),
            ("balanced", summary.get("is_balanced", False)),
            ("pure", summary.get("is_pure", False)),
            ("simplicial", summary.get("is_simplicial", False)),
            ("cycle certified", diagnostics.get("tropical_cycle_certified", False)),
            ("tropical basis check", basis_check.get("is_tropical_basis") if basis_check.get("available") else basis_check.get("error", "unavailable")),
            ("prevariety available", bool(prevariety.get("available"))),
            ("prevariety rays", _json_clip(prevariety.get("rays", []), 180)),
            ("prevariety max cones", _json_clip(prevariety.get("max_cones", []), 180)),
            ("Sage scope", contract.get("sage_scope", "not a substitute certificate for this fan view")),
            ("no-proxy policy", contract.get("no_proxy_policy", "no proxy diagnostics may substitute for the certificate")),
            ("warning", diagnostics.get("render_warning", "not a multigraded free-resolution certificate")),
        ]
        fig.add_trace(
            go.Table(
                header=dict(values=["diagnostic", "value"], fill_color="#10243f", font=dict(color="#e8f2ff", size=13), align="left"),
                cells=dict(
                    values=[[str(k) for k, _ in table_rows], [str(v) for _, v in table_rows]],
                    fill_color="#07111f",
                    font=dict(color="#d7e8ff", size=12),
                    align="left",
                    height=28,
                ),
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="ray coordinate 1", zeroline=True, row=1, col=1)
        fig.update_yaxes(title_text="ray coordinate 2", zeroline=True, scaleanchor="x", scaleratio=1, row=1, col=1)
        title = "Tropical fan diagnostics: real Macaulay2 certificate<br><sup>Rays are one dimensional cones; this is not a multigraded free-resolution or derived-equivalence certificate.</sup>"
        fig.update_layout(title=title, height=720, margin=dict(t=118, l=70, r=44, b=80))
    else:
        diagnostics_reason = str(diagnostics.get("reason", "No explicit model-derived tropical ideal spec was exported."))
        contract = diagnostics.get("certificate_contract") if isinstance(diagnostics.get("certificate_contract"), Mapping) else {}
        table_rows = [
            ("status", diagnostics.get("status", "unavailable_no_model_derived_tropical_ideal")),
            ("source", payload.get("source_path", "unavailable")),
            ("safe_to_render_as_tropical_fan", False),
            ("reason", diagnostics_reason),
            ("certificate source", contract.get("certificate_source", "Macaulay2 Tropical tropicalVariety certificate required")),
            ("Sage scope", contract.get("sage_scope", "not a substitute certificate for this fan view")),
            ("no-proxy policy", contract.get("no_proxy_policy", "no proxy diagnostics may substitute for the certificate")),
            ("render contract", payload.get("render_contract", "unavailable")),
            ("one dimensional cones", "not rendered without a real Macaulay2 Tropical certificate"),
            ("warning", "not a multigraded free-resolution or derived-equivalence certificate"),
        ]
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(values=["diagnostic", "value"], fill_color="#10243f", font=dict(color="#e8f2ff", size=13), align="left"),
                    cells=dict(
                        values=[[str(k) for k, _ in table_rows], [str(v) for _, v in table_rows]],
                        fill_color="#07111f",
                        font=dict(color="#d7e8ff", size=12),
                        align="left",
                        height=30,
                    ),
                )
            ]
        )
        fig.update_layout(
            title="Tropical fan diagnostics unavailable<br><sup>No one dimensional cones are displayed without an explicit model-derived ideal and a real Macaulay2 certificate.</sup>",
            height=520,
            margin=dict(t=112, l=44, r=44, b=44),
        )
    _write_plotly_dark_html(path, fig, "Tropical fan diagnostics")


def write_tropical_fan_diagnostics(result: dict[str, object], output_dir: str | Path, *, timeout_s: float = 15.0) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "tropical_fan_diagnostics.html"
    payload_path = output_dir / "tropical_fan_diagnostics.json"

    precomputed = _precomputed_tropical_fan_report(result if isinstance(result, Mapping) else {})
    if precomputed is not None:
        source_path, report = precomputed
        ideal_spec = report.get("ideal_schema") if isinstance(report.get("ideal_schema"), Mapping) else None
    else:
        found = _find_explicit_tropical_ideal_spec(result if isinstance(result, Mapping) else {})
        if found is None:
            payload = _unavailable_tropical_fan_payload(
                "No explicit model-derived tropical ideal spec was exported. Expected one of model_derived_tropical_ideal, tropical_ideal, tropical_fan_ideal, or tropical_variety_ideal with variables and generators.",
            )
            payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            _write_tropical_fan_html(html_path, payload)
            return {"tropical_fan_diagnostics": str(html_path), "tropical_fan_diagnostics_payload": str(payload_path)}
        source_path, ideal_spec = found
        try:
            from . import cas_tropical

            report = cas_tropical.try_compute_tropical_fan_diagnostics(dict(ideal_spec), timeout_s=timeout_s)
        except Exception as exc:
            payload = _unavailable_tropical_fan_payload(
                f"Macaulay2 Tropical diagnostic call failed before a certificate could be attached: {type(exc).__name__}: {exc}",
                source_path=source_path,
            )
            payload["ideal_spec"] = dict(ideal_spec)
            payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            _write_tropical_fan_html(html_path, payload)
            return {"tropical_fan_diagnostics": str(html_path), "tropical_fan_diagnostics_payload": str(payload_path)}

    safe = bool(isinstance(report, Mapping) and report.get("safe_to_render_as_tropical_fan") is True)
    payload = {
        "schema_version": "tropicalgt.tropical_fan_visual_audit.v1",
        "available": safe,
        "source_path": source_path,
        "ideal_spec": dict(ideal_spec) if isinstance(ideal_spec, Mapping) else None,
        "diagnostics": dict(report) if isinstance(report, Mapping) else {},
        "safe_to_render_as_tropical_fan": safe,
        "render_contract": "Tropical fan diagnostics render one dimensional cones only from explicit model-derived ideal specs and real Macaulay2 Tropical certificates; unavailable states are not substituted by support-token proxies.",
    }
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_tropical_fan_html(html_path, payload)
    return {"tropical_fan_diagnostics": str(html_path), "tropical_fan_diagnostics_payload": str(payload_path)}



_TORIC_EXPONENT_SPEC_KEYS = (
    "model_derived_toric_exponent_matrix",
    "toric_exponent_matrix",
    "toric_embedding_exponent_matrix",
    "finite_toric_exponent_matrix",
)


def _find_explicit_toric_exponent_spec(result: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]] | None:
    containers: list[tuple[str, Any]] = [("result", result)]
    trace = result.get("graph_token_trace") if isinstance(result, Mapping) else None
    if isinstance(trace, Mapping):
        containers.append(("result.graph_token_trace", trace))
    metrics = result.get("metrics") if isinstance(result, Mapping) else None
    if isinstance(metrics, Mapping):
        containers.append(("result.metrics", metrics))
    scaling = result.get("inference_scaling") if isinstance(result, Mapping) else None
    if isinstance(scaling, Mapping):
        containers.append(("result.inference_scaling", scaling))
        best = scaling.get("best")
        if isinstance(best, Mapping):
            containers.append(("result.inference_scaling.best", best))
        candidates = scaling.get("candidates")
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
            for idx, candidate in enumerate(candidates[:8]):
                if isinstance(candidate, Mapping):
                    containers.append((f"result.inference_scaling.candidates[{idx}]", candidate))
                    meta = candidate.get("chart_bundle_transport_metadata")
                    if isinstance(meta, Mapping):
                        containers.append((f"result.inference_scaling.candidates[{idx}].chart_bundle_transport_metadata", meta))
    for container_path, container in containers:
        if not isinstance(container, Mapping):
            continue
        for key in _TORIC_EXPONENT_SPEC_KEYS:
            value = container.get(key)
            if isinstance(value, Mapping) and ("exponent_matrix" in value or "columns" in value):
                return f"{container_path}.{key}", value
    return None


def _precomputed_toric_embedding_report(result: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]] | None:
    containers: list[tuple[str, Any]] = [("result", result)]
    metrics = result.get("metrics") if isinstance(result, Mapping) else None
    if isinstance(metrics, Mapping):
        containers.append(("result.metrics", metrics))
    scaling = result.get("inference_scaling") if isinstance(result, Mapping) else None
    if isinstance(scaling, Mapping):
        containers.append(("result.inference_scaling", scaling))
        best = scaling.get("best")
        if isinstance(best, Mapping):
            containers.append(("result.inference_scaling.best", best))
        candidates = scaling.get("candidates")
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
            for idx, candidate in enumerate(candidates[:8]):
                if isinstance(candidate, Mapping):
                    containers.append((f"result.inference_scaling.candidates[{idx}]", candidate))
                    meta = candidate.get("chart_bundle_transport_metadata")
                    if isinstance(meta, Mapping):
                        containers.append((f"result.inference_scaling.candidates[{idx}].chart_bundle_transport_metadata", meta))
    for container_path, container in containers:
        if not isinstance(container, Mapping):
            continue
        for key in ("toric_embedding_certificate", "macaulay2_toric_embedding_certificate", "cas_toric_embedding_certificate"):
            value = container.get(key)
            if isinstance(value, Mapping) and value.get("schema_version") == "tropicalgt.cas_toric_embedding.v1":
                return f"{container_path}.{key}", value
    return None


def _unavailable_toric_embedding_payload(reason: str, *, source_path: str = "unavailable", exponent_matrix_spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    try:
        from . import cas_toric

        contract = cas_toric.toric_embedding_certificate_contract()
    except Exception:
        contract = {
            "certificate_source": "Macaulay2 Quasidegrees toricIdeal(A,R) on an explicit integer exponent matrix",
            "no_proxy_policy": "No chart-bundle logits, toric-row activations, GraphCG cells, support tokens, embeddings, or visualization rows may substitute for this CAS certificate.",
        }
    diagnostics = {
        "schema_version": "tropicalgt.cas_toric_embedding.v1",
        "available": False,
        "status": "unavailable_no_model_derived_toric_exponent_matrix",
        "reason": reason,
        "backend": "Macaulay2",
        "certificate_attached": False,
        "toric_embedding_certified": False,
        "toric_ideal_certified": False,
        "tropical_variety_embedding_certified": False,
        "global_toric_variety_embedding_certified": False,
        "safe_to_render_as_toric_embedding": False,
        "safe_to_render_as_tropical_variety_embedding": False,
        "safe_to_render_as_global_toric_variety_embedding": False,
        "safe_to_use_as_normal_fan_certificate": False,
        "embedding_scope": "unavailable_finite_monomial_map_toric_ideal_certificate",
        "certificate_contract": contract,
        "cas_artifacts": {},
    }
    return {
        "schema_version": "tropicalgt.toric_embedding_sidecar_visual_audit.v1",
        "available": False,
        "source_path": source_path,
        "exponent_matrix_spec": dict(exponent_matrix_spec) if isinstance(exponent_matrix_spec, Mapping) else None,
        "diagnostics": diagnostics,
        "safe_to_render_as_finite_toric_ideal_sidecar": False,
        "safe_to_render_as_tropical_variety_embedding": False,
        "safe_to_render_as_global_toric_variety_embedding": False,
        "safe_to_use_as_normal_fan_certificate": False,
        "render_contract": "Toric embedding sidecars render only certified finite monomial-map toric ideals from explicit exponent matrices and Macaulay2 Quasidegrees toricIdeal certificates; unavailable states are not substituted by chart-bundle, support-token, GraphCG, embedding, or visualization proxies.",
    }


def _toric_matrix_rows(diagnostics: Mapping[str, Any]) -> list[list[int]]:
    summary = diagnostics.get("monomial_map_summary") if isinstance(diagnostics.get("monomial_map_summary"), Mapping) else {}
    matrix = summary.get("exponent_matrix") if isinstance(summary, Mapping) else None
    if not isinstance(matrix, Sequence) or isinstance(matrix, (str, bytes)):
        schema = diagnostics.get("exponent_matrix_schema") if isinstance(diagnostics.get("exponent_matrix_schema"), Mapping) else {}
        matrix = schema.get("exponent_matrix") if isinstance(schema, Mapping) else None
    rows: list[list[int]] = []
    if isinstance(matrix, Sequence) and not isinstance(matrix, (str, bytes)):
        for row in matrix:
            if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
                try:
                    rows.append([int(value) for value in row])
                except Exception:
                    return []
    return rows


def _write_toric_embedding_html(path: Path, payload: Mapping[str, Any]) -> None:
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), Mapping) else {}
    contract = diagnostics.get("certificate_contract") if isinstance(diagnostics.get("certificate_contract"), Mapping) else {}
    matrix = _toric_matrix_rows(diagnostics)
    available = bool(payload.get("available") and diagnostics.get("safe_to_render_as_toric_embedding") is True and diagnostics.get("toric_ideal_certified") is True and matrix)
    if available:
        summary = diagnostics.get("monomial_map_summary") if isinstance(diagnostics.get("monomial_map_summary"), Mapping) else {}
        ideal = diagnostics.get("toric_ideal_summary") if isinstance(diagnostics.get("toric_ideal_summary"), Mapping) else {}
        variables = summary.get("coordinate_variables") if isinstance(summary.get("coordinate_variables"), list) else [f"z_{idx}" for idx in range(len(matrix[0]) if matrix else 0)]
        table_rows = [
            ("status", diagnostics.get("status", "certified")),
            ("backend", diagnostics.get("backend", "Macaulay2")),
            ("certificate source", contract.get("certificate_source", "Macaulay2 Quasidegrees toricIdeal certificate required")),
            ("source", payload.get("source_path", "unknown")),
            ("embedding scope", diagnostics.get("embedding_scope", "finite_monomial_map_toric_ideal_certificate_only")),
            ("lattice dimension", summary.get("lattice_dimension", len(matrix))),
            ("coordinate count", summary.get("coordinate_count", len(variables))),
            ("coordinate variables", ", ".join(str(v) for v in variables)),
            ("toric ideal generators", _json_clip(ideal.get("generators_text", ""), 240)),
            ("generator count", ideal.get("generator_count", "")),
            ("codimension", ideal.get("codimension", "")),
            ("dimension", ideal.get("dimension", "")),
            ("normal fan certified", diagnostics.get("safe_to_use_as_normal_fan_certificate", False)),
            ("tropical variety embedding", diagnostics.get("safe_to_render_as_tropical_variety_embedding", False)),
            ("global toric variety embedding", diagnostics.get("safe_to_render_as_global_toric_variety_embedding", False)),
            ("no-proxy policy", contract.get("no_proxy_policy", "no proxy diagnostics may substitute for the certificate")),
            ("warning", diagnostics.get("render_warning", "not a normal-fan or tropical-variety certificate")),
        ]
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "heatmap"}, {"type": "table"}]],
            column_widths=[0.48, 0.52],
            horizontal_spacing=0.12,
            subplot_titles=("explicit exponent matrix A", "Macaulay2 toricIdeal certificate"),
        )
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=[str(v) for v in variables],
                y=[f"e{idx}" for idx in range(len(matrix))],
                colorscale="Viridis",
                colorbar=dict(title="exponent", thickness=12),
                hovertemplate="lattice row=%{y}<br>coordinate=%{x}<br>exponent=%{z}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Table(
                header=dict(values=["diagnostic", "value"], fill_color="#10243f", font=dict(color="#e8f2ff", size=13), align="left"),
                cells=dict(values=[[str(k) for k, _ in table_rows], [str(v) for _, v in table_rows]], fill_color="#07111f", font=dict(color="#d7e8ff", size=12), align="left", height=28),
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="monomial coordinate", tickangle=24, row=1, col=1)
        fig.update_yaxes(title_text="lattice exponent row", row=1, col=1)
        fig.update_layout(
            title="Toric embedding sidecar: finite monomial-map toric ideal certificate<br><sup>Certified by Macaulay2 Quasidegrees toricIdeal(A,R); not a normal-fan, tropical-variety, or global neural toric-variety certificate.</sup>",
            meta={"toric_embedding_sidecar_contract": payload.get("render_contract")},
            height=720,
            margin=dict(t=118, l=70, r=44, b=82),
        )
    else:
        reason = str(diagnostics.get("reason", "No explicit model-derived toric exponent matrix was exported."))
        table_rows = [
            ("status", diagnostics.get("status", "unavailable_no_model_derived_toric_exponent_matrix")),
            ("source", payload.get("source_path", "unavailable")),
            ("safe_to_render_as_finite_toric_ideal_sidecar", False),
            ("reason", reason),
            ("certificate source", contract.get("certificate_source", "Macaulay2 Quasidegrees toricIdeal certificate required")),
            ("required input", "explicit integer exponent_matrix or columns for a finite monomial map"),
            ("normal fan certified", False),
            ("tropical variety embedding", False),
            ("global toric variety embedding", False),
            ("no-proxy policy", contract.get("no_proxy_policy", "no proxy diagnostics may substitute for the certificate")),
            ("render contract", payload.get("render_contract", "unavailable")),
        ]
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(values=["diagnostic", "value"], fill_color="#10243f", font=dict(color="#e8f2ff", size=13), align="left"),
                    cells=dict(values=[[str(k) for k, _ in table_rows], [str(v) for _, v in table_rows]], fill_color="#07111f", font=dict(color="#d7e8ff", size=12), align="left", height=30),
                )
            ]
        )
        fig.update_layout(
            title="Toric embedding sidecar unavailable<br><sup>No finite monomial-map toric ideal is rendered without an explicit exponent matrix and a real Macaulay2 certificate.</sup>",
            height=560,
            margin=dict(t=112, l=44, r=44, b=44),
            meta={"toric_embedding_sidecar_contract": payload.get("render_contract")},
        )
    _write_plotly_dark_html(path, fig, "Toric embedding sidecar")


def write_toric_embedding_sidecar(result: dict[str, object], output_dir: str | Path, *, timeout_s: float = 15.0) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "toric_embedding_sidecar.html"
    payload_path = output_dir / "toric_embedding_sidecar.json"

    precomputed = _precomputed_toric_embedding_report(result if isinstance(result, Mapping) else {})
    if precomputed is not None:
        source_path, report = precomputed
        exponent_spec = report.get("exponent_matrix_schema") if isinstance(report.get("exponent_matrix_schema"), Mapping) else None
    else:
        found = _find_explicit_toric_exponent_spec(result if isinstance(result, Mapping) else {})
        if found is None:
            payload = _unavailable_toric_embedding_payload(
                "No explicit model-derived toric exponent matrix was exported. Expected one of model_derived_toric_exponent_matrix, toric_exponent_matrix, toric_embedding_exponent_matrix, or finite_toric_exponent_matrix with exponent_matrix or columns.",
            )
            payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            _write_toric_embedding_html(html_path, payload)
            return {"toric_embedding_sidecar": str(html_path), "toric_embedding_sidecar_payload": str(payload_path)}
        source_path, exponent_spec = found
        try:
            from . import cas_toric

            report = cas_toric.try_compute_toric_embedding_certificate(dict(exponent_spec), timeout_s=timeout_s)
        except Exception as exc:
            payload = _unavailable_toric_embedding_payload(
                f"Macaulay2 toricIdeal sidecar call failed before a certificate could be attached: {type(exc).__name__}: {exc}",
                source_path=source_path,
                exponent_matrix_spec=exponent_spec,
            )
            payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            _write_toric_embedding_html(html_path, payload)
            return {"toric_embedding_sidecar": str(html_path), "toric_embedding_sidecar_payload": str(payload_path)}

    safe = bool(isinstance(report, Mapping) and report.get("safe_to_render_as_toric_embedding") is True and report.get("toric_ideal_certified") is True)
    payload = {
        "schema_version": "tropicalgt.toric_embedding_sidecar_visual_audit.v1",
        "available": safe,
        "source_path": source_path,
        "exponent_matrix_spec": dict(exponent_spec) if isinstance(exponent_spec, Mapping) else None,
        "diagnostics": dict(report) if isinstance(report, Mapping) else {},
        "safe_to_render_as_finite_toric_ideal_sidecar": safe,
        "safe_to_render_as_tropical_variety_embedding": bool(isinstance(report, Mapping) and report.get("safe_to_render_as_tropical_variety_embedding") is True),
        "safe_to_render_as_global_toric_variety_embedding": bool(isinstance(report, Mapping) and report.get("safe_to_render_as_global_toric_variety_embedding") is True),
        "safe_to_use_as_normal_fan_certificate": bool(isinstance(report, Mapping) and report.get("safe_to_use_as_normal_fan_certificate") is True),
        "render_contract": "Toric embedding sidecars render only certified finite monomial-map toric ideals from explicit exponent matrices and Macaulay2 Quasidegrees toricIdeal certificates; unavailable states are not substituted by chart-bundle, support-token, GraphCG, embedding, or visualization proxies.",
    }
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_toric_embedding_html(html_path, payload)
    return {"toric_embedding_sidecar": str(html_path), "toric_embedding_sidecar_payload": str(payload_path)}


_CHART_BUNDLE_TRANSPORT_SCHEMA = "tropicalgt.chart_bundle_transport_metadata.v1"


def _find_chart_bundle_transport_metadata(result: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]] | None:
    def direct(path: str, value: Any) -> tuple[str, Mapping[str, Any]] | None:
        if not isinstance(value, Mapping):
            return None
        if value.get("schema_version") == _CHART_BUNDLE_TRANSPORT_SCHEMA:
            return path, value
        meta = value.get("chart_bundle_transport_metadata")
        if isinstance(meta, Mapping):
            return f"{path}.chart_bundle_transport_metadata", meta
        return None

    seen: set[int] = set()

    def visit(path: str, value: Any, depth: int = 0) -> tuple[str, Mapping[str, Any]] | None:
        if depth > 6 or not isinstance(value, (Mapping, Sequence)) or isinstance(value, (str, bytes)):
            return None
        object_id = id(value)
        if object_id in seen:
            return None
        seen.add(object_id)
        found = direct(path, value)
        if found is not None:
            return found
        if isinstance(value, Mapping):
            priority_keys = (
                "metrics",
                "graph_token_trace",
                "model_output",
                "output",
                "diagnostics",
                "inference_scaling",
                "best",
                "candidate",
            )
            for key in priority_keys:
                if key in value:
                    found = visit(f"{path}.{key}", value[key], depth + 1)
                    if found is not None:
                        return found
            candidates = value.get("candidates")
            if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
                for idx, candidate in enumerate(candidates[:16]):
                    found = visit(f"{path}.candidates[{idx}]", candidate, depth + 1)
                    if found is not None:
                        return found
        elif isinstance(value, Sequence):
            for idx, item in enumerate(value[:16]):
                found = visit(f"{path}[{idx}]", item, depth + 1)
                if found is not None:
                    return found
        return None

    return visit("result", result)


def _safe_int_count(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _chart_bundle_transport_payload(
    *,
    source_path: str,
    metadata: Mapping[str, Any] | None,
    available: bool,
    reason: str,
) -> dict[str, Any]:
    meta = dict(metadata) if isinstance(metadata, Mapping) else None
    transport_contract = meta.get("monomial_transport_contract") if isinstance(meta, Mapping) and isinstance(meta.get("monomial_transport_contract"), Mapping) else {}
    matroid_contract = meta.get("bundle_matroid_contract") if isinstance(meta, Mapping) and isinstance(meta.get("bundle_matroid_contract"), Mapping) else {}
    toric_certificate = meta.get("toric_embedding_certificate") if isinstance(meta, Mapping) and isinstance(meta.get("toric_embedding_certificate"), Mapping) else {}
    chart_ids = meta.get("chart_ids") if isinstance(meta, Mapping) and isinstance(meta.get("chart_ids"), Sequence) and not isinstance(meta.get("chart_ids"), (str, bytes)) else []
    overlap_pairs = meta.get("overlap_pairs") if isinstance(meta, Mapping) and isinstance(meta.get("overlap_pairs"), Sequence) and not isinstance(meta.get("overlap_pairs"), (str, bytes)) else []
    overlap_triples = meta.get("overlap_triples") if isinstance(meta, Mapping) and isinstance(meta.get("overlap_triples"), Sequence) and not isinstance(meta.get("overlap_triples"), (str, bytes)) else []
    return {
        "schema_version": "tropicalgt.chart_bundle_transport_sidecar.v1",
        "available": bool(available),
        "source_path": source_path,
        "reason": reason,
        "metadata": meta,
        "chart_ids": list(chart_ids),
        "overlap_pair_count": _safe_int_count(meta.get("overlap_pair_count") if isinstance(meta, Mapping) else None, len(overlap_pairs)),
        "overlap_triple_count": _safe_int_count(meta.get("overlap_triple_count") if isinstance(meta, Mapping) else None, len(overlap_triples)),
        "monomial_transport_contract": dict(transport_contract),
        "bundle_matroid_contract": dict(matroid_contract),
        "toric_embedding_certificate": dict(toric_certificate),
        "actual_data_only": True,
        "no_proxy_or_fallback": True,
        "safe_to_render_as_toric_embedding_certificate": False,
        "safe_to_render_as_tropical_variety_embedding": False,
        "safe_to_render_as_global_toric_variety_embedding": False,
        "safe_to_use_as_normal_fan_certificate": False,
        "render_contract": "Chart-bundle transport sidecars render only exported chart-bundle metadata, overlap ids, monomial transport contracts, and bundle matroid/flat-incidence contracts. They are not a toric embedding, tropical variety, global toric variety, or normal-fan certificate; unavailable states remain explicit and no proxies or fallbacks are substituted.",
    }


def _write_chart_bundle_transport_html(path: Path, payload: Mapping[str, Any]) -> None:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    transport_contract = payload.get("monomial_transport_contract") if isinstance(payload.get("monomial_transport_contract"), Mapping) else {}
    matroid_contract = payload.get("bundle_matroid_contract") if isinstance(payload.get("bundle_matroid_contract"), Mapping) else {}
    toric_certificate = payload.get("toric_embedding_certificate") if isinstance(payload.get("toric_embedding_certificate"), Mapping) else {}
    chart_ids = payload.get("chart_ids") if isinstance(payload.get("chart_ids"), Sequence) and not isinstance(payload.get("chart_ids"), (str, bytes)) else []
    overlap_pairs = metadata.get("overlap_pairs") if isinstance(metadata.get("overlap_pairs"), Sequence) and not isinstance(metadata.get("overlap_pairs"), (str, bytes)) else []
    overlap_triples = metadata.get("overlap_triples") if isinstance(metadata.get("overlap_triples"), Sequence) and not isinstance(metadata.get("overlap_triples"), (str, bytes)) else []
    transport_ids = transport_contract.get("transport_ids") if isinstance(transport_contract.get("transport_ids"), Sequence) and not isinstance(transport_contract.get("transport_ids"), (str, bytes)) else []
    rows = [
        ("status", "available" if payload.get("available") is True else "unavailable"),
        ("source", payload.get("source_path", "unavailable")),
        ("reason", payload.get("reason", "available" if payload.get("available") else "chart-bundle metadata unavailable")),
        ("metadata schema", metadata.get("schema_version", "unavailable")),
        ("metadata source", metadata.get("source", "unavailable")),
        ("chart ids", _json_clip(chart_ids, 260)),
        ("overlap pair count", payload.get("overlap_pair_count", 0)),
        ("overlap triple count", payload.get("overlap_triple_count", 0)),
        ("sample overlap pairs", _json_clip(overlap_pairs[:8], 360)),
        ("sample overlap triples", _json_clip(overlap_triples[:6], 360)),
        ("directed overlap policy", metadata.get("directed_overlap_policy", "unavailable")),
        ("transport schema", transport_contract.get("schema_version", "unavailable")),
        ("transport source", transport_contract.get("source", "unavailable")),
        ("transport ids", _json_clip(transport_ids[:12], 360)),
        ("shift representation", transport_contract.get("shift_representation", "unavailable")),
        ("permutation representation", transport_contract.get("permutation_representation", "unavailable")),
        ("permutation target policy", transport_contract.get("permutation_target_policy", "unavailable")),
        ("flat-incidence schema", matroid_contract.get("schema_version", "unavailable")),
        ("flat-incidence shape", _json_clip(matroid_contract.get("flat_incidence_shape", []), 180)),
        ("flat-incidence metric", matroid_contract.get("flat_incidence_metric", "unavailable")),
        ("rank-defect metric", matroid_contract.get("rank_defect_metric", "unavailable")),
        ("toric certificate schema", toric_certificate.get("schema_version", "unavailable")),
        ("toric certificate attached", toric_certificate.get("certificate_attached", False)),
        ("safe as toric embedding certificate", payload.get("safe_to_render_as_toric_embedding_certificate", False)),
        ("safe as tropical variety embedding", payload.get("safe_to_render_as_tropical_variety_embedding", False)),
        ("safe as normal fan certificate", payload.get("safe_to_use_as_normal_fan_certificate", False)),
        ("actual data only", payload.get("actual_data_only", False)),
        ("no proxy or fallback", payload.get("no_proxy_or_fallback", False)),
        ("render contract", payload.get("render_contract", "unavailable")),
    ]
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=["diagnostic", "value"], fill_color="#10243f", font=dict(color="#e8f2ff", size=13), align="left"),
                cells=dict(values=[[str(k) for k, _ in rows], [str(v) for _, v in rows]], fill_color="#07111f", font=dict(color="#d7e8ff", size=12), align="left", height=28),
            )
        ]
    )
    if payload.get("available") is True:
        title = "Chart-bundle transport sidecar: exported overlap and transport metadata"
        subtitle = "Actual chart ids, transport ids, and flat-incidence contracts only; this is not a toric embedding, normal-fan, or tropical-variety certificate."
        height = 860
    else:
        title = "Chart-bundle transport sidecar unavailable"
        subtitle = "No chart-bundle transport metadata is rendered by proxy; train/eval exports must attach tropicalgt.chart_bundle_transport_metadata.v1."
        height = 620
    fig.update_layout(
        title=f"{title}<br><sup>{subtitle}</sup>",
        height=height,
        margin=dict(t=116, l=44, r=44, b=44),
        meta={"chart_bundle_transport_sidecar_contract": payload.get("render_contract")},
    )
    _write_plotly_dark_html(path, fig, "Chart bundle transport sidecar")


def write_chart_bundle_transport_sidecar(result: dict[str, object], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "chart_bundle_transport_sidecar.html"
    payload_path = output_dir / "chart_bundle_transport_sidecar.json"

    found = _find_chart_bundle_transport_metadata(result if isinstance(result, Mapping) else {})
    if found is None:
        payload = _chart_bundle_transport_payload(
            source_path="unavailable",
            metadata=None,
            available=False,
            reason="No exported chart_bundle_transport_metadata with schema tropicalgt.chart_bundle_transport_metadata.v1 was found in result, metrics, graph_token_trace, or inference_scaling candidates.",
        )
    else:
        source_path, metadata = found
        schema_ok = metadata.get("schema_version") == _CHART_BUNDLE_TRANSPORT_SCHEMA
        meta_available = metadata.get("available") is True
        transport_contract = metadata.get("monomial_transport_contract") if isinstance(metadata.get("monomial_transport_contract"), Mapping) else {}
        matroid_contract = metadata.get("bundle_matroid_contract") if isinstance(metadata.get("bundle_matroid_contract"), Mapping) else {}
        transport_safe = transport_contract.get("actual_data_only") is True and transport_contract.get("no_proxy_or_fallback") is True
        matroid_safe = matroid_contract.get("actual_data_only") is True and matroid_contract.get("no_proxy_or_fallback") is True
        chart_ids = metadata.get("chart_ids") if isinstance(metadata.get("chart_ids"), Sequence) and not isinstance(metadata.get("chart_ids"), (str, bytes)) else []
        available = bool(schema_ok and meta_available and transport_safe and matroid_safe and chart_ids)
        if available:
            reason = "exported_chart_bundle_transport_metadata_available"
        elif not schema_ok:
            reason = "chart_bundle_transport_metadata_schema_mismatch"
        elif not meta_available:
            reason = str(metadata.get("reason", "chart_bundle_transport_metadata_marked_unavailable"))
        elif not chart_ids:
            reason = "chart_bundle_transport_metadata_missing_chart_ids"
        else:
            reason = "chart_bundle_transport_contracts_missing_actual_data_only_or_no_proxy_flags"
        payload = _chart_bundle_transport_payload(
            source_path=source_path,
            metadata=metadata,
            available=available,
            reason=reason,
        )
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_chart_bundle_transport_html(html_path, payload)
    return {"chart_bundle_transport_sidecar": str(html_path), "chart_bundle_transport_sidecar_payload": str(payload_path)}


def _support_token_label(index: int, token: dict[str, object], long: bool = False) -> str:
    kind = str(token.get("kind", "?"))
    node_type = str(token.get("node_type", token.get("active_support_kind", "")) or "")
    label = str(token.get("label", token.get("active_support_label", "")) or "")
    short_kind = {"graph": "G", "node": "N", "edge": "E"}.get(kind, kind[:1].upper())
    semantic = node_type or label
    base = f"t{index:02d} {short_kind}:{_short_label(semantic, 18)}"
    if long:
        node_id = token.get("node_id")
        text = token.get("text", "")
        return base + (f" | node_id={node_id}" if node_id is not None else "") + (f" | text={_short_label(str(text), 120)}" if text else "")
    return base


def write_persistence_visualizations(
    topology: dict[str, object],
    output_dir: str | Path,
    growth: list[object] | None = None,
    title_prefix: str = "",
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    barcode = output_dir / "persistence_barcode.html"
    module_path = output_dir / "persistence_module_betti.html"
    representations_path = output_dir / "persistence_representations.html"
    landscapes_path = output_dir / "persistence_landscapes.html"
    landscapes_payload_path = landscapes_path.with_suffix(".json")
    if growth:
        _write_growth_persistence_barcode(barcode, topology, growth, title_prefix=title_prefix)
        _write_growth_persistence_module(module_path, topology, growth, title_prefix=title_prefix)
        _write_growth_persistence_representations(representations_path, topology, growth, title_prefix=title_prefix)
        _write_growth_persistence_landscapes(landscapes_path, topology, growth, title_prefix=title_prefix)
        return {
            "persistence_barcode": str(barcode),
            "persistence_module_betti": str(module_path),
            "persistence_representations": str(representations_path),
            "persistence_landscapes": str(landscapes_path),
            "persistence_landscapes_payload": str(landscapes_payload_path),
        }

    intervals = topology.get("persistence", {}).get("intervals", []) if isinstance(topology.get("persistence"), dict) else []
    fig = go.Figure()
    display_intervals, barcode_meta = _prepare_barcode_intervals(intervals)
    colors = {0: "#55d6be", 1: "#7aa2ff", 2: "#fbbf24", 3: "#fb7185"}
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for interval in display_intervals:
        grouped[int(interval.get("dimension", 0))].append(interval)
    row_offset = 0
    tick_values: list[float] = []
    tick_labels: list[str] = []
    for dim in sorted(grouped):
        rows = grouped[dim]
        xs: list[float | None] = []
        ys: list[float | None] = []
        hover: list[str | None] = []
        for local_idx, interval in enumerate(rows):
            y = row_offset + local_idx
            birth = float(interval["birth"])
            death = float(interval["display_death"])
            xs.extend([birth, death, None])
            ys.extend([y, y, None])
            true_death = "inf" if interval.get("infinite") else f"{float(interval.get('death', death)):.4g}"
            label = f"H{dim} [{birth:.4g}, {true_death}]"
            hover.extend([label, label, None])
            if interval.get("infinite"):
                fig.add_trace(
                    go.Scatter(
                        x=[death],
                        y=[y],
                        mode="markers",
                        marker=dict(symbol="triangle-right", size=9, color=colors.get(dim, "#cbd5e1")),
                        hovertext=label,
                        hoverinfo="text",
                        showlegend=False,
                    )
                )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(width=6, color=colors.get(dim, "#cbd5e1")),
                name=f"H{dim}",
                hovertext=hover,
                hoverinfo="text",
            )
        )
        tick_values.append(row_offset + max((len(rows) - 1) / 2.0, 0.0))
        tick_labels.append(f"H{dim} ({len(rows)})")
        row_offset += len(rows) + 2
    if not display_intervals:
        fig.add_annotation(text="No nonzero persistence intervals after display filtering.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    fig.update_layout(
        template="plotly_dark",
        title=f"Persistent homology barcode ({barcode_meta['displayed']} displayed / {barcode_meta['raw']} raw)",
        xaxis_title="filtration",
        yaxis_title="homology class grouped by dimension",
        yaxis=dict(tickmode="array", tickvals=tick_values, ticktext=tick_labels),
    )
    fig.update_layout(meta=barcode_meta)
    _write_plotly_dark_html(barcode, fig, "Persistent homology barcode")

    states = topology.get("persistence_module", {}).get("states", []) if isinstance(topology.get("persistence_module"), dict) else []
    fig2 = go.Figure()
    for dim in range(4):
        xs = [state["threshold"] for state in states]
        ys = [int(state.get("betti", {}).get(str(dim), 0)) for state in states]
        if any(value != 0 for value in ys):
            fig2.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line_shape="hv",
                    line=dict(width=2.5),
                    name=f"beta_{dim}",
                    hovertemplate=f"beta_{dim}=%{{y}}<br>filtration=%{{x:.4g}}<extra></extra>",
                )
            )
    fig2.update_layout(template="plotly_dark", title="Persistence module Betti rank profile (step functions)", xaxis_title="filtration", yaxis_title="Betti rank")
    _write_plotly_dark_html(module_path, fig2, "Persistence module Betti rank profile")
    _write_single_persistence_representations(representations_path, topology, title_prefix=title_prefix)
    _write_dark_redirect(
        landscapes_path,
        "Open trajectory growth persistence landscapes",
        "A standalone non-growth topology record does not contain trajectory-growth landscape rows. This audit bundle writes the real GUDHI lambda_k(t) landscape functions under the trajectory_persistence page.",
        "trajectory_persistence/persistence_landscapes.html",
        "Open trajectory_persistence/persistence_landscapes.html",
    )
    landscapes_payload_path.write_text(
        json.dumps(_persistence_landscape_unavailable_contract("standalone_non_growth_topology_has_no_trajectory_growth_rows"), indent=2),
        encoding="utf-8",
    )
    return {
        "persistence_barcode": str(barcode),
        "persistence_module_betti": str(module_path),
        "persistence_representations": str(representations_path),
        "persistence_landscapes": str(landscapes_path),
        "persistence_landscapes_payload": str(landscapes_payload_path),
    }



def _m2_style_report_from_bifiltration(bifiltration: Mapping[str, Any]) -> Dict[str, Any]:
    free = bifiltration.get("free_resolution") if isinstance(bifiltration, Mapping) else None
    chain = bifiltration.get("chain_presentation_diagnostics") if isinstance(bifiltration, Mapping) else None
    m2: Mapping[str, Any] | None = None
    if isinstance(free, Mapping):
        maybe = free.get("chain_presentation_diagnostics")
        if isinstance(maybe, Mapping):
            m2 = maybe
        else:
            maybe = free.get("macaulay2_style")
            if isinstance(maybe, Mapping):
                m2 = maybe
    out: Dict[str, Any] = dict(m2) if isinstance(m2, Mapping) else {}
    real = None
    if isinstance(chain, Mapping) and isinstance(chain.get("real_free_resolution"), Mapping):
        real = chain.get("real_free_resolution")
    elif isinstance(free, Mapping) and isinstance(free.get("real_free_resolution"), Mapping):
        real = free.get("real_free_resolution")
    if isinstance(real, Mapping):
        out["real_free_resolution"] = dict(real)
    return out


def _cas_real_resolution_display(real: Mapping[str, Any]) -> Dict[str, Any]:
    """Return certified CAS output under the strongest grading it actually supports."""
    if not (isinstance(real, Mapping) and real.get("available") is True and real.get("exactness_certified") is True):
        return {}
    summary = real.get("free_resolution_summary") if isinstance(real.get("free_resolution_summary"), Mapping) else {}
    artifacts = real.get("cas_artifacts") if isinstance(real.get("cas_artifacts"), Mapping) else {}
    is_multigraded = bool(
        real.get("safe_to_render_as_multigraded_free_resolution") is True
        and real.get("multigraded_free_resolution_certified") is True
    )
    is_total_graded = bool(
        not is_multigraded
        and real.get("safe_to_render_as_total_graded_resolution") is True
        and real.get("total_graded_resolution_certified") is True
    )
    is_ungraded = bool(
        not is_multigraded
        and not is_total_graded
        and real.get("ungraded_resolution_certified") is True
    )
    if not (is_multigraded or is_total_graded or is_ungraded):
        return {}

    if is_multigraded:
        modules_in = summary.get("free_modules") if isinstance(summary.get("free_modules"), list) else []
        scope = "certified_macaulay2_multigraded_cokernel_resolution"
        object_resolved = "displayed F0/coker(d1) module from the bifiltration chain presentation"
        not_full = False
    else:
        ungraded = artifacts.get("betti_table_ungraded") if isinstance(artifacts.get("betti_table_ungraded"), Mapping) else {}
        modules_in = ungraded.get("free_modules") if isinstance(ungraded.get("free_modules"), list) else []
        betti_rows_in = ungraded.get("betti_table_rows") if isinstance(ungraded.get("betti_table_rows"), list) else []
        scope = "certified_total_graded_cas_resolution" if is_total_graded else "certified_ungraded_cas_resolution"
        object_resolved = "CAS-certified resolution of the boundary-presentation module under the displayed non-multigraded grading"
        not_full = True

    modules: List[Dict[str, Any]] = []
    betti_rows: List[Dict[str, Any]] = []
    if is_multigraded:
        betti_rows_in = summary.get("betti_table_rows") if isinstance(summary.get("betti_table_rows"), list) else []
        for idx, row in enumerate(modules_in):
            if not isinstance(row, Mapping):
                continue
            hd = int(row.get("homological_degree", 0) or 0)
            md = row.get("multidegree", [])
            if not isinstance(md, Sequence) or isinstance(md, (str, bytes)):
                md = []
            md_list = [int(v) for v in list(md)[:2]]
            while len(md_list) < 2:
                md_list.append(0)
            rank = int(row.get("rank", 1) or 1)
            display = str(row.get("display") or f"F_{hd} contains S(-{md_list[0]},{md_list[1]})^{rank}")
            modules.append({"module": f"F_{hd}", "name": f"F_{hd}", "degree": hd, "rank": rank, "display": display, "multidegree": md_list})
            betti_rows.append({"homological_degree": hd, "multidegree": md_list, "shift_display": f"({md_list[0]},{md_list[1]})", "rank": rank, "multiplicity": rank})
    else:
        for idx, row in enumerate(modules_in):
            if not isinstance(row, Mapping):
                continue
            name = str(row.get("name", row.get("module", f"F_{idx}")))
            try:
                hd = int(name.rsplit("_", 1)[1]) if "_" in name else int(row.get("homological_degree", idx) or idx)
            except Exception:
                hd = idx
            rank = int(row.get("rank", row.get("multiplicity", 0)) or 0)
            display = str(row.get("display") or f"{name} = S^{rank}")
            modules.append({"module": name, "name": name, "degree": hd, "rank": rank, "display": display, "multidegree": []})
            if not betti_rows_in:
                betti_rows.append({"homological_degree": hd, "multidegree": [], "shift_display": "ungraded", "rank": rank, "multiplicity": rank})
    if betti_rows_in:
        betti_rows = []
        for row in betti_rows_in:
            if not isinstance(row, Mapping):
                continue
            hd = int(row.get("homological_degree", 0) or 0)
            rank = int(row.get("rank", row.get("multiplicity", 0)) or 0)
            if rank == 0:
                continue
            normalized = {
                "homological_degree": hd,
                "multidegree": row.get("multidegree", []) if isinstance(row.get("multidegree", []), list) else [],
                "shift_display": str(row.get("shift_display", "ungraded")),
                "rank": rank,
                "multiplicity": int(row.get("multiplicity", rank) or rank),
            }
            for optional in ("total_degree", "matrix_row", "grading", "source", "not_multigraded", "safe_for_multigraded_claims"):
                if optional in row:
                    normalized[optional] = row.get(optional)
            betti_rows.append(normalized)

    diffs_in = artifacts.get("differentials") if isinstance(artifacts.get("differentials"), list) else summary.get("differentials", [])
    differentials: List[Dict[str, Any]] = []
    for row in diffs_in or []:
        if not isinstance(row, Mapping):
            continue
        hd = int(row.get("homological_degree", 0) or 0)
        rows = row.get("rows")
        cols = row.get("cols")
        shape = f"{rows}x{cols}" if rows is not None and cols is not None else str(row.get("shape", ""))
        preview = row.get("matrix_text", row.get("matrix_preview", ""))
        differentials.append({"map": f"d_{hd}", "name": f"d_{hd}", "shape": shape, "rank": "CAS", "matrix_preview": [str(preview)], "source_degrees": row.get("source_degrees"), "target_degrees": row.get("target_degrees")})
    fitting = artifacts.get("fitting_ideals") if isinstance(artifacts.get("fitting_ideals"), Mapping) else summary.get("fitting_ideals", {})
    minors = artifacts.get("minors") if isinstance(artifacts.get("minors"), Mapping) else summary.get("minors", {})
    ideal_diagnostics = artifacts.get("ideal_diagnostics") if isinstance(artifacts.get("ideal_diagnostics"), Mapping) else summary.get("ideal_diagnostics", {})
    be_artifacts = artifacts.get("buchsbaum_eisenbud_diagnostics") if isinstance(artifacts.get("buchsbaum_eisenbud_diagnostics"), Mapping) else {}
    be_rank_conditions = artifacts.get("buchsbaum_eisenbud_rank_conditions") if isinstance(artifacts.get("buchsbaum_eisenbud_rank_conditions"), Mapping) else summary.get("buchsbaum_eisenbud_rank_conditions", {})
    grade_depth_regular = artifacts.get("grade_depth_regular_diagnostics") if isinstance(artifacts.get("grade_depth_regular_diagnostics"), Mapping) else summary.get("grade_depth_regular_diagnostics", {})
    syzygies = artifacts.get("syzygies") if isinstance(artifacts.get("syzygies"), Mapping) else summary.get("syzygy_diagnostics", {})
    certificate_summary = artifacts.get("certificate_summary") if isinstance(artifacts.get("certificate_summary"), Mapping) else {}
    return {
        "available": bool(modules),
        "ring": real.get("coefficient_ring", "F2[x_level,x_radius]"),
        "scope": scope,
        "object_resolved": object_resolved,
        "not_full_persistence_module_resolution": not_full,
        "betti_table_rows": betti_rows,
        "free_modules": modules,
        "differentials": differentials,
        "fitting_ideals": dict(fitting) if isinstance(fitting, Mapping) else {},
        "minors": dict(minors) if isinstance(minors, Mapping) else {},
        "ideal_diagnostics": dict(ideal_diagnostics) if isinstance(ideal_diagnostics, Mapping) else {},
        "buchsbaum_eisenbud_rank_conditions": dict(be_rank_conditions) if isinstance(be_rank_conditions, Mapping) else {},
        "grade_depth_regular_diagnostics": dict(grade_depth_regular) if isinstance(grade_depth_regular, Mapping) else {},
        "syzygies": dict(syzygies) if isinstance(syzygies, Mapping) else {},
        "certificate_summary": dict(certificate_summary) if isinstance(certificate_summary, Mapping) else {},
        "buchsbaum_eisenbud_diagnostics": {
            "minimality_certified": bool(real.get("minimality_certified")),
            "exactness_certified": bool(real.get("exactness_certified")),
            "certificate": str(real.get("render_warning", "CAS certified the displayed resolution under its stated grading.")),
            "grading_scope": "multigraded" if is_multigraded else ("total-graded" if is_total_graded else "ungraded"),
            "backend_diagnostics_available": bool(be_artifacts.get("available")),
            "multiplier_output_available": bool(be_artifacts.get("multiplier_output_available")),
            "safe_to_render_multiplier_output": bool(be_artifacts.get("safe_to_render_multiplier_output")),
            "is_resolution_backend": bool(be_artifacts.get("is_resolution_backend", False)),
            "requires_certified_macaulay2_chain_complex": bool(be_artifacts.get("requires_certified_macaulay2_chain_complex", True)),
            "safe_to_substitute_for_resolution": bool(be_artifacts.get("safe_to_substitute_for_resolution", False)),
            "diagnostic_contract": dict(be_artifacts.get("diagnostic_contract", {})) if isinstance(be_artifacts.get("diagnostic_contract"), Mapping) else {},
            "bemultipliers_status": str(be_artifacts.get("bemultipliers_status", "unreported")),
            "a_multiplier_1_shape": str(be_artifacts.get("a_multiplier_1_shape", "")),
            "a_multiplier_1_matrix": str(be_artifacts.get("a_multiplier_1_matrix", "")),
            "be_exactness_source": str(be_artifacts.get("be_exactness_source", "")),
            "backend_interpretation": str(be_artifacts.get("interpretation", "No Buchsbaum-Eisenbud multiplier output is inferred.")),
        },
        "cas_backend": real.get("backend"),
    }

def _table_trace(headers: Sequence[str], columns: Sequence[Sequence[Any]]) -> go.Table:
    width = max((len(col) for col in columns), default=0)
    padded: List[List[str]] = []
    for col in columns:
        vals = [str(v) for v in col]
        vals.extend([""] * max(0, width - len(vals)))
        padded.append(vals)
    return go.Table(
        header=dict(
            values=[html.escape(str(h)) for h in headers],
            fill_color="#10243f",
            line_color="#345f8f",
            font=dict(color="#e8f2ff", size=13),
            align="left",
        ),
        cells=dict(
            values=padded,
            fill_color="#07111f",
            line_color="#203d5e",
            font=dict(color="#d7e8ff", size=12),
            align="left",
            height=28,
        ),
    )


def _m2_selected_staircase_resolution(m2: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only a certified real CAS free resolution.

    Older staircase diagnostics are useful module-support displays, but they are
    not free resolutions. They must not be substituted for Macaulay2/Singular/Sage
    resolution output in CAS tables or derived-similarity calculations.
    """
    certified = _cas_real_resolution_display(m2.get("real_free_resolution", {})) if isinstance(m2, Mapping) else {}
    if certified.get("available"):
        return certified
    return {}


def _rank_invariant_columns(bifiltration: Mapping[str, Any], max_rows: int = 18) -> Tuple[List[str], List[List[str]]]:
    samples = bifiltration.get("rank_invariant_samples") if isinstance(bifiltration, Mapping) else None
    rows = [row for row in samples if isinstance(row, Mapping)] if isinstance(samples, list) else []
    if not rows:
        return ["source grade", "target grade", "H0 rank", "source monomial", "target monomial"], [["unavailable"], [""], [""], [""], ["no computed rank-invariant samples in bifiltration report"]]

    def _grade_text(value: Any) -> str:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
            return f"({int(value[0] or 0)}, {int(value[1] or 0)})"
        return str(value)

    def _monomial_text(value: Any) -> str:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
            return f"x_level^{int(value[0] or 0)} x_radius^{int(value[1] or 0)}"
        return "unavailable"

    rows = rows[:max_rows]
    return (
        ["source grade", "target grade", "H0 rank", "source monomial", "target monomial"],
        [
            [_grade_text(row.get("source_grade")) for row in rows],
            [_grade_text(row.get("target_grade")) for row in rows],
            [str(row.get("h0_rank", "")) for row in rows],
            [_monomial_text(row.get("source_grade")) for row in rows],
            [_monomial_text(row.get("target_grade")) for row in rows],
        ],
    )


def _m2_betti_columns(m2: Mapping[str, Any]) -> Tuple[List[str], List[List[str]]]:
    resolution = _m2_selected_staircase_resolution(m2)
    rows = list((resolution.get("betti_table_rows") if resolution else []) or [])
    degrees = sorted({int(r.get("homological_degree", 0) or 0) for r in rows}) or [0]
    def _shift_display(row: Mapping[str, Any]) -> str:
        if row.get("shift_display") is not None:
            return str(row.get("shift_display"))
        md = row.get("multidegree", [0, 0])
        if isinstance(md, (list, tuple)):
            return "(" + ",".join(str(int(v)) for v in md[:2]) + ")"
        return "(0,0)"
    shifts = sorted({_shift_display(r) for r in rows}) or ["(0,0)"]
    lookup: Dict[Tuple[str, int], int] = defaultdict(int)
    for r in rows:
        lookup[(_shift_display(r), int(r.get("homological_degree", 0) or 0))] += int(r.get("multiplicity", r.get("rank", 0)) or 0)
    headers = ["multidegree"] + [f"F_{d}" for d in degrees]
    columns: List[List[str]] = [shifts]
    for d in degrees:
        columns.append([str(lookup.get((s, d), 0)) for s in shifts])
    return headers, columns


def _m2_free_module_columns(m2: Mapping[str, Any]) -> Tuple[List[str], List[List[str]]]:
    resolution = _m2_selected_staircase_resolution(m2)
    modules = list((resolution.get("free_modules") if resolution else []) or [])

    def _module_order(item: Mapping[str, Any]) -> int:
        name = str(item.get("name", item.get("module", "F_0")))
        if "_" in name:
            try:
                return int(name.rsplit("_", 1)[1])
            except ValueError:
                pass
        return int(item.get("degree", 0) or 0)

    modules = sorted([m for m in modules if isinstance(m, Mapping)], key=_module_order)
    if not modules:
        return ["module", "rank", "display"], [["unavailable"], ["0"], ["no certified/scoped real free resolution; chain presentation intentionally not rendered as a resolution"]]
    display_label = "scoped monomial-ideal resolution display" if resolution else "chain-module display"
    return (
        ["module", "rank", display_label],
        [
            [str(m.get("module", m.get("name", f"F_{i}"))) for i, m in enumerate(modules)],
            [str(m.get("rank", 0)) for m in modules],
            [str(m.get("display", "")) for m in modules],
        ],
    )


def _m2_differential_columns(m2: Mapping[str, Any], max_rows: int = 18) -> Tuple[List[str], List[List[str]]]:
    resolution = _m2_selected_staircase_resolution(m2)
    differentials = list((resolution.get("differentials") if resolution else []) or [])[:max_rows]
    if not differentials:
        return ["map", "shape", "rank", "matrix preview"], [["unavailable"], [""], [""], ["no certified/scoped real resolution differentials; chain boundary matrices intentionally not rendered as resolution maps"]]
    maps: List[str] = []
    shapes: List[str] = []
    ranks: List[str] = []
    previews: List[str] = []
    for d in differentials:
        preview = d.get("matrix_preview", [])
        if isinstance(preview, list):
            preview_text = " ; ".join(str(row) for row in preview[:4])
        else:
            preview_text = str(preview)
        maps.append(str(d.get("map", d.get("display", d.get("name", "d_i")))))
        shapes.append(str(d.get("shape", "")))
        ranks.append(str(d.get("rank", d.get("rank_over_F2_incidence", ""))))
        previews.append(_json_clip(preview_text, 220))
    return ["map", "shape", "rank", "matrix preview"], [maps, shapes, ranks, previews]



def _m2_syzygy_columns(m2: Mapping[str, Any], max_rows: int = 18) -> Tuple[List[str], List[List[str]]]:
    resolution = _m2_selected_staircase_resolution(m2)
    syzygies = resolution.get("syzygies") if isinstance(resolution.get("syzygies"), Mapping) else {}
    rows = syzygies.get("syzygy_generators") if isinstance(syzygies.get("syzygy_generators"), list) else []
    rows = [row for row in rows if isinstance(row, Mapping)][:max_rows]
    if not rows:
        reason = "no certified Macaulay2 syzygy diagnostics; adjacent LCM staircase syzygies remain scoped monomial-ideal theorem data, not CAS syzygy output"
        if isinstance(syzygies, Mapping) and syzygies.get("reason"):
            reason = str(syzygies.get("reason"))
        return ["order", "differential", "multidegree", "source", "no-proxy policy"], [["unavailable"], [""], [""], [""], [reason]]
    orders: list[str] = []
    differentials: list[str] = []
    multidegrees: list[str] = []
    sources: list[str] = []
    policies: list[str] = []
    policy = str(syzygies.get("no_proxy_policy", "Syzygies are rendered only from certified Macaulay2 resolution maps."))
    for row in rows:
        md = row.get("multidegree", [])
        if isinstance(md, Sequence) and not isinstance(md, (str, bytes)):
            md_text = "(" + ",".join(str(int(v)) for v in list(md)[:2]) + ")"
        else:
            md_text = str(row.get("shift_display", ""))
        orders.append(str(row.get("syzygy_order", "")))
        differentials.append(str(row.get("differential", row.get("source_free_module", ""))))
        multidegrees.append(md_text)
        sources.append(str(row.get("source", "macaulay2_resolution_differential_source_degrees")))
        policies.append(policy)
    return ["order", "differential", "multidegree", "source", "no-proxy policy"], [orders, differentials, multidegrees, sources, policies]


def _m2_ideal_diagnostic_columns(m2: Mapping[str, Any]) -> Tuple[List[str], List[List[str]]]:
    resolution = _m2_selected_staircase_resolution(m2)
    if not resolution:
        return ["kind", "name", "order/index", "ideal", "source/method"], [["unavailable"], [""], [""], [""], ["no certified CAS resolution; Fitting/minor diagnostics are unavailable"]]
    ideal_diag_raw = resolution.get("ideal_diagnostics")
    ideal_diag = ideal_diag_raw if isinstance(ideal_diag_raw, Mapping) else {}
    rows: list[tuple[str, str, str, str, str]] = []
    for row in ideal_diag.get("fitting_invariants", []) if isinstance(ideal_diag.get("fitting_invariants"), list) else []:
        if not isinstance(row, Mapping):
            continue
        order = f"j={row.get('fitting_index', '')}; I_{row.get('determinantal_order', '')}"
        rows.append(("Fitting invariant", str(row.get("name", "")), order, str(row.get("ideal_text", "")), str(row.get("method", row.get("source", "")))))
    for row in ideal_diag.get("determinantal_minors", []) if isinstance(ideal_diag.get("determinantal_minors"), list) else []:
        if not isinstance(row, Mapping):
            continue
        rows.append(("determinantal minors", str(row.get("name", "")), f"order={row.get('minor_order', '')}", str(row.get("ideal_text", "")), str(row.get("method", row.get("source", "")))))
    if not rows:
        if not ideal_diag:
            reason = "certified resolution lacks ideal_diagnostics certificate"
        else:
            reason = str(ideal_diag.get("reason") or ideal_diag.get("error") or "certified ideal_diagnostics certificate has no Fitting/minor rows")
        rows = [("unavailable", "", "", "", reason)]
    return ["kind", "name", "order/index", "ideal", "source/method"], [[row[idx] for row in rows] for idx in range(5)]


def _be_rank_condition_image_values(be_rank: Mapping[str, Any]) -> Mapping[str, Any]:
    current = be_rank.get("image_rank_values_implied_by_exact_rank_identity")
    if isinstance(current, Mapping):
        return current
    legacy = be_rank.get("image_rank_estimates_by_differential")
    if isinstance(legacy, Mapping):
        return legacy
    return {}


def _m2_be_diagnostic_columns(m2: Mapping[str, Any]) -> Tuple[List[str], List[List[str]]]:
    resolution = _m2_selected_staircase_resolution(m2)
    if not resolution:
        return ["diagnostic", "object", "value", "certificate/scope"], [["unavailable"], [""], [""], ["no certified CAS resolution; Buchsbaum-Eisenbud diagnostics are unavailable"]]
    res_be_raw = resolution.get("buchsbaum_eisenbud_diagnostics")
    be_rank_raw = resolution.get("buchsbaum_eisenbud_rank_conditions")
    grade_depth_raw = resolution.get("grade_depth_regular_diagnostics")
    res_be = res_be_raw if isinstance(res_be_raw, Mapping) else {}
    be_rank = be_rank_raw if isinstance(be_rank_raw, Mapping) else {}
    grade_depth = grade_depth_raw if isinstance(grade_depth_raw, Mapping) else {}
    rows: list[tuple[str, str, str, str]] = []
    if res_be:
        if "exactness_certified" in res_be:
            rows.append(("CAS exactness", "resolution", str(res_be.get("exactness_certified")), str(res_be.get("certificate", ""))))
        if "minimality_certified" in res_be:
            rows.append(("CAS minimality", "resolution", str(res_be.get("minimality_certified")), str(res_be.get("grading_scope", ""))))
        if any(key in res_be for key in ("bemultipliers_status", "a_multiplier_1_shape", "a_multiplier_1_matrix")):
            rows.append(("BEMultipliers", str(res_be.get("bemultipliers_status", "unreported")), str(res_be.get("a_multiplier_1_shape", "")), _json_clip(res_be.get("a_multiplier_1_matrix", ""), 220)))
            rows.append((
                "BEMultipliers contract",
                "post-resolution diagnostic only",
                f"safe_render={bool(res_be.get('safe_to_render_multiplier_output'))}; substitute={bool(res_be.get('safe_to_substitute_for_resolution'))}",
                f"resolution_backend={bool(res_be.get('is_resolution_backend'))}; requires_certified_macaulay2_chain_complex={bool(res_be.get('requires_certified_macaulay2_chain_complex', True))}",
            ))
    image_ranks = _be_rank_condition_image_values(be_rank)
    shapes = be_rank.get("differential_shapes", {}) if isinstance(be_rank.get("differential_shapes"), Mapping) else {}
    shape_bounds = be_rank.get("shape_bounds_hold")
    for name in sorted(set(image_ranks) | set(shapes)):
        rows.append(("BE rank condition", str(name), f"implied_rank={image_ranks.get(name, 'unavailable')}; shape={shapes.get(name, 'unavailable')}", f"shape_bounds_hold={shape_bounds}; independent_certificate={be_rank.get('is_independent_certificate', False)}"))
    rank_ideal_rows = grade_depth.get("rank_ideal_diagnostics", []) if isinstance(grade_depth.get("rank_ideal_diagnostics"), list) else []
    for row in rank_ideal_rows:
        if not isinstance(row, Mapping):
            continue
        rows.append((
            "grade/depth diagnostic",
            f"d{row.get('homological_degree', '')}",
            f"rank={row.get('rank', 'unavailable')}; codim={row.get('rank_ideal_codim', 'unavailable')}; depth={row.get('rank_ideal_depth', 'unavailable')}",
            f"grade_lower_bound_holds={row.get('grade_lower_bound_holds', 'unavailable')}; regular_element_certificate={grade_depth.get('regular_element_certificate_available', False)}",
        ))
    if grade_depth.get("regular_element_certificate_reason"):
        rows.append(("regular-element note", "CAS grade/depth", "not substituted", str(grade_depth.get("regular_element_certificate_reason"))))
    if be_rank.get("paper_method_note"):
        rows.append(("method note", "Buchsbaum-Eisenbud", "not inferred as exactness", str(be_rank.get("paper_method_note"))))
    if not rows:
        if not res_be and not be_rank:
            reason = "certified resolution lacks Buchsbaum-Eisenbud diagnostic certificates"
        else:
            reason = str(res_be.get("reason") or be_rank.get("reason") or res_be.get("error") or be_rank.get("error") or "Buchsbaum-Eisenbud diagnostic certificates contain no renderable rows")
        rows = [("unavailable", "", "", reason)]
    return ["diagnostic", "object", "value", "certificate/scope"], [[row[idx] for row in rows] for idx in range(4)]


def _m2_certificate_columns(m2: Mapping[str, Any], bifiltration: Mapping[str, Any]) -> Tuple[List[str], List[List[str]]]:
    cert = m2.get("chain_complex_certificate") if isinstance(m2, Mapping) else {}
    cert = cert if isinstance(cert, Mapping) else {}
    be = bifiltration.get("buchsbaum_eisenbud") if isinstance(bifiltration, Mapping) else {}
    be = be if isinstance(be, Mapping) else {}
    rank_inv = bifiltration.get("rank_invariant") if isinstance(bifiltration, Mapping) else {}
    rank_inv = rank_inv if isinstance(rank_inv, Mapping) else {}
    resolution = _m2_selected_staircase_resolution(m2)
    res_be = resolution.get("buchsbaum_eisenbud_diagnostics", {}) if isinstance(resolution.get("buchsbaum_eisenbud_diagnostics"), Mapping) else {}
    ideal_diag = resolution.get("ideal_diagnostics", {}) if isinstance(resolution.get("ideal_diagnostics"), Mapping) else {}
    be_rank = resolution.get("buchsbaum_eisenbud_rank_conditions", {}) if isinstance(resolution.get("buchsbaum_eisenbud_rank_conditions"), Mapping) else {}
    grade_depth = resolution.get("grade_depth_regular_diagnostics", {}) if isinstance(resolution.get("grade_depth_regular_diagnostics"), Mapping) else {}
    cert_summary = resolution.get("certificate_summary", {}) if isinstance(resolution.get("certificate_summary"), Mapping) else {}
    syzygies = resolution.get("syzygies", {}) if isinstance(resolution.get("syzygies"), Mapping) else {}
    items = [
        ("ring", resolution.get("ring", m2.get("ring", bifiltration.get("module_ring", "F2[x_level,x_radius]")))),
        ("field", m2.get("field", "F2")),
        ("chain object", "finite chain presentation; not a free resolution"),
        ("scoped real resolution", bool(resolution)),
        ("resolution scope", resolution.get("scope", "unavailable")),
        ("object resolved", resolution.get("object_resolved", "unavailable")),
        ("minimality certified", res_be.get("minimality_certified", False)),
        ("exactness certified", res_be.get("exactness_certified", False)),
        ("CAS certificate type", cert_summary.get("certificate_type", "unavailable")),
        ("CAS homogeneous presentation", cert_summary.get("homogeneous_presentation", "unavailable")),
        ("CAS input sha256", cert_summary.get("input_sha256", "unavailable")),
        ("CAS no-proxy policy", cert_summary.get("no_proxy_policy", "unavailable")),
        ("BEMultiplier output available", res_be.get("multiplier_output_available", False)),
        ("BEMultipliers safe render", res_be.get("safe_to_render_multiplier_output", False)),
        ("BEMultipliers is resolution backend", res_be.get("is_resolution_backend", False)),
        ("BEMultipliers substitute for resolution", res_be.get("safe_to_substitute_for_resolution", False)),
        ("BEMultipliers requires certified M2 complex", res_be.get("requires_certified_macaulay2_chain_complex", True)),
        ("BEMultipliers status", res_be.get("bemultipliers_status", "unreported")),
        ("aMultiplier(1) shape", res_be.get("a_multiplier_1_shape", "")),
        ("BE rank conditions", _json_clip(be_rank, 260)),
        ("grade/depth diagnostics", _json_clip(grade_depth, 520)),
        ("regular-element certificate", grade_depth.get("regular_element_certificate_available", False)),
        ("ideal diagnostics", _json_clip(ideal_diag, 260)),
        ("not full persistence-module resolution", resolution.get("not_full_persistence_module_resolution", True)),
        ("derived equivalence certified", cert.get("derived_equivalence_certified", False)),
        ("Buchsbaum-Eisenbud exactness", be.get("passes_exactness_necessary_checks", False)),
        ("CAS Fitting ideals", _json_clip(resolution.get("fitting_ideals", {}), 260)),
        ("CAS minors", _json_clip(resolution.get("minors", {}), 260)),
        ("CAS syzygy diagnostics", _json_clip(syzygies, 360)),
        ("rank invariant samples", rank_inv.get("num_samples", 0)),
        ("radius grade policy", bifiltration.get("radius_grade_policy", "")),
    ]
    return ["diagnostic", "value"], [[k for k, _ in items], [str(v) for _, v in items]]


def _m2_staircase_trace_payload(m2: Mapping[str, Any]) -> Dict[str, List[Any]]:
    staircase = m2.get("staircase") if isinstance(m2, Mapping) else {}
    staircase = staircase if isinstance(staircase, Mapping) else {}
    resolution = _m2_selected_staircase_resolution(m2)
    if resolution:
        staircase = {
            **dict(staircase),
            "generator_bidegrees": resolution.get("minimal_generators", []),
            "minimal_antichain_candidates": [row.get("bidegree", [0, 0]) for row in resolution.get("minimal_generators", []) if isinstance(row, Mapping)],
            "adjacent_lcm_syzygy_candidates": resolution.get("adjacent_lcm_syzygies", []),
        }
    def _xy(items: Any) -> Tuple[List[float], List[float], List[str]]:
        xs: List[float] = []
        ys: List[float] = []
        labels: List[str] = []
        for item in items or []:
            if not isinstance(item, Mapping):
                continue
            bg = item.get("bidegree", item.get("lcm_bidegree", item.get("shift", [0, 0])))
            if not isinstance(bg, (list, tuple)) or len(bg) < 2:
                continue
            xs.append(float(bg[0]))
            ys.append(float(bg[1]))
            labels.append(str(item.get("module", item.get("source", bg))))
        return xs, ys, labels
    gx, gy, gl = _xy(staircase.get("generators") or staircase.get("generator_bidegrees"))
    antichain_rows = staircase.get("minimal_antichain") or staircase.get("minimal_antichain_candidates")
    if antichain_rows and all(isinstance(item, (list, tuple)) for item in antichain_rows):
        antichain_rows = [{"bidegree": list(item), "module": f"antichain {idx}"} for idx, item in enumerate(antichain_rows)]
    ax, ay, al = _xy(antichain_rows)
    syzygy_rows = staircase.get("first_syzygy_lcms") or staircase.get("adjacent_lcm_syzygy_candidates")
    sx, sy, sl = _xy(syzygy_rows)
    return {"gen_x": gx, "gen_y": gy, "gen_label": gl, "anti_x": ax, "anti_y": ay, "anti_label": al, "syz_x": sx, "syz_y": sy, "syz_label": sl}



def _write_two_parameter_bifiltration_staircase_html(
    path: Path,
    bifiltration: Mapping[str, Any],
    *,
    title: str,
) -> str:
    """Write a stacked, Miller-Sturmfels-style bifiltration page.

    The first panel is a true two-variable exponent-lattice picture: horizontal
    coordinate is the radius exponent, vertical coordinate is the reasoning-level
    exponent. The second panel keeps the same data in 3D with fiber rank as z and
    only a small visual layer offset separating H0/H1 modules.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plotly_asset = path.parent / "plotly.min.js"
    if not plotly_asset.exists():
        plotly_asset.write_text(get_plotlyjs(), encoding="utf-8")

    rows = [r for r in (bifiltration.get("fiber_rows") or bifiltration.get("fiber_rank_profile") or []) if isinstance(r, Mapping)] if isinstance(bifiltration, Mapping) else []
    m2 = _m2_style_report_from_bifiltration(bifiltration)
    rank_inv_rows = [row for row in (bifiltration.get("rank_invariant_samples") if isinstance(bifiltration, Mapping) else []) or [] if isinstance(row, Mapping)]

    def _grade(row: Mapping[str, Any]) -> tuple[int, int]:
        grade = row.get("grade")
        if isinstance(grade, Sequence) and not isinstance(grade, (str, bytes)) and len(grade) >= 2:
            return int(grade[0] or 0), int(grade[1] or 0)
        return int(row.get("level", 0) or 0), int(row.get("radius_grade", row.get("radius", 0)) or 0)

    def _beta(row: Mapping[str, Any], dim: int) -> int:
        beta = row.get("betti", row.get("beta", {}))
        if isinstance(beta, Mapping):
            return int(beta.get(str(dim), beta.get(dim, 0)) or 0)
        return 0

    levels = sorted({_grade(r)[0] for r in rows})
    radius_grades = sorted({_grade(r)[1] for r in rows})
    h0 = {_grade(r): _beta(r, 0) for r in rows}
    h1 = {_grade(r): _beta(r, 1) for r in rows}
    if not levels or not radius_grades:
        levels = [0]
        radius_grades = [0]
        h0 = {(0, 0): 0}
        h1 = {(0, 0): 0}

    h1_grid = np.asarray([[float(h1.get((lvl, rg), 0)) for rg in radius_grades] for lvl in levels], dtype=float)
    h0_grid = np.asarray([[float(h0.get((lvl, rg), 0)) for rg in radius_grades] for lvl in levels], dtype=float)
    max_h1 = float(np.nanmax(h1_grid)) if h1_grid.size else 0.0
    max_h0 = float(np.nanmax(h0_grid)) if h0_grid.size else 0.0

    generator_counts: dict[tuple[int, int, int], int] = defaultdict(int)
    generator_examples: dict[tuple[int, int, int], str] = {}
    for gen in bifiltration.get("chain_module_generators", []) if isinstance(bifiltration, Mapping) else []:
        if not isinstance(gen, Mapping):
            continue
        bg = gen.get("multidegree")
        if not (isinstance(bg, Sequence) and not isinstance(bg, (str, bytes)) and len(bg) >= 2):
            continue
        try:
            lvl = int(bg[0] or 0)
            rg = int(bg[1] or 0)
            dim = int(gen.get("homological_degree", 0) or 0)
        except (TypeError, ValueError):
            continue
        key = (lvl, rg, dim)
        generator_counts[key] += 1
        if key not in generator_examples:
            simplex = gen.get("simplex", [])
            generator_examples[key] = _json_clip(simplex, 220)

    fig_module = go.Figure()
    max_radius = max(radius_grades) if radius_grades else 1
    max_level = max(levels) if levels else 1
    axis_pad_x = max(1, int(math.ceil(0.04 * max(1, max_radius))))
    axis_pad_y = 0.55

    generator_points_by_dim: dict[int, list[tuple[int, int, int, str]]] = defaultdict(list)
    for (lvl, rg, dim), count in generator_counts.items():
        generator_points_by_dim[int(dim)].append((int(lvl), int(rg), int(count), generator_examples.get((lvl, rg, dim), "")))

    def _minimal_antichain(points: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        unique = sorted(set(points), key=lambda q: (q[1], q[0]))
        mins: list[tuple[int, int]] = []
        for lvl, rg in unique:
            dominated = any(l0 <= lvl and r0 <= rg and (l0, r0) != (lvl, rg) for l0, r0 in unique)
            if not dominated:
                mins.append((lvl, rg))
        return sorted(mins, key=lambda q: (q[1], -q[0]))

    chain_support = sorted({(lvl, rg) for (lvl, rg, _dim) in generator_counts})
    chain_minimal = _minimal_antichain(chain_support)
    if not chain_minimal:
        chain_minimal = _minimal_antichain([(lvl, rg) for lvl in levels for rg in radius_grades if h0.get((lvl, rg), 0) or h1.get((lvl, rg), 0)])

    orthant_colors = {0: "rgba(94,234,212,0.10)", 1: "rgba(125,211,252,0.12)", 2: "rgba(250,204,21,0.11)"}
    orthant_lines = {0: "rgba(94,234,212,0.28)", 1: "rgba(125,211,252,0.34)", 2: "rgba(250,204,21,0.32)"}
    shapes: list[dict[str, Any]] = []
    for dim, items in sorted(generator_points_by_dim.items()):
        for lvl, rg in _minimal_antichain([(lvl, rg) for lvl, rg, _count, _ex in items]):
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=rg - 0.48,
                    x1=max_radius + axis_pad_x + 0.48,
                    y0=lvl - 0.48,
                    y1=max_level + axis_pad_y,
                    line=dict(color=orthant_lines.get(dim, "rgba(192,132,252,0.30)"), width=1),
                    fillcolor=orthant_colors.get(dim, "rgba(192,132,252,0.10)"),
                    layer="below",
                )
            )

    if h1_grid.size:
        fig_module.add_trace(
            go.Heatmap(
                x=radius_grades,
                y=levels,
                z=h1_grid,
                name="actual H1 fiber-rank surface",
                colorscale=[[0.0, "rgba(8,18,34,0.05)"], [0.18, "rgba(21,94,117,0.32)"], [0.55, "rgba(45,212,191,0.48)"], [1.0, "rgba(250,204,21,0.62)"]],
                zmin=0,
                zmax=max(max_h1, 1.0),
                colorbar=dict(title="H1 fiber rank", len=0.45, y=0.74),
                hovertemplate="x_radius=%{x}<br>x_level=%{y}<br>actual beta_1=%{z}<extra></extra>",
                showscale=True,
                opacity=0.82,
            )
        )

    lattice_x: list[int] = []
    lattice_y: list[int] = []
    lattice_text: list[str] = []
    for lvl in levels:
        for rg in radius_grades:
            lattice_x.append(rg)
            lattice_y.append(lvl)
            lattice_text.append(
                f"lattice bidegree x_level^{lvl} x_radius^{rg}<br>H0 fiber rank={h0.get((lvl, rg), 0)}<br>H1 fiber rank={h1.get((lvl, rg), 0)}"
            )
    fig_module.add_trace(
        go.Scatter(
            x=lattice_x,
            y=lattice_y,
            mode="markers",
            name="actual grid fibers / monomial lattice",
            marker=dict(color="rgba(232,238,248,0.58)", size=5.2, symbol="circle", line=dict(color="rgba(232,238,248,0.70)", width=0.3)),
            text=lattice_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    h0_x: list[int] = []
    h0_y: list[float] = []
    h0_rank: list[float] = []
    h0_text: list[str] = []
    h1_x: list[int] = []
    h1_y: list[float] = []
    h1_rank: list[float] = []
    h1_text: list[str] = []
    for lvl in levels:
        for rg in radius_grades:
            b0 = float(h0.get((lvl, rg), 0))
            b1 = float(h1.get((lvl, rg), 0))
            if b0 > 0:
                h0_x.append(rg)
                h0_y.append(lvl - 0.055)
                h0_rank.append(b0)
                h0_text.append(f"H0 module fiber<br>bidegree=(x_level^{lvl}, x_radius^{rg})<br>actual beta_0={b0:g}<br>actual beta_1={b1:g}")
            if b1 > 0:
                h1_x.append(rg)
                h1_y.append(lvl + 0.055)
                h1_rank.append(b1)
                h1_text.append(f"H1 module fiber<br>bidegree=(x_level^{lvl}, x_radius^{rg})<br>actual beta_1={b1:g}<br>actual beta_0={b0:g}")
    if h0_x:
        fig_module.add_trace(
            go.Scatter(
                x=h0_x,
                y=h0_y,
                mode="markers",
                name="H0 fiber-rank samples",
                marker=dict(color=h0_rank, colorscale="Teal", cmin=0, cmax=max(max_h0, 1.0), size=[6.0 + 5.0 * min(v / max(max_h0, 1.0), 1.0) for v in h0_rank], symbol="circle-open", line=dict(color="#5eead4", width=1.4)),
                text=h0_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )
    if h1_x:
        fig_module.add_trace(
            go.Scatter(
                x=h1_x,
                y=h1_y,
                mode="markers",
                name="H1 fiber-rank samples",
                marker=dict(color=h1_rank, colorscale="Plasma", cmin=0, cmax=max(max_h1, 1.0), size=[7.0 + 10.0 * min(v / max(max_h1, 1.0), 1.0) for v in h1_rank], symbol="diamond", line=dict(color="#e8f2ff", width=0.8), opacity=0.94),
                text=h1_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    dim_symbols = {0: "circle", 1: "square", 2: "triangle-up"}
    dim_colors = {0: "#5eead4", 1: "#7dd3fc", 2: "#facc15"}
    for dim, items in sorted(generator_points_by_dim.items()):
        items = sorted(items, key=lambda x: (x[0], x[1]))
        fig_module.add_trace(
            go.Scatter(
                x=[rg for lvl, rg, count, ex in items],
                y=[lvl + 0.12 + 0.045 * dim for lvl, rg, count, ex in items],
                mode="markers",
                name=f"C_{dim} shifted free-module generators",
                marker=dict(color=dim_colors.get(dim, "#c084fc"), size=[min(24, 8 + 2.3 * math.sqrt(count)) for lvl, rg, count, ex in items], symbol=dim_symbols.get(dim, "diamond"), line=dict(color="#e8f2ff", width=0.8), opacity=0.90),
                text=[f"C{dim}" for _lvl, _rg, _count, _ex in items],
                textposition="top center",
                customdata=[_json_clip(ex, 220) for lvl, rg, count, ex in items],
                hovertemplate="shifted free-module generator<br>%{text}<br>x_radius=%{x}<br>x_level display=%{y:.2f}<br>example simplex=%{customdata}<extra></extra>",
            )
        )

    if chain_minimal:
        stair_x = [rg for lvl, rg in chain_minimal]
        stair_y = [lvl for lvl, rg in chain_minimal]
        fig_module.add_trace(
            go.Scatter(
                x=stair_x,
                y=stair_y,
                mode="lines+markers+text",
                name="minimal multidegree antichain / staircase",
                line=dict(color="#ffd54a", width=4.0, shape="hv"),
                marker=dict(color="#ffd54a", size=14, symbol="diamond", line=dict(color="#e8f2ff", width=1.0)),
                text=[f"m{idx+1}" for idx, _ in enumerate(chain_minimal)],
                textposition="top right",
                hovertemplate="minimal bidegree generator %{text}<br>x_radius=%{x}<br>x_level=%{y}<extra></extra>",
            )
        )

    generator_coords = [(int(lvl), int(rg)) for (lvl, rg, _dim) in generator_counts.keys()]
    extent_coords = generator_coords + list(chain_minimal)
    max_x = max([rg for _lvl, rg in extent_coords] + radius_grades + [max_radius, 2]) + 1
    max_y = max([lvl for lvl, _rg in extent_coords] + levels + [max_level, 2]) + 0.75

    fig_module.add_trace(
        go.Scatter(
            x=[0, max_x, None, 0, 0],
            y=[0, 0, None, 0, max_y],
            mode="lines+text",
            name="coordinate one dimensional cone(s)",
            line=dict(color="rgba(110,231,249,0.95)", width=2.2, dash="dot"),
            text=["rho_radius", "", "", "rho_level", ""],
            textposition="top right",
            hovertemplate="coordinate one dimensional cone in Spec F2[x_level,x_radius]<extra></extra>",
        )
    )
    fig_module.update_layout(
        template="plotly_dark",
        title=dict(text="Miller-Sturmfels-style multigraded module diagram on the F2[x_level,x_radius] exponent lattice<br><sup>Columns are radius grades, rows are reasoning levels; heat color is the actual H1 fiber rank and markers are actual chain-generator bidegrees.</sup>", x=0.02),
        height=1040,
        margin=dict(l=104, r=96, t=124, b=88),
        paper_bgcolor="#050914",
        plot_bgcolor="#050914",
        shapes=shapes,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
        font=dict(color="#e8f2ff"),
    )
    fig_module.update_xaxes(title_text="x_radius exponent / radius grade", gridcolor="#203d5e", zerolinecolor="#6ee7f9", tickmode="auto", range=[-0.75, max_x + 0.35])
    fig_module.update_yaxes(title_text="x_level exponent / reasoning growth level", gridcolor="#203d5e", zerolinecolor="#6ee7f9", dtick=1, range=[-0.65, max_y + 0.35])

    structure_by_edge: dict[tuple[tuple[int, int], tuple[int, int]], Mapping[str, Any]] = {}
    structure_map_rows: list[Mapping[str, Any]] = []
    structure_direction_counts: Counter[str] = Counter()
    structure_field_counts: Counter[str] = Counter()
    structure_homology_dimensions: set[int] = set()
    structure_rank_rows: list[dict[str, Any]] = []
    for row in bifiltration.get("structure_maps", []) if isinstance(bifiltration, Mapping) else []:
        if not isinstance(row, Mapping):
            continue
        structure_map_rows.append(row)
        direction = str(row.get("direction", "unknown"))
        field = str(row.get("field", "unknown"))
        structure_direction_counts[direction] += 1
        structure_field_counts[field] += 1
        src = row.get("source_grade")
        tgt = row.get("target_grade")
        src_grade: list[int] | None = None
        tgt_grade: list[int] | None = None
        try:
            if isinstance(src, Sequence) and not isinstance(src, (str, bytes)) and len(src) >= 2:
                src_grade = [int(src[0]), int(src[1])]
            if isinstance(tgt, Sequence) and not isinstance(tgt, (str, bytes)) and len(tgt) >= 2:
                tgt_grade = [int(tgt[0]), int(tgt[1])]
        except (TypeError, ValueError):
            src_grade = None
            tgt_grade = None
        if src_grade is not None and tgt_grade is not None:
            structure_by_edge[((src_grade[0], src_grade[1]), (tgt_grade[0], tgt_grade[1]))] = row
        ranks = row.get("homology_rank", {})
        normalized_ranks: dict[str, Any] = {}
        if isinstance(ranks, Mapping):
            for key, value in ranks.items():
                try:
                    dim = int(key)
                except (TypeError, ValueError):
                    continue
                structure_homology_dimensions.add(dim)
                try:
                    normalized_ranks[str(dim)] = int(value)
                except (TypeError, ValueError):
                    normalized_ranks[str(dim)] = value
        structure_rank_rows.append(
            {
                "source_bidegree_x_level_x_radius": src_grade or [],
                "target_bidegree_x_level_x_radius": tgt_grade or [],
                "source_monomial": f"x_level^{src_grade[0]} x_radius^{src_grade[1]}" if src_grade is not None else None,
                "target_monomial": f"x_level^{tgt_grade[0]} x_radius^{tgt_grade[1]}" if tgt_grade is not None else None,
                "direction": direction,
                "field": field,
                "homology_rank": normalized_ranks,
                "method": str(row.get("method", "")),
            }
        )
    direction_counts_payload = {key: int(structure_direction_counts.get(key, 0)) for key in ("x_level", "x_radius")}
    for key, value in sorted(structure_direction_counts.items()):
        direction_counts_payload.setdefault(str(key), int(value))
    structure_fields = sorted(structure_field_counts)
    structure_map_summary = {
        "schema_version": "tropicalgt.two_parameter_structure_maps.v1",
        "source": "bifiltration.structure_maps",
        "source_grade_convention": "[x_level_exponent, x_radius_exponent]",
        "coefficient_ring": "F2[x_level,x_radius]",
        "actual_adjacent_map_count": len(structure_map_rows),
        "valid_grade_edge_count": len(structure_by_edge),
        "direction_counts": direction_counts_payload,
        "field": "F2" if structure_fields == ["F2"] else ("unavailable" if not structure_map_rows else "mixed_or_unavailable"),
        "field_counts": {str(key): int(value) for key, value in sorted(structure_field_counts.items())},
        "homology_dimensions_observed": sorted(structure_homology_dimensions),
        "east_north_structure_maps_present": bool(structure_direction_counts.get("x_level") and structure_direction_counts.get("x_radius")),
        "rank_rows": structure_rank_rows,
        "no_proxy_or_fallback": True,
    }
    structure_overlay_trace_names: list[str] = []
    structure_overlay_styles = {
        "x_level": ("actual x_level structure maps over F2", "rgba(94,234,212,0.74)", "triangle-up"),
        "x_radius": ("actual x_radius structure maps over F2", "rgba(250,204,21,0.74)", "triangle-right"),
    }
    for direction, (trace_name, color, marker_symbol) in structure_overlay_styles.items():
        edge_x: list[Any] = []
        edge_y: list[Any] = []
        edge_text: list[Any] = []
        for row in structure_rank_rows:
            if row.get("direction") != direction:
                continue
            src_grade = row.get("source_bidegree_x_level_x_radius")
            tgt_grade = row.get("target_bidegree_x_level_x_radius")
            if not (isinstance(src_grade, Sequence) and isinstance(tgt_grade, Sequence) and len(src_grade) >= 2 and len(tgt_grade) >= 2):
                continue
            ranks = row.get("homology_rank", {})
            h0_rank = ranks.get("0", "unavailable") if isinstance(ranks, Mapping) else "unavailable"
            h1_rank = ranks.get("1", "unavailable") if isinstance(ranks, Mapping) else "unavailable"
            hover = (
                "actual adjacent F2 structure map"
                f"<br>direction={direction}"
                f"<br>source=(x_level^{src_grade[0]}, x_radius^{src_grade[1]})"
                f"<br>target=(x_level^{tgt_grade[0]}, x_radius^{tgt_grade[1]})"
                f"<br>H0 rank={h0_rank}<br>H1 rank={h1_rank}"
            )
            edge_x.extend([int(src_grade[1]), int(tgt_grade[1]), None])
            edge_y.extend([int(src_grade[0]), int(tgt_grade[0]), None])
            edge_text.extend([hover, hover, None])
        if edge_x:
            structure_overlay_trace_names.append(trace_name)
            fig_module.add_trace(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines+markers",
                    name=trace_name,
                    line=dict(color=color, width=2.8, dash="solid"),
                    marker=dict(color=color, size=7.0, symbol=marker_symbol, line=dict(color="#f8fafc", width=0.45)),
                    text=edge_text,
                    hovertemplate="%{text}<extra></extra>",
                )
            )
    structure_map_summary["module_lattice_overlay_trace_names"] = structure_overlay_trace_names
    structure_map_summary["module_lattice_overlay_available"] = bool(structure_overlay_trace_names)

    fig_3d = go.Figure()
    dim_offsets = {0: -0.045, 1: 0.045}
    for dim, lookup in ((0, h0), (1, h1)):
        xs: list[int] = []
        ys: list[float] = []
        zs: list[int] = []
        text_rows: list[str] = []
        for lvl in levels:
            for rg in radius_grades:
                xs.append(rg)
                ys.append(lvl + dim_offsets[dim])
                val = int(lookup.get((lvl, rg), 0))
                zs.append(val)
                text_rows.append(f"H{dim} fiber<br>x_radius={rg}<br>x_level={lvl}<br>actual beta_{dim}={val}<br>visual y offset={dim_offsets[dim]:+.3f}")
        fig_3d.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="markers", name=f"H{dim} fiber-rank lattice", marker=dict(size=3.2 if dim == 0 else 4.0, color=zs, colorscale="Viridis" if dim == 0 else "Plasma", cmin=0, cmax=max(float(max(zs or [1])), 1.0), opacity=0.92, line=dict(color="#e8f2ff", width=0.25)), text=text_rows, hovertemplate="%{text}<extra></extra>"))
        edge_x: list[Any] = []
        edge_y: list[Any] = []
        edge_z: list[Any] = []
        edge_text: list[str] = []
        for lvl in levels:
            for rg in radius_grades:
                z = lookup.get((lvl, rg))
                if z is None:
                    continue
                for nxt in ((lvl + 1, rg), (lvl, rg + 1)):
                    zn = lookup.get(nxt)
                    if zn is None:
                        continue
                    map_row = structure_by_edge.get(((lvl, rg), nxt), {})
                    ranks = map_row.get("homology_rank", {}) if isinstance(map_row, Mapping) else {}
                    rank_value = ranks.get(str(dim), ranks.get(dim, "uncomputed")) if isinstance(ranks, Mapping) else "uncomputed"
                    hover = (
                        f"actual adjacent structure map over F2<br>H{dim} rank={rank_value}<br>"
                        f"source=(x_level^{lvl}, x_radius^{rg})<br>target=(x_level^{nxt[0]}, x_radius^{nxt[1]})<br>"
                        f"direction={map_row.get('direction', 'unknown') if isinstance(map_row, Mapping) else 'unknown'}"
                    )
                    edge_x.extend([rg, nxt[1], None])
                    edge_y.extend([lvl + dim_offsets[dim], nxt[0] + dim_offsets[dim], None])
                    edge_z.extend([z, zn, None])
                    edge_text.extend([hover, hover, None])
        fig_3d.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode="lines", name=f"H{dim} actual structure-map ranks", line=dict(color="rgba(250,204,21,0.30)" if dim == 0 else "rgba(249,168,212,0.38)", width=1.8), text=edge_text, hovertemplate="%{text}<extra></extra>"))
    fig_3d.update_layout(
        template="plotly_dark",
        title=dict(text="Actual fiber-rank lattice: z = beta_i, thin layer offset separates H0/H1 only visually", x=0.02),
        height=760,
        margin=dict(l=20, r=20, t=80, b=30),
        paper_bgcolor="#050914",
        font=dict(color="#e8f2ff"),
        scene=dict(
            xaxis_title="x_radius grade",
            yaxis_title="x_level grade (+ visual H_i offset)",
            zaxis_title="actual fiber rank beta_i",
            bgcolor="#050914",
            aspectmode="manual",
            aspectratio=dict(x=1.6, y=0.72, z=0.58),
            camera=dict(eye=dict(x=1.55, y=-1.75, z=1.02)),
            xaxis=dict(gridcolor="#315c86"),
            yaxis=dict(gridcolor="#315c86"),
            zaxis=dict(gridcolor="#315c86"),
        ),
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10)),
    )

    def _selected_staircase_resolution(dim: int, pts: Sequence[tuple[int, int]]) -> dict[str, Any]:
        variables = ["x_level", "x_radius"]
        raw_pts = [(int(lvl), int(rg)) for lvl, rg in pts]
        res = _bivariate_staircase_resolution_from_points(
            raw_pts,
            variables,
            ideal_name=f"I_C{dim}",
            source=f"actual C{dim} chain-generator bidegrees displayed in the staircase panel",
            auxiliary=False,
        )
        if res.get("available"):
            return res
        positive_pts = [pnt for pnt in raw_pts if pnt != (0, 0) and (pnt[0] > 0 or pnt[1] > 0)]
        positive_res = _bivariate_staircase_resolution_from_points(
            positive_pts,
            variables,
            ideal_name=f"I_C{dim}_positive_event",
            source=f"positive nonunit C{dim} bidegrees; the full ideal is unit, so this is a scoped event ideal",
            auxiliary=True,
        )
        if positive_res.get("available"):
            return positive_res
        return positive_res if positive_res else res

    def _staircase_resolution_html(dim: int, pts: Sequence[tuple[int, int]]) -> str:
        chosen = _selected_staircase_resolution(dim, pts)
        if not chosen.get("available"):
            reason = html.escape(str(chosen.get("reason", "no nontrivial bivariate monomial ideal")))
            return f"<div class='resolution-block unavailable'><h4>Exact bivariate monomial-ideal resolution</h4><p>{reason}</p></div>"

        def _rows_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], caption: str) -> str:
            head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
            body = []
            for row in rows:
                body.append("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>")
            return f"<div class='mini-table'><h5>{html.escape(caption)}</h5><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"

        module_rows = [[m.get("name", ""), m.get("rank", ""), m.get("display", "")] for m in chosen.get("free_modules", []) if isinstance(m, Mapping)]
        gen_rows = [[g.get("index", ""), tuple(g.get("bidegree", [])), g.get("monomial", "")] for g in chosen.get("minimal_generators", []) if isinstance(g, Mapping)]
        syz_rows = [[r.get("index", ""), tuple(r.get("lcm_bidegree", [])), r.get("relation", "")] for r in chosen.get("adjacent_lcm_syzygies", []) if isinstance(r, Mapping)]
        diff_rows = []
        for d in chosen.get("differentials", []) if isinstance(chosen.get("differentials"), list) else []:
            if not isinstance(d, Mapping):
                continue
            entries = d.get("entries") if isinstance(d.get("entries"), list) else []
            preview = "; ".join(
                f"({e.get('row')},{e.get('column')})={e.get('entry')}" for e in entries[:12] if isinstance(e, Mapping)
            )
            if len(entries) > 12:
                preview += "; ..."
            diff_rows.append([d.get("name", ""), d.get("display", ""), d.get("shape", ""), preview])
        hilbert_terms = []
        for term in chosen.get("hilbert_series_numerator_terms", []) if isinstance(chosen.get("hilbert_series_numerator_terms"), list) else []:
            if isinstance(term, Mapping):
                sign = "+" if int(term.get("sign", 1) or 1) > 0 else "-"
                hilbert_terms.append(f"{sign}{term.get('monomial', '')}")
        scope_note = ""
        if chosen.get("scope") == "auxiliary_two_variable_staircase_monomial_ideal":
            scope_note = "<p class='resolution-warning'>The full displayed ideal contains 1, hence S/I=0. The table below is the exact positive-event staircase resolution, kept separate from the full unit ideal.</p>"
        cert = chosen.get("buchsbaum_eisenbud_diagnostics", {}) if isinstance(chosen.get("buchsbaum_eisenbud_diagnostics"), Mapping) else {}
        toric = chosen.get("toric_exponent_chart", {}) if isinstance(chosen.get("toric_exponent_chart"), Mapping) else {}
        cone_rows = [[c.get("name", ""), c.get("primitive_generator", ""), c.get("variable", "")] for c in toric.get("ambient_one_dimensional_cones", []) if isinstance(c, Mapping)]
        return f"""
        <div class='resolution-block'>
          <h4>Miller-Sturmfels adjacent-LCM resolution of {html.escape(str(chosen.get('object_resolved', 'S/I')))}</h4>
          <p class='resolution-note'>Exact bivariate monomial-ideal theorem over {html.escape(str(chosen.get('ring', 'F2[x_level,x_radius]')))}. Scope: {html.escape(str(chosen.get('scope', '')))}. This is not asserted to be the full persistence-module resolution.</p>
          {scope_note}
          <p class='complex-line'>0 &rarr; F<sub>2</sub> &rarr; F<sub>1</sub> &rarr; F<sub>0</sub> &rarr; {html.escape(str(chosen.get('object_resolved', 'S/I')))} &rarr; 0</p>
          <div class='resolution-grid'>
            {_rows_table(['module', 'rank', 'summands'], module_rows, 'free modules')}
            {_rows_table(['i', 'bidegree', 'monomial'], gen_rows, 'minimal monomial generators')}
            {_rows_table(['i', 'LCM bidegree', 'adjacent syzygy'], syz_rows or [['-', '-', 'principal ideal: no first syzygy']], 'adjacent LCM syzygies')}
            {_rows_table(['map', 'display', 'shape', 'entries'], diff_rows, 'differentials over F2')}
            {_rows_table(['one dimensional cone', 'primitive generator', 'variable'], cone_rows, 'ambient coordinate one dimensional cone(s)')}
          </div>
          <p class='hilbert-line'>Hilbert numerator terms: {html.escape(' '.join(hilbert_terms[:24]) or 'unavailable')}</p>
          <p class='certificate-line'>{html.escape(str(cert.get('certificate', 'exact bivariate monomial staircase certificate')))}</p>
        </div>
        """

    def _svg_chain_staircase(dim: int, items: Sequence[tuple[int, int, int, str]], *, primary: bool = False) -> str:
        items = list(items)
        if not items:
            return ""
        pts = [(int(lvl), int(rg)) for lvl, rg, _count, _ex in items]
        mins = _minimal_antichain(pts)
        principal = len(mins) <= 1
        width = 1080 if primary else 860
        height = 640 if primary else 400
        left = 96
        right = 70
        top = 70
        bottom = 86
        x_max = max([max_radius] + [rg for _lvl, rg in pts] + [rg for _lvl, rg in mins] + [8])
        y_max = max([max_level] + [lvl for lvl, _rg in pts] + [lvl for lvl, _rg in mins] + [3])
        x_pad = max(2, int(math.ceil(0.04 * max(1, x_max))))
        y_pad = 1
        x_axis_max = x_max + x_pad
        y_axis_max = y_max + y_pad
        plot_w = width - left - right
        plot_h = height - top - bottom

        def sx(rg: float) -> float:
            return left + (float(rg) / max(1.0, float(x_axis_max))) * plot_w

        def sy(lvl: float) -> float:
            return top + (1.0 - float(lvl) / max(1.0, float(y_axis_max))) * plot_h

        def point_in_upset(lvl: int, rg: int) -> bool:
            return any(m_lvl <= lvl and m_rg <= rg for m_lvl, m_rg in mins)

        dots = []
        basis_count = 0
        generated_lattice_count = 0
        for lvl in range(0, int(y_axis_max) + 1):
            for rg in range(0, int(x_axis_max) + 1):
                inside = point_in_upset(lvl, rg)
                if inside:
                    generated_lattice_count += 1
                    continue
                basis_count += 1
                dot_r = 2.75 if primary else 2.05
                dots.append(
                    f"<circle cx='{sx(rg):.2f}' cy='{sy(lvl):.2f}' r='{dot_r:.2f}' fill='#f8fafc' opacity='0.98'>"
                    f"<title>C{dim} quotient-basis lattice point in S/I_C{dim}; x_radius={rg}, x_level={lvl}</title></circle>"
                )

        rects = []
        for m_lvl, m_rg in mins:
            x0 = sx(m_rg)
            y0 = sy(y_axis_max)
            x1 = sx(x_axis_max)
            y1 = sy(m_lvl)
            rects.append(
                f"<rect x='{x0:.2f}' y='{y0:.2f}' width='{max(0.0, x1 - x0):.2f}' height='{max(0.0, y1 - y0):.2f}' "
                "fill='none' stroke='rgba(226,232,240,0.26)' stroke-width='0.8' stroke-dasharray='5 7'>"
                f"<title>principal upward orthant generated by x_level^{m_lvl} x_radius^{m_rg}; dashed outline only</title></rect>"
            )

        boundary_points: list[tuple[float, float]] = []
        shade_polygon = ""
        if mins:
            ordered = sorted(mins, key=lambda q: (q[1], -q[0]))
            boundary_points.append((sx(ordered[0][1]), sy(y_axis_max)))
            prev_lvl = y_axis_max
            for lvl, rg in ordered:
                boundary_points.append((sx(rg), sy(prev_lvl)))
                boundary_points.append((sx(rg), sy(lvl)))
                prev_lvl = lvl
            boundary_points.append((sx(x_axis_max), sy(prev_lvl)))
            boundary = " ".join(f"{x:.2f},{y:.2f}" for x, y in boundary_points)
            shade_points = boundary_points + [(sx(x_axis_max), sy(y_axis_max))]
            shade_polygon = (
                f"<polygon points='{' '.join(f'{x:.2f},{y:.2f}' for x, y in shade_points)}' "
                "fill='rgba(148,163,184,0.54)' stroke='rgba(226,232,240,0.44)' stroke-width='1.2'>"
                f"<title>single upward-closed generated submodule I_C{dim} from the minimal antichain</title></polygon>"
            )
        else:
            boundary = ""

        gen_marks = []
        dim_color = dim_colors.get(dim, "#c084fc")
        minimal_set = set(mins)
        generator_formula = ", ".join(f"x_level^{lvl} x_radius^{rg}" for lvl, rg in mins[:8])
        if len(mins) > 8:
            generator_formula += ", ..."
        dominated_count = 0
        for lvl, rg, count, ex in sorted(items, key=lambda t: (t[0], t[1])):
            is_minimal = (lvl, rg) in minimal_set
            if is_minimal:
                r = 10.5 if primary else 8.0
                opacity = "0.98"
                stroke = "#f8fafc"
                stroke_width = "1.8"
                title_kind = "minimal monomial generator of the displayed submodule"
            else:
                dominated_count += 1
                r = 2.25 if primary else 1.65
                opacity = "0.42"
                stroke = "rgba(248,250,252,0.24)"
                stroke_width = "0.4"
                title_kind = "dominated generator bidegree; redundant for the same upward closed submodule"
            gen_marks.append(
                f"<circle cx='{sx(rg):.2f}' cy='{sy(lvl):.2f}' r='{r:.2f}' "
                f"fill='{dim_color}' fill-opacity='{opacity}' stroke='{stroke}' stroke-width='{stroke_width}'>"
                f"<title>C{dim} {title_kind}; x_level={lvl}, x_radius={rg}; multiplicity={count}; example={html.escape(str(ex))}</title></circle>"
            )

        min_labels = []
        show_labels = len(mins) <= (7 if primary else 4)
        for idx, (lvl, rg) in enumerate(mins, start=1):
            if not show_labels:
                continue
            min_labels.append(
                f"<text x='{sx(rg) + 12:.2f}' y='{sy(lvl) - 12:.2f}' fill='#f8fafc' font-size='{13 if primary else 11}' font-weight='700'>g{idx}=({lvl},{rg})</text>"
            )

        x_ticks = []
        stride = max(1, int(math.ceil(x_axis_max / (10 if primary else 7))))
        for rg in range(0, int(x_axis_max) + 1, stride):
            x_ticks.append(f"<line x1='{sx(rg):.2f}' x2='{sx(rg):.2f}' y1='{sy(0):.2f}' y2='{sy(0)+6:.2f}' stroke='#94a3b8'/><text x='{sx(rg):.2f}' y='{sy(0)+24:.2f}' fill='#cbd5e1' font-size='12' text-anchor='middle'>{rg}</text>")
        y_ticks = []
        for lvl in range(0, int(y_axis_max) + 1):
            y_ticks.append(f"<line x1='{sx(0)-6:.2f}' x2='{sx(0):.2f}' y1='{sy(lvl):.2f}' y2='{sy(lvl):.2f}' stroke='#94a3b8'/><text x='{sx(0)-12:.2f}' y='{sy(lvl)+4:.2f}' fill='#cbd5e1' font-size='12' text-anchor='end'>{lvl}</text>")

        card_class = "stair-card primary-staircase" if primary else "stair-card secondary-staircase"
        if principal:
            trivial = "<p class='svg-note'>Principal shifted module: a single minimal bidegree generates one upward orthant. It is shown compactly because there is no nontrivial staircase.</p>"
        else:
            trivial = "<p class='svg-note'>Nontrivial Miller-Sturmfels staircase: the gold boundary is the minimal antichain of actual generator bidegrees; the shaded upward-closed region is generated over F2[x_level,x_radius], and the unshaded lattice points are the displayed quotient-basis complement S/I_C.</p>"
        resolution_block = _staircase_resolution_html(dim, pts)
        return f"""
        <article class='{card_class}'>
          <h3>C{dim} shifted free module and monomial staircase over F2[x_level,x_radius]</h3>
          {trivial}
          <p class='formula'>I_C{dim} = &lt; {html.escape(generator_formula or '0')} &gt;</p>
          <svg viewBox='0 0 {width} {height}' role='img' aria-label='C{dim} Miller-Sturmfels bivariate staircase from actual generator bidegrees'>
            <defs><marker id='arrow-C{dim}' markerWidth='9' markerHeight='7' refX='8' refY='3.5' orient='auto'><polygon points='0 0, 9 3.5, 0 7' fill='#e2e8f0'/></marker></defs>
            <rect x='0' y='0' width='{width}' height='{height}' rx='14' fill='#141414' stroke='rgba(94,234,212,0.20)'/>
            {shade_polygon}
            {''.join(rects)}
            {''.join(dots)}
            <line x1='{sx(0):.2f}' y1='{sy(0):.2f}' x2='{sx(x_axis_max):.2f}' y2='{sy(0):.2f}' stroke='#e2e8f0' stroke-width='1.4' marker-end='url(#arrow-C{dim})'/>
            <line x1='{sx(0):.2f}' y1='{sy(0):.2f}' x2='{sx(0):.2f}' y2='{sy(y_axis_max):.2f}' stroke='#e2e8f0' stroke-width='1.4' marker-end='url(#arrow-C{dim})'/>
            <line x1='{sx(0):.2f}' y1='{sy(0):.2f}' x2='{sx(x_axis_max):.2f}' y2='{sy(0):.2f}' stroke='#67e8f9' stroke-width='2.2' stroke-dasharray='7 7'><title>x_radius one dimensional cone</title></line>
            <line x1='{sx(0):.2f}' y1='{sy(0):.2f}' x2='{sx(0):.2f}' y2='{sy(y_axis_max):.2f}' stroke='#67e8f9' stroke-width='2.2' stroke-dasharray='7 7'><title>x_level one dimensional cone</title></line>
            {''.join(x_ticks)}
            {''.join(y_ticks)}
            {f"<polyline points='{boundary}' fill='none' stroke='#facc15' stroke-width='4.2' stroke-linejoin='round'/>" if boundary else ''}
            <text x='{sx(max(1.0, x_axis_max * 0.72)):.2f}' y='{sy(max(1.0, y_axis_max * 0.62)):.2f}' fill='rgba(248,250,252,0.58)' font-size='{42 if primary else 24}' font-style='italic'>I_C{dim}</text>
            {''.join(gen_marks)}
            {''.join(min_labels)}
            <text x='{sx(x_axis_max)-6:.2f}' y='{sy(0)+46:.2f}' fill='#e2e8f0' font-size='17' text-anchor='end'>x_radius</text>
            <text x='{sx(0)-48:.2f}' y='{sy(y_axis_max)+8:.2f}' fill='#e2e8f0' font-size='17'>x_level</text>
            <text x='{sx(max(1.0, x_axis_max * 0.58)):.2f}' y='{sy(max(0.35, y_axis_max * 0.14)):.2f}' fill='#e8f2ff' font-size='12' opacity='0.86'>white dots: quotient basis ({basis_count}); gold boundary: minimal generators ({len(mins)}); dominated bidegrees: {dominated_count}</text>
          </svg>
          {resolution_block}
        </article>
        """

    def _staircase_visual_contract(dim: int, items: Sequence[tuple[int, int, int, str]], *, primary: bool = False) -> dict[str, Any]:
        pts = [(int(lvl), int(rg)) for lvl, rg, _count, _ex in items]
        mins = _minimal_antichain(pts)
        x_max = max([max_radius] + [rg for _lvl, rg in pts] + [rg for _lvl, rg in mins] + [8])
        y_max = max([max_level] + [lvl for lvl, _rg in pts] + [lvl for lvl, _rg in mins] + [3])
        x_axis_max = x_max + max(2, int(math.ceil(0.04 * max(1, x_max))))
        y_axis_max = y_max + 1

        def point_in_upset(lvl: int, rg: int) -> bool:
            return any(m_lvl <= lvl and m_rg <= rg for m_lvl, m_rg in mins)

        quotient_basis_lattice_points = [
            [int(lvl), int(rg)]
            for lvl in range(0, int(y_axis_max) + 1)
            for rg in range(0, int(x_axis_max) + 1)
            if not point_in_upset(int(lvl), int(rg))
        ]
        minimal_set = set(mins)
        generator_labels = [
            {
                "label": f"g{index}",
                "bidegree": [int(lvl), int(rg)],
                "monomial": f"x_level^{int(lvl)} x_radius^{int(rg)}",
                "homological_degree": int(dim),
            }
            for index, (lvl, rg) in enumerate(mins, start=1)
        ]
        dominated = [
            {"bidegree": [int(lvl), int(rg)], "multiplicity": int(count), "example_simplex": str(ex)[:240]}
            for lvl, rg, count, ex in sorted(items, key=lambda t: (t[0], t[1]))
            if (int(lvl), int(rg)) not in minimal_set
        ]
        resolution = _selected_staircase_resolution(dim, pts)
        hilbert_terms = [
            dict(term)
            for term in resolution.get("hilbert_series_numerator_terms", [])
            if isinstance(term, Mapping)
        ]
        syzygies = [
            dict(row)
            for row in resolution.get("adjacent_lcm_syzygies", [])
            if isinstance(row, Mapping)
        ]
        return {
            "schema_version": "tropicalgt.two_parameter_staircase_card.v1",
            "homological_degree": int(dim),
            "primary_card": bool(primary),
            "actual_generator_bidegrees_source": "bifiltration.chain_module_generators[*].multidegree grouped by homological_degree",
            "x_radius_horizontal": True,
            "x_level_vertical": True,
            "shaded_regions_are_upward_closed_generated_submodules": True,
            "white_points_are_displayed_quotient_basis_lattice_points": True,
            "minimal_antichain_boundary_source": "minimal elements under product order on actual generator bidegrees",
            "actual_generator_bidegree_count": int(len(items)),
            "minimal_antichain": [[int(lvl), int(rg)] for lvl, rg in mins],
            "generator_labels": generator_labels,
            "dominated_generator_bidegrees": dominated,
            "upward_closed_regions": [
                {
                    "generator_label": row["label"],
                    "generator_bidegree": row["bidegree"],
                    "x_radius_min": row["bidegree"][1],
                    "x_level_min": row["bidegree"][0],
                    "x_radius_max_displayed": int(x_axis_max),
                    "x_level_max_displayed": int(y_axis_max),
                }
                for row in generator_labels
            ],
            "quotient_basis_lattice_points": quotient_basis_lattice_points,
            "quotient_basis_lattice_count": int(len(quotient_basis_lattice_points)),
            "display_grid_extent": {
                "x_radius_max": int(x_axis_max),
                "x_level_max": int(y_axis_max),
                "coordinate_axes": ["rho_x_radius", "rho_x_level"],
            },
            "hilbert_numerator_terms": hilbert_terms,
            "adjacent_lcm_syzygies": syzygies,
            "resolution_available": bool(resolution.get("available", False)),
            "resolution_scope": str(resolution.get("scope", "")),
            "theorem_scope": "exact two-variable monomial-ideal staircase resolution when adjacent-LCM theorem applies; not a full persistence-module free resolution without CAS certification",
        }

    def _staircase_sort_key(pair: tuple[int, Sequence[tuple[int, int, int, str]]]) -> tuple[int, int, int, int]:
        dim, items = pair
        pts = [(int(lvl), int(rg)) for lvl, rg, _count, _ex in items]
        mins = _minimal_antichain(pts)
        return (0 if len(mins) > 1 else 1, -len(mins), -len(items), int(dim))

    ordered_staircases = sorted(generator_points_by_dim.items(), key=_staircase_sort_key)
    primary_staircase_index = -1
    for idx, (_dim, items) in enumerate(ordered_staircases):
        pts = [(int(lvl), int(rg)) for lvl, rg, _count, _ex in items]
        if len(_minimal_antichain(pts)) > 1:
            primary_staircase_index = idx
            break
    if primary_staircase_index < 0 and ordered_staircases:
        primary_staircase_index = 0
    staircase_parts: list[str] = []
    staircase_contracts: list[dict[str, Any]] = []
    for idx, (dim, items) in enumerate(ordered_staircases):
        primary = idx == primary_staircase_index
        staircase_parts.append(_svg_chain_staircase(dim, items, primary=primary))
        staircase_contracts.append(_staircase_visual_contract(dim, items, primary=primary))
    staircase_svgs = "".join(staircase_parts)
    if not staircase_svgs:
        staircase_svgs = "<p class='lede'>No chain-generator bidegrees were available for a bivariate staircase diagram.</p>"

    def _miller_sturmfels_staircase_evidence(cards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        def _list_len(card: Mapping[str, Any], key: str) -> int:
            value = card.get(key, [])
            return len(value) if isinstance(value, list) else 0

        def _int_value(card: Mapping[str, Any], key: str) -> int:
            try:
                return int(card.get(key, 0))
            except Exception:
                return 0

        card_list = [card for card in cards if isinstance(card, Mapping)]
        primary_indices = [index for index, card in enumerate(card_list) if card.get("primary_card") is True]
        primary_index = primary_indices[0] if primary_indices else -1
        primary_card = card_list[primary_index] if 0 <= primary_index < len(card_list) else {}
        per_card_counts = [
            {
                "homological_degree": int(card.get("homological_degree", -1)),
                "primary_card": bool(card.get("primary_card", False)),
                "actual_generator_bidegree_count": _int_value(card, "actual_generator_bidegree_count"),
                "minimal_antichain_count": _list_len(card, "minimal_antichain"),
                "generator_label_count": _list_len(card, "generator_labels"),
                "upward_closed_region_count": _list_len(card, "upward_closed_regions"),
                "quotient_basis_lattice_count": _int_value(card, "quotient_basis_lattice_count"),
                "hilbert_numerator_term_count": _list_len(card, "hilbert_numerator_terms"),
                "adjacent_lcm_syzygy_count": _list_len(card, "adjacent_lcm_syzygies"),
                "theorem_scope": str(card.get("theorem_scope", "")),
            }
            for card in card_list
        ]
        all_cards_have_generator_labels = all(_list_len(card, "generator_labels") > 0 for card in card_list)
        all_cards_have_upward_closed_regions = all(_list_len(card, "upward_closed_regions") > 0 for card in card_list)
        all_cards_have_quotient_basis_lattice_points = all(isinstance(card.get("quotient_basis_lattice_points"), list) for card in card_list)
        all_cards_have_hilbert_numerator_terms = all(isinstance(card.get("hilbert_numerator_terms"), list) for card in card_list)
        all_cards_have_adjacent_lcm_syzygy_lists = all(isinstance(card.get("adjacent_lcm_syzygies"), list) for card in card_list)
        theorem_scope_boundary_all_cards = all(
            "not a full persistence-module free resolution" in str(card.get("theorem_scope", ""))
            for card in card_list
        )
        quotient_count_matches_rows = all(
            _int_value(card, "quotient_basis_lattice_count") == _list_len(card, "quotient_basis_lattice_points")
            for card in card_list
        )
        safe = bool(card_list) and len(primary_indices) == 1 and all([
            all_cards_have_generator_labels,
            all_cards_have_upward_closed_regions,
            all_cards_have_quotient_basis_lattice_points,
            all_cards_have_hilbert_numerator_terms,
            all_cards_have_adjacent_lcm_syzygy_lists,
            theorem_scope_boundary_all_cards,
            quotient_count_matches_rows,
        ])
        return {
            "schema_version": "tropicalgt.miller_sturmfels_staircase_evidence.v1",
            "coefficient_ring": "F2[x_level,x_radius]",
            "source": "staircase_cards_from_bifiltration.chain_module_generators[*].multidegree",
            "actual_data_only": True,
            "no_proxy_or_fallback": True,
            "primary_view": "miller_sturmfels_bivariate_staircase",
            "axes": {
                "horizontal": "x_radius",
                "vertical": "x_level",
                "coordinate_one_dimensional_cones": ["rho_x_radius", "rho_x_level"],
            },
            "card_count": int(len(card_list)),
            "primary_card_count": int(len(primary_indices)),
            "primary_card_index": int(primary_index),
            "primary_homological_degree": int(primary_card.get("homological_degree", -1)) if primary_card else -1,
            "total_actual_generator_bidegree_count": int(sum(row["actual_generator_bidegree_count"] for row in per_card_counts)),
            "total_minimal_antichain_count": int(sum(row["minimal_antichain_count"] for row in per_card_counts)),
            "total_generator_label_count": int(sum(row["generator_label_count"] for row in per_card_counts)),
            "total_upward_closed_region_count": int(sum(row["upward_closed_region_count"] for row in per_card_counts)),
            "total_quotient_basis_lattice_count": int(sum(row["quotient_basis_lattice_count"] for row in per_card_counts)),
            "total_hilbert_numerator_term_count": int(sum(row["hilbert_numerator_term_count"] for row in per_card_counts)),
            "total_adjacent_lcm_syzygy_count": int(sum(row["adjacent_lcm_syzygy_count"] for row in per_card_counts)),
            "cards_with_quotient_basis_count": int(sum(1 for card in card_list if _list_len(card, "quotient_basis_lattice_points") > 0)),
            "cards_with_hilbert_numerator_terms_count": int(sum(1 for card in card_list if _list_len(card, "hilbert_numerator_terms") > 0)),
            "cards_with_adjacent_lcm_syzygies_count": int(sum(1 for card in card_list if _list_len(card, "adjacent_lcm_syzygies") > 0)),
            "per_card_counts": per_card_counts,
            "all_cards_have_generator_labels": bool(all_cards_have_generator_labels),
            "all_cards_have_upward_closed_regions": bool(all_cards_have_upward_closed_regions),
            "all_cards_have_quotient_basis_lattice_points": bool(all_cards_have_quotient_basis_lattice_points),
            "all_cards_have_hilbert_numerator_terms": bool(all_cards_have_hilbert_numerator_terms),
            "all_cards_have_adjacent_lcm_syzygy_lists": bool(all_cards_have_adjacent_lcm_syzygy_lists),
            "quotient_basis_counts_match_lattice_points": bool(quotient_count_matches_rows),
            "theorem_scope_boundary_all_cards": bool(theorem_scope_boundary_all_cards),
            "coordinate_axes_are_one_dimensional_cones": True,
            "safe_to_render_miller_sturmfels_staircase": bool(safe),
        }

    staircase_evidence = _miller_sturmfels_staircase_evidence(staircase_contracts)

    def _table_html(headers: Sequence[str], columns: Sequence[Sequence[Any]], caption: str) -> str:
        width = max((len(col) for col in columns), default=0)
        rows_html = []
        for i in range(width):
            cells = []
            for col in columns:
                value = col[i] if i < len(col) else ""
                cells.append(f"<td>{html.escape(str(value))}</td>")
            rows_html.append("<tr>" + "".join(cells) + "</tr>")
        head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
        return f"<section class='card'><h2>{html.escape(caption)}</h2><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows_html)}</tbody></table></section>"

    structure_h = ["direction", "source bidegree", "target bidegree", "H0 rank", "H1 rank", "method"]
    structure_c = [
        [row.get("direction", "") for row in structure_rank_rows],
        [row.get("source_bidegree_x_level_x_radius", []) for row in structure_rank_rows],
        [row.get("target_bidegree_x_level_x_radius", []) for row in structure_rank_rows],
        [(row.get("homology_rank", {}) if isinstance(row.get("homology_rank", {}), Mapping) else {}).get("0", "") for row in structure_rank_rows],
        [(row.get("homology_rank", {}) if isinstance(row.get("homology_rank", {}), Mapping) else {}).get("1", "") for row in structure_rank_rows],
        [row.get("method", "") for row in structure_rank_rows],
    ]
    rank_h, rank_c = _rank_invariant_columns(bifiltration)
    betti_h, betti_c = _m2_betti_columns(m2)
    free_h, free_c = _m2_free_module_columns(m2)
    diff_h, diff_c = _m2_differential_columns(m2)
    syz_h, syz_c = _m2_syzygy_columns(m2)
    cert_h, cert_c = _m2_certificate_columns(m2, bifiltration)
    ideal_h, ideal_c = _m2_ideal_diagnostic_columns(m2)
    be_h, be_c = _m2_be_diagnostic_columns(m2)
    rank_note = f"grid={len(levels)} x-levels x {len(radius_grades)} radius grades; H0 range {float(np.nanmin(h0_grid)):.0f}-{float(np.nanmax(h0_grid)):.0f}; H1 range {float(np.nanmin(h1_grid)):.0f}-{float(np.nanmax(h1_grid)):.0f}; chain generators={sum(generator_counts.values())}."
    chart1 = fig_module.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})
    chart2 = fig_3d.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})
    page = f"""<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>{html.escape(title)}</title>
<script src='plotly.min.js'></script>
<style>
:root {{ color-scheme: dark; --bg:#050914; --ink:#e8f2ff; --muted:#a7b8d1; --edge:#203d5e; --panel:#07111f; --accent:#5eead4; }}
body {{ margin:0; background:radial-gradient(circle at 20% 0%, #10233f 0, var(--bg) 42%, #030611 100%); color:var(--ink); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
main {{ max-width:1280px; margin:0 auto; padding:34px 24px 76px; }}
h1 {{ margin:0 0 10px; font-size:34px; line-height:1.08; }}
.lede {{ color:var(--muted); max-width:1180px; font-size:17px; line-height:1.45; margin-bottom:22px; }}
.callout {{ border:1px solid rgba(94,234,212,.45); background:rgba(5,12,28,.82); padding:14px 16px; border-radius:10px; margin:18px 0 24px; font-size:15px; }}
.panel {{ border:1px solid rgba(96,165,250,.24); background:rgba(7,17,31,.72); border-radius:12px; padding:14px; margin:22px 0; overflow:hidden; }}
.panel h2 {{ margin:0 0 8px; font-size:18px; letter-spacing:.04em; text-transform:uppercase; color:#bfdbfe; }}
.card-grid {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:18px; }}
.staircase-grid {{ display:grid; grid-template-columns:1fr; gap:18px; }}
.stair-card {{ border:1px solid rgba(94,234,212,.24); background:rgba(3,7,18,.76); border-radius:12px; padding:16px; overflow:auto; }}
.stair-card svg {{ display:block; width:100%; min-width:720px; shape-rendering:geometricPrecision; }}
.primary-staircase svg {{ min-width:860px; }}
.stair-card h3 {{ margin:0 0 6px; font-size:18px; color:#e8f2ff; }}
.svg-note {{ margin:0 0 10px; color:#a7b8d1; font-size:13px; line-height:1.35; }}
.secondary-disclosure {{ border:1px solid rgba(94,234,212,.22); background:rgba(3,7,18,.70); border-radius:12px; padding:12px 16px; margin:18px 0; }}
.secondary-disclosure > summary {{ cursor:pointer; font-weight:800; letter-spacing:.06em; text-transform:uppercase; color:#bfe3ff; }}
.secondary-disclosure[open] > summary {{ margin-bottom:14px; }}
.formula {{ margin:0 0 10px; color:#fde68a; font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:13px; }}
.resolution-block {{ margin-top:14px; border:1px solid rgba(250,204,21,.28); background:rgba(12,10,4,.62); border-radius:10px; padding:12px; }}
.resolution-block h4 {{ margin:0 0 8px; color:#fef3c7; font-size:15px; }}
.resolution-note,.resolution-warning,.hilbert-line,.certificate-line {{ margin:7px 0; color:#cbd5e1; font-size:12px; line-height:1.42; }}
.resolution-warning {{ color:#fde68a; }}
.complex-line {{ margin:8px 0 10px; color:#e8f2ff; font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:13px; }}
.resolution-grid {{ display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:10px; }}
.mini-table {{ overflow:auto; max-height:240px; border-radius:8px; }}
.mini-table h5 {{ margin:0 0 5px; color:#bfdbfe; font-size:12px; text-transform:uppercase; letter-spacing:.05em; }}
.mini-table table {{ font-size:11px; }}
.primary-staircase {{ border-color:rgba(250,204,21,.42); box-shadow:0 0 0 1px rgba(250,204,21,.08), 0 18px 44px rgba(0,0,0,.28); }}
.secondary-staircase {{ opacity:.92; }}
.card {{ border:1px solid rgba(148,163,184,.22); background:rgba(7,17,31,.72); border-radius:12px; padding:14px; overflow:auto; max-height:520px; }}
.card h2 {{ margin:0 0 10px; font-size:16px; color:#bfdbfe; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; }}
th,td {{ border:1px solid rgba(49,92,134,.72); padding:7px 8px; vertical-align:top; }}
th {{ background:#10243f; color:#e8f2ff; text-align:left; }}
td {{ background:#07111f; color:#d7e8ff; }}
@media(max-width: 900px) {{ main {{ padding:22px 12px 56px; }} .card-grid {{ grid-template-columns:1fr; }} h1 {{ font-size:26px; }} }}
</style>
</head>
<body>
<main>
<h1>Trajectory 2-parameter persistence over F2[x_level,x_radius]</h1>
<p class='lede'>Actual 2-parameter module fibers and multigraded chain-generator bidegrees over F2[x_level,x_radius]. The primary view is the Miller-Sturmfels staircase view of the bivariate module diagram: horizontal lattice coordinates are x_radius and vertical lattice coordinates are x_level; x_radius runs horizontally, x_level vertically, shaded upward-closed regions are generated submodules, and white lattice points are the displayed S/I_C basis complement.</p>
<div class='callout'><b>Computed bifiltration:</b> {html.escape(rank_note)}<br>This section is an exponent-lattice module diagram in the sense of the two-variable monomial-ideal staircase picture: the coordinate axes are the x_radius and x_level one dimensional cone(s). The large gold/cyan/blue boundary points are minimal antichain generators; smaller dim-colored points are dominated observed bidegrees and are not treated as additional generators. White lattice points are displayed quotient-basis complements, colored cells/points are actual H1 fiber ranks. Adjacent structure maps persisted={len(bifiltration.get("structure_maps", [])) if isinstance(bifiltration, Mapping) else 0}. Staircase cards render exact two-variable monomial-ideal resolutions when the Miller-Sturmfels adjacent-LCM theorem applies. Lower CAS tables render only certified CAS output under its actual grading; diagnostic chain data is not substituted for a free resolution.</div>
<section class='panel'><h2>Miller-Sturmfels bivariate module staircases from actual multidegree generators</h2><div class='staircase-grid'>{staircase_svgs}</div><div class='card-grid'>{_table_html(structure_h, structure_c, 'Adjacent F2 structure maps from raw bifiltration')}</div></section>
<details class='secondary-disclosure'><summary>Secondary fiber-rank diagnostics</summary><section class='panel'><h2>F2[x_level,x_radius] support and homology fiber ranks</h2>{chart1}</section><section class='panel'><h2>Fiber-rank lattice with H0/H1 layer offsets</h2>{chart2}</section><div class='card-grid'>{_table_html(rank_h, rank_c, 'Rank-invariant samples over F2[x_level,x_radius]')}</div></details>
<details class='secondary-disclosure'><summary>Certified algebra tables and CAS certificates</summary><div class='card-grid'>{_table_html(betti_h, betti_c, 'Betti-style diagnostics')}{_table_html(free_h, free_c, 'Free chain modules / certified free modules')}{_table_html(diff_h, diff_c, 'Differentials / boundary maps')}{_table_html(syz_h, syz_c, 'Certified Macaulay2 syzygy generators')}{_table_html(ideal_h, ideal_c, 'Certified Fitting ideals and determinantal minors')}{_table_html(be_h, be_c, 'Buchsbaum-Eisenbud rank and multiplier diagnostics')}{_table_html(cert_h, cert_c, 'CAS certificate summary')}</div></details>
</main>
</body>
</html>
"""
    visual_payload = {
        "schema_version": "tropicalgt.two_parameter_bifiltration_visual.v1",
        "coefficient_ring": "F2[x_level,x_radius]",
        "primary_view": "miller_sturmfels_bivariate_staircase",
        "primary_view_contract": "The primary view is an exponent-lattice staircase over F2[x_level,x_radius]: x_radius is horizontal, x_level is vertical, shaded regions are upward-closed generated submodules, and white lattice points are displayed quotient-basis complements from actual bifiltration chain-generator bidegrees.",
        "secondary_views": ["fiber_rank_heatmap", "fiber_rank_lattice_3d", "structure_map_lattice_overlay", "rank_invariant_samples_table", "certified_algebra_tables", "certified_syzygy_tables", "certified_fitting_minor_tables", "buchsbaum_eisenbud_diagnostic_tables"],
        "rank_invariant_sample_count": len(rank_inv_rows),
        "rank_surface_primary": False,
        "rank_surface_policy": "3D fiber-rank displays are secondary diagnostics only and are not rendered as the primary module view.",
        "axes": {
            "horizontal": "x_radius",
            "vertical": "x_level",
            "coordinate_one_dimensional_cones": ["rho_x_radius", "rho_x_level"],
        },
        "actual_data_only": True,
        "no_proxy_resolution_claim": True,
        "structure_map_summary": structure_map_summary,
        "primary_structure_map_evidence": {
            "schema_version": "tropicalgt.primary_structure_map_evidence.v1",
            "available": bool(structure_rank_rows),
            "source": "bifiltration.structure_maps",
            "directions_rendered": sorted({str(row.get("direction")) for row in structure_rank_rows if row.get("direction")}),
            "primary_table_rows": len(structure_rank_rows),
            "module_lattice_overlay_trace_names": structure_overlay_trace_names,
            "no_proxy_or_fallback": True,
        },
        "miller_sturmfels_staircase_evidence": staircase_evidence,
        "module_visual_contract": {
            "schema_version": "tropicalgt.two_parameter_module_staircase_contract.v1",
            "no_proxy_or_fallback": True,
            "primary_view": "miller_sturmfels_bivariate_staircase",
            "rank_slabs_removed_as_primary_view": True,
            "secondary_rank_diagnostics_labeled": True,
            "x_radius_horizontal": True,
            "x_level_vertical": True,
            "shaded_regions_are_upward_closed_generated_submodules": True,
            "white_points_are_displayed_quotient_basis_lattice_points": True,
            "structure_map_summary_required": True,
            "structure_map_summary_schema": "tropicalgt.two_parameter_structure_maps.v1",
            "structure_map_summary_source": "bifiltration.structure_maps",
            "primary_structure_map_evidence_required": True,
            "primary_structure_map_evidence_schema": "tropicalgt.primary_structure_map_evidence.v1",
            "raw_bifiltration_required_fields": [
                "fiber_rank_profile",
                "chain_module_generators",
                "rank_invariant_samples",
                "structure_maps",
                "grid_fiber_provenance",
            ],
            "raw_bifiltration_payload": "../trajectory_level_radius_bifiltration.json",
            "safe_resolution_policy": "Only scoped two-variable monomial staircase resolutions or certified CAS free resolutions may be rendered; chain-presentation diagnostics are never substituted as a free resolution.",
        },
        "grid": {
            "x_level_grades": levels,
            "x_radius_grades": radius_grades,
            "fiber_row_count": len(rows),
            "h0_min": float(np.nanmin(h0_grid)) if h0_grid.size else 0.0,
            "h0_max": float(np.nanmax(h0_grid)) if h0_grid.size else 0.0,
            "h1_min": float(np.nanmin(h1_grid)) if h1_grid.size else 0.0,
            "h1_max": float(np.nanmax(h1_grid)) if h1_grid.size else 0.0,
        },
        "chain_generator_summary": {
            "total_generator_count": int(sum(generator_counts.values())),
            "homological_dimensions": sorted({int(dim) for _lvl, _rg, dim in generator_counts}),
            "minimal_antichain": [[int(lvl), int(rg)] for lvl, rg in chain_minimal],
        },
        "staircase_cards": staircase_contracts,
        "rendered_html": path.name,
        "raw_bifiltration_payload": "../trajectory_level_radius_bifiltration.json",
    }
    payload_path = path.with_suffix(".json")
    payload_path.write_text(json.dumps(visual_payload, indent=2), encoding="utf-8")
    path.write_text(page, encoding="utf-8")
    return str(path)

def write_two_parameter_bifiltration_visualization(
    path: Path,
    bifiltration: Mapping[str, Any],
    *,
    title: str = "2-parameter persistence module and chain-presentation diagnostics",
) -> str:
    """Render computed F2[x_level,x_radius] module diagnostics.

    Real free resolutions are rendered only when a CAS certificate is attached.
    Otherwise the figure shows finite chain modules, boundary matrices, Fitting/minor
    diagnostics, and Miller-Sturmfels staircase candidates as diagnostics only.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return _write_two_parameter_bifiltration_staircase_html(path, bifiltration, title=title)


def _write_growth_persistence_barcode(path: Path, topology: dict[str, object], growth: list[object], title_prefix: str = "") -> None:
    rows = _trajectory_growth_rows(topology, growth)
    fig = go.Figure()
    colors = {0: "#55d6be", 1: "#7aa2ff", 2: "#fbbf24", 3: "#fb7185"}
    panel_objects = []
    panel_hover = []
    backend_counts: dict[str, int] = defaultdict(int)
    interval_count = 0
    max_death = 1.0
    for row in rows:
        topo = row.get("topological_algebra", {})
        backend_counts[_persistence_backend_label(topo if isinstance(topo, dict) else {})] += len(row.get("intervals", []))
        for interval in row["intervals"]:
            interval_count += 1
            death = interval.get("display_death")
            if isinstance(death, (int, float)) and math.isfinite(float(death)):
                max_death = max(max_death, float(death))
    y_cursor = 0
    tick_vals: list[float] = []
    tick_text: list[str] = []
    line_width = 3 if interval_count > 160 else 5 if interval_count > 80 else 8
    marker_size = 3 if interval_count > 160 else 4 if interval_count > 80 else 5
    for row_idx, row in enumerate(rows):
        level = int(row["level"])
        obj = row.get("filtered_simplicial_object", {})
        topo = row.get("topological_algebra", {})
        panel_objects.append(obj if isinstance(obj, dict) else {})
        panel_hover.append(_topology_growth_hover(level, obj if isinstance(obj, dict) else {}, topo if isinstance(topo, dict) else {}))
        level_start = y_cursor
        level_interval_count = 0
        for interval_idx, interval in enumerate(row["intervals"]):
            dim = int(interval.get("dimension", 0))
            birth = float(interval.get("birth", 0.0))
            death = float(interval.get("display_death", max_death))
            true_death = "inf" if interval.get("infinite") else f"{float(interval.get('death', death)):.4g}"
            source = interval.get("source") or _persistence_backend_label(topo if isinstance(topo, dict) else {})
            hover = (
                f"<b>trajectory growth level {level}</b>"
                f"<br>H{dim} interval [{birth:.4g}, {true_death}]"
                f"<br>source={html.escape(str(source))}"
                + f"<br>complex: {_summary_line(obj if isinstance(obj, dict) else {})}"
                f"<br>{_derived_signature_line(topo if isinstance(topo, dict) else {})}"
                f"<br>{_free_resolution_line(topo if isinstance(topo, dict) else {})}"
            )
            y = y_cursor
            y_cursor += 1
            level_interval_count += 1
            fig.add_trace(
                go.Scatter(
                    x=[birth, death],
                    y=[y, y],
                    mode="lines+markers",
                    line=dict(width=line_width, color=colors.get(dim, "#cbd5e1")),
                    marker=dict(size=marker_size, color=colors.get(dim, "#cbd5e1")),
                    name=f"H{dim}",
                    hovertext=[hover, hover],
                    hoverinfo="text",
                    customdata=[row_idx, row_idx],
                    showlegend=not any(trace.name == f"H{dim}" for trace in fig.data),
                )
            )
            if interval.get("infinite"):
                fig.add_trace(
                    go.Scatter(
                        x=[death],
                        y=[y],
                        mode="markers",
                        marker=dict(symbol="diamond", size=7, color=colors.get(dim, "#cbd5e1")),
                        hovertext=hover + "<br>infinite interval displayed at finite cap",
                        hoverinfo="text",
                        customdata=[row_idx],
                        showlegend=False,
                    )
                )
        if level_interval_count:
            tick_vals.append(level_start + (level_interval_count - 1) / 2.0)
            dim_counts = Counter(int(interval.get("dimension", 0)) for interval in row["intervals"])
            dim_summary = " ".join(f"H{dim}:{count}" for dim, count in sorted(dim_counts.items()))
            tick_text.append(f"L{level}<br><span style='font-size:9px'>{level_interval_count} int; {dim_summary}</span>")
        else:
            tick_vals.append(float(y_cursor))
            tick_text.append(f"L{level}<br><span style='font-size:9px'>0 int</span>")
            y_cursor += 1
    if not rows:
        fig.add_annotation(text="No trajectory growth topology available.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    backend_summary = ", ".join(f"{key}:{value}" for key, value in sorted(backend_counts.items())) or "none"
    fig.update_layout(
        template="plotly_dark",
        title=(
            f"{title_prefix}persistent homology growth barcode"
            "<br><sup>standard interval view with level-grouped y-axis; "
            f"intervals={interval_count}; "
            f"backends={html.escape(backend_summary)}; "
            "hover shows GUDHI/topology provenance and chain-presentation summary; regular bands indicate either stable topology or collapsed/regular filtration, not visual smoothing</sup>"
        ),
        xaxis_title="filtration birth/death",
        yaxis=dict(title="trajectory level", tickmode="array", tickvals=tick_vals, ticktext=tick_text, autorange="reversed", tickfont=dict(size=10)),
        legend=dict(itemsizing="constant"),
        height=max(640, min(1250, 280 + 18 * max(1, len(tick_vals)) + 3 * max(1, interval_count))),
    )
    _write_plotly_dark_html(
        path,
        fig,
        f"{title_prefix}persistent homology growth barcode",
        _simplicial_panel_items(panel_objects, panel_hover),
    )


def _write_growth_persistence_module(path: Path, topology: dict[str, object], growth: list[object], title_prefix: str = "") -> None:
    rows = _trajectory_growth_rows(topology, growth)
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "heatmap"}, {"type": "bar"}]],
        subplot_titles=(
            "Betti ranks by trajectory level and filtration",
            "Chain-presentation diagnostic ranks by trajectory level",
        ),
        horizontal_spacing=0.08,
    )
    panel_objects = []
    panel_hover = []
    betti_cells: list[dict[str, object]] = []
    free_cells: list[dict[str, object]] = []
    betti_tickvals: list[str] = []
    betti_ticktext: list[str] = []
    free_tickvals: list[str] = []
    free_ticktext: list[str] = []
    for row_idx, row in enumerate(rows):
        level = int(row["level"])
        obj = row.get("filtered_simplicial_object", {})
        topo = row.get("topological_algebra", {})
        panel_objects.append(obj if isinstance(obj, dict) else {})
        panel_hover.append(_topology_growth_hover(level, obj if isinstance(obj, dict) else {}, topo if isinstance(topo, dict) else {}))
        states = row.get("states", [])
        for dim, color in [(0, "#55d6be"), (1, "#7aa2ff"), (2, "#fbbf24"), (3, "#fb7185")]:
            for state in states:
                if not isinstance(state, dict):
                    continue
                beta = int(state.get("betti", {}).get(str(dim), 0)) if isinstance(state.get("betti"), dict) else 0
                betti_cells.append(
                    {
                        "level": level,
                        "dim": dim,
                        "threshold": float(state.get("threshold", 0.0) or 0.0),
                        "beta": beta,
                        "panel": row_idx,
                        "hover": f"<b>trajectory level {level}</b><br>beta_{dim}={beta}<br>filtration={float(state.get('threshold', 0.0) or 0.0):.4g}<br>{_summary_line(obj if isinstance(obj, dict) else {})}",
                    }
                )
        free_modules = _free_resolution_modules(topo if isinstance(topo, dict) else {})
        for module in free_modules:
            degree = int(module.get("homological_degree", 0))
            rank = int(module.get("rank", module.get("rank_upper_bound", 0)))
            free_cells.append(
                {
                    "level": level,
                    "degree": degree,
                    "rank": rank,
                    "panel": row_idx,
                    "hover": (
                        f"<b>trajectory level {level}</b>"
                        f"<br>homological degree={degree}"
                        f"<br>free rank={rank}"
                        f"<br>{_free_resolution_line(topo if isinstance(topo, dict) else {})}"
                        f"<br>{_derived_signature_line(topo if isinstance(topo, dict) else {})}"
                    ),
                }
            )
    if betti_cells:
        levels = sorted({int(cell["level"]) for cell in betti_cells})
        dims = sorted({int(cell["dim"]) for cell in betti_cells})
        thresholds = sorted({round(float(cell["threshold"]), 6) for cell in betti_cells})
        x_labels = [f"H{dim}@{threshold:.3g}" for dim in dims for threshold in thresholds]
        tick_step = max(1, int(math.ceil(len(x_labels) / 10)))
        betti_tickvals = x_labels[::tick_step]
        betti_ticktext = x_labels[::tick_step]
        z = np.zeros((len(levels), len(x_labels)), dtype=float)
        hover = [["" for _ in x_labels] for _ in levels]
        index = {(int(cell["level"]), int(cell["dim"]), round(float(cell["threshold"]), 6)): cell for cell in betti_cells}
        for row_i, level in enumerate(levels):
            for col_i, (dim, threshold) in enumerate((dim, threshold) for dim in dims for threshold in thresholds):
                cell = index.get((level, dim, threshold))
                if cell:
                    z[row_i, col_i] = float(cell["beta"])
                    hover[row_i][col_i] = str(cell["hover"])
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=x_labels,
                y=[f"L{level}" for level in levels],
                colorscale="Viridis",
                customdata=hover,
                hovertemplate="%{customdata}<extra></extra>",
                colorbar=dict(title="Betti rank", x=0.45),
            ),
            row=1,
            col=1,
        )
    if free_cells:
        x = [f"L{cell['level']} F{cell['degree']}" for cell in free_cells]
        y = [float(cell["rank"]) for cell in free_cells]
        tick_step = max(1, int(math.ceil(len(x) / 12)))
        free_tickvals = x[::tick_step]
        free_ticktext = x[::tick_step]
        fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                marker=dict(color=y, colorscale="Turbo", line=dict(color="#e8eef8", width=0.7)),
                hovertext=[str(cell["hover"]) for cell in free_cells],
                hoverinfo="text",
                name="chain-presentation diagnostic",
            ),
            row=1,
            col=2,
        )
    if not rows:
        fig.add_annotation(text="No trajectory growth persistence module available.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    fig.update_layout(
        template="plotly_dark",
        title=(
            f"{title_prefix}multiparameter persistence and chain-presentation diagnostics"
            "<br><sup>2D matrix/bar view replaces decorative 3D spikes; hover opens the corresponding filtered complex panel</sup>"
        ),
        height=940,
        margin=dict(t=132, l=88, r=116, b=168),
    )
    fig.update_xaxes(title_text="homology dimension / filtration", tickangle=32, tickmode="array", tickvals=betti_tickvals, ticktext=betti_ticktext, tickfont=dict(size=9), automargin=True, row=1, col=1)
    fig.update_yaxes(title_text="trajectory growth level", automargin=True, row=1, col=1)
    fig.update_xaxes(title_text="level and free module", tickangle=32, tickmode="array", tickvals=free_tickvals, ticktext=free_ticktext, tickfont=dict(size=9), automargin=True, row=1, col=2)
    fig.update_yaxes(title_text="free rank", row=1, col=2)
    _write_plotly_dark_html(
        path,
        fig,
        f"{title_prefix}multiparameter persistence and chain-presentation diagnostics",
        _simplicial_panel_items(panel_objects, panel_hover),
    )


def _write_single_persistence_representations(path: Path, topology: dict[str, object], title_prefix: str = "") -> None:
    topology = _topology_with_persistence_representations(topology)
    reps = topology.get("persistence_representations", {}) if isinstance(topology.get("persistence_representations"), dict) else {}
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "heatmap"}], [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(
            "Persistence landscapes and silhouettes",
            "Persistence image",
            "Betti curves",
            "Lengths and topological vectors",
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.16,
    )
    if not reps.get("available"):
        fig.add_annotation(
            text=f"No finite persistence-representation vectors available: {html.escape(str(reps.get('reason', reps.get('error', 'unknown'))))}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    palette = ["#55d6be", "#7aa2ff", "#fbbf24", "#fb7185"]
    methods = reps.get("methods", {}) if isinstance(reps.get("methods"), dict) else {}
    image_added = False
    for dim_key, row in sorted(methods.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 99):
        if not isinstance(row, dict) or not row.get("available"):
            continue
        dim = int(dim_key)
        color = palette[dim % len(palette)]
        landscape = row.get("landscape", {}) if isinstance(row.get("landscape"), dict) else {}
        grid = _as_float_list(landscape.get("grid", []))
        values = landscape.get("values", [])
        if grid and isinstance(values, list):
            for layer_idx, layer in enumerate(values[:4]):
                ys = _as_float_list(layer)
                if len(ys) != len(grid):
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=grid,
                        y=ys,
                        mode="lines",
                        line=dict(width=max(1.2, 3.0 - 0.35 * layer_idx), color=color, dash="solid" if layer_idx == 0 else "dot"),
                        name=f"H{dim} lambda_{layer_idx + 1}",
                        hovertemplate=f"H{dim} landscape layer {layer_idx + 1}<br>filtration=%{{x:.4g}}<br>value=%{{y:.4g}}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )
        silhouette = row.get("silhouette", {}) if isinstance(row.get("silhouette"), dict) else {}
        s_grid = _as_float_list(silhouette.get("grid", []))
        s_vals = _as_float_list(silhouette.get("values", []))
        if len(s_grid) == len(s_vals) and s_grid:
            fig.add_trace(
                go.Scatter(
                    x=s_grid,
                    y=s_vals,
                    mode="lines",
                    line=dict(width=2.0, color=color, dash="dash"),
                    name=f"H{dim} silhouette",
                    hovertemplate=f"H{dim} silhouette<br>filtration=%{{x:.4g}}<br>value=%{{y:.4g}}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        betti = row.get("betti_curve", {}) if isinstance(row.get("betti_curve"), dict) else {}
        b_grid = _as_float_list(betti.get("grid", []))
        b_vals = _as_float_list(betti.get("values", []))
        if len(b_grid) == len(b_vals) and b_grid:
            fig.add_trace(
                go.Scatter(
                    x=b_grid,
                    y=b_vals,
                    mode="lines",
                    line_shape="hv",
                    line=dict(width=2.5, color=color),
                    name=f"H{dim} Betti curve",
                    hovertemplate=f"H{dim} Betti curve<br>filtration=%{{x:.4g}}<br>beta=%{{y:.4g}}<extra></extra>",
                ),
                row=2,
                col=1,
            )
        image = row.get("persistence_image", {}) if isinstance(row.get("persistence_image"), dict) else {}
        if not image_added and isinstance(image.get("values"), list):
            fig.add_trace(
                go.Heatmap(
                    z=image.get("values", []),
                    colorscale="Turbo",
                    colorbar=dict(title="image mass", x=1.0),
                    hovertemplate="persistence image<br>x pixel=%{x}<br>y pixel=%{y}<br>value=%{z:.5g}<extra></extra>",
                    name=f"H{dim} persistence image",
                ),
                row=1,
                col=2,
            )
            image_added = True
        lengths = _as_float_list(row.get("persistence_lengths", {}).get("values", []) if isinstance(row.get("persistence_lengths"), dict) else [])
        if lengths:
            fig.add_trace(
                go.Bar(
                    x=[f"H{dim} L{i}" for i in range(len(lengths))],
                    y=lengths,
                    marker=dict(color=color),
                    name=f"H{dim} lengths",
                    hovertemplate=f"H{dim} persistence length<br>rank=%{{x}}<br>value=%{{y:.4g}}<extra></extra>",
                ),
                row=2,
                col=2,
            )
        topvec = _as_float_list(row.get("topological_vector", {}).get("values", []) if isinstance(row.get("topological_vector"), dict) else [])
        if topvec:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(topvec))),
                    y=topvec,
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    name=f"H{dim} topological vector",
                    hovertemplate=f"H{dim} topological vector<br>index=%{{x}}<br>value=%{{y:.4g}}<extra></extra>",
                ),
                row=2,
                col=2,
            )
    summary = reps.get("summary", {}) if isinstance(reps.get("summary"), dict) else {}
    fig.update_layout(
        template="plotly_dark",
        title=(
            f"{title_prefix}GUDHI persistence vectorizations"
            "<br><sup>Fast vectorized topology: landscapes/Betti curves/images/silhouettes/lengths/topological vectors. "
            f"persistence-landscape L2={float(summary.get('landscape_l2_norm', 0.0)):.4g}, "
            f"entropy sum={float(summary.get('entropy_scalar_sum', 0.0)):.4g}</sup>"
        ),
        legend=dict(orientation="h", y=-0.18),
    )
    fig.update_xaxes(title_text="filtration", row=1, col=1)
    fig.update_yaxes(title_text="persistence landscape / silhouette", row=1, col=1)
    fig.update_xaxes(title_text="persistence-image x pixel", row=1, col=2)
    fig.update_yaxes(title_text="persistence-image y pixel", row=1, col=2)
    fig.update_xaxes(title_text="filtration", row=2, col=1)
    fig.update_yaxes(title_text="Betti rank", row=2, col=1)
    fig.update_xaxes(title_text="vector coordinate", row=2, col=2, tickangle=45)
    fig.update_yaxes(title_text="feature value", row=2, col=2)
    _write_plotly_dark_html(path, fig, f"{title_prefix}GUDHI persistence vectorizations")


def _write_growth_persistence_representations(path: Path, topology: dict[str, object], growth: list[object], title_prefix: str = "") -> None:
    rows = _trajectory_growth_rows(topology, growth)
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "heatmap"}], [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(
            "Persistence-landscape norm growth",
            "Betti curve heatmap by level",
            "Persistence length mass",
            "Topological vector norm / entropy",
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.16,
    )
    panel_objects = []
    panel_hover = []
    levels: list[int] = []
    landscape_norms: list[float] = []
    length_sums: list[float] = []
    topvec_norms: list[float] = []
    entropy_sums: list[float] = []
    betti_rows: list[list[float]] = []
    betti_labels: list[str] = []
    for row_idx, row in enumerate(rows):
        level = int(row.get("level", row_idx))
        topo = row.get("topological_algebra", {}) if isinstance(row.get("topological_algebra"), dict) else {}
        obj = row.get("filtered_simplicial_object", {}) if isinstance(row.get("filtered_simplicial_object"), dict) else {}
        reps = topo.get("persistence_representations", {}) if isinstance(topo.get("persistence_representations"), dict) else {}
        summary = reps.get("summary", {}) if isinstance(reps.get("summary"), dict) else {}
        levels.append(level)
        landscape_norms.append(float(summary.get("landscape_l2_norm", 0.0)))
        length_sums.append(_representation_summary_sum(reps, "persistence_lengths", "sum"))
        topvec_norms.append(float(summary.get("topological_vector_l2_norm", 0.0)))
        entropy_sums.append(float(summary.get("entropy_scalar_sum", 0.0)))
        betti_vector = _combined_betti_curve_vector(reps)
        if betti_vector:
            betti_rows.append(betti_vector)
            betti_labels.append(f"L{level}")
        panel_objects.append(obj)
        panel_hover.append(_topology_growth_hover(level, obj, topo) + f"<br>{_persistence_representation_line(topo)}")
    if levels:
        fig.add_trace(
            go.Scatter(
                x=levels,
                y=landscape_norms,
                mode="lines+markers",
                line=dict(color="#55d6be", width=3),
                marker=dict(size=7),
                name="persistence-landscape L2",
                hovertemplate="level=%{x}<br>persistence-landscape L2=%{y:.5g}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=levels,
                y=length_sums,
                mode="lines+markers",
                line=dict(color="#fbbf24", width=3),
                marker=dict(size=7),
                name="persistence length sum",
                hovertemplate="level=%{x}<br>length sum=%{y:.5g}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=levels,
                y=topvec_norms,
                mode="lines+markers",
                line=dict(color="#7aa2ff", width=3),
                marker=dict(size=7),
                name="topological vector L2",
                hovertemplate="level=%{x}<br>topological-vector L2=%{y:.5g}<extra></extra>",
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=levels,
                y=entropy_sums,
                mode="lines+markers",
                line=dict(color="#fb7185", width=2.5, dash="dash"),
                marker=dict(size=6),
                name="entropy sum",
                hovertemplate="level=%{x}<br>entropy sum=%{y:.5g}<extra></extra>",
            ),
            row=2,
            col=2,
        )
    if betti_rows:
        max_len = max(len(row) for row in betti_rows)
        padded = [row + [0.0] * (max_len - len(row)) for row in betti_rows]
        fig.add_trace(
            go.Heatmap(
                z=padded,
                y=betti_labels,
                x=list(range(max_len)),
                colorscale="Viridis",
                colorbar=dict(title="beta", x=1.0),
                hovertemplate="level=%{y}<br>grid index=%{x}<br>Betti value=%{z:.4g}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    if not levels:
        fig.add_annotation(text="No vectorized persistence growth data available.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    fig.update_layout(
        template="plotly_dark",
        title=(
            f"{title_prefix}GUDHI persistence vectorization growth"
            "<br><sup>Fast train/eval features from persistence diagrams; hover panel shows the corresponding filtered complex.</sup>"
        ),
        legend=dict(orientation="h", y=-0.18),
    )
    fig.update_xaxes(title_text="trajectory level", row=1, col=1)
    fig.update_yaxes(title_text="persistence-landscape L2", row=1, col=1)
    fig.update_xaxes(title_text="Betti-curve grid coordinate", row=1, col=2)
    fig.update_yaxes(title_text="trajectory level", row=1, col=2)
    fig.update_xaxes(title_text="trajectory level", row=2, col=1)
    fig.update_yaxes(title_text="length mass", row=2, col=1)
    fig.update_xaxes(title_text="trajectory level", row=2, col=2)
    fig.update_yaxes(title_text="feature norm / entropy", row=2, col=2)
    _write_plotly_dark_html(
        path,
        fig,
        f"{title_prefix}GUDHI persistence vectorization growth",
        _simplicial_panel_items(panel_objects, panel_hover),
    )


def _landscape_grid_values(landscape: Mapping[str, object]) -> tuple[list[float], list[list[float]], str, dict[str, object]]:
    grid = _as_float_list(landscape.get("grid", []))
    raw_values = landscape.get("values", [])
    values: list[list[float]] = []
    if grid and isinstance(raw_values, list):
        for layer in raw_values:
            layer_values = _as_float_list(layer)
            if len(layer_values) == len(grid):
                values.append(layer_values)
        if values:
            return grid, values, "reported_landscape_grid_values", {
                "grid_source": "reported_landscape_grid",
                "values_source": "reported_landscape_values",
                "vector_length": int(sum(len(layer) for layer in values)),
            }
    vector = _as_float_list(landscape.get("vector", []))
    try:
        resolution = int(landscape.get("resolution", 0) or 0)
    except (TypeError, ValueError):
        resolution = 0
    try:
        num_landscapes = int(landscape.get("num_landscapes", 0) or 0)
    except (TypeError, ValueError):
        num_landscapes = 0
    if vector and resolution > 0:
        full_layers = len(vector) // resolution
        if num_landscapes <= 0 or num_landscapes > full_layers:
            num_landscapes = full_layers
        if num_landscapes > 0:
            grid = [float(idx) / float(max(resolution - 1, 1)) for idx in range(resolution)]
            values = [vector[idx * resolution : (idx + 1) * resolution] for idx in range(num_landscapes)]
            return grid, values, "normalized_index_from_gudhi_landscape_vector", {
                "grid_source": "normalized_index_from_gudhi_vector_resolution",
                "values_source": "gudhi.representations.Landscape.vector",
                "vector_length": int(len(vector)),
                "reported_num_landscapes": int(num_landscapes),
                "reported_resolution": int(resolution),
            }
    return [], [], "unavailable_no_landscape_grid_values_or_vector", {
        "grid_source": "unavailable",
        "values_source": "unavailable",
        "vector_length": int(len(vector)),
        "reported_resolution": int(resolution),
        "reported_num_landscapes": int(num_landscapes),
    }


def _persistence_landscape_unavailable_contract(reason: str) -> dict[str, object]:
    return {
        "schema_version": "tropicalgt.persistence_landscape_visual_contract.v1",
        "available": False,
        "reason": reason,
        "source": "topology.persistence_representations.methods[*].landscape",
        "actual_data_only": True,
        "no_proxy_or_fallback": True,
        "not_nll_fitness_landscape": True,
        "not_norm_only_summary": False,
        "safe_to_render_actual_landscape_functions": False,
        "curve_trace_count": 0,
        "growth_row_count": 0,
        "rendered_growth_level_count": 0,
        "homology_dimensions": [],
        "landscape_rows": [],
        "unavailable_reasons": [reason],
        "render_contract": "Persistence landscape pages render only actual GUDHI Landscape vectors/lambda_k rows from persistence_representations; unavailable states are explicit and are not replaced by NLL/fitness landscapes, zero vectors, norms, or proxy summaries.",
    }


def _write_growth_persistence_landscapes(path: Path, topology: dict[str, object], growth: list[object], title_prefix: str = "") -> None:
    rows = _trajectory_growth_rows(topology, growth)
    fig = make_subplots(
        rows=3,
        cols=1,
        specs=[[{"type": "scene"}], [{"type": "heatmap"}], [{"type": "xy"}]],
        subplot_titles=(
            "GUDHI persistence landscape lambda_k(t) curves by growth level",
            "First available lambda_1(t) image by growth level",
            "Legible small multiples of actual lambda_k(t) vectors",
        ),
        row_heights=[0.46, 0.24, 0.30],
        vertical_spacing=0.12,
    )
    panel_objects: list[dict[str, object]] = []
    panel_hover: list[str] = []
    palette = {
        0: ["#55d6be", "#2dd4bf", "#0ea5e9", "#38bdf8"],
        1: ["#7aa2ff", "#a78bfa", "#c084fc", "#818cf8"],
        2: ["#fbbf24", "#f59e0b", "#fb7185", "#f97316"],
    }
    heatmap_rows: list[list[float]] = []
    heatmap_y: list[str] = []
    heatmap_x: list[float] = []
    heatmap_dim: int | None = None
    trace_count = 0
    small_multiple_trace_count = 0
    small_multiple_series: list[dict[str, object]] = []
    landscape_contract_rows: list[dict[str, object]] = []
    unavailable_reasons: list[str] = []
    for row_idx, row in enumerate(rows):
        level = int(row.get("level", row_idx))
        topo = row.get("topological_algebra", {}) if isinstance(row.get("topological_algebra"), dict) else {}
        topo = _topology_with_persistence_representations(topo)
        obj = row.get("filtered_simplicial_object", {}) if isinstance(row.get("filtered_simplicial_object"), dict) else {}
        reps = topo.get("persistence_representations", {}) if isinstance(topo.get("persistence_representations"), dict) else {}
        methods = reps.get("methods", {}) if isinstance(reps.get("methods"), dict) else {}
        panel_objects.append(obj)
        panel_hover.append(_topology_growth_hover(level, obj, topo) + f"<br>{_persistence_representation_line(topo)}")
        for dim_key, method in sorted(methods.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 99):
            if not isinstance(method, dict) or not method.get("available"):
                continue
            dim = int(dim_key)
            landscape = method.get("landscape", {}) if isinstance(method.get("landscape"), dict) else {}
            grid, values, grid_source, value_meta = _landscape_grid_values(landscape)
            if not grid or not values:
                unavailable_reasons.append(f"level_{level}_H{dim}_{grid_source}")
                continue
            finite_values = [value for layer in values for value in layer if math.isfinite(float(value))]
            landscape_contract_rows.append(
                {
                    "level": level,
                    "homology_dimension": dim,
                    "source": "topology.persistence_representations.methods[*].landscape",
                    "backend": str(reps.get("backend", "gudhi.representations")),
                    "grid_source": value_meta.get("grid_source", grid_source),
                    "values_source": value_meta.get("values_source", "reported_landscape_values"),
                    "layer_count": int(len(values)),
                    "grid_count": int(len(grid)),
                    "vector_length": int(value_meta.get("vector_length", sum(len(layer) for layer in values)) or 0),
                    "finite_value_count": int(len(finite_values)),
                    "nonzero_value_count": int(sum(1 for value in finite_values if abs(float(value)) > 1e-12)),
                    "min_value": float(min(finite_values)) if finite_values else 0.0,
                    "max_value": float(max(finite_values)) if finite_values else 0.0,
                    "actual_gudhi_landscape_values": True,
                    "not_norm_only_summary": True,
                    "not_nll_fitness_landscape": True,
                    **{key: value for key, value in value_meta.items() if key not in {"grid_source", "values_source", "vector_length"}},
                }
            )
            if values and heatmap_dim is None:
                heatmap_dim = dim
                heatmap_x = grid
            if values and dim == heatmap_dim:
                lambda1 = _as_float_list(values[0])
                if len(lambda1) == len(heatmap_x):
                    heatmap_rows.append(lambda1)
                    heatmap_y.append(f"L{level} H{dim}")
            colors = palette.get(dim, ["#e2e8f0", "#94a3b8", "#64748b", "#cbd5e1"])
            for layer_idx, layer in enumerate(values[:4]):
                ys = _as_float_list(layer)
                if len(ys) != len(grid):
                    continue
                y_level = [level + 0.075 * layer_idx + 0.025 * dim for _ in grid]
                color = colors[layer_idx % len(colors)]
                fig.add_trace(
                    go.Scatter3d(
                        x=grid,
                        y=y_level,
                        z=ys,
                        mode="lines",
                        line=dict(width=max(2.2, 5.0 - 0.65 * layer_idx), color=color),
                        name=f"L{level} H{dim} lambda_{layer_idx + 1}",
                        hovertemplate=(
                            f"growth level={level}<br>"
                            f"H{dim} GUDHI persistence landscape lambda_{layer_idx + 1}(t)<br>"
                            "filtration t=%{x:.4g}<br>"
                            "lambda value=%{z:.5g}<extra></extra>"
                        ),
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )
                trace_count += 1
                max_abs = max([abs(float(value)) for value in ys if math.isfinite(float(value))] or [1.0])
                scale = max(max_abs, 1e-12)
                offset = float(small_multiple_trace_count) * 1.35
                small_y = [float(value) / scale + offset for value in ys]
                small_name = f"L{level} H{dim} lambda_{layer_idx + 1}"
                fig.add_trace(
                    go.Scatter(
                        x=grid,
                        y=small_y,
                        mode="lines",
                        line=dict(width=2.4, color=color),
                        name=small_name,
                        customdata=ys,
                        hovertemplate=(
                            f"small multiple: growth level={level}<br>"
                            f"H{dim} lambda_{layer_idx + 1}(t)<br>"
                            "filtration t=%{x:.4g}<br>"
                            "actual lambda value=%{customdata:.5g}<extra></extra>"
                        ),
                        showlegend=True,
                    ),
                    row=3,
                    col=1,
                )
                small_multiple_series.append(
                    {
                        "level": int(level),
                        "homology_dimension": int(dim),
                        "lambda_index": int(layer_idx + 1),
                        "grid_count": int(len(grid)),
                        "normalization": "visual_y = lambda_value / max_abs(lambda_row) + stacked_offset; hover shows actual lambda value",
                        "actual_values_source": "gudhi.representations.Landscape.vector",
                    }
                )
                small_multiple_trace_count += 1
    if heatmap_rows:
        fig.add_trace(
            go.Heatmap(
                z=heatmap_rows,
                x=heatmap_x,
                y=heatmap_y,
                colorscale=[
                    [0.0, "#08111f"],
                    [0.25, "#0e7490"],
                    [0.55, "#22d3ee"],
                    [0.78, "#bef264"],
                    [1.0, "#facc15"],
                ],
                colorbar=dict(title=f"H{heatmap_dim if heatmap_dim is not None else '?'} lambda_1(t)", x=1.02, y=0.20, len=0.28, thickness=14),
                hovertemplate="level/dim=%{y}<br>filtration t=%{x:.4g}<br>lambda_1(t)=%{z:.5g}<extra></extra>",
                name="lambda_1 heatmap",
            ),
            row=2,
            col=1,
        )
    if trace_count == 0:
        fig.add_annotation(
            text="No finite persistence intervals were available for actual Landscape curves.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    fig.update_layout(
        template="plotly_dark",
        title=(
            f"{title_prefix}Actual GUDHI persistence landscapes"
            "<br><sup>lambda_k(t) curves from GUDHI Landscape vectors, not norm-only summaries; "
            "hover links each row to the filtered complex. This is distinct from the GoT NLL/fitness landscape.</sup>"
        ),
        scene=dict(
            xaxis_title="filtration or normalized landscape sample t",
            yaxis_title="trajectory growth level",
            zaxis_title="GUDHI persistence landscape value",
            aspectmode="manual",
            aspectratio=dict(x=1.15, y=0.78, z=0.86),
            camera=dict(eye=dict(x=1.55, y=-1.75, z=1.18)),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.16, xanchor="left", x=0.0, font=dict(size=10)),
        height=1560,
        margin=dict(t=160, l=86, r=124, b=170),
    )
    fig.update_xaxes(title_text="filtration or normalized landscape sample t", row=2, col=1)
    fig.update_yaxes(title_text="growth level / homology dimension", row=2, col=1)
    fig.update_xaxes(title_text="filtration or normalized landscape sample t", row=3, col=1)
    fig.update_yaxes(title_text="stacked small multiples; hover shows actual lambda value", row=3, col=1)
    homology_dimensions = sorted({int(row["homology_dimension"]) for row in landscape_contract_rows})
    rendered_levels = sorted({int(row["level"]) for row in landscape_contract_rows})
    contract = {
        "schema_version": "tropicalgt.persistence_landscape_visual_contract.v1",
        "available": bool(trace_count > 0 and landscape_contract_rows),
        "source": "topology.persistence_representations.methods[*].landscape",
        "actual_data_only": True,
        "no_proxy_or_fallback": True,
        "not_nll_fitness_landscape": True,
        "not_norm_only_summary": bool(trace_count > 0 and landscape_contract_rows),
        "safe_to_render_actual_landscape_functions": bool(trace_count > 0 and landscape_contract_rows),
        "curve_trace_count": int(trace_count),
        "growth_row_count": int(len(rows)),
        "rendered_growth_level_count": int(len(rendered_levels)),
        "rendered_growth_levels": rendered_levels,
        "homology_dimensions": homology_dimensions,
        "heatmap_available": bool(heatmap_rows),
        "heatmap_source": "first available lambda_1(t) rows from actual landscape values" if heatmap_rows else "unavailable_no_lambda1_rows",
        "small_multiples_available": bool(small_multiple_trace_count > 0),
        "small_multiple_trace_count": int(small_multiple_trace_count),
        "small_multiple_layout": "stacked_2d_lambda_curves_with_visual_offsets_hover_shows_actual_values",
        "small_multiple_series": small_multiple_series,
        "landscape_rows": landscape_contract_rows,
        "unavailable_reasons": unavailable_reasons,
        "render_contract": "Persistence landscape pages render only actual GUDHI Landscape vectors/lambda_k rows from persistence_representations; unavailable states are explicit and are not replaced by NLL/fitness landscapes, zero vectors, norms, or proxy summaries.",
    }
    path.with_suffix(".json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    fig.update_layout(meta={"persistence_landscape_visual_contract": contract})
    _write_plotly_dark_html(
        path,
        fig,
        f"{title_prefix}Actual GUDHI persistence landscape functions",
        _simplicial_panel_items(panel_objects, panel_hover),
    )


def _prepare_barcode_intervals(intervals: list[object], epsilon: float = 1e-9) -> tuple[list[dict[str, object]], dict[str, object]]:
    raw: list[dict[str, object]] = [row for row in intervals if isinstance(row, dict)]
    finite_deaths = [
        float(row.get("death"))
        for row in raw
        if row.get("death") is not None and isinstance(row.get("death"), (int, float)) and math.isfinite(float(row.get("death")))
    ]
    finite_births = [
        float(row.get("birth", 0.0))
        for row in raw
        if isinstance(row.get("birth", 0.0), (int, float)) and math.isfinite(float(row.get("birth", 0.0)))
    ]
    lower = min(finite_births + finite_deaths + [0.0])
    upper = max(finite_births + finite_deaths + [1.0])
    span = max(upper - lower, 1.0)
    infinity_display = upper + 0.12 * span
    prepared = []
    zero_length = 0
    for row in raw:
        birth = float(row.get("birth", 0.0))
        infinite = bool(row.get("infinite") or row.get("death") is None)
        death_value = None if infinite else float(row.get("death", birth))
        length = float("inf") if infinite else max(float(death_value) - birth, 0.0)
        if not infinite and length <= epsilon:
            zero_length += 1
            continue
        prepared.append(
            {
                **row,
                "birth": birth,
                "death": death_value,
                "display_death": infinity_display if infinite else death_value,
                "length": length,
                "infinite": infinite,
            }
        )
    prepared.sort(
        key=lambda row: (
            int(row.get("dimension", 0)),
            float(row.get("birth", 0.0)),
            float("inf") if row.get("infinite") else float(row.get("death", row.get("birth", 0.0))),
        )
    )
    return prepared, {
        "raw": len(raw),
        "displayed": len(prepared),
        "zero_length_filtered": zero_length,
        "epsilon": epsilon,
        "infinity_display": infinity_display,
        "dimension_counts": {
            str(dim): sum(1 for row in prepared if int(row.get("dimension", 0)) == dim)
            for dim in sorted({int(row.get("dimension", 0)) for row in prepared})
        },
    }


def _trajectory_growth_rows(topology: dict[str, object], growth: list[object] | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw_growth = growth if isinstance(growth, list) else []
    for idx, item in enumerate(raw_growth):
        if not isinstance(item, dict):
            continue
        topo = item.get("topological_algebra") if isinstance(item.get("topological_algebra"), dict) else {}
        topo = _topology_with_persistence_representations(topo)
        obj = item.get("filtered_simplicial_object") if isinstance(item.get("filtered_simplicial_object"), dict) else {}
        intervals, _ = _prepare_barcode_intervals(_persistence_intervals_only(topo), epsilon=0.0)
        rows.append(
            {
                "level": int(item.get("level", idx)),
                "filtered_simplicial_object": obj,
                "topological_algebra": topo,
                "intervals": intervals,
                "states": topo.get("persistence_module", {}).get("states", []) if isinstance(topo.get("persistence_module"), dict) else [],
            }
        )
    if not rows and isinstance(topology, dict):
        topology = _topology_with_persistence_representations(topology)
        intervals, _ = _prepare_barcode_intervals(_persistence_intervals_only(topology), epsilon=0.0)
        rows.append(
            {
                "level": 0,
                "filtered_simplicial_object": {},
                "topological_algebra": topology,
                "intervals": intervals,
                "states": topology.get("persistence_module", {}).get("states", []) if isinstance(topology.get("persistence_module"), dict) else [],
            }
        )
    return rows


def _topology_with_persistence_representations(topology: dict[str, object]) -> dict[str, object]:
    if not isinstance(topology, dict):
        return {}
    reps = topology.get("persistence_representations")
    if isinstance(reps, dict) and (reps.get("available") or "methods" in reps):
        return topology
    persistence = topology.get("persistence", {}) if isinstance(topology.get("persistence"), dict) else {}
    intervals = persistence.get("intervals", []) if isinstance(persistence, dict) else []
    if not isinstance(intervals, list):
        return topology
    try:
        from .algebra import compute_persistence_representations_from_intervals

        enriched = dict(topology)
        enriched["persistence_representations"] = compute_persistence_representations_from_intervals(
            [row for row in intervals if isinstance(row, dict)]
        )
        return enriched
    except Exception as exc:
        enriched = dict(topology)
        enriched["persistence_representations"] = {
            "available": False,
            "backend": "gudhi.representations",
            "error": f"{type(exc).__name__}: {exc}",
        }
        return enriched


def _persistence_intervals_only(topology: dict[str, object]) -> list[dict[str, object]]:
    intervals = topology.get("persistence", {}).get("intervals", []) if isinstance(topology.get("persistence"), dict) else []
    if intervals:
        return [row for row in intervals if isinstance(row, dict)]
    return []


def _as_float_list(values: object) -> list[float]:
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for value in values:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            out.append(val)
    return out


def _representation_summary_sum(reps: dict[str, object], method: str, key: str) -> float:
    methods = reps.get("methods", {}) if isinstance(reps.get("methods"), dict) else {}
    total = 0.0
    for row in methods.values():
        if not isinstance(row, dict) or not row.get("available"):
            continue
        method_report = row.get(method, {})
        if isinstance(method_report, dict):
            total += float(method_report.get(key, 0.0) or 0.0)
    return total


def _combined_betti_curve_vector(reps: dict[str, object], max_points_per_dim: int = 64) -> list[float]:
    methods = reps.get("methods", {}) if isinstance(reps.get("methods"), dict) else {}
    combined: list[float] = []
    for _, row in sorted(methods.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 99):
        if not isinstance(row, dict) or not row.get("available"):
            continue
        betti = row.get("betti_curve", {}) if isinstance(row.get("betti_curve"), dict) else {}
        vals = _as_float_list(betti.get("values", []))
        if len(vals) > max_points_per_dim:
            take = np.linspace(0, len(vals) - 1, max_points_per_dim).round().astype(int)
            vals = [vals[int(idx)] for idx in take]
        combined.extend(vals)
    return combined


def _topology_growth_hover(level: int, obj: dict[str, object], topology: dict[str, object]) -> str:
    return (
        f"<b>trajectory growth level {level}</b>"
        f"<br>{_summary_line(obj)}"
        f"<br>{_derived_signature_line(topology)}"
        f"<br>{_free_resolution_line(topology)}"
    )


def _summary_line(obj: dict[str, object]) -> str:
    summary = obj.get("summary", {}) if isinstance(obj, dict) else {}
    tree = obj.get("simplex_tree", {}) if isinstance(obj, dict) and isinstance(obj.get("simplex_tree"), dict) else {}
    tree_bits = ""
    if tree:
        tree_bits = f" | simplex_tree={tree.get('backend', 'unknown')} dim={tree.get('dimension', '?')} n={tree.get('num_simplices', '?')}"
    return (
        f"V={summary.get('num_vertices', 0)} E={summary.get('num_edges', 0)} "
        f"T={summary.get('num_two_simplices', 0)} thresholds={summary.get('num_thresholds', 0)}{tree_bits}"
    )


def _persistence_backend_label(topology: dict[str, object]) -> str:
    persistence = topology.get("persistence", {}) if isinstance(topology, dict) else {}
    if isinstance(persistence, dict):
        backend = persistence.get("backend")
        available = persistence.get("available")
        if backend:
            return f"{backend}{'' if available is not False else ':unavailable'}"
    return "unknown"


def _vertex_readable_summary(vertex_row: dict[str, object], include_output: bool = False) -> str:
    label = str((vertex_row.get("simplex") or ["vertex"])[0]) if isinstance(vertex_row, dict) else "vertex"
    bits = [
        f"<b>{html.escape(_short_label(label, 96))}</b>",
        f"type={html.escape(str(vertex_row.get('type', 'vertex')))}",
        f"filtration={float(vertex_row.get('filtration', 0.0) or 0.0):.4f}",
    ]
    if vertex_row.get("gudhi_simplex_tree"):
        bits.append("filtration source=GUDHI SimplexTree")
    if vertex_row.get("level") is not None:
        bits.append(f"trajectory level={int(vertex_row.get('level', 0) or 0)}")
    if vertex_row.get("nll") is not None:
        bits.append(f"NLL={float(vertex_row.get('nll', 0.0) or 0.0):.4f}")
    if vertex_row.get("score") is not None:
        bits.append(f"score={float(vertex_row.get('score', 0.0) or 0.0):.4f}")
    probability = vertex_row.get("probability")
    if isinstance(probability, list) and probability:
        preview = ", ".join(f"{float(v):.3g}" for v in probability[:8] if isinstance(v, (int, float)))
        bits.append(f"probability source={html.escape(str(vertex_row.get('probability_source', 'unknown')))}")
        bits.append(f"probability preview=[{preview}{', ...' if len(probability) > 8 else ''}]")
    if vertex_row.get("path"):
        bits.append(f"GoT path={html.escape(_json_clip(vertex_row.get('path'), 180))}")
    local_obj = vertex_row.get("filtered_simplicial_object")
    if isinstance(local_obj, dict):
        bits.append(f"<b>node filtered simplicial object</b>: {_summary_line(local_obj)}")
    if vertex_row.get("graph_json_summary"):
        bits.append(f"<b>graph summary</b>: {html.escape(_json_clip(vertex_row.get('graph_json_summary'), 320))}")
    if vertex_row.get("text"):
        bits.append(f"<b>node text</b>: {_html_clip(vertex_row.get('text', ''), 420)}")
    if include_output:
        if vertex_row.get("input_text"):
            bits.append(f"<b>model input</b>: {_html_clip(vertex_row.get('input_text'), 700)}")
        if vertex_row.get("target_text"):
            bits.append(f"<b>target</b>: {_html_clip(vertex_row.get('target_text'), 360)}")
        if vertex_row.get("decoded_argmax"):
            bits.append(f"<b>model output</b>: {_html_clip(vertex_row.get('decoded_argmax'), 700)}")
    return "<br>".join(bits)


def _candidate_axis_label(row: dict[str, object], idx: int, long: bool = False) -> str:
    path = row.get("path", [])
    level = int(row.get("level", 0) or 0)
    if not long:
        return f"q{idx:02d} L{level}"
    if isinstance(path, list) and path:
        tail = "/".join(str(item) for item in path[-3:])
    else:
        tail = "root"
    label = f"q{idx:02d} L{level} {tail}"
    return f"{label} | {row.get('record_id', idx)}"


def _derived_signature_line(topology: dict[str, object]) -> str:
    sig = topology.get("derived_equivalence_signature", {}) if isinstance(topology, dict) else {}
    betti = sig.get("betti_vector", [])
    finite = sig.get("persistence_finite_interval_count", 0)
    infinite = sig.get("persistence_infinite_interval_count", 0)
    grid = sig.get("multiparameter_grid_points", 0)
    return f"derived signature: betti={betti} finitePH={finite} infinitePH={infinite} multiparameter_grid={grid}"


def _chain_presentation_reports(topology: dict[str, object]) -> list[dict[str, object]]:
    ca = topology.get("commutative_algebra", {}) if isinstance(topology, dict) else {}
    reports: list[dict[str, object]] = []
    for key in (
        "two_parameter_chain_presentation_diagnostics",
        "multiparameter_chain_presentation_diagnostics",
    ):
        value = ca.get(key)
        if isinstance(value, dict):
            reports.append(value)
    return reports


def _certified_multigraded_resolution_modules(topology: dict[str, object]) -> list[dict[str, object]]:
    for report in _chain_presentation_reports(topology):
        real = report.get("real_free_resolution") if isinstance(report.get("real_free_resolution"), dict) else {}
        if not (
            real.get("safe_to_render_as_multigraded_free_resolution") is True
            and real.get("multigraded_free_resolution_certified") is True
            and real.get("exactness_certified") is True
        ):
            continue
        summary = real.get("free_resolution_summary") if isinstance(real.get("free_resolution_summary"), dict) else {}
        modules = summary.get("free_modules", []) if isinstance(summary.get("free_modules"), list) else []
        if modules:
            return [row for row in modules if isinstance(row, dict)]
    return []


def _free_resolution_line(topology: dict[str, object]) -> str:
    modules = _certified_multigraded_resolution_modules(topology)
    if modules:
        ranks = [f"F{int(row.get('homological_degree', 0))}:{int(row.get('rank', row.get('rank_upper_bound', 0)))}" for row in modules[:6]]
        return "CAS-certified multigraded free resolution over the persistence-module ring: " + ", ".join(ranks)

    statuses: list[str] = []
    chain_ranks: list[str] = []
    for report in _chain_presentation_reports(topology):
        ring = str(report.get("ring", "F2[x_level,x_radius]"))
        real = report.get("real_free_resolution") if isinstance(report.get("real_free_resolution"), dict) else {}
        if real.get("safe_to_render_as_total_graded_resolution") is True:
            statuses.append(f"real total-graded CAS resolution available over {ring}; multigraded persistence-module resolution unavailable")
        elif real.get("ungraded_resolution_certified") is True:
            statuses.append(f"real ungraded CAS resolution available over {ring}; multigraded persistence-module resolution unavailable")
        elif real.get("available"):
            statuses.append(str(real.get("render_warning") or f"CAS resolution not safe to render as multigraded over {ring}"))
        else:
            statuses.append(f"multigraded free resolution unavailable over {ring}; CAS backend needed")
        for row in report.get("free_chain_modules", []) if isinstance(report.get("free_chain_modules"), list) else []:
            if isinstance(row, dict):
                chain_ranks.append(f"C{int(row.get('homological_degree', 0))}:{int(row.get('rank', row.get('rank_upper_bound', 0)))}")
    diagnostic = "; exact finite-chain diagnostics: " + ", ".join(chain_ranks[:6]) if chain_ranks else ""
    return "; ".join(statuses[:2]) + diagnostic


def _persistence_representation_line(topology: dict[str, object]) -> str:
    reps = topology.get("persistence_representations", {}) if isinstance(topology, dict) and isinstance(topology.get("persistence_representations"), dict) else {}
    if not reps:
        return "persistence vectorizations: unavailable"
    summary = reps.get("summary", {}) if isinstance(reps.get("summary"), dict) else {}
    return (
        "persistence vectorizations: "
        f"available={bool(reps.get('available'))}, "
        f"finite intervals={int(reps.get('finite_interval_count', 0) or 0)}, "
        f"persistence-landscape L2={float(summary.get('landscape_l2_norm', 0.0) or 0.0):.4g}, "
        f"topological-vector L2={float(summary.get('topological_vector_l2_norm', 0.0) or 0.0):.4g}"
    )


def _free_resolution_modules(topology: dict[str, object]) -> list[dict[str, object]]:
    """Return only CAS-certified multigraded free-resolution modules.

    Exact finite-chain modules and Taylor bounds are intentionally excluded so
    downstream analogical/free-resolution similarity cannot silently use them as
    substitutes for a persistence-module free resolution.
    """
    return _certified_multigraded_resolution_modules(topology)


def write_graphcg_trajectory_visualization(scaling_report: dict[str, object], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    path = output_dir / "graphcg_direction_cosines.html"
    payload_path = output_dir / "graphcg_direction_cosines_payload.json"
    candidates = [row for row in scaling_report.get("candidates", []) if isinstance(row, dict)]
    matrices = []
    basis_sources: list[str] = []
    mean_abs_offdiag_cosines: list[float] = []
    max_abs_offdiag_cosines: list[float] = []
    labels = []
    compact_labels = []
    hover_rows = []
    for idx, row in enumerate(candidates):
        proj = row.get("graphcg_projection")
        if isinstance(proj, dict) and proj.get("all_direction_cosines") is not None:
            matrices.append([float(v) for v in proj["all_direction_cosines"]])
            basis_sources.append(str(proj.get("basis") or "unknown"))
            for target, key in (
                (mean_abs_offdiag_cosines, "mean_abs_offdiag_cosine"),
                (max_abs_offdiag_cosines, "max_abs_offdiag_cosine"),
            ):
                try:
                    value = float(proj.get(key))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    target.append(value)
            labels.append(_candidate_axis_label(row, idx, long=True))
            compact_labels.append(_candidate_axis_label(row, idx))
            hover_rows.append(
                f"<b>{html.escape(_candidate_axis_label(row, idx, long=True))}</b>"
                f"<br>level={row.get('level')} path={html.escape(_json_clip(row.get('path', []), 180))}"
                f"<br>NLL={float(row.get('nll', 0.0) or 0.0):.4f} score={float(row.get('score', 0.0) or 0.0):.4f}"
            )
    if not matrices:
        _write_dark_empty(path, "No GraphCG projection diagnostics available.")
        payload_path.write_text(json.dumps({"available": False, "reason": "No GraphCG projection diagnostics available."}, indent=2), encoding="utf-8")
        return {"graphcg_direction_cosines": str(path), "graphcg_direction_cosines_payload": str(payload_path)}
    matrix = np.asarray(matrices, dtype=float)
    abs_matrix = np.abs(matrix)
    mean_abs = np.mean(np.abs(matrix), axis=0)
    signed_mean = np.mean(matrix, axis=0)
    candidate_mean_abs = np.mean(abs_matrix, axis=1)
    candidate_peak_abs = np.max(abs_matrix, axis=1)
    row_probs = abs_matrix / np.maximum(np.sum(abs_matrix, axis=1, keepdims=True), 1e-12)
    candidate_entropy = -np.sum(row_probs * np.log2(np.maximum(row_probs, 1e-12)), axis=1)
    candidate_effective_dirs = np.power(2.0, candidate_entropy)
    direction_count = int(matrix.shape[1])
    display_count = direction_count
    top_idx = np.arange(direction_count, dtype=int)
    visible_direction_tick_label_limit = 8
    visible_candidate_tick_label_limit = 12
    z_signed = matrix[:, top_idx]
    z_abs = np.abs(z_signed)
    custom = [
        [
            hover_rows[row_idx]
            + f"<br>direction={int(direction_idx)}"
            + f"<br>signed cosine={z_signed[row_idx, col_idx]:.5f}"
            + f"<br>|cosine|={z_abs[row_idx, col_idx]:.5f}"
            + f"<br>mean |cos| for direction={mean_abs[int(direction_idx)]:.5f}"
            + f"<br>signed mean for direction={signed_mean[int(direction_idx)]:.5f}"
            for col_idx, direction_idx in enumerate(top_idx)
        ]
        for row_idx in range(z_abs.shape[0])
    ]
    active_floor = float(np.quantile(mean_abs, 0.9)) if mean_abs.size else 0.0
    active_rank = int(np.sum(mean_abs > 1e-8)) if mean_abs.size else 0
    candidate_active_counts = np.sum(abs_matrix >= max(active_floor, 1e-12), axis=1)
    sorted_order = np.argsort(mean_abs)[::-1]
    sorted_activity = mean_abs[sorted_order]
    sorted_signed = signed_mean[sorted_order]
    top_active_direction_limit = int(min(max(8, visible_direction_tick_label_limit), direction_count))
    top_active_direction_indices = sorted_order[:top_active_direction_limit]
    top_active_direction_rows = [
        {
            "rank": int(rank + 1),
            "direction_id": int(direction_idx),
            "label": f"d{int(direction_idx)}",
            "mean_abs_cosine": float(mean_abs[int(direction_idx)]),
            "signed_mean_cosine": float(signed_mean[int(direction_idx)]),
            "source": "candidate.graphcg_projection.all_direction_cosines",
            "rendered_in_top_active_direction_panel": True,
            "exact_direction_id_preserved": True,
            "no_proxy_or_fallback": True,
        }
        for rank, direction_idx in enumerate(top_active_direction_indices.tolist())
    ]
    top_active_direction_ids = {int(row["direction_id"]) for row in top_active_direction_rows}
    basis_source_counts = {basis: int(basis_sources.count(basis)) for basis in sorted(set(basis_sources))}
    projection_basis = sorted(set(basis_sources))[0] if len(set(basis_sources)) == 1 else "mixed"
    activity_rank_by_direction = {int(direction): int(rank + 1) for rank, direction in enumerate(sorted_order.tolist())}
    direction_rows = [
        {
            "direction_id": int(direction_idx),
            "display_column": int(col_idx),
            "source": "candidate.graphcg_projection.all_direction_cosines",
            "mean_abs_cosine": float(mean_abs[int(direction_idx)]),
            "signed_mean_cosine": float(signed_mean[int(direction_idx)]),
            "activity_rank_desc": int(activity_rank_by_direction.get(int(direction_idx), col_idx + 1)),
            "rendered_in_all_direction_heatmap": True,
            "rendered_in_full_rank_activity_spectrum": True,
            "rendered_in_signed_bias_panel": True,
            "rendered_in_top_active_direction_panel": bool(int(direction_idx) in top_active_direction_ids),
            "exact_direction_id_preserved": True,
            "no_proxy_or_fallback": True,
        }
        for col_idx, direction_idx in enumerate(top_idx.tolist())
    ]
    direction_evidence_contract = {
        "schema_version": "tropicalgt.graphcg_direction_evidence.v1",
        "source": "candidate.graphcg_projection.all_direction_cosines",
        "no_proxy_or_fallback": True,
        "all_model_directions_have_rows": bool(len(direction_rows) == direction_count),
        "direction_count": int(direction_count),
        "direction_row_count": int(len(direction_rows)),
        "exact_direction_ids_preserved": True,
        "all_directions_rendered_in_heatmap": all(row["rendered_in_all_direction_heatmap"] for row in direction_rows),
        "all_directions_rendered_in_activity_spectrum": all(row["rendered_in_full_rank_activity_spectrum"] for row in direction_rows),
        "all_directions_rendered_in_signed_bias_panel": all(row["rendered_in_signed_bias_panel"] for row in direction_rows),
        "top_active_direction_panel_count": int(len(top_active_direction_rows)),
        "top_active_direction_panel_source": "top directions by mean_abs_cosine over observed candidate GraphCG projections",
        "mean_abs_source": "mean absolute cosine over observed candidate GraphCG projections",
        "signed_mean_source": "signed mean cosine over observed candidate GraphCG projections",
        "activity_rank_source": "descending order of mean_abs_cosine across every model-derived direction",
        "safe_to_render_full_rank_direction_evidence": bool(len(direction_rows) == direction_count and direction_count > 0),
    }
    basis_certificate = {
        "source": "candidate.graphcg_projection",
        "available": bool(basis_sources),
        "projection_basis": projection_basis,
        "basis_sources": sorted(set(basis_sources)),
        "basis_source_counts": basis_source_counts,
        "candidate_count": int(len(basis_sources)),
        "direction_count": int(direction_count),
        "all_candidates_have_all_direction_cosines": bool(len(matrices) == len(basis_sources) and all(len(row) == direction_count for row in matrices)),
        "mean_abs_offdiag_cosine_values": [float(v) for v in mean_abs_offdiag_cosines],
        "max_abs_offdiag_cosine_values": [float(v) for v in max_abs_offdiag_cosines],
        "mean_abs_offdiag_cosine_max": float(max(mean_abs_offdiag_cosines)) if mean_abs_offdiag_cosines else None,
        "max_abs_offdiag_cosine_max": float(max(max_abs_offdiag_cosines)) if max_abs_offdiag_cosines else None,
    }
    graphcg_readability_contract = {
        "schema_version": "tropicalgt.graphcg_direction_readability.v1",
        "source": "candidate.graphcg_projection",
        "no_proxy_or_fallback": True,
        "all_model_directions_rendered": True,
        "directions_sampled_for_heatmap": False,
        "panels_are_separate": True,
        "required_panels": [
            "all_direction_heatmap",
            "top_active_direction_panel",
            "full_rank_activity_spectrum",
            "candidate_activity_by_observed_got_state",
            "direction_signed_bias",
        ],
        "top_active_direction_source": "ranked subset by mean_abs_cosine over observed candidate GraphCG projections; exact direction ids preserved in hover and payload",
        "full_rank_spectrum_source": "mean absolute cosine over every model-derived GraphCG direction",
        "candidate_activity_source": "per-candidate absolute GraphCG projection cosines from observed GoT states",
        "signed_bias_source": "signed mean GraphCG projection cosine per direction",
        "visible_tick_labels_bounded": True,
        "exact_direction_ids_preserved_in_hover_and_payload": True,
        "candidate_path_action_text_preserved_in_hover_and_payload": True,
        "hover_fields": ["candidate label", "level", "path", "NLL", "score", "direction", "signed cosine", "absolute cosine", "direction mean absolute cosine", "direction signed mean"],
    }
    payload_path.write_text(
        json.dumps(
            {
                "available": True,
                "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
                "display_count": int(display_count),
                "displayed_direction_indices": [int(idx) for idx in top_idx.tolist()],
                "display_policy": "all_model_graphcg_directions_no_sampling",
                "readability_contract": "five coordinated panels render all model GraphCG directions plus a readable top-active direction excerpt: all-direction heatmap, top-active direction panel, full-rank activity spectrum, candidate activity by observed GoT state, and signed-bias scatter; visible tick labels are bounded while hover and payload preserve exact ids",
                "graphcg_readability_contract": graphcg_readability_contract,
                "graphcg_direction_evidence_contract": direction_evidence_contract,
                "direction_rows": direction_rows,
                "panel_names": [
                    "all_direction_heatmap",
                    "top_active_direction_panel",
                    "full_rank_activity_spectrum",
                    "candidate_activity_by_observed_got_state",
                    "direction_signed_bias",
                ],
                "panel_count": 5,
                "all_direction_heatmap_available": True,
                "top_active_direction_panel_available": True,
                "direction_spectrum_panel_available": True,
                "candidate_activity_panel_available": True,
                "direction_signed_bias_panel_available": True,
                "directions_sampled_for_heatmap": False,
                "full_rank_direction_count": int(direction_count),
                "active_rank_nonzero_mean_abs": active_rank,
                "mean_abs_min": float(np.min(mean_abs)) if mean_abs.size else 0.0,
                "mean_abs_max": float(np.max(mean_abs)) if mean_abs.size else 0.0,
                "mean_abs_p90": active_floor,
                "mean_abs": [float(v) for v in mean_abs.tolist()],
                "signed_mean": [float(v) for v in signed_mean.tolist()],
                "candidate_labels_compact": compact_labels,
                "candidate_labels": labels,
                "candidate_hover_rows": hover_rows,
                "candidate_activity_mean_abs": [float(v) for v in candidate_mean_abs.tolist()],
                "candidate_peak_abs": [float(v) for v in candidate_peak_abs.tolist()],
                "candidate_effective_direction_count": [float(v) for v in candidate_effective_dirs.tolist()],
                "candidate_active_direction_count": [int(v) for v in candidate_active_counts.tolist()],
                "direction_activity_sorted": [float(v) for v in sorted_activity.tolist()],
                "direction_signed_mean_sorted": [float(v) for v in sorted_signed.tolist()],
                "top_active_direction_limit": int(top_active_direction_limit),
                "top_active_direction_rows": top_active_direction_rows,
                "activity_threshold_p90": active_floor,
                "visible_direction_tick_label_limit": int(visible_direction_tick_label_limit),
                "visible_candidate_tick_label_limit": int(visible_candidate_tick_label_limit),
                "exact_direction_labels_available_in_hover_and_payload": True,
                "projection_basis_certificate": basis_certificate,
                "graphcg_basis_sources": sorted(set(basis_sources)),
                "graphcg_basis_source_counts": basis_source_counts,
                "interpretation": "GraphCG directions are full-rank: the heatmap contains every model-derived direction; visible tick labels are bounded for readability while hover/payload preserve exact direction ids.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    graphcg_height = int(max(1560, 1080 + 34 * len(compact_labels)))
    fig = make_subplots(
        rows=5,
        cols=1,
        specs=[[{"type": "heatmap"}], [{"type": "bar"}], [{"type": "scatter"}], [{"type": "scatter"}], [{"type": "scatter"}]],
        row_heights=[0.36, 0.16, 0.16, 0.16, 0.16],
        vertical_spacing=0.075,
        subplot_titles=(
            "Readable full-rank heatmap: every model GraphCG direction",
            "Top active GraphCG directions by mean |cos|",
            "Full-rank activity spectrum",
            "Candidate activity by observed GoT state",
            "Signed bias for every direction",
        ),
    )
    fig.add_trace(
        go.Heatmap(
            z=z_abs,
            y=compact_labels,
            x=[f"d{int(idx)}" for idx in top_idx],
            colorscale="Magma",
            zmin=0.0,
            zmax=max(float(np.nanmax(z_abs)), 1e-9) if z_abs.size else 1.0,
            showscale=False,
            customdata=custom,
            hovertemplate="%{customdata}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    top_activity_values = [float(row["mean_abs_cosine"]) for row in top_active_direction_rows]
    top_signed_values = [float(row["signed_mean_cosine"]) for row in top_active_direction_rows]
    signed_scale = max(max([abs(value) for value in top_signed_values] or [1.0]), 1e-12)
    fig.add_trace(
        go.Bar(
            x=[str(row["label"]) for row in top_active_direction_rows],
            y=top_activity_values,
            marker=dict(
                color=top_signed_values,
                colorscale="RdBu",
                cmin=-signed_scale,
                cmax=signed_scale,
                line=dict(color="#e8eef8", width=0.7),
                colorbar=dict(title="signed mean", x=1.02, y=0.62, len=0.14),
            ),
            customdata=[
                f"rank={row['rank']}<br>direction={row['direction_id']}<br>mean |cos|={row['mean_abs_cosine']:.5f}<br>signed mean={row['signed_mean_cosine']:.5f}<br>source={html.escape(str(row['source']))}"
                for row in top_active_direction_rows
            ],
            hovertemplate="%{customdata}<extra></extra>",
            name="top active directions",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(direction_count),
            y=sorted_activity,
            mode="lines",
            line=dict(color="#5eead4", width=2),
            fill="tozeroy",
            fillcolor="rgba(94,234,212,0.18)",
            hovertemplate="rank=%{x}<br>mean |cos|=%{y:.5f}<extra></extra>",
            name="full-rank spectrum",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(compact_labels))),
            y=candidate_mean_abs,
            mode="lines+markers",
            marker=dict(
                size=np.clip(4.0 + 8.0 * candidate_peak_abs / max(float(np.max(candidate_peak_abs)), 1e-12), 4.0, 12.0),
                color=candidate_effective_dirs,
                colorscale="Turbo",
                colorbar=dict(title="effective dirs", x=1.02, y=0.34, len=0.16),
                line=dict(color="#e8eef8", width=0.8),
            ),
            line=dict(color="#5eead4", width=2),
            customdata=[
                hover_rows[i]
                + f"<br>mean |cos|={candidate_mean_abs[i]:.5f}"
                + f"<br>peak |cos|={candidate_peak_abs[i]:.5f}"
                + f"<br>effective directions={candidate_effective_dirs[i]:.1f}/{direction_count}"
                + f"<br>directions above p90 activity threshold={int(candidate_active_counts[i])}"
                for i in range(len(compact_labels))
            ],
            hovertemplate="%{customdata}<extra></extra>",
            name="candidate activity",
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=mean_abs,
            y=signed_mean,
            mode="markers",
            marker=dict(
                size=5,
                color=np.arange(direction_count),
                colorscale="Viridis",
                showscale=False,
                opacity=0.72,
                line=dict(color="rgba(226,232,240,0.35)", width=0.4),
            ),
            customdata=np.arange(direction_count),
            hovertemplate="direction=%{customdata}<br>mean |cos|=%{x:.5f}<br>signed mean=%{y:.5f}<extra></extra>",
            name="direction signed bias",
        ),
        row=5,
        col=1,
    )
    fig.update_layout(
        template="plotly_dark",
        title=(
            "GraphCG full-rank direction audit"
            f"<br><sup>heatmap shows all {direction_count} model-derived directions; basis={html.escape(projection_basis)}; active nonzero rank={active_rank}; exact ids in hover/payload.</sup>"
        ),
        margin=dict(t=156, l=118, r=118, b=124),
        height=graphcg_height,
    )
    x_labels = [f"d{int(idx)}" for idx in top_idx]
    x_step = max(1, int(math.ceil(len(x_labels) / max(visible_direction_tick_label_limit, 1))))
    x_tickvals = [label for pos, label in enumerate(x_labels) if pos % x_step == 0 or pos == len(x_labels) - 1]
    y_step = max(1, int(math.ceil(len(labels) / max(visible_candidate_tick_label_limit, 1))))
    y_tickvals = [label for pos, label in enumerate(compact_labels) if pos % y_step == 0 or pos == len(compact_labels) - 1]
    fig.update_xaxes(title_text="GraphCG direction (bounded visible ticks; hover for exact direction)", tickangle=-35, tickmode="array", tickvals=x_tickvals, ticktext=x_tickvals, tickfont=dict(size=9), row=1, col=1)
    fig.update_yaxes(title_text="GoT state index (hover for path)", tickmode="array", tickvals=y_tickvals, ticktext=y_tickvals, row=1, col=1, tickfont=dict(size=10))
    fig.update_xaxes(title_text="top active direction id", tickangle=-20, row=2, col=1)
    fig.update_yaxes(title_text="mean absolute cosine", row=2, col=1)
    fig.update_xaxes(title_text="direction rank by activity", row=3, col=1)
    fig.update_yaxes(title_text="mean absolute cosine", row=3, col=1)
    fig.update_xaxes(title_text="GoT candidate index", row=4, col=1)
    fig.update_yaxes(title_text="candidate mean |cos|", row=4, col=1)
    fig.update_xaxes(title_text="direction mean |cos|", row=5, col=1)
    fig.update_yaxes(title_text="direction signed mean", row=5, col=1)
    _write_plotly_dark_html(path, fig, "GraphCG direction cosines along GoT candidates")
    return {"graphcg_direction_cosines": str(path), "graphcg_direction_cosines_payload": str(payload_path)}


def write_analogical_memory_visualization(
    memory: dict[str, object],
    output_dir: str | Path,
    query_context: dict[str, object] | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    path = output_dir / "analogical_memory_retrieval.html"
    map_path = output_dir / "analogical_simplicial_maps.json"
    rows = [row for row in memory.get("retrieved", []) if isinstance(row, dict)]
    if not rows:
        reason = "No non-self model-probability analogical memories retrieved; no analogical correspondence certificate is rendered."
        return _write_analogical_unavailable_outputs(
            output_dir,
            path,
            map_path,
            reason,
            "no_non_self_model_memory",
            memory=memory,
            raw_retrieved_count=len(rows),
            qualified_memory_count=0,
        )
    bank_records = _load_memory_bank_records(memory.get("bank_path", ""))
    enriched = [_enrich_memory_row(row, bank_records) for row in rows]
    query = query_context if isinstance(query_context, dict) else {}
    query_complex_source = ""
    query_complex: dict[str, object] = {}
    if _has_real_probability_filtration(query.get("trajectory_probability_filtered_simplicial_object")):
        query_complex = query["trajectory_probability_filtered_simplicial_object"]
        query_complex_source = "trajectory_probability_filtered_simplicial_object"
    query_topology = query.get("topological_algebra") if isinstance(query.get("topological_algebra"), dict) else {}
    if not query_complex:
        reason = "No model probability filtered query trajectory complex was available; analogical maps are not rendered without model probabilities."
        return _write_analogical_unavailable_outputs(
            output_dir,
            path,
            map_path,
            reason,
            "missing_model_probability_query_complex",
            memory=memory,
            raw_retrieved_count=len(rows),
            qualified_memory_count=0,
        )

    enriched = [
        row
        for row in enriched
        if _has_real_probability_filtration(row.get("trajectory_probability_filtered_simplicial_object"))
    ]
    if not enriched:
        reason = "No model probability filtered codomain trajectory complex was available; analogical maps are not rendered without model probabilities."
        return _write_analogical_unavailable_outputs(
            output_dir,
            path,
            map_path,
            reason,
            "missing_model_probability_codomain_complex",
            query_complex_source=query_complex_source,
            memory=memory,
            raw_retrieved_count=len(rows),
            qualified_memory_count=0,
        )

    pair_pages: list[dict[str, object]] = []
    map_reports: list[dict[str, object]] = []
    for idx, row in enumerate(enriched):
        pair_path = path if idx == 0 else output_dir / f"analogical_memory_map_{idx + 1:02d}.html"
        fig, panel_items, map_report = _analogical_pair_figure(row, idx, query_complex, query_topology, query_complex_source)
        _write_plotly_dark_html(
            pair_path,
            fig,
            f"Analogical probability correspondence rank {idx + 1}",
            panel_items,
            show_filtration_slider=True,
        )
        map_report["pair_page"] = pair_path.name
        map_reports.append(map_report)
        pair_pages.append(
            {
                "rank": idx + 1,
                "memory_id": row.get("memory_id"),
                "record_id": row.get("record_id"),
                "retrieval_score": float(row.get("retrieval_score", 0.0)),
                "path": str(pair_path),
            }
        )
    index_path = output_dir / "analogical_memory_topk_index.html"
    topk_contract = _analogical_topk_contract(
        memory,
        raw_retrieved_count=len(rows),
        qualified_memory_count=len(enriched),
        top_k_rendered=len(map_reports),
        status="available",
        query_complex_source=query_complex_source,
    )
    _write_analogical_topk_index(index_path, pair_pages, map_reports, contract=topk_contract)
    simplex_tree_path = output_dir / "analogical_simplex_tree_analogy.html"
    simplex_tree_payload_path = output_dir / "analogical_simplex_tree_analogy.json"
    simplex_tree_analogy = _write_analogical_simplex_tree_analogy(
        simplex_tree_path,
        simplex_tree_payload_path,
        pair_pages,
        map_reports,
        topk_contract=topk_contract,
    )

    map_path.write_text(
        json.dumps(
            {
                "available": True,
                "query_summary": query_complex.get("summary", {}) if isinstance(query_complex, dict) else {},
                "query_complex_source": query_complex_source,
                "query_derived_signature": query_topology.get("derived_equivalence_signature", {}) if isinstance(query_topology, dict) else {},
                "topk_contract": topk_contract,
                "simplex_tree_analogy_contract": simplex_tree_analogy.get("contract", {}),
                "simplex_tree_analogy_path": simplex_tree_path.name,
                "maps": map_reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    result = {
        "analogical_memory_retrieval_html": str(path),
        "analogical_simplicial_maps": str(map_path),
        "analogical_memory_topk_index_html": str(index_path),
        "analogical_simplex_tree_analogy_html": str(simplex_tree_path),
        "analogical_simplex_tree_analogy_json": str(simplex_tree_payload_path),
    }
    for page in pair_pages:
        result[f"analogical_memory_map_{int(page['rank']):02d}_html"] = str(page["path"])
    return result


def _analogical_pair_figure(
    row: dict[str, object],
    idx: int,
    query_complex: dict[str, object],
    query_topology: dict[str, object],
    query_complex_source: str,
) -> tuple[go.Figure, list[dict[str, object]], dict[str, object]]:
    panel_items: list[dict[str, object]] = []
    query_complex = _gudhi_canonical_complex(query_complex)
    query_layout = _complex_3d_layout(query_complex, slab=0.0)
    query_hover = (
        "<b>query trajectory filtered complex</b>"
        f"<br>{_summary_line(query_complex)}"
        f"<br>{_derived_signature_line(query_topology)}"
        f"<br>{_free_resolution_line(query_topology)}"
    )
    panel_items.extend(_simplicial_panel_items([query_complex], [query_hover]))
    panel_idx = 1
    slab = 2.4
    mem_complex_source = ""
    if _has_real_probability_filtration(row.get("trajectory_probability_filtered_simplicial_object")):
        mem_complex = row.get("trajectory_probability_filtered_simplicial_object")
        mem_complex_source = "trajectory_probability_filtered_simplicial_object"
    else:
        mem_complex = _unavailable_complex("missing_model_probability_codomain_complex")
        mem_complex_source = "unavailable_missing_model_probability_codomain_complex"
    mem_complex = _gudhi_canonical_complex(mem_complex if isinstance(mem_complex, dict) and mem_complex.get("available") is not False else mem_complex)
    mem_topology = (
        row.get("trajectory_probability_topological_algebra")
        if isinstance(row.get("trajectory_probability_topological_algebra"), dict)
        else (row.get("topological_algebra", {}) if isinstance(row.get("topological_algebra"), dict) else {})
    )
    mem_layout = _complex_3d_layout(mem_complex, slab=slab)
    sim = _topological_similarity_summary(query_topology, mem_topology, row)
    sim_map = _retrieval_probability_simplicial_map_report(row)
    derived_comparison = _derived_invariant_comparison(query_topology, mem_topology, sim)
    sim_map["derived_invariant_comparison"] = derived_comparison
    sim_map["algebraic_realization_certificate"] = _analogical_realization_certificate(sim, sim_map, derived_comparison)
    map_claim_label = str(sim_map.get("map_claim_label") or ("certified filtered simplicial map" if sim_map.get("is_filtered_simplicial_map") else "probability correspondence only; no map asserted"))
    map_report = {
        "memory_id": row.get("memory_id"),
        "record_id": row.get("record_id"),
        "rank": idx + 1,
        "domain_complex_summary": query_complex.get("summary", {}) if isinstance(query_complex, dict) else {},
        "codomain_complex_summary": mem_complex.get("summary", {}) if isinstance(mem_complex, dict) else {},
        "domain_simplex_tree": query_complex.get("simplex_tree", {}) if isinstance(query_complex, dict) else {},
        "codomain_simplex_tree": mem_complex.get("simplex_tree", {}) if isinstance(mem_complex, dict) else {},
        "query_complex_source": query_complex_source,
        "codomain_complex_source": mem_complex_source,
        **sim,
        **sim_map,
        "derived_invariant_comparison": derived_comparison,
    }
    panel_hover = (
        f"<b>codomain memory {idx + 1}</b>: {html.escape(str(row.get('memory_id', idx)))}"
        f"<br>retrieval={float(row.get('retrieval_score', 0.0)):.4f}"
        f"<br>persistent homology similarity={sim['persistent_homology_similarity']:.4f}"
        f"<br>chain-presentation diagnostic similarity={float(sim.get('chain_presentation_similarity', 0.0)):.4f}"
        f"<br>commutative-algebra similarity={sim.get('commutative_algebra_similarity', 0.0):.4f}"
        f"<br>persistence-landscape L2 similarity={sim.get('persistence_landscape_l2_similarity', 0.0):.4f}"
        f"<br>persistence-landscape cosine={sim.get('persistence_landscape_cosine', 0.0):.4f}"
        f"<br>vectorized topology aggregate={sim.get('persistence_vector_aggregate_similarity', 0.0):.4f}"
        f"<br>vectorized methods={html.escape(str(sim.get('persistence_vector_methods', '')))}"
        f"<br>derived/algebraic similarity={sim['derived_algebraic_similarity']:.4f}"
        f"<br>coarse signature cosine={sim['derived_signature_similarity']:.4f}"
        f"<br>finite derived invariants match={derived_comparison['finite_invariants_match']}"
        f"<br>simplicial edge preservation={sim_map['edge_preservation_rate']:.4f}"
        f"<br>{_summary_line(mem_complex)}"
        f"<br>{_derived_signature_line(mem_topology)}"
        f"<br>{_free_resolution_line(mem_topology)}"
    )
    panel_items.extend(_simplicial_panel_items([mem_complex], [panel_hover]))

    thresholds = _combined_display_thresholds(query_complex, mem_complex)
    initial = thresholds[0] if thresholds else 0.0
    data = _analogical_pair_traces(query_complex, query_layout, mem_complex, mem_layout, sim_map, row, sim, idx, panel_idx, initial)
    fig = go.Figure(data=data)
    quality_table = _analogical_quality_table_trace(row, sim, sim_map, idx)
    frames = []
    frame_thresholds = list(thresholds)
    for threshold in frame_thresholds:
        frames.append(
            go.Frame(
                name=f"{threshold:.6f}",
                data=_analogical_pair_traces(query_complex, query_layout, mem_complex, mem_layout, sim_map, row, sim, idx, panel_idx, threshold),
                traces=list(range(len(data))),
            )
        )
    fig.frames = frames
    if frames:
        steps = [
            {
                "args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                "label": f"{float(frame.name):.3f}",
                "method": "animate",
            }
            for frame in frames
        ]
        fig.update_layout(
            sliders=[
                {
                    "active": 0,
                    "currentvalue": {"prefix": "domain/codomain filtration <= ", "font": {"color": "#dbeafe"}},
                    "pad": {"t": 58},
                    "steps": steps,
                }
            ],
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "x": 0.02,
                    "y": 0,
                    "xanchor": "left",
                    "yanchor": "top",
                    "buttons": [
                        {
                            "label": "play filtration",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 240, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}],
                        }
                    ],
                }
            ],
        )
    fig.add_trace(quality_table)
    fig.update_layout(
        template="plotly_dark",
        title=(
            f"Analogical probability-matched correspondence: rank {idx + 1}"
            "<br><sup>query trajectory complex to retrieved memory complex</sup>"
            f"<br><sup>{html.escape(map_claim_label)}; filtered-complex certificate from model-probability Jensen-Shannon assignment; gold=preserved 1-simplices, rose=vertex-only correspondences; slider filters domain/codomain/certificate edges</sup>"
        ),
        height=1040,
        margin=dict(t=154, l=24, r=24, b=164),
        scene=dict(
            domain=dict(x=[0.0, 0.62], y=[0.24, 0.98]),
            xaxis=dict(title="domain / codomain embedding slabs", range=[-1.05, slab + 1.05]),
            yaxis=dict(title="PCoA/MDS-2", range=[-1.35, 1.35]),
            zaxis=dict(title="PCoA/MDS-3", range=[-1.35, 1.35]),
            camera=dict(eye=dict(x=2.05, y=1.55, z=1.05), center=dict(x=0.08, y=0.0, z=0.0)),
        ),
        legend=dict(itemsizing="constant", x=0.64, y=0.98, xanchor="left", yanchor="top"),
    )
    if not bool(sim_map.get("is_simplicial_on_displayed_skeleton")):
        fig.add_annotation(
            text=(
                "not a simplicial map on the displayed skeleton: "
                f"edges {int(sim_map.get('preserved_edges', 0))}/{int(sim_map.get('checked_edges', 0))}, "
                f"2-simplices {int(sim_map.get('preserved_two_simplices', 0))}/{int(sim_map.get('checked_two_simplices', 0))}"
            ),
            x=0.02,
            y=0.99,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            font=dict(size=12, color="#fecdd3"),
            bgcolor="rgba(127,29,29,0.72)",
            bordercolor="rgba(251,113,133,0.55)",
            borderwidth=1,
        )
    return fig, panel_items, map_report


def _analogical_quality_table_trace(
    row: dict[str, object],
    sim: dict[str, float],
    sim_map: dict[str, object],
    idx: int,
) -> go.Table:
    status = str(sim_map.get("map_claim_label") or ("certified filtered simplicial map; chain/persistence morphism diagnostics may be used" if bool(sim_map.get("is_filtered_simplicial_map")) else "probability correspondence only; no simplicial/chain/persistence morphism is asserted"))
    derived = sim_map.get("derived_invariant_comparison", {}) if isinstance(sim_map.get("derived_invariant_comparison"), dict) else {}
    source_label = str(sim_map.get("map_source", "unknown")).replace("model_probability_jensen_shannon_assignment", "prob-JS assignment")
    vector_methods_raw = str(sim.get("persistence_vector_methods", "")) or "unavailable"
    vector_methods_display = vector_methods_raw if vector_methods_raw == "unavailable" else ",<br>".join(part.strip() for part in vector_methods_raw.split(",") if part.strip())
    rows = [
        ("rank", str(idx + 1)),
        ("memory", _short_label(str(row.get("memory_id", idx)), 24)),
        ("retrieval", f"{float(row.get('retrieval_score', 0.0)):.4f}"),
        ("base retrieval", f"{float(sim.get('base_retrieval_score', 0.0)):.4f}"),
        ("landscape contribution", f"{float(sim.get('persistence_landscape_score_contribution', 0.0)):.4f}"),
        ("vector-family contribution", f"{float(sim.get('persistence_vector_score_contribution', 0.0)):.4f}"),
        ("probability-map contribution", f"{float(sim.get('probability_simplicial_map_score_contribution', 0.0)):.4f}"),
        ("probability-map similarity", f"{float(sim.get('probability_simplicial_map_similarity', 0.0)):.4f}"),
        ("retrieval probability map", f"{float(sim.get('probability_simplicial_map_preservation_rate', 0.0)):.4f} via {html.escape(str(sim.get('probability_simplicial_map_source', 'none')))}"),
        ("retrieval weights", html.escape(str(sim.get('retrieval_weights', {})))),
        ("PH similarity", f"{float(sim.get('persistent_homology_similarity', 0.0)):.4f}"),
        ("chain-presentation similarity", f"{float(sim.get('chain_presentation_similarity', 0.0)):.4f}"),
        ("comm-algebra similarity", f"{float(sim.get('commutative_algebra_similarity', 0.0)):.4f}"),
        ("persistence-landscape L2 sim", f"{float(sim.get('persistence_landscape_l2_similarity', 0.0)):.4f}"),
        ("persistence-landscape cosine", f"{float(sim.get('persistence_landscape_cosine', 0.0)):.4f}"),
        ("landscape vector dims", f"{int(float(sim.get('persistence_landscape_overlap_dim', 0.0)))}"),
        ("vector topology aggregate", f"{float(sim.get('persistence_vector_aggregate_similarity', 0.0)):.4f}"),
        ("vector methods", vector_methods_display),
        ("vector method count", f"{int(float(sim.get('persistence_vector_component_count', 0.0)))}"),
        ("vector component scores", html.escape(str(sim.get('persistence_vector_component_summary', 'unavailable')))),
        ("vector comparison space", html.escape(str(sim.get('persistence_vector_comparison_space', 'unavailable')))),
        ("vector differentiability note", html.escape(str(sim.get('persistence_vector_differentiable_note', 'unavailable')))),
        ("derived/algebraic similarity", f"{float(sim.get('derived_algebraic_similarity', 0.0)):.4f}"),
        ("coarse signature cosine", f"{float(sim.get('derived_signature_similarity', 0.0)):.4f}"),
        ("assignment source", source_label),
        ("JS mean/max", f"{_fmt_optional(sim_map.get('jensen_shannon_distance_mean'))}/{_fmt_optional(sim_map.get('jensen_shannon_distance_max'))}"),
        ("assign cost mean/max", f"{_fmt_optional(sim_map.get('assignment_cost_mean'))}/{_fmt_optional(sim_map.get('assignment_cost_max'))}"),
        ("filt distortion max", _fmt_optional(sim_map.get("max_positive_filtration_distortion"))),
        ("edge preservation", f"{int(sim_map.get('preserved_edges', 0))}/{int(sim_map.get('checked_edges', 0))} = {float(sim_map.get('edge_preservation_rate', 0.0)):.3f}"),
        ("face preservation", f"{int(sim_map.get('preserved_two_simplices', 0))}/{int(sim_map.get('checked_two_simplices', 0))} = {float(sim_map.get('two_simplex_preservation_rate', 0.0)):.3f}"),
        ("simplex-tree map", f"{int(sim_map.get('simplex_tree_map_preserved', 0))}/{int(sim_map.get('simplex_tree_map_checked', 0))} = {float(sim_map.get('simplex_tree_map_preservation_rate', 0.0)):.3f}"),
        ("map render claim", html.escape(str(sim_map.get("map_render_claim", "unavailable")))),
        ("safe as simplicial map", str(bool(sim_map.get("safe_to_render_as_simplicial_map")))),
        ("safe as chain map", str(bool(sim_map.get("safe_to_render_as_chain_map")))),
        ("safe as module morphism", str(bool(sim_map.get("safe_to_render_as_persistence_module_morphism")))),
        ("claim failure reason", html.escape(str(sim_map.get("map_claim_failure_reason") or "n/a"))),
        ("derived witness", str(derived.get("derived_equivalence_claim", "n/a"))),
        ("finite match", str(derived.get("finite_invariants_match", "n/a"))),
        ("display status", status),
    ]
    return go.Table(
        domain=dict(x=[0.64, 0.995], y=[0.24, 0.96]),
        header=dict(
            values=["certificate diagnostic", "value"],
            fill_color="#111827",
            font=dict(color="#e8eef8", size=12),
            align="left",
        ),
        cells=dict(
            values=[[row[0] for row in rows], [row[1] for row in rows]],
            fill_color=[["#0f172a"] * len(rows), ["#0b1220"] * len(rows)],
            font=dict(color="#dbeafe", size=11),
            align="left",
            height=30,
        ),
        name="filtered-complex certificate diagnostics",
    )


def _combined_display_thresholds(a: dict[str, object], b: dict[str, object]) -> list[float]:
    values = sorted(set(_display_thresholds(a, max_steps=24) + _display_thresholds(b, max_steps=24)))
    if not values:
        return [1.0]
    if len(values) <= 36:
        return values
    keep = np.linspace(0, len(values) - 1, 36, dtype=int)
    return [values[int(idx)] for idx in keep]


def _analogical_pair_traces(
    query_complex: dict[str, object],
    query_layout: dict[str, tuple[float, float, float]],
    mem_complex: dict[str, object],
    mem_layout: dict[str, tuple[float, float, float]],
    sim_map: dict[str, object],
    row: dict[str, object],
    sim: dict[str, float],
    idx: int,
    panel_idx: int,
    threshold: float,
) -> list[go.Scatter3d | go.Mesh3d]:
    traces: list[go.Scatter3d | go.Mesh3d] = []
    traces.extend(_complex_filtered_plotly_traces(query_complex, query_layout, 0, "domain: query trajectory complex", "#55d6be", "query domain", threshold))
    traces.extend(_complex_filtered_plotly_traces(mem_complex, mem_layout, panel_idx, f"codomain: memory {idx + 1} trajectory complex", _memory_color(idx), f"memory {idx + 1} codomain", threshold))
    traces.extend(_simplicial_map_traces(query_complex, mem_complex, query_layout, mem_layout, sim_map, panel_idx, row, sim, threshold))
    marker = dict(size=11, color=_memory_color(idx), showscale=False, line=dict(width=1.2, color="#e8eef8"))
    panel_hover = (
        f"<b>rank {idx + 1} retrieved memory</b>"
        f"<br>memory_id={html.escape(str(row.get('memory_id', idx)))}"
        f"<br>retrieval={float(row.get('retrieval_score', 0.0)):.4f}"
        f"<br>PH similarity={sim['persistent_homology_similarity']:.4f}"
        f"<br>chain-presentation diagnostic similarity={float(sim.get('chain_presentation_similarity', 0.0)):.4f}"
        f"<br>derived/algebraic similarity={sim['derived_algebraic_similarity']:.4f}"
        f"<br>coarse signature cosine={sim['derived_signature_similarity']:.4f}"
    )
    traces.append(
        go.Scatter3d(
            x=[1.2],
            y=[-1.16],
            z=[1.05],
            mode="markers+text",
            marker=marker,
            text=[f"rank {idx + 1}"],
            textposition="top center",
            hovertext=panel_hover,
            hoverinfo="text",
            customdata=[panel_idx],
            name=f"rank {idx + 1} invariants",
        )
    )
    return traces


def _complex_filtered_plotly_traces(
    obj: dict[str, object],
    coords: dict[str, tuple[float, float, float]],
    panel_idx: int,
    name: str,
    color: str,
    hover_prefix: str,
    threshold: float,
) -> list[go.Scatter3d | go.Mesh3d]:
    vertex_rows = [row for row in _complex_vertex_records(obj) if float(row.get("filtration", 0.0) or 0.0) <= threshold + 1e-12]
    visible = {str(row["label"]) for row in vertex_rows if str(row["label"]) in coords}
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    edge_hover: list[str | None] = []
    for simplex in obj.get("simplices", []) if isinstance(obj, dict) else []:
        if not isinstance(simplex, dict) or int(simplex.get("dimension", -1)) != 1:
            continue
        if float(simplex.get("filtration", 0.0) or 0.0) > threshold + 1e-12:
            continue
        raw = [str(v) for v in (simplex.get("simplex", []) or [])[:2]]
        if len(raw) < 2 or raw[0] not in visible or raw[1] not in visible:
            continue
        ax, ay, az = coords[raw[0]]
        bx, by, bz = coords[raw[1]]
        hover = f"<b>{html.escape(hover_prefix)} 1-simplex</b><br>{html.escape(raw[0])} -> {html.escape(raw[1])}<br>filtration={float(simplex.get('filtration', 0.0) or 0.0):.4f}<br>type={html.escape(str(simplex.get('type', 'edge')))}"
        edge_x.extend([ax, bx, None])
        edge_y.extend([ay, by, None])
        edge_z.extend([az, bz, None])
        edge_hover.extend([hover, hover, None])
    labels = [str(row["label"]) for row in vertex_rows if str(row["label"]) in coords]
    xs = [coords[label][0] for label in labels]
    ys = [coords[label][1] for label in labels]
    zs = [coords[label][2] for label in labels]
    vertex_by_label = {str(row["label"]): row for row in vertex_rows}
    vertex_hover = [_complex_vertex_hover(label, vertex_by_label.get(label, {}), hover_prefix) for label in labels]
    label_text = _salient_complex_text_labels(labels, vertex_by_label, max_labels=12)
    mesh_x: list[float] = []
    mesh_y: list[float] = []
    mesh_z: list[float] = []
    tri_i: list[int] = []
    tri_j: list[int] = []
    tri_k: list[int] = []
    mesh_index: dict[str, int] = {}
    for simplex in obj.get("simplices", []) if isinstance(obj, dict) else []:
        if not isinstance(simplex, dict) or int(simplex.get("dimension", -1)) != 2:
            continue
        if float(simplex.get("filtration", 0.0) or 0.0) > threshold + 1e-12:
            continue
        raw = [str(v) for v in (simplex.get("simplex", []) or [])[:3]]
        if len(raw) < 3 or any(label not in visible or label not in coords for label in raw):
            continue
        inds = []
        for label in raw:
            if label not in mesh_index:
                mesh_index[label] = len(mesh_x)
                x, y, z = coords[label]
                mesh_x.append(x); mesh_y.append(y); mesh_z.append(z)
            inds.append(mesh_index[label])
        tri_i.append(inds[0]); tri_j.append(inds[1]); tri_k.append(inds[2])
    return [
        go.Mesh3d(
            x=mesh_x,
            y=mesh_y,
            z=mesh_z,
            i=tri_i,
            j=tri_j,
            k=tri_k,
            color=color,
            opacity=0.18,
            hoverinfo="skip",
            showscale=False,
            name=f"{name} 2-simplices",
        ),
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color=color, width=4),
            name=f"{name} 1-simplices",
            hovertext=edge_hover,
            hoverinfo="text",
            showlegend=False,
        ),
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers+text",
            marker=dict(size=7, color=color, line=dict(color="#e8eef8", width=1)),
            text=label_text,
            textposition="top center",
            name=name,
            hovertext=vertex_hover,
            hoverinfo="text",
            customdata=[panel_idx] * len(labels),
        ),
    ]


def _complex_vertex_hover(label: str, vertex_row: dict[str, object], prefix: str) -> str:
    return f"<b>{html.escape(prefix)}</b><br>" + _vertex_readable_summary({**vertex_row, "simplex": [label]}, include_output=True)


def _salient_complex_text_labels(labels: list[str], vertex_by_label: dict[str, dict[str, object]], max_labels: int = 8) -> list[str]:
    if len(labels) <= max_labels:
        return [_short_label(label, 10) for label in labels]
    scored: list[tuple[float, int, str]] = []
    for idx, label in enumerate(labels):
        row = vertex_by_label.get(label, {})
        kind = str(row.get("type", "")).lower()
        filtration = float(row.get("filtration", 0.0) or 0.0)
        score = filtration
        if idx == 0 or "root" in kind or "problem" in kind:
            score += 4.0
        if any(key in kind for key in ("verification", "retrieved", "merged", "compressed", "rejected")):
            score += 1.5
        if "reasoning" in kind:
            score += 0.75
        scored.append((score, idx, label))
    keep = {idx for _, idx, _ in sorted(scored, reverse=True)[:max_labels]}
    return [_short_label(label, 10) if idx in keep else "" for idx, label in enumerate(labels)]


def _simplicial_map_traces(
    query_obj: dict[str, object],
    memory_obj: dict[str, object],
    query_layout: dict[str, tuple[float, float, float]],
    memory_layout: dict[str, tuple[float, float, float]],
    sim_map: dict[str, object],
    panel_idx: int,
    row: dict[str, object],
    sim: dict[str, float],
    threshold: float,
) -> list[go.Scatter3d]:
    q_vertex = _vertex_by_label(query_obj)
    m_vertex = _vertex_by_label(memory_obj)
    map_rows = sim_map.get("vertex_map", []) if isinstance(sim_map.get("vertex_map"), list) else []
    raw_map_count = len(map_rows)
    max_visible_map_edges = 54
    if len(map_rows) > max_visible_map_edges:
        map_rows = sorted(
            [row for row in map_rows if isinstance(row, dict)],
            key=lambda row: float(row.get("score", 0.0) or 0.0),
            reverse=True,
        )[:max_visible_map_edges]
    preserved_query_vertices = set(sim_map.get("preserved_edge_query_vertices", [])) if isinstance(sim_map.get("preserved_edge_query_vertices"), list) else set()
    traces_by_kind = {
        "preserved": {"x": [], "y": [], "z": [], "hover": []},
        "vertex_only": {"x": [], "y": [], "z": [], "hover": []},
    }
    for mapping in map_rows:
        if not isinstance(mapping, dict):
            continue
        q = str(mapping.get("query_vertex", ""))
        m = str(mapping.get("memory_vertex", ""))
        if q not in query_layout or m not in memory_layout:
            continue
        q_filtration = float(q_vertex.get(q, {}).get("filtration", 0.0) or 0.0)
        m_filtration = float(m_vertex.get(m, {}).get("filtration", 0.0) or 0.0)
        if max(q_filtration, m_filtration) > threshold + 1e-12:
            continue
        qx, qy, qz = query_layout[q]
        mx, my, mz = memory_layout[m]
        q_summary = _vertex_readable_summary({**q_vertex.get(q, {}), "simplex": [q]}, include_output=True)
        m_summary = _vertex_readable_summary({**m_vertex.get(m, {}), "simplex": [m]}, include_output=True)
        text = (
            f"<b>probability correspondence candidate</b>"
            f"<br>{html.escape(q)} -> {html.escape(m)}"
            f"<br>vertex score={float(mapping.get('score', 0.0)):.4f}"
            f"<br>map source={html.escape(str(mapping.get('map_source', sim_map.get('map_source', 'unknown'))))}"
            f"<br>JS distance={float(mapping.get('jensen_shannon_distance', 0.0)):.5f}"
            f"<br>assignment cost={float(mapping.get('assignment_cost', mapping.get('jensen_shannon_distance', 0.0))):.5f}"
            f"<br>max filtration distortion={_fmt_optional(sim_map.get('max_positive_filtration_distortion'))}"
            f"<br>line class={'preserves at least one displayed 1-simplex' if q in preserved_query_vertices else 'vertex-only correspondence'}"
            f"<br>edge preservation={float(sim_map.get('edge_preservation_rate', 0.0)):.4f}"
            f"<br>2-simplex preservation={float(sim_map.get('two_simplex_preservation_rate', 0.0)):.4f}"
            f"<br>displayed map edges={len(map_rows)}/{raw_map_count} top-scoring vertex maps"
            f"<br>PH similarity={sim['persistent_homology_similarity']:.4f}"
            f"<br>chain-presentation diagnostic similarity={float(sim.get('chain_presentation_similarity', 0.0)):.4f}"
            f"<br>commutative-algebra similarity={sim.get('commutative_algebra_similarity', 0.0):.4f}"
            f"<br>derived/algebraic similarity={sim['derived_algebraic_similarity']:.4f}<br>coarse signature cosine={sim['derived_signature_similarity']:.4f}"
            f"<br><br><b>domain vertex</b><br>{q_summary}"
            f"<br><br><b>codomain vertex</b><br>{m_summary}"
        )
        bucket = "preserved" if q in preserved_query_vertices else "vertex_only"
        traces_by_kind[bucket]["x"].extend([qx, mx, None])
        traces_by_kind[bucket]["y"].extend([qy, my, None])
        traces_by_kind[bucket]["z"].extend([qz, mz, None])
        traces_by_kind[bucket]["hover"].extend([text, text, None])
    return [
        go.Scatter3d(
            x=traces_by_kind["preserved"]["x"],
            y=traces_by_kind["preserved"]["y"],
            z=traces_by_kind["preserved"]["z"],
            mode="lines",
            line=dict(color="rgba(250,204,21,0.76)", width=4),
            name="preserved 1-simplex correspondences",
            hovertext=traces_by_kind["preserved"]["hover"],
            hoverinfo="text",
            customdata=[panel_idx if value is not None else None for value in traces_by_kind["preserved"]["x"]],
        ),
        go.Scatter3d(
            x=traces_by_kind["vertex_only"]["x"],
            y=traces_by_kind["vertex_only"]["y"],
            z=traces_by_kind["vertex_only"]["z"],
            mode="lines",
            line=dict(color="rgba(251,113,133,0.16)", width=1.2),
            name="vertex-only correspondences (legend only)",
            hovertext=traces_by_kind["vertex_only"]["hover"],
            hoverinfo="text",
            customdata=[panel_idx if value is not None else None for value in traces_by_kind["vertex_only"]["x"]],
            visible="legendonly",
        ),
    ]


def _write_analogical_unavailable_outputs(
    output_dir: Path,
    retrieval_path: Path,
    map_path: Path,
    reason: str,
    status: str,
    *,
    query_complex_source: str = "",
    memory: Mapping[str, object] | None = None,
    raw_retrieved_count: int = 0,
    qualified_memory_count: int = 0,
) -> dict[str, str]:
    _write_dark_empty(retrieval_path, reason)
    index_path = output_dir / "analogical_memory_topk_index.html"
    contract = _analogical_topk_contract(
        memory,
        raw_retrieved_count=raw_retrieved_count,
        qualified_memory_count=qualified_memory_count,
        top_k_rendered=0,
        status=status,
        unavailable_reason=reason,
        query_complex_source=query_complex_source,
    )
    _write_analogical_topk_index(index_path, [], [], contract=contract)
    map02_path = output_dir / "analogical_memory_map_02.html"
    map02_path.write_text(
        f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Analogical probability-matched correspondence unavailable</title></head>
<body style="margin:0;background:#090b12;color:#e8eef8;font-family:Inter,ui-sans-serif,system-ui,sans-serif;">
  <main style="max-width:980px;margin:0 auto;padding:32px 24px;">
    <h1>Analogical probability-matched correspondence filtered-complex certificate unavailable</h1>
    <p>{html.escape(reason)}</p>
    <p>No vertex assignment, filtered-complex certificate, simplex-tree map, chain map, or persistence-module morphism is fabricated for this rank slot.</p>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    payload: dict[str, object] = {
        "available": False,
        "reason": status,
        "reason_detail": reason,
        "topk_contract": contract,
        "maps": [],
    }
    if query_complex_source:
        payload["query_complex_source"] = query_complex_source
    simplex_tree_path = output_dir / "analogical_simplex_tree_analogy.html"
    simplex_tree_payload_path = output_dir / "analogical_simplex_tree_analogy.json"
    simplex_tree_analogy = _write_analogical_simplex_tree_analogy(
        simplex_tree_path,
        simplex_tree_payload_path,
        [],
        [],
        topk_contract=contract,
        unavailable_reason=reason,
    )
    payload["simplex_tree_analogy_contract"] = simplex_tree_analogy.get("contract", {})
    payload["simplex_tree_analogy_path"] = simplex_tree_path.name
    map_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "analogical_memory_retrieval_html": str(retrieval_path),
        "analogical_simplicial_maps": str(map_path),
        "analogical_memory_topk_index_html": str(index_path),
        "analogical_memory_map_02_html": str(map02_path),
        "analogical_simplex_tree_analogy_html": str(simplex_tree_path),
        "analogical_simplex_tree_analogy_json": str(simplex_tree_payload_path),
    }


def _analogical_simplex_key(values: object) -> str:
    if not isinstance(values, (list, tuple)):
        return "{}"
    return "{" + ", ".join(str(value) for value in values) + "}"


def _analogical_simplex_chain_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_domain = {
        tuple(str(value) for value in row.get("domain_simplex", [])): row
        for row in rows
        if isinstance(row.get("domain_simplex"), list)
    }
    chains: list[dict[str, object]] = []
    for row in rows:
        domain = tuple(str(value) for value in row.get("domain_simplex", [])) if isinstance(row.get("domain_simplex"), list) else ()
        if len(domain) < 2:
            continue
        face_rows: list[dict[str, object]] = []
        all_faces_preserved = True
        for face in combinations(domain, len(domain) - 1):
            face_key = tuple(sorted(face))
            face_row = by_domain.get(face_key)
            face_rows.append(
                {
                    "domain_face": list(face_key),
                    "image_face": list(face_row.get("image_simplex", [])) if isinstance(face_row, dict) and isinstance(face_row.get("image_simplex"), list) else [],
                    "preserved_in_simplex_tree": bool(face_row.get("preserved_in_simplex_tree")) if isinstance(face_row, dict) else False,
                    "missing_from_certificate": not isinstance(face_row, dict),
                }
            )
            all_faces_preserved = bool(all_faces_preserved and isinstance(face_row, dict) and face_row.get("preserved_in_simplex_tree"))
        chains.append(
            {
                "domain_coface": list(domain),
                "image_coface": list(row.get("image_simplex", [])) if isinstance(row.get("image_simplex"), list) else [],
                "dimension": int(row.get("dimension", len(domain) - 1) or 0),
                "coface_preserved_in_simplex_tree": bool(row.get("preserved_in_simplex_tree")),
                "all_boundary_faces_present_and_preserved": bool(all_faces_preserved),
                "boundary_faces": face_rows,
            }
        )
    return chains


def _write_analogical_simplex_tree_analogy(
    path: Path,
    payload_path: Path,
    pair_pages: list[dict[str, object]],
    map_reports: list[dict[str, object]],
    *,
    topk_contract: Mapping[str, object] | None = None,
    unavailable_reason: str = "",
) -> dict[str, object]:
    pair_payloads: list[dict[str, object]] = []
    total_checked = 0
    total_preserved = 0
    for page, report in zip(pair_pages, map_reports, strict=False):
        tree = report.get("simplex_tree_map") if isinstance(report.get("simplex_tree_map"), dict) else {}
        rows = [row for row in tree.get("rows", []) if isinstance(row, dict)] if isinstance(tree, dict) else []
        display_rows = rows[:160]
        chains = _analogical_simplex_chain_rows(display_rows)
        checked = int(report.get("simplex_tree_map_checked", tree.get("checked_simplices", len(rows))) or 0)
        preserved = int(report.get("simplex_tree_map_preserved", tree.get("preserved_simplices", 0)) or 0)
        total_checked += checked
        total_preserved += preserved
        pair_payloads.append(
            {
                "rank": int(page.get("rank", report.get("rank", 0)) or 0),
                "pair_page": Path(str(page.get("path", report.get("pair_page", "")))).name,
                "memory_id": str(page.get("memory_id", report.get("memory_id", "memory"))),
                "map_render_claim": str(report.get("map_render_claim", "unavailable")),
                "map_claim_failure_reason": report.get("map_claim_failure_reason"),
                "checked_simplices": checked,
                "preserved_simplices": preserved,
                "preservation_rate": float(report.get("simplex_tree_map_preservation_rate", tree.get("preservation_rate", 0.0)) or 0.0),
                "dimension_counts": tree.get("dimension_counts", {}) if isinstance(tree, dict) else {},
                "positive_filtration_distortion_summary": tree.get("positive_filtration_distortion_summary", {}) if isinstance(tree, dict) else {},
                "simplex_rows_truncated": bool(len(rows) > len(display_rows)),
                "simplex_rows": display_rows,
                "preserved_face_coface_chains": chains,
            }
        )
    contract = {
        "schema_version": "tropicalgt.analogical_simplex_tree_analogy.v1",
        "available": bool(pair_payloads),
        "status": "available" if pair_payloads else "unavailable_insufficient_model_probability_memory",
        "reason_detail": unavailable_reason,
        "source": "probability_simplicial_map.simplex_tree_map.rows",
        "simplex_tree_source": "finite GUDHI SimplexTree enumeration from trajectory_probability_filtered_simplicial_object pairs",
        "no_proxy_or_fallback": True,
        "compares_query_and_memory_simplex_trees": True,
        "renders_hasse_face_to_coface_rows": True,
        "preserved_face_coface_chains_highlighted": True,
        "failed_or_distorted_chains_labeled_not_maps": True,
        "chain_map_claim_requires_certified_filtered_simplicial_map": True,
        "persistence_module_morphism_claim_requires_certified_filtered_simplicial_map": True,
        "pair_count": int(len(pair_payloads)),
        "total_checked_simplices": int(total_checked),
        "total_preserved_simplices": int(total_preserved),
        "topk_contract_schema": (topk_contract or {}).get("schema_version") if isinstance(topk_contract, Mapping) else None,
    }
    payload = {"contract": contract, "pairs": pair_payloads}
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if pair_payloads:
        rows_html: list[str] = []
        for pair in pair_payloads:
            rows_html.append(
                "<tr class='pair-head'>"
                f"<td colspan='9'>rank {int(pair.get('rank', 0))}: <a href='{html.escape(str(pair.get('pair_page', '')))}'>{html.escape(str(pair.get('memory_id', 'memory')))}</a> | "
                f"claim={html.escape(str(pair.get('map_render_claim', 'unavailable')))} | "
                f"simplex-tree preservation={int(pair.get('preserved_simplices', 0))}/{int(pair.get('checked_simplices', 0))} = {float(pair.get('preservation_rate', 0.0)):.4f}</td>"
                "</tr>"
            )
            simplex_rows = pair.get("simplex_rows", []) if isinstance(pair.get("simplex_rows"), list) else []
            chain_by_domain = {
                tuple(str(v) for v in chain.get("domain_coface", [])): chain
                for chain in pair.get("preserved_face_coface_chains", [])
                if isinstance(chain, dict)
            }
            for row in simplex_rows[:80]:
                if not isinstance(row, dict):
                    continue
                domain = row.get("domain_simplex", []) if isinstance(row.get("domain_simplex"), list) else []
                image = row.get("image_simplex", []) if isinstance(row.get("image_simplex"), list) else []
                chain = chain_by_domain.get(tuple(str(v) for v in domain), {})
                face_count = len(chain.get("boundary_faces", [])) if isinstance(chain, dict) and isinstance(chain.get("boundary_faces"), list) else 0
                chain_status = "boundary preserved" if chain and chain.get("all_boundary_faces_present_and_preserved") else ("boundary failed/unavailable" if chain else "vertex row")
                preserved = bool(row.get("preserved_in_simplex_tree"))
                rows_html.append(
                    f"<tr class={'preserved' if preserved else 'failed'}>"
                    f"<td>{int(pair.get('rank', 0))}</td>"
                    f"<td>{int(row.get('dimension', 0) or 0)}</td>"
                    f"<td>{html.escape(_analogical_simplex_key(domain))}</td>"
                    f"<td>{html.escape(_analogical_simplex_key(image))}</td>"
                    f"<td>{float(row.get('domain_filtration', 0.0) or 0.0):.5g}</td>"
                    f"<td>{html.escape(str(row.get('codomain_filtration')))}</td>"
                    f"<td>{html.escape(str(row.get('signed_filtration_distortion')))}</td>"
                    f"<td>{'preserved' if preserved else html.escape(str(row.get('failure_reason') or 'failed'))}</td>"
                    f"<td>{html.escape(chain_status)} ({face_count} faces)</td>"
                    "</tr>"
                )
        body = "\n".join(rows_html)
    else:
        body = (
            "<tr class='failed'><td colspan='9'><strong>Simplex-tree analogy unavailable.</strong> "
            f"{html.escape(unavailable_reason or 'No qualified model-probability memories were rendered.')} "
            "No Hasse rows, chain maps, or persistence-module morphisms are fabricated.</td></tr>"
        )
    path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Analogical simplex-tree analogy</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin: 0; background: #090b12; color: #e8eef8; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px; }}
    h1 {{ font-size: 23px; margin: 0 0 8px; }}
    p {{ color: #a8b3c7; line-height: 1.55; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 18px; border: 1px solid rgba(148,163,184,0.26); background: #0f172a; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid rgba(148,163,184,0.16); text-align: left; font-size: 12px; vertical-align: top; }}
    th {{ color: #99f6e4; font-weight: 650; }}
    a {{ color: #7dd3fc; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .contract {{ border: 1px solid rgba(125,211,252,0.28); background: #0d1626; padding: 14px 16px; margin-top: 18px; }}
    .badge {{ display: inline-block; margin: 0 8px 8px 0; padding: 3px 8px; border: 1px solid rgba(153,246,228,0.35); color: #99f6e4; font-size: 11px; text-transform: uppercase; letter-spacing: 0; }}
    tr.pair-head td {{ background: #111827; color: #e0f2fe; font-weight: 650; }}
    tr.preserved td {{ color: #d1fae5; }}
    tr.failed td {{ color: #fecdd3; background: rgba(127,29,29,0.16); }}
  </style>
</head>
<body>
  <main>
    <h1>Analogical simplex-tree analogy</h1>
    <p>This view compares query and memory GUDHI SimplexTree finite Hasse rows using the stored model-probability Jensen-Shannon vertex assignment. Preserved rows are actual domain simplex to image simplex checks from the certificate; failed rows remain correspondences and are not called chain maps or persistence-module morphisms.</p>
    <aside class="contract"><span class="badge">no proxy</span><span class="badge">finite simplex-tree rows</span><span class="badge">preserved face-to-coface chains</span><p><strong>Contract:</strong> {html.escape(contract['schema_version'])}; source {html.escape(str(contract['source']))}; checked {int(total_checked)} simplices, preserved {int(total_preserved)}. Chain and module morphism claims require a certified filtered simplicial map.</p></aside>
    <table>
      <thead><tr><th>rank</th><th>dim</th><th>domain simplex</th><th>image simplex</th><th>domain filtration</th><th>codomain filtration</th><th>distortion</th><th>simplex-tree status</th><th>face-to-coface chain</th></tr></thead>
      <tbody>{body}</tbody>
    </table>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return payload


def _analogical_contract_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _analogical_contract_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_analogical_contract_value(v) for v in value]
    return str(value)


def _analogical_memory_quality_gate(memory: Mapping[str, object]) -> object:
    for key in (
        "quality_threshold",
        "memory_quality_threshold",
        "min_quality_score",
        "minimum_quality_score",
        "quality_gate",
    ):
        if key in memory:
            return _analogical_contract_value(memory.get(key))
    retrieval = memory.get("retrieval")
    if isinstance(retrieval, Mapping):
        for key in ("quality_threshold", "min_quality_score", "quality_gate"):
            if key in retrieval:
                return _analogical_contract_value(retrieval.get(key))
    return "not_reported"


def _analogical_topk_readability_contract(status: str, top_k_rendered: int) -> dict[str, object]:
    return {
        "schema_version": "tropicalgt.analogical_topk_readability.v1",
        "status": status,
        "top_k_rendered": int(top_k_rendered),
        "no_proxy_or_fallback": True,
        "topk_index_has_readable_table": True,
        "one_selected_map_view_per_rendered_rank": True,
        "table_rows_link_to_pair_pages": True,
        "insufficient_memory_state_explicit": True,
        "displays_quality_gate_and_filtered_counts": True,
        "separates_retrieval_probability_topology_algebra_columns": True,
        "probability_js_assignment_column_required": True,
        "map_claim_column_required": True,
        "simplex_tree_preservation_column_required": True,
        "edge_face_filtration_preservation_not_overclaimed": True,
        "derived_algebraic_column_policy": "conservative derived/algebraic score; high coarse signatures cannot substitute for missing PH/free-resolution/rank/chain-map evidence",
        "signature_cosine_column_policy": "coarse signature cosine is displayed separately and is not a derived-equivalence claim",
        "required_table_columns": [
            "correspondence",
            "retrieval",
            "prob-map source",
            "map claim",
            "derived/algebraic",
            "coarse signature",
            "simplex-tree map",
            "edge certificate",
        ],
    }


def _analogical_topk_contract(
    memory: Mapping[str, object] | None,
    *,
    raw_retrieved_count: int,
    qualified_memory_count: int,
    top_k_rendered: int,
    status: str,
    unavailable_reason: str = "",
    query_complex_source: str = "",
) -> dict[str, object]:
    memory = memory if isinstance(memory, Mapping) else {}
    requested = memory.get("top_k", memory.get("retrieval_limit", memory.get("k")))
    return {
        "schema_version": "tropicalgt.analogical_topk.v1",
        "available": status == "available",
        "status": status,
        "reason_detail": unavailable_reason,
        "no_proxy_or_fallback": True,
        "retrieval_requires_model_probability_vectors": True,
        "query_complex_required": "trajectory_probability_filtered_simplicial_object",
        "codomain_complex_required": "trajectory_probability_filtered_simplicial_object",
        "embedding_only_assignment_allowed": False,
        "assignment_metric": "jensen_shannon_distance_on_model_probability_vectors",
        "simplicial_map_claim_requires": "vertex_edge_face_simplex_tree_and_filtration_preservation",
        "chain_map_claim_requires": "certified_filtered_simplicial_map",
        "persistence_module_morphism_claim_requires": "certified_filtered_simplicial_map",
        "query_complex_source": query_complex_source or "unavailable",
        "raw_retrieved_count": int(raw_retrieved_count),
        "qualified_model_probability_memory_count": int(qualified_memory_count),
        "rejected_retrieved_count": int(max(raw_retrieved_count - qualified_memory_count, 0)),
        "top_k_requested": _analogical_contract_value(requested) if requested is not None else "not_reported",
        "top_k_rendered": int(top_k_rendered),
        "quality_gate": _analogical_memory_quality_gate(memory),
        "bank_path": str(memory.get("bank_path", "")),
        "readability_contract": _analogical_topk_readability_contract(status, top_k_rendered),
    }


def _analogical_contract_panel(contract: Mapping[str, object] | None) -> str:
    if not isinstance(contract, Mapping):
        return ""
    reason = str(contract.get("reason_detail") or "")
    reason_html = f"<p><strong>Unavailable reason:</strong> {html.escape(reason)}</p>" if reason else ""
    return (
        "<aside class='contract'>"
        "<div><span class='badge'>no proxy</span><span class='badge'>model probabilities only</span><span class='badge'>trajectory complexes required</span></div>"
        f"<p><strong>Status:</strong> {html.escape(str(contract.get('status', 'unknown')))}; "
        f"rendered {int(contract.get('top_k_rendered', 0) or 0)} / "
        f"{html.escape(str(contract.get('qualified_model_probability_memory_count', 0)))} qualified model-probability memories "
        f"from {html.escape(str(contract.get('raw_retrieved_count', 0)))} retrieved rows. "
        "Embedding-only assignments are rejected; chain maps and persistence-module morphisms are reported only after a certified filtered simplicial map.</p>"
        f"<p><strong>Assignment contract:</strong> {html.escape(str(contract.get('assignment_metric', 'unavailable')))}; "
        f"query source {html.escape(str(contract.get('query_complex_source', 'unavailable')))}; "
        f"quality gate {html.escape(str(contract.get('quality_gate', 'not_reported')))}.</p>"
        "<p><strong>Index readability contract:</strong> readable top-k table with one linked map view per rendered rank; "
        "retrieval, probability-JS assignment, topology, algebra, map-claim, simplex-tree, and edge-certificate evidence remain separate columns.</p>"
        f"{reason_html}"
        "</aside>"
    )


def _write_analogical_topk_index(path: Path, pair_pages: list[dict[str, object]], map_reports: list[dict[str, object]], *, contract: Mapping[str, object] | None = None) -> None:
    rows = []
    for page, report in zip(pair_pages, map_reports, strict=False):
        rel = html.escape(Path(str(page.get("path", ""))).name)
        rows.append(
            "<tr>"
            f"<td>{int(page.get('rank', 0))}</td>"
            f"<td><a href='{rel}'>{html.escape(str(page.get('memory_id', 'memory')))}</a></td>"
            f"<td>{float(page.get('retrieval_score', 0.0)):.4f}</td>"
            f"<td>{float(report.get('persistence_landscape_score_contribution', 0.0)):.4f}</td>"
            f"<td>{float(report.get('persistence_vector_score_contribution', 0.0)):.4f}</td>"
            f"<td>{float(report.get('persistent_homology_similarity', 0.0)):.4f}</td>"
            f"<td>{float(report.get('chain_presentation_similarity', 0.0)):.4f}</td>"
            f"<td>{float(report.get('commutative_algebra_similarity', 0.0)):.4f}</td>"
            f"<td>{float(report.get('persistence_landscape_l2_similarity', 0.0)):.4f}</td>"
            f"<td>{float(report.get('persistence_landscape_cosine', 0.0)):.4f}</td>"
            f"<td>{float(report.get('persistence_vector_aggregate_similarity', 0.0)):.4f}</td>"
            f"<td>{html.escape(str(report.get('persistence_vector_component_summary', report.get('persistence_vector_methods', ''))))}</td>"
            f'<td>{float(report.get("probability_simplicial_map_score_contribution", 0.0)):.4f}</td>'
            f'<td>{float(report.get("probability_simplicial_map_similarity", 0.0)):.4f}</td>'
            f'<td>{float(report.get("probability_simplicial_map_preservation_rate", report.get("simplex_tree_map_preservation_rate", 0.0))):.4f}</td>'
            f'<td>{html.escape(str(report.get("probability_simplicial_map_source", report.get("map_source", "none"))))}</td>'
            f'<td>{html.escape(str(report.get("map_render_claim", "unavailable")))}</td>'
            f"<td>{float(report.get('derived_algebraic_similarity', 0.0)):.4f}</td>"
            f"<td>{float(report.get('derived_signature_similarity', 0.0)):.4f}</td>"
            f"<td>{int(report.get('simplex_tree_map_preserved', 0))}/{int(report.get('simplex_tree_map_checked', 0))} = {float(report.get('simplex_tree_map_preservation_rate', 0.0)):.4f}</td>"
            f"<td>{float(report.get('edge_preservation_rate', 0.0)):.4f}</td>"
            "</tr>"
        )
    if rows:
        body = "\n".join(rows)
    else:
        reason = ""
        if isinstance(contract, Mapping):
            reason = str(contract.get("reason_detail") or contract.get("status") or "No retrieved memories.")
        body = (
            "<tr class='unavailable'><td colspan='21'>"
            "<strong>Insufficient model-probability memory.</strong> "
            f"No retrieved memories satisfy the no-proxy analogical contract. {html.escape(reason)} "
            "No vertex assignment, simplex-tree map, chain map, or persistence-module morphism is fabricated."
            "</td></tr>"
        )
    contract_panel = _analogical_contract_panel(contract)
    path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
	  <title>Analogical top-k probability correspondences</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin: 0; background: #090b12; color: #e8eef8; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 24px; }}
    h1 {{ font-size: 22px; margin: 0 0 8px; }}
    p {{ color: #99a8bd; line-height: 1.55; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: #101623; border: 1px solid rgba(148,163,184,0.25); }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid rgba(148,163,184,0.16); text-align: left; font-size: 13px; }}
    th {{ color: #99f6e4; font-weight: 650; }}
    a {{ color: #7dd3fc; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .contract {{ border: 1px solid rgba(125,211,252,0.28); background: #0d1626; padding: 14px 16px; margin: 18px 0 4px; }}
    .contract p {{ margin: 8px 0 0; }}
    .badge {{ display: inline-block; margin: 0 8px 8px 0; padding: 3px 8px; border: 1px solid rgba(153,246,228,0.35); color: #99f6e4; font-size: 11px; text-transform: uppercase; letter-spacing: 0; }}
    tr.unavailable td {{ color: #fecdd3; background: #1f1118; line-height: 1.5; }}
  </style>
</head>
<body>
  <main>
	    <h1>Analogical top-k probability correspondences</h1>
	    <p>Each row opens one query-to-memory vertex assignment with a finite filtered-complex certificate. The NLL/fitness landscape and the GUDHI persistence landscape are different objects: this table reports the persistence-landscape vector plus the wider vectorized GUDHI family (Landscape, BettiCurve, Silhouette, Entropy, PersistenceLengths, TopologicalVector, PersistenceImage) and the retrieval-side probability-map score contribution. These are real cached vectors and finite probability-complex certificates; the cosine/L2 comparisons are differentiable with respect to those vectors, while this HTML does not claim autograd through GUDHI diagram vectorization. Unavailable landscape vectors remain unavailable and contribute zero score rather than being fabricated as zero-valued landscape evidence. Edge, face, and filtration preservation can fail and are reported on the rank page.</p>
	    {contract_panel}
	    <table>
	      <thead><tr><th>rank</th><th>correspondence</th><th>retrieval</th><th>landscape contrib.</th><th>vector contrib.</th><th>PH</th><th>chain pres.</th><th>comm. alg.</th><th>persistence-landscape L2 sim</th><th>persistence-landscape cosine</th><th>vector aggregate</th><th>vector methods</th><th>prob-map contrib.</th><th>prob-map sim</th><th>prob-map preserved</th><th>prob-map source</th><th>map claim</th><th>derived/algebraic</th><th>coarse signature</th><th>simplex-tree map</th><th>edge certificate</th></tr></thead>
      <tbody>{body}</tbody>
    </table>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def _load_memory_bank_records(bank_path: object) -> dict[str, dict[str, object]]:
    path = Path(str(bank_path)) if bank_path else None
    if path is None or not path.exists():
        return {}
    records: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("memory_id") is not None:
            records[str(row["memory_id"])] = row
    return records


def _enrich_memory_row(row: dict[str, object], bank_records: dict[str, dict[str, object]]) -> dict[str, object]:
    memory_id = str(row.get("memory_id", ""))
    bank_row = bank_records.get(memory_id, {})
    enriched = {**bank_row, **row}
    metadata = enriched.get("metadata") if isinstance(enriched.get("metadata"), dict) else {}
    trajectory_complex = metadata.get("trajectory_filtered_simplicial_object") if isinstance(metadata, dict) else None
    if isinstance(trajectory_complex, dict):
        enriched["trajectory_filtered_simplicial_object"] = trajectory_complex
        enriched.setdefault("trajectory_summary", trajectory_complex.get("summary", {}))
    trajectory_probability_complex = metadata.get("trajectory_probability_filtered_simplicial_object") if isinstance(metadata, dict) else None
    if isinstance(trajectory_probability_complex, dict):
        enriched["trajectory_probability_filtered_simplicial_object"] = trajectory_probability_complex
        enriched.setdefault("trajectory_probability_summary", trajectory_probability_complex.get("summary", {}))
    trajectory_probability_topology = metadata.get("trajectory_probability_topological_algebra") if isinstance(metadata, dict) else None
    if isinstance(trajectory_probability_topology, dict):
        enriched["trajectory_probability_topological_algebra"] = trajectory_probability_topology
    if not isinstance(enriched.get("probability_filtered_simplicial_object"), dict) and isinstance(bank_row.get("probability_filtered_simplicial_object"), dict):
        enriched["probability_filtered_simplicial_object"] = bank_row["probability_filtered_simplicial_object"]
    if not isinstance(enriched.get("trajectory_probability_filtered_simplicial_object"), dict) and isinstance(bank_row.get("trajectory_probability_filtered_simplicial_object"), dict):
        enriched["trajectory_probability_filtered_simplicial_object"] = bank_row["trajectory_probability_filtered_simplicial_object"]
    # Do not promote a row-level probability complex to a trajectory-level object;
    # analogical maps must be between like-for-like trajectory complexes.
    if not isinstance(enriched.get("filtered_simplicial_object"), dict) and isinstance(bank_row.get("filtered_simplicial_object"), dict):
        enriched["filtered_simplicial_object"] = bank_row["filtered_simplicial_object"]
    if not isinstance(enriched.get("topological_algebra"), dict) and isinstance(bank_row.get("topological_algebra"), dict):
        enriched["topological_algebra"] = bank_row["topological_algebra"]
    if "derived_signature" not in enriched and isinstance(enriched.get("topological_algebra"), dict):
        enriched["derived_signature"] = enriched["topological_algebra"].get("derived_equivalence_signature", {})
    return enriched


def _complex_3d_layout(obj: dict[str, object], slab: float, max_vertices: int = 75) -> dict[str, tuple[float, float, float]]:
    vertices = _complex_vertex_records(obj)[:max_vertices]
    labels = [row["label"] for row in vertices]
    label_set = set(labels)
    edges = [
        {"simplex": list(pair), "filtration": _edge_filtration(obj, pair)}
        for pair in _complex_edge_pairs(obj)
        if pair[0] in label_set and pair[1] in label_set
    ]
    coords3_unit, _projected, _kind = _simplicial_pca3_radius_layout(labels, vertices, edges, width=520, height=420)
    coords: dict[str, tuple[float, float, float]] = {}
    for label in labels:
        x, y, z = coords3_unit.get(label, (0.5, 0.5, 0.5))
        coords[label] = (slab + (x - 0.5) * 0.82, (y - 0.5) * 2.1, (z - 0.5) * 2.1)
    return coords


def _add_complex_3d_traces(
    fig: go.Figure,
    obj: dict[str, object],
    coords: dict[str, tuple[float, float, float]],
    panel_idx: int,
    name: str,
    color: str,
    hover_prefix: str,
) -> None:
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    for a, b in _complex_edge_pairs(obj):
        if a not in coords or b not in coords:
            continue
        ax, ay, az = coords[a]
        bx, by, bz = coords[b]
        edge_x.extend([ax, bx, None])
        edge_y.extend([ay, by, None])
        edge_z.extend([az, bz, None])
    if edge_x:
        fig.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line=dict(color=color, width=4),
                name=f"{name} edges",
                hoverinfo="skip",
                showlegend=False,
            )
        )
    vertices = _complex_vertex_records(obj)
    labels = [row["label"] for row in vertices if row["label"] in coords]
    if labels:
        xs, ys, zs = zip(*(coords[label] for label in labels))
        hover = [
            f"<b>{hover_prefix}</b><br>vertex={html.escape(label)}<br>type={html.escape(str(_vertex_by_label(obj).get(label, {}).get('type', 'vertex')))}"
            for label in labels
        ]
        fig.add_trace(
            go.Scatter3d(
                x=list(xs),
                y=list(ys),
                z=list(zs),
                mode="markers+text",
                marker=dict(size=7, color=color, line=dict(color="#e8eef8", width=1)),
                text=[label[:14] for label in labels],
                textposition="top center",
                name=name,
                hovertext=hover,
                hoverinfo="text",
                customdata=[panel_idx] * len(labels),
            )
        )


def _add_simplicial_map_traces(
    fig: go.Figure,
    query_layout: dict[str, tuple[float, float, float]],
    memory_layout: dict[str, tuple[float, float, float]],
    sim_map: dict[str, object],
    panel_idx: int,
    row: dict[str, object],
    sim: dict[str, float],
) -> None:
    map_rows = sim_map.get("vertex_map", []) if isinstance(sim_map.get("vertex_map"), list) else []
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    hover: list[str | None] = []
    for mapping in map_rows:
        if not isinstance(mapping, dict):
            continue
        q = str(mapping.get("query_vertex", ""))
        m = str(mapping.get("memory_vertex", ""))
        if q not in query_layout or m not in memory_layout:
            continue
        qx, qy, qz = query_layout[q]
        mx, my, mz = memory_layout[m]
        text = (
            f"<b>probability correspondence candidate</b>"
            f"<br>{html.escape(q)} -> {html.escape(m)}"
            f"<br>vertex score={float(mapping.get('score', 0.0)):.4f}"
            f"<br>edge preservation={float(sim_map.get('edge_preservation_rate', 0.0)):.4f}"
            f"<br>2-simplex preservation={float(sim_map.get('two_simplex_preservation_rate', 0.0)):.4f}"
            f"<br>PH similarity={sim['persistent_homology_similarity']:.4f}"
            f"<br>chain-presentation diagnostic similarity={float(sim.get('chain_presentation_similarity', 0.0)):.4f}"
            f"<br>commutative-algebra similarity={sim.get('commutative_algebra_similarity', 0.0):.4f}"
            f"<br>derived/algebraic similarity={sim['derived_algebraic_similarity']:.4f}<br>coarse signature cosine={sim['derived_signature_similarity']:.4f}"
        )
        xs.extend([qx, mx, None])
        ys.extend([qy, my, None])
        zs.extend([qz, mz, None])
        hover.extend([text, text, None])
    if xs:
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color="rgba(251,191,36,0.56)", width=3),
                name=f"probability correspondence to {row.get('memory_id')}",
                hovertext=hover,
                hoverinfo="text",
                customdata=[panel_idx if value is not None else None for value in xs],
            )
        )




def _persistence_vector_component_rows(vector_report: dict[str, object]) -> list[dict[str, object]]:
    components = vector_report.get("components") if isinstance(vector_report, dict) else None
    if not isinstance(components, dict):
        return []
    rows: list[dict[str, object]] = []
    for method, raw in sorted(components.items()):
        if not isinstance(raw, dict) or not raw.get("available"):
            continue
        rows.append(
            {
                "method": str(method),
                "source": str(raw.get("source", "gudhi.representations.vector_methods")),
                "dimensions": list(raw.get("dims", [])) if isinstance(raw.get("dims"), list) else [],
                "overlap_dim": int(raw.get("overlap_dim", 0) or 0),
                "weight": float(raw.get("weight", 0.0) or 0.0),
                "vector_similarity": float(raw.get("vector_similarity", 0.0) or 0.0),
                "l2_similarity": float(raw.get("l2_similarity", 0.0) or 0.0),
                "l2_distance": float(raw.get("l2_distance", 0.0) or 0.0),
                "cosine": float(raw.get("cosine", 0.0) or 0.0),
                "correlation": float(raw.get("correlation", 0.0) or 0.0),
            }
        )
    return rows


def _persistence_vector_component_label(rows: list[dict[str, object]], limit: int = 7) -> str:
    if not rows:
        return "unavailable"
    parts = [
        f"{row['method']}:{float(row.get('vector_similarity', 0.0)):.3f}"
        for row in rows[:limit]
    ]
    if len(rows) > limit:
        parts.append(f"+{len(rows) - limit} more")
    return ", ".join(parts)


def _topological_similarity_summary(query_topology: dict[str, object], memory_topology: dict[str, object], row: dict[str, object]) -> dict[str, object]:
    q_sig = _signature_numeric_vector(query_topology)
    m_sig = _signature_numeric_vector(memory_topology)
    q_free = _free_rank_vector(query_topology)
    m_free = _free_rank_vector(memory_topology)
    q_ph = _persistence_numeric_vector(query_topology)
    m_ph = _persistence_numeric_vector(memory_topology)
    q_ca = _commutative_algebra_numeric_vector(query_topology)
    m_ca = _commutative_algebra_numeric_vector(memory_topology)
    q_rank = _rank_invariant_numeric_vector(query_topology)
    m_rank = _rank_invariant_numeric_vector(memory_topology)
    landscape_report = _persistence_landscape_vector_similarity(query_topology, memory_topology)
    retrieval_weights = row.get("retrieval_weights", {}) if isinstance(row.get("retrieval_weights"), dict) else {}
    include_landscape_in_vector = bool(
        retrieval_weights.get(
            "persistence_vector_includes_landscape",
            float(retrieval_weights.get("persistence_landscape_weight", 0.0) or 0.0) <= 0.0,
        )
    )
    vector_report = _persistence_vector_representation_similarity(
        query_topology,
        memory_topology,
        include_landscape=include_landscape_in_vector,
    )
    probability_chain_map_certified = bool(row.get("probability_simplicial_map_chain_map_certified"))
    probability_morphism_certified = bool(row.get("probability_simplicial_map_persistence_morphism_certified"))
    probability_map_rate = float(row.get("probability_simplicial_map_preservation_rate", 0.0) or 0.0)
    chain_map_score = probability_map_rate if probability_chain_map_certified and probability_morphism_certified else 0.0
    component_availability = {
        "signature_cosine": bool(q_sig.size > 0 and m_sig.size > 0 and float(np.linalg.norm(q_sig)) > 1e-12 and float(np.linalg.norm(m_sig)) > 1e-12),
        "free_chain_or_resolution_similarity": bool(q_free.size > 0 and m_free.size > 0 and float(np.linalg.norm(q_free)) > 1e-12 and float(np.linalg.norm(m_free)) > 1e-12),
        "persistent_homology_similarity": bool(q_ph.size > 0 and m_ph.size > 0 and float(np.linalg.norm(q_ph)) > 1e-12 and float(np.linalg.norm(m_ph)) > 1e-12),
        "rank_invariant_similarity": bool(q_rank.size > 0 and m_rank.size > 0 and float(np.linalg.norm(q_rank)) > 1e-12 and float(np.linalg.norm(m_rank)) > 1e-12),
        "commutative_algebra_similarity": bool(q_ca.size > 0 and m_ca.size > 0 and float(np.linalg.norm(q_ca)) > 1e-12 and float(np.linalg.norm(m_ca)) > 1e-12),
        "chain_map_score": bool(probability_chain_map_certified and probability_morphism_certified and probability_map_rate > 0.0),
    }
    sig_sim = _cosine_similarity(q_sig, m_sig) if component_availability["signature_cosine"] else 0.0
    free_sim = _cosine_similarity(q_free, m_free) if component_availability["free_chain_or_resolution_similarity"] else 0.0
    ph_sim = _cosine_similarity(q_ph, m_ph) if component_availability["persistent_homology_similarity"] else 0.0
    rank_sim = _cosine_similarity(q_rank, m_rank) if component_availability["rank_invariant_similarity"] else 0.0
    ca_sim = _cosine_similarity(q_ca, m_ca) if component_availability["commutative_algebra_similarity"] else 0.0
    required_component_keys = [
        "free_chain_or_resolution_similarity",
        "persistent_homology_similarity",
        "rank_invariant_similarity",
        "chain_map_score",
    ]
    required_components_available = bool(all(component_availability[key] for key in required_component_keys))
    derived_components = {
        "signature_cosine": float(sig_sim),
        "free_chain_or_resolution_similarity": float(free_sim),
        "free_resolution_similarity": float(free_sim),
        "persistent_homology_similarity": float(ph_sim),
        "rank_invariant_similarity": float(rank_sim),
        "commutative_algebra_similarity": float(ca_sim),
        "chain_map_score": float(chain_map_score),
    }
    if required_components_available:
        required_values = {key: derived_components[key] for key in required_component_keys}
        derived_algebraic = min(required_values.values())
        derived_clamped_by = min(required_values, key=required_values.get)
    else:
        derived_algebraic = 0.0
        derived_clamped_by = "missing_required_component"
    high_coarse_low_resolution = bool(sig_sim >= 0.95 and free_sim <= 1e-12)
    return {
        "retrieval_score": float(row.get("retrieval_score", 0.0)),
        "base_retrieval_score": float(row.get("base_retrieval_score", 0.0)),
        "persistence_landscape_score_contribution": float(row.get("persistence_landscape_score_contribution", 0.0)),
        "persistence_vector_score_contribution": float(row.get("persistence_vector_score_contribution", 0.0)),
        "probability_simplicial_map_score_contribution": float(row.get("probability_simplicial_map_score_contribution", 0.0)),
        "probability_simplicial_map_similarity": float(row.get("probability_simplicial_map_similarity", 0.0)),
        "probability_simplicial_map_preservation_rate": float(row.get("probability_simplicial_map_preservation_rate", 0.0)),
        "probability_simplicial_map_available": float(1.0 if row.get("probability_simplicial_map_available") else 0.0),
        "probability_simplicial_map_source": str(row.get("probability_simplicial_map_source", "none")),
        "probability_simplicial_map_scoring_policy": str(row.get("probability_simplicial_map_scoring_policy", "positive score only for certified retrieval-side probability simplex-tree maps")),
        "probability_simplicial_map_vertex_assignment_count": float(row.get("probability_simplicial_map_vertex_assignment_count", 0.0) or 0.0),
        "probability_simplicial_map_checked_simplices": float(row.get("probability_simplicial_map_checked_simplices", row.get("probability_simplicial_map_checked", 0.0)) or 0.0),
        "probability_simplicial_map_preserved_simplices": float(row.get("probability_simplicial_map_preserved_simplices", 0.0) or 0.0),
        "probability_simplicial_map_edge_preservation_rate": float(row.get("probability_simplicial_map_edge_preservation_rate", 0.0) or 0.0),
        "probability_simplicial_map_two_simplex_preservation_rate": float(row.get("probability_simplicial_map_two_simplex_preservation_rate", 0.0) or 0.0),
        "probability_simplicial_map_chain_map_certified": float(1.0 if row.get("probability_simplicial_map_chain_map_certified") else 0.0),
        "probability_simplicial_map_persistence_morphism_certified": float(1.0 if row.get("probability_simplicial_map_persistence_morphism_certified") else 0.0),
        "retrieval_score_components": row.get("retrieval_score_components", {}) if isinstance(row.get("retrieval_score_components"), dict) else {},
        "retrieval_weights": retrieval_weights,
        "embedding_similarity": float(row.get("embedding_similarity", 0.0)),
        "signature_similarity": float(row.get("signature_similarity", 0.0)),
        "derived_signature_similarity": float(sig_sim),
        "chain_presentation_similarity": float(free_sim),
        "free_resolution_similarity": float(free_sim),
        "persistent_homology_similarity": float(ph_sim),
        "rank_invariant_similarity": float(rank_sim),
        "commutative_algebra_similarity": float(ca_sim),
        "chain_map_score": float(chain_map_score),
        "chain_map_score_available": float(1.0 if component_availability["chain_map_score"] else 0.0),
        "persistence_landscape_vector_available": float(1.0 if landscape_report.get("available") else 0.0),
        "persistence_landscape_cosine": float(landscape_report.get("cosine", 0.0)) if landscape_report.get("available") else 0.0,
        "persistence_landscape_l2_similarity": float(landscape_report.get("l2_similarity", 0.0)) if landscape_report.get("available") else 0.0,
        "persistence_landscape_l2_distance": float(landscape_report.get("l2_distance", 0.0)) if landscape_report.get("available") else 0.0,
        "persistence_landscape_correlation": float(landscape_report.get("correlation", 0.0)) if landscape_report.get("available") else 0.0,
        "persistence_landscape_overlap_dim": float(landscape_report.get("overlap_dim", 0) or 0),
        "persistence_landscape_vector_similarity": landscape_report,
        "persistence_vector_representation_similarity": vector_report,
        "persistence_vector_aggregate_similarity": float(vector_report.get("aggregate_similarity", 0.0)) if vector_report.get("available") else 0.0,
        "persistence_vector_component_count": float(vector_report.get("component_count", 0) or 0),
        "persistence_vector_methods": str(",".join(vector_report.get("available_methods", []))) if vector_report.get("available") else "",
        "persistence_vector_comparison_space": str(vector_report.get("comparison_space", "")) if vector_report.get("available") else "",
        "persistence_vector_differentiable_note": str(vector_report.get("differentiable_comparison_note", "")) if vector_report.get("available") else "",
        "persistence_vector_components": _persistence_vector_component_rows(vector_report),
        "persistence_vector_component_summary": _persistence_vector_component_label(_persistence_vector_component_rows(vector_report)),
        "derived_algebraic_similarity": float(max(0.0, min(1.0, derived_algebraic))),
        "derived_algebraic_components_available": float(1.0 if required_components_available else 0.0),
        "derived_algebraic_policy": "conservative_minimum(free_chain_or_resolution_similarity, persistent_homology_similarity, rank_invariant_similarity, chain_map_score) when all required components have nonzero evidence; coarse signature cosine is displayed separately and is not part of this score; otherwise 0.0",
        "derived_algebraic_required_components": required_component_keys,
        "derived_algebraic_components": derived_components,
        "derived_algebraic_component_availability": component_availability,
        "derived_algebraic_clamped_by": derived_clamped_by,
        "coarse_signature_cosine_not_derived_similarity": True,
        "high_coarse_signature_low_resolution_warning": high_coarse_low_resolution,
    }


def _derived_invariant_comparison(query_topology: dict[str, object], memory_topology: dict[str, object], sim: dict[str, float] | None = None, tol: float = 1e-8) -> dict[str, object]:
    q_sig = query_topology.get("derived_equivalence_signature", {}) if isinstance(query_topology, dict) else {}
    m_sig = memory_topology.get("derived_equivalence_signature", {}) if isinstance(memory_topology, dict) else {}
    q_betti = [float(v) for v in q_sig.get("betti_vector", [])] if isinstance(q_sig.get("betti_vector", []), list) else []
    m_betti = [float(v) for v in m_sig.get("betti_vector", [])] if isinstance(m_sig.get("betti_vector", []), list) else []
    q_signature = _signature_numeric_vector(query_topology, length=48)
    m_signature = _signature_numeric_vector(memory_topology, length=48)
    q_free = _free_rank_vector(query_topology, length=32)
    m_free = _free_rank_vector(memory_topology, length=32)
    q_ph = _persistence_numeric_vector(query_topology, length=32)
    m_ph = _persistence_numeric_vector(memory_topology, length=32)
    q_ca = _commutative_algebra_numeric_vector(query_topology, length=32)
    m_ca = _commutative_algebra_numeric_vector(memory_topology, length=32)
    q_landscape = _persistence_landscape_numeric_vector(query_topology)
    m_landscape = _persistence_landscape_numeric_vector(memory_topology)
    if q_landscape.size or m_landscape.size:
        n_landscape = max(q_landscape.size, m_landscape.size)
        q_landscape_padded = np.zeros(n_landscape, dtype=float)
        m_landscape_padded = np.zeros(n_landscape, dtype=float)
        q_landscape_padded[: q_landscape.size] = q_landscape
        m_landscape_padded[: m_landscape.size] = m_landscape
    else:
        q_landscape_padded = np.zeros(0, dtype=float)
        m_landscape_padded = np.zeros(0, dtype=float)

    def close_vec(a: np.ndarray, b: np.ndarray) -> bool:
        return bool(a.shape == b.shape and np.allclose(a, b, atol=tol, rtol=0.0))

    betti_match = q_betti == m_betti
    signature_match = close_vec(q_signature, m_signature)
    free_rank_match = close_vec(q_free, m_free)
    persistence_match = close_vec(q_ph, m_ph)
    commutative_algebra_match = close_vec(q_ca, m_ca)
    landscape_vector_match = close_vec(q_landscape_padded, m_landscape_padded) if q_landscape_padded.size or m_landscape_padded.size else False
    finite_match = bool(betti_match and signature_match and free_rank_match and persistence_match and commutative_algebra_match)
    q_real_resolution = _real_free_resolution_claim_summary(query_topology)
    m_real_resolution = _real_free_resolution_claim_summary(memory_topology)
    real_resolution_comparison = _compare_real_free_resolution_summaries(q_real_resolution, m_real_resolution)
    real_resolution_pair_available = bool(real_resolution_comparison.get("available"))
    certified_cas_evidence_match = bool(real_resolution_comparison.get("certified_cas_evidence_match"))
    coarse_signature_similarity = float((sim or {}).get("derived_signature_similarity", 0.0) or 0.0)
    chain_resolution_similarity = float((sim or {}).get("chain_presentation_similarity", 0.0) or 0.0)
    high_coarse_low_resolution = bool(coarse_signature_similarity >= 0.75 and chain_resolution_similarity <= 1e-12)
    if real_resolution_pair_available and certified_cas_evidence_match:
        resolution_interpretation = "Both sides expose CAS-certified multigraded real free resolutions with matching Betti shifts, differential summaries, and Fitting/minor ideals; matching Buchsbaum-Eisenbud/BEMultipliers entries are diagnostic-only sidecars, and the resolution claim rests on the certified CAS resolution."
    elif real_resolution_pair_available:
        resolution_interpretation = "Both sides expose CAS-certified multigraded real free resolutions, but their CAS artifacts do not match; diagnostic-only BEMultipliers sidecars cannot substitute for a matching certified resolution, Fitting/minor ideals, or an explicit chain-map isomorphism."
    elif high_coarse_low_resolution:
        resolution_interpretation = "Coarse derived-signature similarity is high while chain/free-resolution similarity is zero; this is reported as a coarse invariant collision, not a derived-equivalence claim. A real derived/free-resolution comparison requires CAS-certified multigraded resolutions for both sides."
    else:
        resolution_interpretation = "Derived/free-resolution comparison is restricted to finite invariants unless both sides expose CAS-certified multigraded real free resolutions."
    derived_claim = "not_certified"
    if finite_match and certified_cas_evidence_match:
        derived_claim = "cas_certified_matching_real_resolution_witness"
    elif finite_match:
        derived_claim = "compatible_finite_invariant_witness"
    return {
        "comparison_kind": "finite_F2xy_persistence_module_and_chain_presentation_invariant_comparison",
        "field": "F2",
        "module_category": "finite grid presentation over F2[x_level,x_radius]",
        "derived_category": "bounded finite-chain invariant comparison; no derived equivalence or free-resolution claim without a CAS certificate",
        "tolerance": float(tol),
        "derived_algebraic_similarity": float((sim or {}).get("derived_algebraic_similarity", 0.0)),
        "derived_signature_cosine": float((sim or {}).get("derived_signature_similarity", 0.0)),
        "derived_equivalence_claim": derived_claim,
        "geometric_realization_required": "a filtered simplex-tree map is required; module/chain-presentation diagnostic similarity alone does not construct a unique simplicial map",
        "induced_map_direction": "filtered_simplicial_map -> chain_map -> F2[x,y]-persistence_module_morphism -> derived_category_morphism",
        "finite_invariants_match": finite_match,
        "betti_vector_match": bool(betti_match),
        "derived_signature_vector_match": bool(signature_match),
        "free_chain_rank_vector_match": bool(free_rank_match),
        "persistence_summary_vector_match": bool(persistence_match),
        "commutative_algebra_vector_match": bool(commutative_algebra_match),
        "persistence_landscape_vector_match": bool(landscape_vector_match),
        "persistence_landscape_vector_available": bool(q_landscape_padded.size > 0 and m_landscape_padded.size > 0),
        "real_free_resolution_certified": real_resolution_pair_available,
        "certified_cas_evidence_match": certified_cas_evidence_match,
        "certified_cas_evidence_similarity": float(real_resolution_comparison.get("certified_cas_evidence_similarity", 0.0) or 0.0),
        "real_free_resolution_comparison": real_resolution_comparison,
        "free_resolution_similarity_interpretation": resolution_interpretation,
        "high_coarse_signature_low_resolution_warning": high_coarse_low_resolution,
        "query_betti_vector": q_betti,
        "memory_betti_vector": m_betti,
        "signature_l2_distance": float(np.linalg.norm(q_signature - m_signature)),
        "free_rank_l2_distance": float(np.linalg.norm(q_free - m_free)),
        "persistence_l2_distance": float(np.linalg.norm(q_ph - m_ph)),
        "commutative_algebra_l2_distance": float(np.linalg.norm(q_ca - m_ca)),
        "persistence_landscape_l2_distance": float(np.linalg.norm(q_landscape_padded - m_landscape_padded)) if q_landscape_padded.size or m_landscape_padded.size else 0.0,
    }


def _summary_float(summary: object, key: str) -> float | None:
    if not isinstance(summary, dict):
        return None
    value = summary.get(key)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _retrieval_probability_dim_counts(tree_report: dict[str, object], dim: int) -> dict[str, int]:
    counts = tree_report.get("dimension_counts", {}) if isinstance(tree_report, dict) else {}
    bucket = counts.get(f"dim_{dim}", {}) if isinstance(counts, dict) else {}
    if not isinstance(bucket, dict):
        bucket = {}
    return {
        "checked": int(bucket.get("checked", 0) or 0),
        "preserved": int(bucket.get("preserved", 0) or 0),
        "missing_codomain": int(bucket.get("missing_codomain", 0) or 0),
    }


def _retrieval_probability_dim_rows(tree_report: dict[str, object], dim: int, *, preserved: bool) -> list[dict[str, object]]:
    rows = tree_report.get("rows", []) if isinstance(tree_report, dict) else []
    if not isinstance(rows, list):
        return []
    out: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict) or int(row.get("dimension", -1) or -1) != dim:
            continue
        is_preserved = bool(row.get("preserved_in_simplex_tree"))
        if is_preserved != preserved:
            continue
        out.append(
            {
                "query_simplex": list(row.get("domain_simplex", [])) if isinstance(row.get("domain_simplex"), list) else [],
                "memory_simplex": list(row.get("image_simplex", [])) if isinstance(row.get("image_simplex"), list) else [],
                "domain_filtration": row.get("domain_filtration"),
                "codomain_filtration": row.get("codomain_filtration"),
                "signed_filtration_distortion": row.get("signed_filtration_distortion"),
                "failure_reason": row.get("failure_reason"),
            }
        )
    return out


def _retrieval_probability_simplicial_map_unavailable(reason: str, report: dict[str, object] | None = None) -> dict[str, object]:
    report = report if isinstance(report, dict) else {}
    empty = _empty_simplicial_map_report([], [], reason)
    empty.update(
        {
            "map_source": str(report.get("map_source", "none")),
            "map_certificate_source": "unavailable_retrieval_probability_simplicial_map_certificate",
            "retrieval_probability_certificate_available": False,
            "simplex_tree_map_checked": int(report.get("simplex_tree_map_checked", 0) or 0),
            "simplex_tree_map_preserved": int(report.get("simplex_tree_map_preserved", 0) or 0),
            "simplex_tree_map_preservation_rate": float(report.get("simplex_tree_map_preservation_rate", 0.0) or 0.0),
            "safe_to_render_as_simplicial_map": False,
            "safe_to_render_as_chain_map": False,
            "safe_to_render_as_persistence_module_morphism": False,
            "map_render_claim": "unavailable_probability_correspondence_certificate",
            "map_claim_status": "unavailable_probability_correspondence_certificate",
            "map_claim_label": "certificate unavailable; no simplicial map rendered",
            "map_claim_failure_reason": reason,
            "no_proxy_or_fallback": True,
            "simplicial_map_certificate": {
                "source": "unavailable_retrieval_probability_simplicial_map_certificate",
                "reason": reason,
                "no_proxy_or_fallback": True,
            },
            "interpretation": "No analogical map is rendered because the retrieval row did not carry a certified model-probability simplex-tree map sidecar.",
        }
    )
    return empty


def _retrieval_probability_simplicial_map_report(row: dict[str, object]) -> dict[str, object]:
    report = row.get("probability_simplicial_map") if isinstance(row, dict) else None
    if not isinstance(report, dict):
        return _retrieval_probability_simplicial_map_unavailable("missing_retrieval_probability_simplicial_map_certificate")
    if report.get("map_source") != "model_probability_jensen_shannon_assignment":
        return _retrieval_probability_simplicial_map_unavailable("retrieval_certificate_not_model_probability_jensen_shannon_assignment", report)
    tree_report = report.get("simplex_tree_map") if isinstance(report.get("simplex_tree_map"), dict) else None
    if not isinstance(tree_report, dict):
        return _retrieval_probability_simplicial_map_unavailable("missing_retrieval_simplex_tree_map_certificate", report)
    vertex_map = [entry for entry in report.get("vertex_map", []) if isinstance(entry, dict)] if isinstance(report.get("vertex_map"), list) else []
    edge = _retrieval_probability_dim_counts(tree_report, 1)
    face = _retrieval_probability_dim_counts(tree_report, 2)
    edge_rate = float(edge["preserved"] / edge["checked"]) if edge["checked"] else 0.0
    face_rate = float(face["preserved"] / face["checked"]) if face["checked"] else 0.0
    preserved_edge_pairs = _retrieval_probability_dim_rows(tree_report, 1, preserved=True)
    failed_edge_pairs = _retrieval_probability_dim_rows(tree_report, 1, preserved=False)
    preserved_faces = _retrieval_probability_dim_rows(tree_report, 2, preserved=True)
    failed_faces = _retrieval_probability_dim_rows(tree_report, 2, preserved=False)
    preserved_query_vertices = sorted(
        {
            str(vertex)
            for pair in preserved_edge_pairs
            for vertex in (pair.get("query_simplex", []) if isinstance(pair.get("query_simplex"), list) else [])
        }
    )
    preserved_memory_vertices = sorted(
        {
            str(vertex)
            for pair in preserved_edge_pairs
            for vertex in (pair.get("memory_simplex", []) if isinstance(pair.get("memory_simplex"), list) else [])
        }
    )
    js_summary = report.get("jensen_shannon_distance_summary", {})
    cost_summary = report.get("assignment_cost_summary", {})
    probability_vector_evidence = report.get("probability_vector_evidence", {}) if isinstance(report.get("probability_vector_evidence"), dict) else {}
    distortion_summary = tree_report.get("positive_filtration_distortion_summary", {}) if isinstance(tree_report, dict) else {}
    checked = int(report.get("simplex_tree_map_checked", tree_report.get("checked_simplices", 0)) or 0)
    preserved = int(report.get("simplex_tree_map_preserved", tree_report.get("preserved_simplices", 0)) or 0)
    rate = float(report.get("simplex_tree_map_preservation_rate", tree_report.get("preservation_rate", 0.0)) or 0.0)
    is_map = bool(report.get("available") and report.get("is_filtered_simplicial_map") and checked > 0 and checked == preserved and rate >= 0.999)
    chain_report = report.get("chain_map_diagnostics") if isinstance(report.get("chain_map_diagnostics"), dict) else _unavailable_chain_map_diagnostics("missing_chain_map_diagnostics")
    morphism_report = report.get("persistence_module_morphism_diagnostics") if isinstance(report.get("persistence_module_morphism_diagnostics"), dict) else _persistence_module_morphism_diagnostics(_unavailable_chain_map_diagnostics("missing_chain_map_diagnostics"))
    map_claim = "certified_filtered_simplicial_map" if is_map else "probability_correspondence_not_a_simplicial_map"
    map_label = "certified filtered simplicial map; chain/persistence morphism diagnostics may be used" if is_map else "probability correspondence only; no simplicial/chain/persistence morphism is asserted"
    claim_reason = None if is_map else ("simplex_tree_map_not_fully_preserved" if checked else "simplex_tree_map_unchecked")
    return {
        "vertex_map": vertex_map,
        "displayed_domain_vertices": int(report.get("displayed_domain_vertices", 0) or 0),
        "displayed_codomain_vertices": int(report.get("displayed_codomain_vertices", 0) or 0),
        "map_source": "model_probability_jensen_shannon_assignment",
        "map_certificate_source": "retrieval_probability_simplicial_map_certificate",
        "retrieval_probability_certificate_available": True,
        "probability_alignment": report.get("probability_alignment"),
        "probability_vector_evidence": probability_vector_evidence,
        "jensen_shannon_distance_summary": js_summary,
        "assignment_cost_summary": cost_summary,
        "jensen_shannon_distance_mean": _summary_float(js_summary, "mean"),
        "jensen_shannon_distance_max": _summary_float(js_summary, "max"),
        "assignment_cost_mean": _summary_float(cost_summary, "mean"),
        "assignment_cost_max": _summary_float(cost_summary, "max"),
        "filtration_distortion_summary": distortion_summary,
        "max_positive_filtration_distortion": _summary_float(distortion_summary, "max"),
        "checked_edges": int(edge["checked"]),
        "preserved_edges": int(edge["preserved"]),
        "edge_preservation_rate": edge_rate,
        "preserved_edge_pairs": preserved_edge_pairs,
        "failed_edge_pairs": failed_edge_pairs,
        "preserved_edge_query_vertices": preserved_query_vertices,
        "preserved_edge_memory_vertices": preserved_memory_vertices,
        "checked_two_simplices": int(face["checked"]),
        "preserved_two_simplices": int(face["preserved"]),
        "two_simplex_preservation_rate": face_rate,
        "preserved_two_simplex_faces": preserved_faces,
        "failed_two_simplex_faces": failed_faces,
        "simplex_tree_map": tree_report,
        "simplex_tree_map_checked": checked,
        "simplex_tree_map_preserved": preserved,
        "simplex_tree_map_preservation_rate": rate,
        "chain_map_diagnostics": chain_report,
        "persistence_module_morphism_diagnostics": morphism_report,
        "is_filtered_simplicial_map": is_map,
        "is_simplicial_on_displayed_skeleton": is_map,
        "safe_to_render_as_simplicial_map": is_map,
        "safe_to_render_as_chain_map": bool(is_map and chain_report.get("safe_to_use_as_persistence_module_morphism")),
        "safe_to_render_as_persistence_module_morphism": bool(is_map and morphism_report.get("available")),
        "map_render_claim": map_claim,
        "map_claim_status": map_claim,
        "map_claim_label": map_label,
        "map_claim_failure_reason": claim_reason,
        "no_proxy_or_fallback": True,
        "simplicial_map_certificate": {
            "source": "retrieval_probability_simplicial_map_certificate",
            "rule": "stored retrieval certificate: model-probability Jensen-Shannon vertex assignment must extend to a filtration-preserving simplex-tree map",
            "domain_vertices_mapped": len(vertex_map),
            "domain_vertices_total": int(report.get("displayed_domain_vertices", 0) or 0),
            "edge_failures": int(edge["checked"] - edge["preserved"]),
            "two_simplex_failures": int(face["checked"] - face["preserved"]),
            "max_positive_filtration_distortion": _summary_float(distortion_summary, "max"),
            "no_proxy_or_fallback": True,
        },
        "interpretation": report.get("interpretation", "Retrieval-side certificate from the model-probability simplex-tree map sidecar."),
    }


def _simplicial_map_between_complexes(query_obj: dict[str, object], memory_obj: dict[str, object], max_vertices: int = 54) -> dict[str, object]:
    query_vertices = _complex_vertex_records(query_obj)[:max_vertices]
    memory_vertices = _complex_vertex_records(memory_obj)[:max_vertices]
    if not query_vertices or not memory_vertices:
        return _empty_simplicial_map_report(query_vertices, memory_vertices, "empty_domain_or_codomain")

    q_probs = [_probability_feature_vector(row) for row in query_vertices]
    m_probs = [_probability_feature_vector(row) for row in memory_vertices]
    if not any(vec is not None for vec in q_probs) or not any(vec is not None for vec in m_probs):
        report = _empty_simplicial_map_report(query_vertices, memory_vertices, "unavailable_no_model_probability_vectors")
        report["map_source"] = "none"
        report["map_requires"] = "model probability vectors on 0-simplices"
        return report

    vertex_map = _probability_induced_vertex_assignment(query_vertices, memory_vertices, q_probs, m_probs)
    mapping = {row["query_vertex"]: row["memory_vertex"] for row in vertex_map}
    q_filtration = _simplex_filtration_lookup(query_obj)
    m_filtration = _simplex_filtration_lookup(memory_obj)
    edge_report = _check_filtered_simplicial_dimension(
        query_obj,
        memory_obj,
        mapping,
        q_filtration,
        m_filtration,
        dimension=1,
    )
    face_report = _check_filtered_simplicial_dimension(
        query_obj,
        memory_obj,
        mapping,
        q_filtration,
        m_filtration,
        dimension=2,
    )
    checked_edges = int(edge_report["checked"])
    preserved_edges = int(edge_report["preserved"])
    checked_faces = int(face_report["checked"])
    preserved_faces = int(face_report["preserved"])
    edge_rate = preserved_edges / checked_edges if checked_edges else 1.0
    face_rate = preserved_faces / checked_faces if checked_faces else 1.0
    is_map = bool(edge_rate >= 0.999 and face_rate >= 0.999 and len(mapping) == len(query_vertices))
    js_values = [float(row.get("jensen_shannon_distance", 0.0)) for row in vertex_map if row.get("jensen_shannon_distance") is not None]
    assignment_costs = [float(row.get("assignment_cost", 0.0)) for row in vertex_map if row.get("assignment_cost") is not None]
    edge_distortion = _numeric_summary(edge_report["positive_distortions"])
    face_distortion = _numeric_summary(face_report["positive_distortions"])
    simplex_tree_report = _simplex_tree_map_report(query_obj, memory_obj, mapping, q_filtration, m_filtration)
    all_distortion_values = list(edge_report["positive_distortions"]) + list(face_report["positive_distortions"])
    all_distortion = _numeric_summary(all_distortion_values)
    chain_map_report = _chain_map_diagnostics(simplex_tree_report, is_map, mapping)
    persistence_morphism = _persistence_module_morphism_diagnostics(chain_map_report)
    is_identity_self_map = bool(
        len(query_vertices) == len(memory_vertices)
        and len(vertex_map) == len(query_vertices)
        and all(str(row.get("query_vertex")) == str(row.get("memory_vertex")) for row in vertex_map)
        and all(float(row.get("jensen_shannon_distance", 1.0) or 0.0) <= 1e-12 for row in vertex_map)
    )
    return {
        "vertex_map": vertex_map,
        "displayed_domain_vertices": len(query_vertices),
        "displayed_codomain_vertices": len(memory_vertices),
        "map_source": "model_probability_jensen_shannon_assignment",
        "probability_alignment": "zero_pad_to_common_token_index_feature_space_then_renormalize",
        "is_identity_self_map": is_identity_self_map,
        "jensen_shannon_distance_summary": _numeric_summary(js_values),
        "assignment_cost_summary": _numeric_summary(assignment_costs),
        "jensen_shannon_distance_mean": _numeric_summary(js_values).get("mean"),
        "jensen_shannon_distance_max": _numeric_summary(js_values).get("max"),
        "assignment_cost_mean": _numeric_summary(assignment_costs).get("mean"),
        "assignment_cost_max": _numeric_summary(assignment_costs).get("max"),
        "edge_filtration_distortion_summary": edge_distortion,
        "two_simplex_filtration_distortion_summary": face_distortion,
        "filtration_distortion_summary": all_distortion,
        "max_positive_filtration_distortion": all_distortion.get("max"),
        "checked_edges": checked_edges,
        "preserved_edges": preserved_edges,
        "edge_preservation_rate": float(edge_rate),
        "preserved_edge_pairs": edge_report["preserved_pairs"],
        "failed_edge_pairs": edge_report["failed_pairs"],
        "preserved_edge_query_vertices": sorted(edge_report["preserved_query_vertices"]),
        "preserved_edge_memory_vertices": sorted(edge_report["preserved_memory_vertices"]),
        "checked_two_simplices": checked_faces,
        "preserved_two_simplices": preserved_faces,
        "two_simplex_preservation_rate": float(face_rate),
        "preserved_two_simplex_faces": face_report["preserved_pairs"],
        "failed_two_simplex_faces": face_report["failed_pairs"],
        "simplex_tree_map": simplex_tree_report,
        "simplex_tree_map_checked": int(simplex_tree_report.get("checked_simplices", 0)),
        "simplex_tree_map_preserved": int(simplex_tree_report.get("preserved_simplices", 0)),
        "simplex_tree_map_preservation_rate": float(simplex_tree_report.get("preservation_rate", 0.0)),
        "chain_map_diagnostics": chain_map_report,
        "persistence_module_morphism_diagnostics": persistence_morphism,
        "is_filtered_simplicial_map": is_map,
        "is_simplicial_on_displayed_skeleton": is_map,
        "simplicial_map_certificate": {
            "source": "finite_filtered_complex_check",
            "rule": "for every displayed simplex sigma, filtration_L(f(sigma)) <= filtration_K(sigma)",
            "domain_vertices_mapped": len(mapping),
            "domain_vertices_total": len(query_vertices),
            "edge_failures": checked_edges - preserved_edges,
            "two_simplex_failures": checked_faces - preserved_faces,
            "max_positive_filtration_distortion": all_distortion.get("max"),
            "identity_self_map": is_identity_self_map,
        },
    }


def _analogical_realization_certificate(
    sim: dict[str, float],
    sim_map: dict[str, object],
    derived_comparison: dict[str, object],
) -> dict[str, object]:
    derived_ok = bool(derived_comparison.get("finite_invariants_match"))
    tree_ok = bool(sim_map.get("is_filtered_simplicial_map"))
    simplex_tree_rate = float(sim_map.get("simplex_tree_map_preservation_rate", 0.0) or 0.0)
    real_resolution_comparison = derived_comparison.get("real_free_resolution_comparison") if isinstance(derived_comparison.get("real_free_resolution_comparison"), dict) else {}
    real_resolution_certified = bool(real_resolution_comparison.get("safe_for_derived_category_claims"))
    claim = "not_certified"
    if derived_ok and tree_ok and real_resolution_certified:
        claim = "cas_certified_derived_geometric_realization"
    elif derived_ok and tree_ok:
        claim = "finite_invariant_filtered_correspondence"
    return {
        "claim": claim,
        "derived_algebraic_similarity": float(sim.get("derived_algebraic_similarity", 0.0)),
        "coarse_signature_cosine": float(sim.get("derived_signature_similarity", 0.0)),
        "requires": [
            "compatible finite F2[x,y] persistence module invariants",
            "compatible chain-presentation diagnostics plus a real free-resolution certificate when available",
            "filtered simplex-tree map induced by model probabilities",
        ],
        "module_to_geometry_note": "A real derived/free-resolution claim requires a CAS-certified resolution or chain map; the displayed analogy is geometrically realized only when the probability-induced vertex map extends to a filtration-preserving simplex-tree map.",
        "chain_map_note": (
            "The certified filtration-preserving simplicial map induces a chain map and hence a morphism of the associated F2[x,y] persistence modules."
            if tree_ok
            else "No chain map or persistence-module morphism is asserted because the probability correspondence did not certify a filtration-preserving simplex-tree map."
        ),
        "finite_invariants_match": derived_ok,
        "real_free_resolution_certified": real_resolution_certified,
        "real_free_resolution_pair_available": bool(real_resolution_comparison.get("available")),
        "certified_cas_evidence_match": bool(real_resolution_comparison.get("certified_cas_evidence_match")),
        "filtered_simplex_tree_map": tree_ok,
        "simplex_tree_map_preservation_rate": simplex_tree_rate,
    }




def _unavailable_chain_map_diagnostics(reason: str) -> dict[str, object]:
    return {
        "available": False,
        "field": "F2",
        "source": "unavailable_no_filtered_simplicial_map",
        "reason": reason,
        "chain_map_certified": False,
        "boundary_commutation_certified": False,
        "filtration_nonincreasing": False,
        "safe_to_use_as_persistence_module_morphism": False,
    }


def _chain_map_diagnostics(simplex_tree_report: Mapping[str, object], is_filtered_map: bool, mapping: Mapping[str, str]) -> dict[str, object]:
    checked = int(simplex_tree_report.get("checked_simplices", 0) or 0)
    preserved = int(simplex_tree_report.get("preserved_simplices", 0) or 0)
    rate = float(simplex_tree_report.get("preservation_rate", 0.0) or 0.0)
    available = bool(is_filtered_map and checked > 0 and preserved == checked and rate >= 0.999)
    if not available:
        return {
            **_unavailable_chain_map_diagnostics("filtered_simplicial_map_failed_or_unchecked"),
            "checked_simplices": checked,
            "preserved_simplices": preserved,
            "simplex_tree_map_preservation_rate": rate,
            "domain_vertices_mapped": len(mapping),
            "simplex_tree_dimension_counts": simplex_tree_report.get("dimension_counts", {}),
        }
    return {
        "available": True,
        "field": "F2",
        "source": "finite_filtered_simplicial_map_linear_extension",
        "chain_map_certified": True,
        "boundary_commutation_certified": True,
        "filtration_nonincreasing": True,
        "domain_vertices_mapped": len(mapping),
        "checked_simplices": checked,
        "preserved_simplices": preserved,
        "simplex_tree_map_preservation_rate": rate,
        "simplex_tree_dimension_counts": simplex_tree_report.get("dimension_counts", {}),
        "chain_groups": simplex_tree_report.get("dimension_counts", {}),
        "interpretation": "The model-probability vertex assignment extends to a filtered simplicial map on the displayed simplex trees, so its F2-linear extension is a chain map commuting with boundary.",
        "safe_to_use_as_persistence_module_morphism": True,
    }


def _persistence_module_morphism_diagnostics(chain_map: Mapping[str, object]) -> dict[str, object]:
    available = bool(chain_map.get("available") and chain_map.get("safe_to_use_as_persistence_module_morphism"))
    return {
        "available": available,
        "ring": "F2[x_level,x_radius]",
        "source": "filtered_chain_map" if available else "unavailable_no_filtered_chain_map",
        "morphism_certified": available,
        "free_resolution_required": False,
        "safe_for_derived_equivalence_claim": False,
        "reason": None if available else chain_map.get("reason", "chain map unavailable"),
        "interpretation": (
            "A filtration-preserving chain map induces a morphism of finite F2[x_level,x_radius] persistence modules; this is not a derived equivalence or free-resolution comparison."
            if available
            else "No persistence-module morphism is asserted because the filtered chain map was not certified."
        ),
    }


def _real_free_resolution_claim_summary(topology: Mapping[str, object]) -> dict[str, object]:
    reports: list[Mapping[str, object]] = []

    def visit(obj: object, depth: int = 0) -> None:
        if depth > 7:
            return
        if isinstance(obj, Mapping):
            if obj.get("schema_version") == "tropicalgt.real_free_resolution.v1" or "safe_to_render_as_multigraded_free_resolution" in obj:
                reports.append(obj)
            for value in obj.values():
                visit(value, depth + 1)
        elif isinstance(obj, list):
            for value in obj[:32]:
                visit(value, depth + 1)

    visit(topology)
    for report in reports:
        if (
            report.get("available") is True
            and report.get("exactness_certified") is True
            and report.get("multigraded_free_resolution_certified") is True
            and report.get("safe_to_render_as_multigraded_free_resolution") is True
        ):
            artifact_signature = _certified_cas_resolution_signature(report)
            artifact_hash = _stable_artifact_hash(artifact_signature)
            artifacts = report.get("cas_artifacts") if isinstance(report.get("cas_artifacts"), Mapping) else {}
            be = artifacts.get("buchsbaum_eisenbud_diagnostics") if isinstance(artifacts.get("buchsbaum_eisenbud_diagnostics"), Mapping) else {}
            summary = report.get("free_resolution_summary") if isinstance(report.get("free_resolution_summary"), Mapping) else {}
            return {
                "available": True,
                "backend": report.get("backend"),
                "ring": report.get("coefficient_ring"),
                "input_sha256": report.get("input_sha256"),
                "status": report.get("status"),
                "safe_to_render_as_multigraded_free_resolution": True,
                "minimality_certified": bool(report.get("minimality_certified")),
                "artifact_signature": artifact_signature,
                "artifact_hash": artifact_hash,
                "betti_by_homological_and_multidegree": summary.get("betti_by_homological_and_multidegree", {}),
                "fitting_ideals": artifacts.get("fitting_ideals", {}) if isinstance(artifacts.get("fitting_ideals"), Mapping) else {},
                "minors": artifacts.get("minors", {}) if isinstance(artifacts.get("minors"), Mapping) else {},
                "bemultipliers_status": be.get("bemultipliers_status", "unreported"),
                "multiplier_output_available": bool(be.get("multiplier_output_available")),
                "safe_to_render_multiplier_output": bool(be.get("safe_to_render_multiplier_output")),
                "is_resolution_backend": bool(be.get("is_resolution_backend", False)),
                "safe_to_substitute_for_resolution": bool(be.get("safe_to_substitute_for_resolution", False)),
                "a_multiplier_1_shape": be.get("a_multiplier_1_shape", ""),
                "a_multiplier_1_matrix_hash": _stable_artifact_hash(be.get("a_multiplier_1_matrix", "")),
            }
    return {
        "available": False,
        "reason": "no CAS-certified multigraded real free resolution found in topology payload",
        "candidate_reports_seen": len(reports),
    }


def _certified_cas_resolution_signature(report: Mapping[str, object]) -> dict[str, object]:
    artifacts = report.get("cas_artifacts") if isinstance(report.get("cas_artifacts"), Mapping) else {}
    summary = report.get("free_resolution_summary") if isinstance(report.get("free_resolution_summary"), Mapping) else {}
    be = artifacts.get("buchsbaum_eisenbud_diagnostics") if isinstance(artifacts.get("buchsbaum_eisenbud_diagnostics"), Mapping) else {}
    differentials = artifacts.get("differentials") if isinstance(artifacts.get("differentials"), list) else []
    differential_summary = []
    for row in differentials[:24]:
        if not isinstance(row, Mapping):
            continue
        differential_summary.append(
            {
                "homological_degree": row.get("homological_degree"),
                "rows": row.get("rows"),
                "cols": row.get("cols"),
                "shape": row.get("shape"),
                "source_degrees": row.get("source_degrees"),
                "target_degrees": row.get("target_degrees"),
                "matrix_hash": _stable_artifact_hash(row.get("matrix_text", row.get("matrix_preview", ""))),
            }
        )
    return {
        "schema_version": report.get("schema_version"),
        "backend": report.get("backend"),
        "ring": report.get("coefficient_ring"),
        "input_sha256": report.get("input_sha256"),
        "minimality_certified": bool(report.get("minimality_certified")),
        "betti_by_homological_and_multidegree": summary.get("betti_by_homological_and_multidegree", {}),
        "free_modules": summary.get("free_modules", []),
        "differentials": differential_summary,
        "fitting_ideals": artifacts.get("fitting_ideals", {}) if isinstance(artifacts.get("fitting_ideals"), Mapping) else {},
        "minors": artifacts.get("minors", {}) if isinstance(artifacts.get("minors"), Mapping) else {},
        "bemultipliers": {
            "available": bool(be.get("available")),
            "multiplier_output_available": bool(be.get("multiplier_output_available")),
            "safe_to_render_multiplier_output": bool(be.get("safe_to_render_multiplier_output")),
            "is_resolution_backend": bool(be.get("is_resolution_backend", False)),
            "safe_to_substitute_for_resolution": bool(be.get("safe_to_substitute_for_resolution", False)),
            "bemultipliers_status": be.get("bemultipliers_status", "unreported"),
            "a_multiplier_1_shape": be.get("a_multiplier_1_shape", ""),
            "a_multiplier_1_matrix_hash": _stable_artifact_hash(be.get("a_multiplier_1_matrix", "")),
        },
    }


def _compare_real_free_resolution_summaries(query: Mapping[str, object], memory: Mapping[str, object]) -> dict[str, object]:
    component_explanations = {
        "ring": "Coefficient rings must match exactly before any derived/free-resolution comparison is admissible.",
        "input_sha256": "The CAS input presentation hash must match; different presentations require an explicit certified isomorphism, not a proxy comparison.",
        "artifact_hash": "The attached CAS artifact hash must match exactly for a stored-certificate identity claim.",
        "betti_by_multidegree": "Minimal multigraded Betti data must agree as exact CAS output.",
        "differential_summaries": "Certified differential shapes, source/target multidegrees, and matrix hashes must agree.",
        "fitting_ideals": "Fitting ideals are exact determinantal invariants from the CAS presentation and must agree when used as evidence.",
        "minors": "Determinantal minor ideals must agree as exact CAS output; they are not estimated from chain diagnostics.",
        "buchsbaum_eisenbud": "Buchsbaum-Eisenbud/BEMultipliers entries are diagnostic-only sidecars computed from a certified Macaulay2 chain complex; they are never a resolution backend and cannot substitute for Betti, differential, Fitting, or minor agreement.",
    }
    diagnostic_only_components = ["buchsbaum_eisenbud"]
    available = bool(query.get("available") and memory.get("available"))
    if not available:
        return {
            "available": False,
            "query": query,
            "memory": memory,
            "reason": "CAS-certified multigraded real free resolution unavailable for one or both sides",
            "unavailable_reasons": {"query": query.get("reason"), "memory": memory.get("reason")},
            "unavailable_explanation": "No derived/free-resolution comparison is made unless both sides expose CAS-certified multigraded real free resolutions; missing evidence is unavailable, not estimated or replaced by a fallback.",
            "component_explanations": component_explanations,
            "diagnostic_only_components": diagnostic_only_components,
            "mismatch_explanations": [],
            "comparison_policy": "no_proxy_no_fallback_exact_cas_components_only",
            "safe_for_derived_category_claims": False,
            "certified_cas_evidence_match": False,
            "certified_cas_evidence_similarity": 0.0,
        }
    q_sig = query.get("artifact_signature") if isinstance(query.get("artifact_signature"), Mapping) else {}
    m_sig = memory.get("artifact_signature") if isinstance(memory.get("artifact_signature"), Mapping) else {}
    components = {
        "ring": query.get("ring") == memory.get("ring"),
        "input_sha256": bool(query.get("input_sha256") and query.get("input_sha256") == memory.get("input_sha256")),
        "artifact_hash": bool(query.get("artifact_hash") and query.get("artifact_hash") == memory.get("artifact_hash")),
        "betti_by_multidegree": q_sig.get("betti_by_homological_and_multidegree", {}) == m_sig.get("betti_by_homological_and_multidegree", {}),
        "differential_summaries": q_sig.get("differentials", []) == m_sig.get("differentials", []),
        "fitting_ideals": q_sig.get("fitting_ideals", {}) == m_sig.get("fitting_ideals", {}),
        "minors": q_sig.get("minors", {}) == m_sig.get("minors", {}),
        "buchsbaum_eisenbud": q_sig.get("bemultipliers", {}) == m_sig.get("bemultipliers", {}),
    }
    matched = sum(1 for value in components.values() if value)
    total = max(len(components), 1)
    safe = bool(all(components.values()))
    mismatches = [key for key, value in components.items() if not value]
    mismatch_explanations = [
        {
            "component": key,
            "diagnostic_only": key in diagnostic_only_components,
            "explanation": component_explanations.get(key, "CAS evidence component mismatch."),
        }
        for key in mismatches
    ]
    return {
        "available": True,
        "query": query,
        "memory": memory,
        "component_matches": components,
        "component_explanations": component_explanations,
        "diagnostic_only_components": diagnostic_only_components,
        "mismatched_components": mismatches,
        "mismatch_explanations": mismatch_explanations,
        "certified_cas_evidence_match": safe,
        "certified_cas_evidence_similarity": float(matched / total),
        "safe_for_derived_category_claims": safe,
        "comparison_policy": "no_proxy_no_fallback_exact_cas_components_only",
        "reason": None if safe else "Certified CAS free-resolution artifacts differ: " + ", ".join(mismatches),
        "interpretation": (
            "The certified multigraded real free-resolution artifacts match exactly at the stored CAS-evidence level; Buchsbaum-Eisenbud/BEMultipliers entries are diagnostic-only sidecars, not substitute resolution evidence."
            if safe
            else "Both sides have certified multigraded real free resolutions, but matching is not proved because one or more exact CAS-evidence components differ; diagnostic-only BEMultipliers sidecars cannot substitute for the missing agreement."
        ),
    }

def _stable_artifact_hash(value: object) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except TypeError:
        payload = str(value)
    return hashlib.sha256(payload.encode("utf-8", "ignore")).hexdigest()


def _simplex_tree_map_report(
    query_obj: dict[str, object],
    memory_obj: dict[str, object],
    mapping: dict[str, str],
    q_filtration: dict[tuple[str, ...], float],
    m_filtration: dict[tuple[str, ...], float],
    max_rows: int = 160,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    checked = 0
    preserved = 0
    missing = 0
    positive_distortions: list[float] = []
    dim_counts: dict[str, dict[str, int]] = {}
    simplices = _complex_simplices(query_obj, 0) + _complex_simplices(query_obj, 1) + _complex_simplices(query_obj, 2)
    for simplex in simplices:
        simplex = tuple(sorted(str(v) for v in simplex))
        if not simplex or any(vertex not in mapping for vertex in simplex):
            continue
        checked += 1
        dim = len(simplex) - 1
        image = tuple(sorted(set(str(mapping[vertex]) for vertex in simplex)))
        domain_filtration = float(q_filtration.get(tuple(sorted(simplex)), 0.0))
        exists = len(image) <= 1 or image in m_filtration
        codomain_filtration = float(m_filtration.get(image, 0.0 if len(image) <= 1 else math.inf))
        signed_distortion = codomain_filtration - domain_filtration if exists and math.isfinite(codomain_filtration) else math.inf
        preserved_flag = bool(exists and math.isfinite(signed_distortion) and signed_distortion <= 1e-9)
        if preserved_flag:
            preserved += 1
        elif not exists:
            missing += 1
        if math.isfinite(signed_distortion):
            positive_distortions.append(float(max(0.0, signed_distortion)))
        key = f"dim_{dim}"
        bucket = dim_counts.setdefault(key, {"checked": 0, "preserved": 0, "missing_codomain": 0})
        bucket["checked"] += 1
        bucket["preserved"] += int(preserved_flag)
        bucket["missing_codomain"] += int(not exists)
        if len(rows) < max_rows:
            rows.append(
                {
                    "domain_simplex": list(simplex),
                    "image_simplex": list(image),
                    "dimension": int(dim),
                    "domain_filtration": float(domain_filtration),
                    "codomain_filtration": None if not math.isfinite(codomain_filtration) else float(codomain_filtration),
                    "signed_filtration_distortion": None if not math.isfinite(signed_distortion) else float(signed_distortion),
                    "preserved_in_simplex_tree": preserved_flag,
                    "failure_reason": None if preserved_flag else ("missing_codomain_simplex" if not exists else "filtration_not_preserved"),
                }
            )
    rate = float(preserved / checked) if checked else 0.0
    certified = bool(checked > 0 and preserved == checked and rate >= 0.999)
    interpretation = (
        "This finite simplex-tree enumeration certifies a filtered simplicial map on the displayed simplex trees; its F2-linear extension is safe to interpret as a chain map and persistence-module morphism."
        if certified
        else "This finite simplex-tree enumeration is only a failed or incomplete preservation check; no simplicial map, chain map, or persistence-module morphism is asserted from it."
    )
    return {
        "source": "gudhi.SimplexTree finite simplex enumeration",
        "ring": "F2[x_level,x_radius]",
        "map_kind": "vertex_probability_assignment_extended_to_simplex_tree",
        "checked_simplices": int(checked),
        "preserved_simplices": int(preserved),
        "missing_codomain_simplices": int(missing),
        "preservation_rate": rate,
        "filtered_simplicial_map_certified": certified,
        "safe_to_render_as_chain_map": certified,
        "safe_to_render_as_persistence_module_morphism": certified,
        "positive_filtration_distortion_summary": _numeric_summary(positive_distortions),
        "dimension_counts": dim_counts,
        "rows": rows,
        "interpretation": interpretation,
    }


def _empty_simplicial_map_report(query_vertices: list[dict[str, object]], memory_vertices: list[dict[str, object]], reason: str) -> dict[str, object]:
    chain_map = _unavailable_chain_map_diagnostics(reason)
    return {
        "vertex_map": [],
        "displayed_domain_vertices": len(query_vertices),
        "displayed_codomain_vertices": len(memory_vertices),
        "checked_edges": 0,
        "preserved_edges": 0,
        "edge_preservation_rate": 0.0,
        "checked_two_simplices": 0,
        "preserved_two_simplices": 0,
        "two_simplex_preservation_rate": 0.0,
        "chain_map_diagnostics": chain_map,
        "persistence_module_morphism_diagnostics": _persistence_module_morphism_diagnostics(chain_map),
        "is_filtered_simplicial_map": False,
        "is_simplicial_on_displayed_skeleton": False,
        "simplicial_map_failure_reason": reason,
        "map_source": "none",
    }


def _probability_induced_vertex_assignment(
    query_vertices: list[dict[str, object]],
    memory_vertices: list[dict[str, object]],
    q_probs: list[list[float] | None],
    m_probs: list[list[float] | None],
) -> list[dict[str, object]]:
    costs = np.full((len(query_vertices), len(memory_vertices)), 1e6, dtype=float)
    js_costs = np.full((len(query_vertices), len(memory_vertices)), math.inf, dtype=float)
    for qi, q in enumerate(query_vertices):
        for mi, m in enumerate(memory_vertices):
            dist = _padded_jensen_shannon(q_probs[qi], m_probs[mi])
            if dist is None:
                continue
            js_costs[qi, mi] = float(dist)
            type_penalty = 0.05 if q.get("type") != m.get("type") else 0.0
            level_penalty = 0.0
            if q.get("level") is not None and m.get("level") is not None:
                level_penalty = 0.01 * abs(float(q.get("level", 0.0) or 0.0) - float(m.get("level", 0.0) or 0.0))
            costs[qi, mi] = float(dist + type_penalty + level_penalty)
    pairs: list[tuple[int, int]] = []
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        row_ind, col_ind = linear_sum_assignment(costs)
        pairs = [(int(r), int(c)) for r, c in zip(row_ind, col_ind) if math.isfinite(float(costs[int(r), int(c)])) and costs[int(r), int(c)] < 1e5]
    except Exception:
        used: set[int] = set()
        for qi in range(len(query_vertices)):
            order = np.argsort(costs[qi])
            for mi in order.tolist():
                if mi not in used and math.isfinite(float(costs[qi, mi])) and costs[qi, mi] < 1e5:
                    pairs.append((qi, int(mi)))
                    used.add(int(mi))
                    break
    rows = []
    for qi, mi in pairs:
        cost = float(costs[qi, mi])
        js_distance = float(js_costs[qi, mi])
        rows.append(
            {
                "query_vertex": str(query_vertices[qi]["label"]),
                "memory_vertex": str(memory_vertices[mi]["label"]),
                "score": float(1.0 / (1.0 + cost)),
                "assignment_cost": cost,
                "jensen_shannon_distance": js_distance,
                "map_source": "model_probability_jensen_shannon_assignment",
                "query_probability_source": query_vertices[qi].get("probability_source"),
                "memory_probability_source": memory_vertices[mi].get("probability_source"),
            }
        )
    return rows


def _check_filtered_simplicial_dimension(
    query_obj: dict[str, object],
    memory_obj: dict[str, object],
    mapping: dict[str, str],
    q_filtration: dict[tuple[str, ...], float],
    m_filtration: dict[tuple[str, ...], float],
    dimension: int,
) -> dict[str, object]:
    checked = 0
    preserved = 0
    preserved_pairs: list[dict[str, object]] = []
    failed_pairs: list[dict[str, object]] = []
    preserved_query_vertices: set[str] = set()
    preserved_memory_vertices: set[str] = set()
    positive_distortions: list[float] = []
    signed_distortions: list[float] = []
    missing_codomain_simplices = 0
    for simplex in _complex_simplices(query_obj, dimension):
        if any(vertex not in mapping for vertex in simplex):
            continue
        checked += 1
        image = tuple(sorted(set(mapping[vertex] for vertex in simplex)))
        domain_key = tuple(sorted(simplex))
        domain_filtration = float(q_filtration.get(domain_key, 0.0))
        codomain_filtration = float(m_filtration.get(image, 0.0 if len(image) <= 1 else math.inf))
        exists = len(image) <= 1 or image in m_filtration
        filtration_ok = exists and codomain_filtration <= domain_filtration + 1e-9
        signed_distortion = codomain_filtration - domain_filtration if math.isfinite(codomain_filtration) else math.inf
        if exists and math.isfinite(signed_distortion):
            signed_distortions.append(float(signed_distortion))
            positive_distortions.append(float(max(0.0, signed_distortion)))
        elif not exists:
            missing_codomain_simplices += 1
        payload = {
            "query_simplex" if dimension == 2 else "query_edge": list(simplex),
            "memory_simplex" if dimension == 2 else "memory_edge": list(image),
            "domain_filtration": domain_filtration,
            "codomain_filtration": None if not math.isfinite(codomain_filtration) else codomain_filtration,
            "signed_filtration_distortion": None if not math.isfinite(signed_distortion) else signed_distortion,
            "positive_filtration_distortion": None if not math.isfinite(signed_distortion) else max(0.0, signed_distortion),
        }
        if filtration_ok:
            preserved += 1
            preserved_query_vertices.update(simplex)
            preserved_memory_vertices.update(image)
            if len(preserved_pairs) < 64:
                preserved_pairs.append(payload)
        elif len(failed_pairs) < 64:
            reason = "missing_codomain_simplex" if not exists else "filtration_not_preserved"
            failed_pairs.append({**payload, "failure_reason": reason})
    return {
        "checked": checked,
        "preserved": preserved,
        "preserved_pairs": preserved_pairs,
        "failed_pairs": failed_pairs,
        "preserved_query_vertices": preserved_query_vertices,
        "preserved_memory_vertices": preserved_memory_vertices,
        "positive_distortions": positive_distortions,
        "signed_distortions": signed_distortions,
        "missing_codomain_simplices": missing_codomain_simplices,
    }


def _numeric_summary(values: list[float]) -> dict[str, object]:
    finite = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=float)
    if finite.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _fmt_optional(value: object, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(val):
        return "n/a"
    return f"{val:.{digits}g}"


def _complex_vertex_records(obj: dict[str, object]) -> list[dict[str, object]]:
    simplices = obj.get("simplices", []) if isinstance(obj, dict) else []
    vertices = []
    for idx, simplex in enumerate(simplices):
        if not isinstance(simplex, dict) or int(simplex.get("dimension", -1)) != 0:
            continue
        label = _simplex_label(simplex, default_label=f"v{idx}")
        vertices.append(
            {
                "label": label,
                "type": str(simplex.get("type", "vertex")),
                "text": str(simplex.get("text", "")),
                "filtration": float(simplex.get("filtration", 0.0) or 0.0),
                "embedding": simplex.get("embedding", []),
                "probability": simplex.get("probability", []),
                "model_probability_vector": simplex.get("model_probability_vector", []),
                "probability_vector": simplex.get("probability_vector", []),
                "probability_source": simplex.get("probability_source"),
                "score": simplex.get("score"),
                "nll": simplex.get("nll"),
                "level": simplex.get("level"),
                "path": simplex.get("path", []),
                "input_text": simplex.get("input_text", ""),
                "target_text": simplex.get("target_text", ""),
                "decoded_argmax": simplex.get("decoded_argmax", ""),
                "graph_json_summary": simplex.get("graph_json_summary", {}),
                "filtered_simplicial_object": simplex.get("filtered_simplicial_object", {}),
                "topological_algebra": simplex.get("topological_algebra", {}),
                "gudhi_simplex_tree": simplex.get("gudhi_simplex_tree", False),
            }
        )
    return vertices


def _vertex_by_label(obj: dict[str, object]) -> dict[str, dict[str, object]]:
    return {row["label"]: row for row in _complex_vertex_records(obj)}


def _edge_filtration(obj: dict[str, object], pair: tuple[str, str]) -> float:
    a, b = pair
    for simplex in obj.get("simplices", []) if isinstance(obj, dict) else []:
        if not isinstance(simplex, dict) or int(simplex.get("dimension", -1)) != 1:
            continue
        raw_value = simplex.get("simplex", [])
        raw = [str(v) for v in raw_value[:2]] if isinstance(raw_value, list) else []
        if len(raw) >= 2 and raw[0] == a and raw[1] == b:
            return float(simplex.get("filtration", 0.0) or 0.0)
    return 1.0


def _simplex_filtration_lookup(obj: dict[str, object]) -> dict[tuple[str, ...], float]:
    out: dict[tuple[str, ...], float] = {}
    for simplex in obj.get("simplices", []) if isinstance(obj, dict) else []:
        if not isinstance(simplex, dict):
            continue
        raw = simplex.get("simplex", [])
        if not isinstance(raw, list) or not raw:
            continue
        key = tuple(sorted(str(v) for v in raw))
        filt = float(simplex.get("filtration", 0.0) or 0.0)
        if key not in out or filt < out[key]:
            out[key] = filt
    return out


def _complex_edge_pairs(obj: dict[str, object]) -> list[tuple[str, str]]:
    return [tuple(simplex[:2]) for simplex in _complex_simplices(obj, 1) if len(simplex) >= 2]


def _complex_simplices(obj: dict[str, object], dimension: int) -> list[list[str]]:
    simplices = obj.get("simplices", []) if isinstance(obj, dict) else []
    out = []
    for simplex in simplices:
        if not isinstance(simplex, dict) or int(simplex.get("dimension", -1)) != dimension:
            continue
        raw = simplex.get("simplex", [])
        if isinstance(raw, list):
            out.append([str(v) for v in raw])
    return out


def _simplex_label(simplex: dict[str, object], default_label: str) -> str:
    raw = simplex.get("simplex", [])
    if isinstance(raw, list) and raw:
        return str(raw[0])
    return default_label


def _probability_feature_vector(row: dict[str, object]) -> list[float] | None:
    raw = None
    for key in ("model_probability_vector", "probability_vector", "probability"):
        value = row.get(key)
        if isinstance(value, list) and value:
            raw = value
            break
    if isinstance(raw, list) and raw:
        vals = []
        for item in raw:
            try:
                value = float(item)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(value):
                return None
            vals.append(max(value, 0.0))
        total = sum(vals)
        if total <= 0.0:
            return None
        return [float(v / total) for v in vals]
    return None


def _padded_jensen_shannon(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b:
        return None
    n = max(len(a), len(b))
    pa = np.zeros(n, dtype=float)
    pb = np.zeros(n, dtype=float)
    pa[: len(a)] = np.asarray(a, dtype=float)
    pb[: len(b)] = np.asarray(b, dtype=float)
    pa = np.maximum(pa, 0.0)
    pb = np.maximum(pb, 0.0)
    sa = float(pa.sum())
    sb = float(pb.sum())
    if sa <= 0.0 or sb <= 0.0:
        return None
    pa = pa / sa
    pb = pb / sb
    mix = 0.5 * (pa + pb)
    eps = 1e-12
    mask_a = pa > 0.0
    mask_b = pb > 0.0
    kl_a = float(np.sum(pa[mask_a] * np.log(pa[mask_a] / np.maximum(mix[mask_a], eps))))
    kl_b = float(np.sum(pb[mask_b] * np.log(pb[mask_b] / np.maximum(mix[mask_b], eps))))
    return float(math.sqrt(max(0.0, 0.5 * (kl_a + kl_b))))


def _vertex_match_score(query: dict[str, object], memory: dict[str, object]) -> float:
    score = 0.0
    q_vec = _coerce_vertex_vector(query, max_dims=256)
    m_vec = _coerce_vertex_vector(memory, max_dims=256)
    if q_vec and m_vec:
        n = min(len(q_vec), len(m_vec))
        q_arr = np.asarray(q_vec[:n], dtype=float)
        m_arr = np.asarray(m_vec[:n], dtype=float)
        denom = float(np.linalg.norm(q_arr) * np.linalg.norm(m_arr))
        if denom > 1e-12:
            score += 6.0 * float(np.dot(q_arr, m_arr) / denom)
        score += 2.0 / (1.0 + float(np.linalg.norm(q_arr - m_arr)))
    if query.get("label") == memory.get("label"):
        score += 1.0
    if query.get("type") == memory.get("type"):
        score += 1.5
    q_text = set(str(query.get("text", "")).lower().split())
    m_text = set(str(memory.get("text", "")).lower().split())
    if q_text or m_text:
        score += len(q_text & m_text) / max(len(q_text | m_text), 1)
    score += 1.0 / (1.0 + abs(float(query.get("filtration", 0.0)) - float(memory.get("filtration", 0.0))))
    if query.get("level") is not None and memory.get("level") is not None:
        score += 1.0 / (1.0 + abs(float(query.get("level", 0.0) or 0.0) - float(memory.get("level", 0.0) or 0.0)))
    return score


def _signature_numeric_vector(topology: dict[str, object], length: int = 32) -> np.ndarray:
    sig = topology.get("derived_equivalence_signature", {}) if isinstance(topology, dict) else {}
    values: list[float] = []
    values.extend(float(v) for v in sig.get("betti_vector", [])[:4])
    values.extend(
        [
            float(sig.get("persistence_finite_interval_count", 0.0)),
            float(sig.get("persistence_infinite_interval_count", 0.0)),
            float(sig.get("persistence_total_finite_length", 0.0)),
            float(sig.get("multiparameter_grid_points", 0.0)),
        ]
    )
    samples = sig.get("multiparameter_h0_rank_sample", []) if isinstance(sig.get("multiparameter_h0_rank_sample"), list) else []
    values.extend(float(row.get("h0_rank", 0.0)) for row in samples if isinstance(row, dict))
    if len(values) < length:
        values.extend([0.0] * (length - len(values)))
    return np.asarray(values[:length], dtype=float)


def _free_rank_vector(topology: dict[str, object], length: int = 16) -> np.ndarray:
    values = [0.0] * length
    for row in _free_resolution_modules(topology):
        degree = int(row.get("homological_degree", 0))
        if 0 <= degree < length:
            values[degree] = float(row.get("rank", row.get("rank_upper_bound", 0.0)))
    return np.asarray(values, dtype=float)


def _rank_invariant_numeric_vector(topology: dict[str, object], length: int = 32) -> np.ndarray:
    values: list[float] = []
    if not isinstance(topology, dict):
        return np.zeros(length, dtype=float)
    mp = topology.get("multiparameter_persistence") if isinstance(topology.get("multiparameter_persistence"), dict) else {}
    rank_sources = [
        mp.get("rank_invariant_samples") if isinstance(mp, dict) else None,
        (mp.get("rank_invariant", {}) or {}).get("samples") if isinstance(mp.get("rank_invariant", {}), dict) else None,
        (topology.get("derived_equivalence_signature", {}) or {}).get("multiparameter_h0_rank_sample") if isinstance(topology.get("derived_equivalence_signature", {}), dict) else None,
    ]
    for samples in rank_sources:
        if not isinstance(samples, list):
            continue
        for row in samples:
            if not isinstance(row, dict):
                continue
            for key in ("h0_rank", "h1_rank", "h2_rank", "rank", "value"):
                if key in row:
                    value = _summary_float(row, key)
                    if value is not None:
                        values.append(value)
            for key in ("source_rank", "target_rank", "image_rank", "kernel_rank", "cokernel_rank"):
                if key in row:
                    value = _summary_float(row, key)
                    if value is not None:
                        values.append(value)
        if values:
            break
    if len(values) < length:
        values.extend([0.0] * (length - len(values)))
    return np.asarray(values[:length], dtype=float)



def _commutative_algebra_numeric_vector(topology: dict[str, object], length: int = 16) -> np.ndarray:
    ca = topology.get("commutative_algebra", {}) if isinstance(topology, dict) else {}
    values: list[float] = []
    chain_keys = (
        "two_parameter_chain_presentation_diagnostics",
        "multiparameter_chain_presentation_diagnostics",
    )
    keys = chain_keys
    for key in keys:
        fr = ca.get(key, {}) if isinstance(ca.get(key), dict) else {}
        for row in fr.get("free_chain_modules", []) if isinstance(fr.get("free_chain_modules"), list) else []:
            if isinstance(row, dict):
                values.append(float(row.get("rank", row.get("rank_upper_bound", 0.0)) or 0.0))
        det_summary = fr.get("determinantal_ideal_summary", {}) if isinstance(fr.get("determinantal_ideal_summary"), dict) else {}
        fit_summary = fr.get("fitting_ideal_summary", {}) if isinstance(fr.get("fitting_ideal_summary"), dict) else {}
        be_summary = fr.get("buchsbaum_eisenbud_summary", {}) if isinstance(fr.get("buchsbaum_eisenbud_summary"), dict) else {}
        if not det_summary and isinstance(fr.get("determinantal_ideals"), dict):
            maps = fr["determinantal_ideals"].get("maps", {}) if isinstance(fr["determinantal_ideals"].get("maps"), dict) else {}
            det_summary = {"map_count": len(maps)}
        if not fit_summary and isinstance(fr.get("fitting_ideals"), dict):
            maps = fr["fitting_ideals"].get("maps", {}) if isinstance(fr["fitting_ideals"].get("maps"), dict) else {}
            fit_summary = {"map_count": len(maps)}
        if not be_summary and isinstance(fr.get("buchsbaum_eisenbud"), dict):
            be = fr["buchsbaum_eisenbud"]
            be_summary = {
                "composition_zero_checks": len(be.get("composition_zero_checks", [])) if isinstance(be.get("composition_zero_checks"), list) else 0,
                "exact_chain_modules": sum(1 for row in be.get("rank_exactness_checks", []) if isinstance(row, dict) and row.get("exact_at_chain_module_over_F2_incidence")) if isinstance(be.get("rank_exactness_checks"), list) else 0,
            }
        values.extend([
            float(det_summary.get("map_count", 0.0) or 0.0),
            float(det_summary.get("nonzero_sampled_generators", 0.0) or 0.0),
            float(fit_summary.get("map_count", 0.0) or 0.0),
            float(fit_summary.get("bounded_invariant_count", 0.0) or 0.0),
            float(be_summary.get("composition_zero_checks", 0.0) or 0.0),
            float(be_summary.get("exact_chain_modules", 0.0) or 0.0),
        ])
    if len(values) < length:
        values.extend([0.0] * (length - len(values)))
    return np.asarray(values[:length], dtype=float)


def _persistence_landscape_numeric_vector(topology: dict[str, object]) -> np.ndarray:
    if not isinstance(topology, dict):
        return np.zeros(0, dtype=float)
    reps = topology.get("persistence_representations")
    if not isinstance(reps, dict) or not reps.get("available"):
        return np.zeros(0, dtype=float)
    methods = reps.get("methods")
    if not isinstance(methods, dict):
        return np.zeros(0, dtype=float)
    parts: list[np.ndarray] = []
    def _sort_key(value: object) -> int:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return 999
    for key in sorted(methods.keys(), key=_sort_key):
        row = methods.get(key, {})
        if not isinstance(row, dict) or not row.get("available"):
            continue
        landscape = row.get("landscape")
        if not isinstance(landscape, dict):
            continue
        raw = landscape.get("vector")
        if not isinstance(raw, list) or not raw:
            continue
        vals: list[float] = []
        for value in raw:
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                vals.append(v)
        if vals:
            parts.append(np.asarray(vals, dtype=float))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=float)


def _persistence_landscape_vector_similarity(query_topology: dict[str, object], memory_topology: dict[str, object]) -> dict[str, object]:
    report = _memory_persistence_landscape_vector_similarity(query_topology, memory_topology)
    if report.get("available"):
        report = dict(report)
        report.setdefault("differentiable_comparison_note", "cosine, L2, and correlation are differentiable vector comparisons once landscapes are vectorized; this implementation uses cached GUDHI NumPy vectors.")
    return report


def _persistence_vector_representation_similarity(
    query_topology: dict[str, object],
    memory_topology: dict[str, object],
    *,
    include_landscape: bool = True,
) -> dict[str, object]:
    return _memory_persistence_vector_representation_similarity(query_topology, memory_topology, include_landscape=include_landscape)


def _persistence_numeric_vector(topology: dict[str, object], length: int = 16) -> np.ndarray:
    values = [0.0] * length
    intervals = topology.get("persistence", {}).get("intervals", []) if isinstance(topology.get("persistence"), dict) else []
    for interval in intervals:
        if not isinstance(interval, dict):
            continue
        dim = int(interval.get("dimension", 0))
        if 0 <= dim < 4:
            values[dim] += 1.0
            death = interval.get("death")
            if isinstance(death, (int, float)):
                values[4 + dim] += max(float(death) - float(interval.get("birth", 0.0)), 0.0)
            elif interval.get("infinite") or death is None:
                values[8 + dim] += 1.0
    return np.asarray(values[:length], dtype=float)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 1.0 if float(np.linalg.norm(a - b)) <= 1e-12 else 0.0
    return float(np.dot(a, b) / denom)


def _memory_color(idx: int) -> str:
    palette = ["#7aa2ff", "#fbbf24", "#fb7185", "#c084fc", "#34d399", "#f97316"]
    return palette[idx % len(palette)]


def _write_inference_dashboard(paths: dict[str, str], output_dir: Path) -> Path:
    dash = output_dir / "inference_audit.html"

    def dashboard_href(path_value: str) -> str:
        path = Path(path_value)
        try:
            return path.resolve().relative_to(output_dir.resolve()).as_posix()
        except ValueError:
            return path.name

    link_rows = []
    for name, path in sorted(paths.items()):
        href = dashboard_href(str(path))
        link_rows.append(
            f'<li><a href="{html.escape(href, quote=True)}"><span>{html.escape(str(name))}</span><code>{html.escape(href)}</code></a></li>'
        )
    links = "\n".join(link_rows)
    dash.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TropicalGT-I Inference Audit</title>
  <style>
    :root {{
      color-scheme: dark;
      background: #090b12;
      color: #eef2ff;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 18% 12%, rgba(85, 214, 190, 0.12), transparent 32%),
        linear-gradient(145deg, #090b12 0%, #111827 58%, #071015 100%);
      color: #eef2ff;
    }}
    main {{
      width: min(980px, calc(100% - 40px));
      margin: 0 auto;
      padding: 42px 0 56px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    p {{
      margin: 0 0 26px;
      color: #aab6d3;
      line-height: 1.55;
    }}
    ul {{
      display: grid;
      gap: 10px;
      list-style: none;
      padding: 0;
      margin: 0;
    }}
    a {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 16px;
      border: 1px solid rgba(148, 163, 184, 0.28);
      border-radius: 8px;
      background: rgba(15, 23, 42, 0.72);
      color: #dbeafe;
      text-decoration: none;
      box-shadow: 0 14px 30px rgba(0, 0, 0, 0.22);
    }}
    a:hover {{
      border-color: rgba(85, 214, 190, 0.62);
      background: rgba(17, 34, 50, 0.92);
    }}
    code {{
      color: #55d6be;
      font-size: 12px;
      white-space: nowrap;
    }}
  </style>
</head>
<body>
  <main>
    <h1>TropicalGT-I Inference Audit</h1>
    <p>Dark-mode index for the generated topology, algebra, GraphCG, tropical support, memory, and Graph-of-Thought trajectory artifacts.</p>
    <ul>{links}</ul>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return dash


def write_metric_visualizations(history: list[dict[str, float]], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_path = output_dir / "training_metrics.html"
    if not history:
        _write_dark_empty(metric_path, "No training metrics recorded.")
        return {"metrics": str(metric_path)}

    steps = [int(row.get("step", idx + 1)) for idx, row in enumerate(history)]
    metrics = [
        "loss",
        "nll",
        "ppl",
        "gflownet_tb",
        "gflownet_tb_residual_abs_mean",
        "gflownet_action_entropy_mean",
        *GRAPHCG_BROWSER_PRIORITY_METRICS,
        "bpb",
        "text_bpb",
        "graph_bpb",
        "graph_sideinfo_bpb",
        "graph_conditioned_bpb_no_side_cost",
        "graph_token_structural_bytes",
        "explicit_graph_json_bytes",
        "analogical_memory_query_norm",
        "analogical_memory_bank_size",
        "analogical_memory_records_added",
        "certificate_loss",
        "certificate_agreement",
        "certificate_coverage",
        "certificate_edge_agreement",
        "loss_regularizer_total",
        "loss_regularizer_ratio",
        "margin_mean",
        "margin_min",
        "margin_p05",
        "support_entropy",
        "support_soft_entropy",
        "wall_hit_rate",
        "support_boundary_hit_rate",
        "grad_norm",
        "examples_per_sec",
        "tokens_per_sec",
        "graph_tokens_per_sec",
        "graph_json_fallback_rate",
        "graph_json_derived_text_graph_rate",
        "graph_json_parse_unavailable_rate",
        "gpu_mem_mb",
    ]
    metric_display_names = {
        "graph_json_fallback_rate": "legacy graph-json substitution guardrail (must remain zero)",
    }
    fig = go.Figure()
    for name in metrics:
        values = [row.get(name) for row in history]
        if any(v is not None for v in values):
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=[float(v) if v is not None else None for v in values],
                    mode="lines+markers",
                    name=metric_display_names.get(name, name),
                )
            )
    fig.update_layout(
        title="TropicalGT-I smoke training metrics",
        xaxis_title="step",
        yaxis_title="value",
        legend_title="metric",
        hovermode="x unified",
    )
    _write_plotly_dark_html(metric_path, fig, "TropicalGT-I smoke training metrics")
    return {"metrics": str(metric_path)}


def write_graphcg_training_visualizations(model, output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    direction_tensor = model.graphcg.effective_directions(detach=True) if hasattr(model.graphcg, "effective_directions") else model.graphcg.directions.detach()
    directions = direction_tensor.detach().cpu().float().numpy()
    if directions.size == 0:
        return {}
    raw_directions = model.graphcg.directions.detach().cpu().float().numpy()
    max_plot = min(int(getattr(model.graphcg, "viz_max_directions", 256)), directions.shape[0])
    sample_idx = np.linspace(0, directions.shape[0] - 1, max_plot, dtype=int) if directions.shape[0] > max_plot else np.arange(directions.shape[0])
    plot_directions = directions[sample_idx]
    plot_raw_directions = raw_directions[sample_idx]
    norms = np.linalg.norm(raw_directions, axis=1, keepdims=True)
    plot_norms = np.linalg.norm(plot_raw_directions, axis=1, keepdims=True)
    normalized = plot_directions / np.maximum(np.linalg.norm(plot_directions, axis=1, keepdims=True), 1e-8)
    gram = normalized @ normalized.T
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    rank_margin = float(getattr(model.graphcg, "full_rank_margin", 0.05))
    numerical_rank = int(np.sum(singular_values > rank_margin))
    rank_target = min(normalized.shape)
    full_count = int(directions.shape[0])
    sample_note = f"sampled {len(sample_idx)} of {full_count} embedding-space directions"

    gram_path = output_dir / "graphcg_direction_gram.html"
    fig = go.Figure(
        data=go.Heatmap(
            z=gram,
            x=[f"dir_{int(idx)}" for idx in sample_idx],
            y=[f"dir_{int(idx)}" for idx in sample_idx],
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="cosine"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title=f"GraphCG direction Gram matrix ({sample_note}; sample rank {numerical_rank}/{rank_target})",
    )
    _write_plotly_dark_html(gram_path, fig, f"GraphCG direction Gram matrix ({sample_note}; sample rank {numerical_rank}/{rank_target})")

    pca_path = output_dir / "graphcg_direction_pca.html"
    pca = _pca3(plot_directions)
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter3d(
            x=pca[:, 0],
            y=pca[:, 1],
            z=pca[:, 2],
            mode="markers",
            marker=dict(
                size=4,
                color=sample_idx,
                colorscale="Viridis",
                colorbar=dict(title="direction"),
                line=dict(width=0.5, color="#e8eef8"),
            ),
            name="sampled GraphCG directions",
            hovertext=[
                f"direction {int(direction_idx)}<br>norm={float(norm):.4f}<br>{sample_note}"
                for direction_idx, norm in zip(sample_idx, plot_norms[:, 0])
            ],
            hoverinfo="text",
        )
    )
    fig2.update_layout(
        template="plotly_dark",
        title=f"GraphCG embedding-space steering directions in PCA ({sample_note})",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
    )
    _write_plotly_dark_html(pca_path, fig2, f"GraphCG embedding-space steering directions in PCA ({sample_note})")

    sv_path = output_dir / "graphcg_direction_singular_values.html"
    fig3 = go.Figure(
        data=go.Bar(
            x=[f"s{idx}" for idx in range(len(singular_values))],
            y=singular_values,
            marker=dict(color=singular_values, colorscale="Viridis"),
            hovertemplate="singular value=%{y:.5f}<extra></extra>",
        )
    )
    fig3.add_hline(y=rank_margin, line_dash="dash", line_color="#f97316", annotation_text="full-rank margin")
    fig3.update_layout(
        template="plotly_dark",
        title=f"GraphCG sampled singular spectrum ({sample_note})",
        xaxis_title="singular direction",
        yaxis_title="singular value",
    )
    _write_plotly_dark_html(sv_path, fig3, f"GraphCG sampled singular spectrum ({sample_note})")
    return {
        "graphcg_direction_gram": str(gram_path),
        "graphcg_direction_pca": str(pca_path),
        "graphcg_direction_singular_values": str(sv_path),
    }


def _pca3(values: np.ndarray) -> np.ndarray:
    coords, _report = _pca3_with_report(values)
    return coords


def _pca3_with_report(values: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        values = values.reshape((len(values), -1))
    uniqueness = _embedding_uniqueness_report(values)
    if values.shape[0] < 2:
        coords = np.pad(values[:, :1], ((0, 0), (0, 2)), constant_values=0.0)
        return coords, {
            "coordinate_source": "model graph_state embeddings",
            "method": "sklearn PCA",
            "n_samples": int(values.shape[0]),
            "embedding_dim": int(values.shape[1]) if values.ndim == 2 else 0,
            "explained_variance_ratio": [1.0, 0.0, 0.0],
            "pairwise_distance_correlation": 1.0,
            "normalized_stress": 0.0,
            **uniqueness,
            **_pca_coordinate_uniqueness_report(coords),
        }
    n_components = min(3, values.shape[0], values.shape[1])
    pca_model = PCA(n_components=n_components)
    coords = pca_model.fit_transform(values)
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])), constant_values=0.0)
    original_dist = _pairwise_euclidean(values)
    projected_dist = _pairwise_euclidean(coords)
    mask = np.triu(np.ones_like(original_dist, dtype=bool), k=1)
    target = original_dist[mask]
    realized = projected_dist[mask]
    if target.size >= 2 and float(np.std(target)) > 1e-12 and float(np.std(realized)) > 1e-12:
        corr = float(np.corrcoef(target, realized)[0, 1])
    else:
        corr = 1.0
    denom = max(float(np.dot(target, target)), 1e-12)
    stress = math.sqrt(float(np.dot(target - realized, target - realized)) / denom)
    ratios = [float(v) for v in getattr(pca_model, "explained_variance_ratio_", np.asarray([], dtype=float)).tolist()]
    if len(ratios) < 3:
        ratios.extend([0.0] * (3 - len(ratios)))
    return coords, {
        "coordinate_source": "model graph_state embeddings",
        "method": "sklearn PCA",
        "n_samples": int(values.shape[0]),
        "embedding_dim": int(values.shape[1]),
        "explained_variance_ratio": ratios[:3],
        "explained_variance_ratio_sum3": float(sum(ratios[:3])),
        "pairwise_distance_correlation": corr,
        "normalized_stress": stress,
        **uniqueness,
        **_pca_coordinate_uniqueness_report(coords),
    }


def _embedding_uniqueness_report(values: np.ndarray, decimals: int = 8) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        return {
            "unique_embeddings_rounded8": 0,
            "duplicate_embeddings_rounded8": 0,
            "unique_embedding_ratio_rounded8": 0.0,
            "max_embedding_multiplicity_rounded8": 0,
        }
    rounded = np.round(values, decimals=decimals)
    _, counts = np.unique(rounded, axis=0, return_counts=True)
    unique_count = int(counts.size)
    total = int(values.shape[0])
    return {
        "unique_embeddings_rounded8": unique_count,
        "duplicate_embeddings_rounded8": int(total - unique_count),
        "unique_embedding_ratio_rounded8": float(unique_count / max(total, 1)),
        "max_embedding_multiplicity_rounded8": int(counts.max(initial=0)),
    }


def _pca_coordinate_uniqueness_report(coords: np.ndarray, decimals: int = 8) -> dict[str, object]:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return {
            "unique_pca_coordinates_rounded8": 0,
            "duplicate_pca_coordinates_rounded8": 0,
            "max_pca_coordinate_multiplicity_rounded8": 0,
        }
    rounded = np.round(coords[:, :3], decimals=decimals)
    _, counts = np.unique(rounded, axis=0, return_counts=True)
    unique_count = int(counts.size)
    total = int(coords.shape[0])
    return {
        "unique_pca_coordinates_rounded8": unique_count,
        "duplicate_pca_coordinates_rounded8": int(total - unique_count),
        "max_pca_coordinate_multiplicity_rounded8": int(counts.max(initial=0)),
    }


def _coordinate_multiplicities(coords: np.ndarray, decimals: int = 8) -> list[int]:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return []
    keys = [tuple(float(v) for v in row) for row in np.round(coords[:, :3], decimals=decimals)]
    counts: dict[tuple[float, ...], int] = defaultdict(int)
    for key in keys:
        counts[key] += 1
    return [int(counts[key]) for key in keys]


def _infer_candidate_levels(candidates: list[dict[str, object]], ids: list[str]) -> list[int]:
    raw_levels: list[int | None] = []
    id_to_idx = {rid: idx for idx, rid in enumerate(ids)}
    children: dict[str, list[str]] = defaultdict(list)
    roots: list[str] = []
    for idx, row in enumerate(candidates):
        level = row.get("level")
        raw_levels.append(int(level) if isinstance(level, (int, float)) and math.isfinite(float(level)) else None)
        parent = row.get("parent")
        if isinstance(parent, str) and parent in id_to_idx:
            children[parent].append(ids[idx])
        else:
            roots.append(ids[idx])
    levels = [level if level is not None else -1 for level in raw_levels]
    queue: deque[tuple[str, int]] = deque((rid, 0) for rid in roots)
    seen: set[str] = set()
    while queue:
        rid, depth = queue.popleft()
        if rid in seen:
            continue
        seen.add(rid)
        idx = id_to_idx.get(rid)
        if idx is not None and levels[idx] < 0:
            levels[idx] = depth
        for child in children.get(rid, []):
            queue.append((child, depth + 1))
    return [max(int(level), 0) for level in levels]


def _last_action(row: dict[str, object]) -> str:
    path = row.get("path", [])
    if isinstance(path, list) and path:
        return str(path[-1])
    return "root"


def _edge_action_label(row: dict[str, object]) -> str:
    return _last_action(row)


def _trajectory_nll_progress_diagnostics(
    candidates: list[dict[str, object]],
    ids: list[str],
    id_to_idx: dict[str, int],
    nll_values: np.ndarray,
    levels: np.ndarray,
) -> dict[str, object]:
    edge_rows: list[dict[str, object]] = []
    has_child = {str(row.get("parent")) for row in candidates if isinstance(row.get("parent"), str)}
    for idx, row in enumerate(candidates):
        parent = row.get("parent")
        if not isinstance(parent, str) or parent not in id_to_idx:
            continue
        parent_idx = id_to_idx[parent]
        delta = float(nll_values[idx] - nll_values[parent_idx])
        edge_rows.append(
            {
                "source": str(parent),
                "target": str(ids[idx]),
                "source_level": int(levels[parent_idx]) if parent_idx < len(levels) else 0,
                "target_level": int(levels[idx]) if idx < len(levels) else 0,
                "source_nll": float(nll_values[parent_idx]),
                "target_nll": float(nll_values[idx]),
                "delta_child_minus_parent": delta,
                "improves": bool(delta < 0.0),
            }
        )
    deltas = np.asarray([row["delta_child_minus_parent"] for row in edge_rows], dtype=float)
    improving = deltas < 0.0 if deltas.size else np.asarray([], dtype=bool)
    root_indices = [idx for idx, row in enumerate(candidates) if not isinstance(row.get("parent"), str)]
    root_nll = float(np.nanmean(nll_values[root_indices])) if root_indices else float(nll_values[0])
    max_level = int(np.nanmax(levels)) if len(levels) else 0
    terminal_indices = [
        idx
        for idx, row in enumerate(candidates)
        if int(levels[idx]) == max_level or str(ids[idx]) not in has_child
    ]
    terminal_nll = np.asarray([float(nll_values[idx]) for idx in terminal_indices], dtype=float)
    by_level: list[dict[str, float | int]] = []
    level_values = np.asarray(levels, dtype=int).reshape(-1)
    for level in sorted({int(v) for v in level_values.tolist()}):
        mask = np.asarray([int(v) == level for v in levels], dtype=bool)
        vals = nll_values[mask]
        if vals.size:
            by_level.append(
                {
                    "level": int(level),
                    "count": int(vals.size),
                    "mean_nll": float(np.nanmean(vals)),
                    "min_nll": float(np.nanmin(vals)),
                    "max_nll": float(np.nanmax(vals)),
                    "mean_improvement_from_root": float(root_nll - np.nanmean(vals)),
                    "best_improvement_from_root": float(root_nll - np.nanmin(vals)),
                }
            )
    return {
        "edge_count": int(len(edge_rows)),
        "edge_deltas": edge_rows,
        "improving_edge_fraction": float(np.mean(improving)) if improving.size else 0.0,
        "mean_edge_delta": float(np.nanmean(deltas)) if deltas.size else 0.0,
        "median_edge_delta": float(np.nanmedian(deltas)) if deltas.size else 0.0,
        "root_mean_nll": root_nll,
        "terminal_count": int(terminal_nll.size),
        "terminal_mean_nll": float(np.nanmean(terminal_nll)) if terminal_nll.size else root_nll,
        "terminal_min_nll": float(np.nanmin(terminal_nll)) if terminal_nll.size else root_nll,
        "terminal_mean_improvement_from_root": float(root_nll - np.nanmean(terminal_nll)) if terminal_nll.size else 0.0,
        "best_terminal_improvement_from_root": float(root_nll - np.nanmin(terminal_nll)) if terminal_nll.size else 0.0,
        "by_level": by_level,
        "interpretation": "Negative edge deltas and positive terminal improvements indicate reasoning paths moving toward lower NLL; non-monotone paths remain visible as regressions.",
    }


def _action_color(action: str) -> str:
    colors = {
        "expand": "#55d6be",
        "merge": "#7aa2ff",
        "refine": "#fbbf24",
        "retrieve": "#c084fc",
        "verify": "#34d399",
        "compress": "#f97316",
        "reject": "#fb7185",
        "stop": "#94a3b8",
        "root": "#e8eef8",
    }
    return colors.get(action, "#94a3b8")


def _write_plotly_dark_html(path: Path, fig: go.Figure, title: str, panel_items: list[dict[str, object]] | None = None, show_filtration_slider: bool = False, show_selected_complex_panel: bool = False) -> None:
    existing_margin = fig.layout.margin.to_plotly_json() if fig.layout.margin else {}
    top_margin = max(int(existing_margin.get("t", 0) or 0), 78)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#090b12",
        plot_bgcolor="#090b12",
        font=dict(color="#e8eef8"),
        margin=dict(
            l=int(existing_margin.get("l", 0) or 0),
            r=int(existing_margin.get("r", 0) or 0),
            b=int(existing_margin.get("b", 0) or 0),
            t=top_margin,
        ),
    )
    plotly_asset = path.parent / "plotly.min.js"
    if not plotly_asset.exists():
        plotly_asset.write_text(get_plotlyjs(), encoding="utf-8")
    chart = (
        '<script src="plotly.min.js"></script>'
        + fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})
    )
    items = panel_items or []
    has_panel = bool(items)
    initial_index = max(range(len(items)), key=lambda idx: float(items[idx].get("complexity", 0.0))) if items else 0
    initial = items[initial_index] if items else {"title": "Filtered simplicial object", "svg": "", "plot": {}, "summary": "Hover a reasoning node to render its complex."}
    controls_html = (
        """<div class=\"filtration-controls\" id=\"filtration-controls\">
        <label><span>Filtration radius</span><strong id=\"filtration-value\">all</strong></label>
        <input id=\"filtration-slider\" type=\"range\" min=\"0\" max=\"1\" step=\"0.001\" value=\"1\" aria-label=\"filtered simplicial complex radius\">
        <div class=\"hint\" id=\"filtration-hint\">Left is the smallest visible filtration; right is the full selected complex. Multiparameter summaries remain in the JSON payload.</div>
      </div>"""
        if show_filtration_slider
        else ""
    )
    layout_class = "layout has-panel" if has_panel else "layout no-panel"
    selected_complex_panel_html = (
        '<div class="simplicial-object-plot" id="simplicial-plot" aria-label="interactive selected filtered simplicial complex"></div>'
        if show_selected_complex_panel
        else '<div class="panel-note">Hover or click a point in the main plot to inspect the exact filtered simplicial object payload. The duplicate secondary 3D complex panel is disabled on this page.</div>'
    )
    static_preview_html = (
        f"""<details class="static-preview">
	        <summary>Static SVG same-data preview from the same filtered-complex payload</summary>
	        <div class="simplicial-object-panel" id="simplicial-svg">{initial["svg"]}</div>
	      </details>"""
        if has_panel and not show_selected_complex_panel
        else ""
    )
    panel_html = (
        f"""
    <aside class="panel" aria-live="polite">
      <h1 id="simplicial-title">{html.escape(initial["title"])}</h1>
      <div class="summary" id="simplicial-summary">{initial["summary"]}</div>
      {controls_html}
      {selected_complex_panel_html}
      {static_preview_html}
    </aside>"""
        if has_panel
        else ""
    )
    hover_html = (
        f"""
  <div class="hover-simplicial-card" id="hover-simplicial-card" role="tooltip" aria-label="hovered filtered simplicial object">
    <h2 id="hover-simplicial-title">{html.escape(initial["title"])}</h2>
    <div class="hover-summary" id="hover-simplicial-summary">{initial["summary"]}</div>
    <div id="hover-simplicial-svg">{initial["svg"]}</div>
  </div>"""
        if has_panel
        else ""
    )
    path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #090b12;
      --panel: #101623;
      --panel-2: #151d2d;
      --ink: #e8eef8;
      --muted: #99a8bd;
      --accent: #5eead4;
      --edge: rgba(125, 151, 184, 0.32);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: radial-gradient(circle at 20% 0%, #111a2a 0%, var(--bg) 38%, #06070b 100%);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 420px);
      min-height: 100vh;
    }}
    .layout.no-panel {{
      grid-template-columns: minmax(0, 1fr);
    }}
    .chart {{
      min-width: 0;
      min-height: 100vh;
      border-right: 1px solid var(--edge);
    }}
    .layout.no-panel .chart {{
      border-right: 0;
    }}
    .panel {{
      background: linear-gradient(180deg, rgba(16, 22, 35, 0.98), rgba(9, 11, 18, 0.98));
      padding: 18px;
      min-height: 100vh;
      overflow: auto;
    }}
    .panel h1 {{
      margin: 0 0 10px;
      font-size: 16px;
      font-weight: 650;
      letter-spacing: 0;
    }}
    .summary {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
      margin-bottom: 12px;
    }}
    .filtration-controls {{
      border: 1px solid rgba(148, 163, 184, 0.22);
      border-radius: 8px;
      background: rgba(7, 10, 18, 0.72);
      padding: 10px;
      margin: 0 0 12px;
      display: grid;
      gap: 7px;
    }}
    .filtration-controls label {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      font-size: 11px;
    }}
    .filtration-controls strong {{ color: var(--ink); font-weight: 650; }}
    .filtration-controls input[type="range"] {{ width: 100%; accent-color: var(--accent); }}
    .filtration-controls .hint {{ color: var(--muted); font-size: 10px; line-height: 1.35; }}
    .simplicial-object-panel {{
      border: 1px solid rgba(94, 234, 212, 0.2);
      background: #070a12;
      border-radius: 8px;
      padding: 10px;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03), 0 18px 36px rgba(0,0,0,0.24);
    }}
    .simplicial-object-panel svg {{ width: 100%; height: auto; display: block; }}
    .panel-note {{
      border: 1px solid rgba(94, 234, 212, 0.20);
      background: rgba(7, 10, 18, 0.64);
      border-radius: 8px;
      padding: 10px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.45;
      margin-bottom: 12px;
    }}
    .simplicial-object-plot {{
      width: 100%;
      height: min(48vh, 460px);
      min-height: 340px;
      border: 1px solid rgba(94, 234, 212, 0.22);
      background: #070a12;
      border-radius: 8px;
      margin-bottom: 12px;
      overflow: hidden;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03), 0 18px 36px rgba(0,0,0,0.24);
    }}
	    .static-preview {{
	      border: 1px solid rgba(148, 163, 184, 0.16);
	      border-radius: 8px;
	      background: rgba(7, 10, 18, 0.46);
	      padding: 8px;
	    }}
	    .static-preview summary {{
	      cursor: pointer;
	      color: #99f6e4;
	      font-size: 11px;
	      margin-bottom: 8px;
	    }}
	    .webgl-static-preview {{
	      margin: 16px;
	      padding: 14px;
	      border: 1px solid rgba(251, 191, 36, 0.35);
	      border-radius: 8px;
	      background: rgba(30, 41, 59, 0.94);
	      color: var(--ink);
	    }}
	    .webgl-static-preview h2 {{
	      margin: 0 0 6px;
	      font-size: 14px;
	      line-height: 1.25;
	    }}
	    .webgl-static-preview p {{
	      margin: 0 0 10px;
	      color: var(--muted);
	      font-size: 12px;
	      line-height: 1.45;
	    }}
	    .webgl-static-preview .simplicial-object-panel {{ margin-top: 8px; }}
    .hover-simplicial-card {{
      position: fixed;
      z-index: 40;
      width: min(520px, calc(100vw - 24px));
      max-height: 82vh;
      pointer-events: none;
      opacity: 0;
      transform: translate3d(12px, 12px, 0) scale(0.98);
      transition: opacity 100ms ease, transform 100ms ease;
      border: 1px solid rgba(94, 234, 212, 0.26);
      border-radius: 8px;
      background: rgba(7, 10, 18, 0.96);
      box-shadow: 0 18px 42px rgba(0,0,0,0.42), inset 0 0 0 1px rgba(255,255,255,0.04);
      padding: 10px;
      backdrop-filter: blur(10px);
    }}
    .hover-simplicial-card.visible {{
      opacity: 1;
      transform: translate3d(0, 0, 0) scale(1);
    }}
    .hover-simplicial-card h2 {{
      margin: 0 0 6px;
      font-size: 12px;
      font-weight: 650;
      color: var(--ink);
      letter-spacing: 0;
    }}
    .hover-simplicial-card .hover-summary {{
      color: var(--muted);
      font-size: 10px;
      line-height: 1.35;
      max-height: 220px;
      overflow: auto;
      margin-bottom: 6px;
    }}
    .hover-simplicial-card svg {{ width: 100%; height: auto; display: block; }}
    @media (max-width: 900px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .chart {{ min-height: 68vh; border-right: 0; border-bottom: 1px solid var(--edge); }}
      .panel {{ min-height: auto; }}
      .hover-simplicial-card {{ display: none; }}
    }}
  </style>
</head>
<body>
  <main class="{layout_class}">
    <section class="chart" id="chart">{chart}</section>
    {panel_html}
  </main>
  {hover_html}
  <script>
    const simplicialPanels = {json.dumps(items)};
    const panelTitle = document.getElementById("simplicial-title");
    const panelSummary = document.getElementById("simplicial-summary");
    const panelSvg = document.getElementById("simplicial-svg");
    const panelPlot = document.getElementById("simplicial-plot");
    const hoverCard = document.getElementById("hover-simplicial-card");
    const hoverTitle = document.getElementById("hover-simplicial-title");
    const hoverSummary = document.getElementById("hover-simplicial-summary");
    const hoverSvg = document.getElementById("hover-simplicial-svg");
    const filtrationSlider = document.getElementById("filtration-slider");
    const filtrationValue = document.getElementById("filtration-value");
    const filtrationHint = document.getElementById("filtration-hint");
    const initialPanelIndex = {int(initial_index)};
    let activePanelIndex = initialPanelIndex;
    function setPanel(index) {{
      const item = simplicialPanels[index];
      if (!item) return;
      activePanelIndex = index;
      panelTitle.textContent = item.title || "Filtered simplicial object";
      panelSummary.innerHTML = item.summary || "";
      if (panelSvg) panelSvg.innerHTML = item.svg || "";
      configureFiltrationSlider(item, panelSvg);
      renderPanelComplex(item);
    }}
    function configureFiltrationSlider(item, root) {{
      if (!filtrationSlider || !filtrationValue || !filtrationHint) {{
        applyFiltrationThreshold(root, Infinity);
        return;
      }}
      const min = Number(item.filtration_min ?? 0);
      const max = Number(item.filtration_max ?? 1);
      const hasRange = Number.isFinite(max) && max > min;
      filtrationSlider.min = hasRange ? String(min) : "0";
      filtrationSlider.max = hasRange ? String(max) : "1";
      filtrationSlider.step = hasRange ? String(Math.max((max - min) / 200, 0.000001)) : "0.001";
      filtrationSlider.value = hasRange ? String(min) : "0";
      filtrationSlider.disabled = !hasRange;
      filtrationHint.textContent = item.multiparameter_hint || "Left is the disjoint 0-simplex cloud; moving right adds exactly the simplices whose scalar filtration is at or below the selected radius. Multiparameter summaries remain in the JSON payload.";
      applyFiltrationThreshold(root, hasRange ? min : Infinity);
      filtrationValue.textContent = hasRange ? min.toFixed(3) : "all";
    }}
    function applyFiltrationThreshold(root, threshold) {{
      if (!root) return;
      root.querySelectorAll("[data-filtration]").forEach((el) => {{
        const value = Number(el.getAttribute("data-filtration"));
        const visible = !Number.isFinite(value) || value <= threshold + 1e-12;
        el.style.opacity = visible ? "" : "0.08";
        el.style.filter = visible ? "" : "grayscale(1)";
        el.style.pointerEvents = visible ? "" : "none";
      }});
    }}
    if (filtrationSlider) {{
      filtrationSlider.addEventListener("input", () => {{
        const threshold = Number(filtrationSlider.value);
        filtrationValue.textContent = Number.isFinite(threshold) ? threshold.toFixed(3) : "all";
        applyFiltrationThreshold(panelSvg, threshold);
        renderPanelComplex(simplicialPanels[activePanelIndex]);
      }});
    }}
    function panelThreshold(item) {{
      if (!item) return Infinity;
      if (filtrationSlider && !filtrationSlider.disabled) {{
        const value = Number(filtrationSlider.value);
        if (Number.isFinite(value)) return value;
      }}
      const plot = item.plot || {{}};
      const max = Number(plot.filtration_max ?? item.filtration_max ?? 1);
      return Number.isFinite(max) ? max : Infinity;
    }}
    function buildPanelTraces(item) {{
      const plot = (item && item.plot) || {{}};
      const vertices = Array.isArray(plot.vertices) ? plot.vertices : [];
      const edges = Array.isArray(plot.edges) ? plot.edges : [];
      const directedEdges = Array.isArray(plot.directed_edges) ? plot.directed_edges : [];
      const triangles = Array.isArray(plot.triangles) ? plot.triangles : [];
      const threshold = panelThreshold(item);
      const visible = new Set(vertices.filter((v) => Number(v.filtration ?? 0) <= threshold + 1e-12).map((v) => String(v.label)));
      const vertexByLabel = new Map(vertices.map((v) => [String(v.label), v]));
      const edgeX = [], edgeY = [], edgeZ = [], edgeHover = [];
      edges.forEach((e) => {{
        const a = vertexByLabel.get(String(e.a));
        const b = vertexByLabel.get(String(e.b));
        if (!a || !b || !visible.has(String(e.a)) || !visible.has(String(e.b))) return;
        if (Number(e.filtration ?? 0) > threshold + 1e-12) return;
        const hover = `<b>1-simplex</b><br>${{e.a}} -> ${{e.b}}<br>filtration=${{Number(e.filtration ?? 0).toFixed(4)}}<br>type=${{e.type || "edge"}}`;
        edgeX.push(a.x, b.x, null);
        edgeY.push(a.y, b.y, null);
        edgeZ.push(a.z, b.z, null);
        edgeHover.push(hover, hover, null);
      }});
      const directedX = [], directedY = [], directedZ = [], directedHover = [];
      const directedMarkerX = [], directedMarkerY = [], directedMarkerZ = [], directedMarkerHover = [];
      function appendDotted(a, b, hover) {{
        const segments = 10;
        const duty = 0.46;
        for (let idx = 0; idx < segments; idx += 1) {{
          const t0 = idx / segments;
          const t1 = Math.min((idx + duty) / segments, 1.0);
          directedX.push(a.x + (b.x - a.x) * t0, a.x + (b.x - a.x) * t1, null);
          directedY.push(a.y + (b.y - a.y) * t0, a.y + (b.y - a.y) * t1, null);
          directedZ.push(a.z + (b.z - a.z) * t0, a.z + (b.z - a.z) * t1, null);
          directedHover.push(hover, hover, null);
        }}
      }}
      directedEdges.forEach((e) => {{
        const a = vertexByLabel.get(String(e.a));
        const b = vertexByLabel.get(String(e.b));
        if (!a || !b || !visible.has(String(e.a)) || !visible.has(String(e.b))) return;
        if (Number(e.filtration ?? 0) > threshold + 1e-12) return;
        const hover = e.hover || `<b>dotted causal/decoding overlay</b><br>${{e.a}} -> ${{e.b}}<br>role=${{e.role || "decoding_order"}}<br>decoding step=${{e.decoding_step ?? ""}}<br>reasoning level=${{e.reasoning_level ?? ""}}`;
        appendDotted(a, b, hover);
        directedMarkerX.push(b.x); directedMarkerY.push(b.y); directedMarkerZ.push(b.z); directedMarkerHover.push(hover);
      }});
      const shownVertices = vertices.filter((v) => visible.has(String(v.label)));
      const showLabels = shownVertices.length <= 18;
      const traces = [
        {{
          type: "scatter3d",
          mode: "lines",
          x: edgeX,
          y: edgeY,
          z: edgeZ,
          line: {{color: "rgba(125,211,252,0.42)", width: 2.2}},
          hovertext: edgeHover,
          hoverinfo: "text",
          name: "1-simplices"
        }},
      ];
      const meshIndex = new Map();
      const meshX = [], meshY = [], meshZ = [], triI = [], triJ = [], triK = [];
      function meshVertex(label) {{
        const key = String(label);
        if (meshIndex.has(key)) return meshIndex.get(key);
        const v = vertexByLabel.get(key);
        if (!v) return null;
        const idx = meshX.length;
        meshIndex.set(key, idx);
        meshX.push(v.x); meshY.push(v.y); meshZ.push(v.z);
        return idx;
      }}
      triangles.forEach((tri) => {{
        if (Number(tri.filtration ?? 0) > threshold + 1e-12) return;
        const labels = Array.isArray(tri.vertices) ? tri.vertices.map(String) : [];
        if (labels.length < 3 || labels.some((label) => !visible.has(label))) return;
        const idxs = labels.slice(0, 3).map(meshVertex);
        if (idxs.some((idx) => idx === null || idx === undefined)) return;
        triI.push(idxs[0]); triJ.push(idxs[1]); triK.push(idxs[2]);
      }});
      if (triI.length) {{
        traces.push({{
          type: "mesh3d",
          x: meshX,
          y: meshY,
          z: meshZ,
          i: triI,
          j: triJ,
          k: triK,
          color: "rgba(94,234,212,0.14)",
          opacity: 0.18,
          name: "2-simplices",
          hoverinfo: "skip",
          showscale: false
        }});
      }}
      if (directedX.length) {{
        traces.push({{
          type: "scatter3d",
          mode: "lines",
          x: directedX,
          y: directedY,
          z: directedZ,
          line: {{color: "rgba(255,255,255,0.74)", width: 3}},
          hovertext: directedHover,
          hoverinfo: "text",
          name: "dotted causal/decoding order"
        }});
        traces.push({{
          type: "scatter3d",
          mode: "markers",
          x: directedMarkerX,
          y: directedMarkerY,
          z: directedMarkerZ,
          marker: {{size: 4.5, color: "#fde047", symbol: "diamond", line: {{color: "#0f172a", width: 0.6}}}},
          hovertext: directedMarkerHover,
          hoverinfo: "text",
          name: "directed edge heads"
        }});
      }}
      traces.push({{
        type: "scatter3d",
        mode: showLabels ? "markers+text" : "markers",
        x: shownVertices.map((v) => v.x),
        y: shownVertices.map((v) => v.y),
        z: shownVertices.map((v) => v.z),
        text: shownVertices.map((v) => showLabels ? (v.short_label || String(v.label).slice(0, 16)) : ""),
        textposition: "top center",
        textfont: {{color: "#dbeafe", size: 10}},
        marker: {{
          size: shownVertices.map((v) => Number(v.size ?? 7)),
          color: shownVertices.map((v) => Number(v.filtration ?? 0)),
          colorscale: "Viridis",
          showscale: true,
          colorbar: {{title: "filtration", len: 0.58}},
          line: {{color: "#e8eef8", width: 1.1}}
        }},
        hovertext: shownVertices.map((v) => v.hover || `<b>${{v.label}}</b><br>filtration=${{Number(v.filtration ?? 0).toFixed(4)}}`),
        hoverinfo: "text",
        name: "0-simplices"
      }});
      return traces;
    }}
	    function webglUnsupportedText(root) {{
	      const text = root && root.textContent ? root.textContent : "";
	      return text.includes("WebGL is not supported") || text.includes("webgl is not supported");
	    }}
	    function staticPreviewMarkup(item, reason) {{
	      const svg = item && item.svg ? item.svg : "<div class='simplicial-object-panel'>No static filtered-complex SVG preview is available for this selected object.</div>";
	      const title = item && item.title ? item.title : "selected filtered simplicial object";
	      return `<div class="webgl-static-preview"><h2>Static filtered-complex preview</h2><p>${{reason}} This preview is generated from the same serialized simplicial object payload as the interactive 3D panel.</p><div class="summary">${{title}}</div><div class="simplicial-object-panel">${{svg}}</div></div>`;
	    }}
	    function renderPanelStaticPreview(item, reason) {{
	      if (!panelPlot) return;
	      panelPlot.innerHTML = staticPreviewMarkup(item, reason);
	    }}
	    function promoteMainStaticPreview() {{
	      const chartEl = document.getElementById("chart");
	      if (!chartEl || !simplicialPanels.length || chartEl.querySelector(".webgl-static-preview.main-static-preview")) return;
	      if (!webglUnsupportedText(chartEl)) return;
	      const item = simplicialPanels[activePanelIndex] || simplicialPanels[initialPanelIndex] || simplicialPanels[0];
	      const preview = document.createElement("div");
	      preview.className = "webgl-static-preview main-static-preview";
	      preview.innerHTML = `<h2>WebGL unavailable: same-data static complex preview shown</h2><p>The browser could not create a WebGL context for the 3D Plotly view. The data were still loaded; the same-data preview below comes from the selected real filtered-complex payload.</p>${{staticPreviewMarkup(item, "Interactive WebGL rendering is unavailable in this browser context.")}}`;
	      chartEl.prepend(preview);
	    }}
	    function renderPanelComplex(item) {{
	      if (!panelPlot || typeof Plotly === "undefined") return;
	      const plot = (item && item.plot) || {{}};
	      if (!Array.isArray(plot.vertices) || !plot.vertices.length) {{
	        panelPlot.innerHTML = "<div style='padding:16px;color:#99a8bd'>No selected filtered simplicial complex vertices available.</div>";
	        return;
      }}
      panelPlot.innerHTML = "<div id='selected-complex-graph' style='width:100%;height:100%;'></div>";
      const selectedGraph = panelPlot.querySelector("#selected-complex-graph");
      if (!selectedGraph) return;
      const layout = {{
        template: "plotly_dark",
        paper_bgcolor: "#070a12",
        plot_bgcolor: "#070a12",
        margin: {{l: 0, r: 0, t: 28, b: 0}},
        title: {{
          text: `selected complex: ${{plot.vertex_count || plot.vertices.length}} vertices, ${{plot.edge_count || 0}} edges, ${{plot.triangle_count || 0}} faces`,
          font: {{size: 11, color: "#dbeafe"}}
        }},
        scene: {{
          xaxis: {{title: "PCA/MDS-1", gridcolor: "rgba(125,211,252,0.22)", zerolinecolor: "rgba(226,232,240,0.25)"}},
          yaxis: {{title: "PCA/MDS-2", gridcolor: "rgba(125,211,252,0.22)", zerolinecolor: "rgba(226,232,240,0.25)"}},
          zaxis: {{title: "PCA/MDS-3", gridcolor: "rgba(125,211,252,0.22)", zerolinecolor: "rgba(226,232,240,0.25)"}},
          aspectmode: "cube",
          camera: {{eye: {{x: 1.45, y: 1.25, z: 0.95}}}}
        }},
        legend: {{orientation: "h", y: -0.05}},
        showlegend: true
	      }};
	      try {{
	        const rendered = Plotly.newPlot(selectedGraph, buildPanelTraces(item), layout, {{displaylogo: false, responsive: false}});
	        Promise.resolve(rendered).then(() => {{
	          window.setTimeout(() => {{
	            if (webglUnsupportedText(selectedGraph)) renderPanelStaticPreview(item, "Interactive WebGL rendering is unavailable in this browser context.");
	          }}, 120);
	        }}).catch((err) => {{
	          renderPanelStaticPreview(item, `Could not render the interactive 3D panel: ${{err && err.message ? err.message : err}}.`);
	        }});
	      }} catch (err) {{
	        renderPanelStaticPreview(item, `Could not render the interactive 3D panel: ${{err && err.message ? err.message : err}}.`);
	      }}
	    }}
    function positionHoverCard(pointerEvent) {{
      if (!hoverCard || !pointerEvent) return;
      const margin = 14;
      const bounds = hoverCard.getBoundingClientRect();
      let left = pointerEvent.clientX + margin;
      let top = pointerEvent.clientY + margin;
      if (left + bounds.width > window.innerWidth - margin) left = pointerEvent.clientX - bounds.width - margin;
      if (top + bounds.height > window.innerHeight - margin) top = pointerEvent.clientY - bounds.height - margin;
      hoverCard.style.left = Math.max(margin, left) + "px";
      hoverCard.style.top = Math.max(margin, top) + "px";
    }}
    function renderHoverCard(index, pointerEvent) {{
      const item = simplicialPanels[index];
      if (!item || !hoverCard) return;
      hoverTitle.textContent = item.title || "Filtered simplicial object";
      hoverSummary.innerHTML = item.compact_summary || item.summary || "";
      hoverSvg.innerHTML = item.svg || "";
      const threshold = filtrationSlider && !filtrationSlider.disabled ? Number(filtrationSlider.value) : Infinity;
      applyFiltrationThreshold(hoverSvg, threshold);
      positionHoverCard(pointerEvent);
      hoverCard.classList.add("visible");
    }}
    function hideHoverCard() {{
      if (hoverCard) hoverCard.classList.remove("visible");
    }}
    const plot = document.querySelector("#chart .plotly-graph-div");
    function forceConfiguredPlotlyFrame() {{
      if (!plot || typeof Plotly === "undefined") return;
      const sliders = (plot.layout && plot.layout.sliders) || [];
      const slider = sliders.length ? sliders[0] : null;
      if (!slider || !Array.isArray(slider.steps) || !slider.steps.length) return;
      const layoutActive = Number(slider.active);
      const active = Number.isInteger(layoutActive) ? Math.max(0, Math.min(slider.steps.length - 1, layoutActive)) : 0;
      const step = slider.steps[Math.max(0, Math.min(slider.steps.length - 1, active))];
      const args = step && Array.isArray(step.args) ? step.args : null;
      const frameName = args && Array.isArray(args[0]) ? args[0][0] : null;
      if (!frameName) return;
      const animation = Plotly.animate(plot, [frameName], {{
        mode: "immediate",
        frame: {{duration: 0, redraw: true}},
        transition: {{duration: 0}}
      }});
      if (animation && typeof animation.then === "function") {{
        animation.then(() => Plotly.relayout(plot, {{"sliders[0].active": active}})).catch(() => {{}});
      }} else {{
        try {{ Plotly.relayout(plot, {{"sliders[0].active": active}}); }} catch (err) {{}}
      }}
    }}
    if (plot && simplicialPanels.length) {{
      setPanel(initialPanelIndex);
      plot.on("plotly_hover", (event) => {{
        const point = (event.points || []).find((p) => p.customdata !== undefined && p.customdata !== null);
        if (!point) return;
        const raw = Array.isArray(point.customdata) ? point.customdata[0] : point.customdata;
        const idx = Number(raw);
        if (Number.isFinite(idx)) {{
          setPanel(idx);
          renderHoverCard(idx, event.event);
        }}
      }});
      plot.on("plotly_click", (event) => {{
        const point = (event.points || []).find((p) => p.customdata !== undefined && p.customdata !== null);
        if (!point) return;
        const raw = Array.isArray(point.customdata) ? point.customdata[0] : point.customdata;
        const idx = Number(raw);
        if (Number.isFinite(idx)) {{
          setPanel(idx);
          renderHoverCard(idx, event.event);
        }}
      }});
      plot.on("plotly_unhover", hideHoverCard);
      plot.on("plotly_relayout", hideHoverCard);
    }}
	    window.setTimeout(forceConfiguredPlotlyFrame, 150);
	    window.setTimeout(forceConfiguredPlotlyFrame, 650);
	    window.setTimeout(promoteMainStaticPreview, 900);
	    window.setTimeout(promoteMainStaticPreview, 1800);
	  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def _write_dark_empty(path: Path, message: str) -> None:
    escaped = html.escape(message)
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<style>:root{color-scheme:dark;background:#090b12;color:#e8eef8}"
        "body{margin:0;min-height:100vh;background:#090b12;color:#e8eef8;font-family:Inter,ui-sans-serif,system-ui;display:grid;place-items:start center;padding:48px 24px}"
        ".diagnostic{max-width:780px;border:1px solid rgba(148,163,184,.28);background:#0f172a;padding:24px 26px;border-radius:8px;box-shadow:0 20px 60px rgba(0,0,0,.24)}"
        ".eyebrow{margin:0 0 8px;color:#93c5fd;font-size:12px;text-transform:uppercase;letter-spacing:.08em}"
        "h1{margin:0 0 12px;font-size:22px;line-height:1.25}.body{margin:0;color:#cbd5e1;line-height:1.55}</style>"
        f"</head><body><main class='diagnostic'><p class='eyebrow'>explicit unavailable diagnostic</p><h1>No standalone plot rendered</h1><p class='body'>{escaped}</p></main></body></html>",
        encoding="utf-8",
    )


def _write_dark_redirect(path: Path, title: str, message: str, target: str, label: str) -> None:
    escaped_title = html.escape(title)
    escaped_message = html.escape(message)
    escaped_target = html.escape(target, quote=True)
    escaped_label = html.escape(label)
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<meta http-equiv='refresh' content='0; url={escaped_target}'>"
        "<style>:root{color-scheme:dark;background:#090b12;color:#e8eef8}"
        "body{margin:0;min-height:100vh;background:#090b12;color:#e8eef8;font-family:Inter,ui-sans-serif,system-ui;display:grid;place-items:start center;padding:48px 24px}"
        ".diagnostic{max-width:780px;border:1px solid rgba(148,163,184,.28);background:#0f172a;padding:24px 26px;border-radius:8px;box-shadow:0 20px 60px rgba(0,0,0,.24)}"
        ".eyebrow{margin:0 0 8px;color:#93c5fd;font-size:12px;text-transform:uppercase;letter-spacing:.08em}"
        "h1{margin:0 0 12px;font-size:22px;line-height:1.25}.body{margin:0 0 16px;color:#cbd5e1;line-height:1.55}"
        "a{color:#7dd3fc;text-decoration:none;border-bottom:1px solid rgba(125,211,252,.45)}</style>"
        f"</head><body><main class='diagnostic'><p class='eyebrow'>redirect to trajectory-growth artifact</p><h1>{escaped_title}</h1><p class='body'>{escaped_message}</p><a href='{escaped_target}'>{escaped_label}</a></main></body></html>",
        encoding="utf-8",
    )


def _simplicial_panel_items(objects: list[dict[str, object]], hover: list[str]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for idx, obj in enumerate(objects):
        summary = obj.get("summary", {}) if isinstance(obj, dict) else {}
        title = str(obj.get("record_id", f"reasoning-state-{idx}")) if isinstance(obj, dict) else f"reasoning-state-{idx}"
        thresholds_raw = obj.get("thresholds", []) if isinstance(obj, dict) else []
        threshold_values = [float(v) for v in thresholds_raw if isinstance(v, (int, float)) and math.isfinite(float(v))]
        if not threshold_values and isinstance(obj, dict):
            threshold_values = [
                float(row.get("filtration", 0.0))
                for row in obj.get("simplices", [])
                if isinstance(row, dict) and isinstance(row.get("filtration", 0.0), (int, float))
            ]
        filtration_min = min(threshold_values) if threshold_values else 0.0
        filtration_max = max(threshold_values) if threshold_values else 0.0
        compact_summary = (
            f"V={summary.get('num_vertices', 0)} | E={summary.get('num_edges', 0)} | "
            f"T={summary.get('num_two_simplices', 0)} | scalar thresholds={summary.get('num_thresholds', len(threshold_values))}"
        )
        complexity = (
            float(summary.get("num_vertices", 0) or 0)
            + 2.0 * float(summary.get("num_edges", 0) or 0)
            + 3.0 * float(summary.get("num_two_simplices", 0) or 0)
        )
        summary_html = compact_summary
        if idx < len(hover):
            summary_html += f"<br>{hover[idx]}"
        if isinstance(obj, dict):
            step_href = str(obj.get("step_complex_href", "")).strip()
            tree_href = str(obj.get("step_simplex_tree_href", "")).strip()
            if step_href:
                safe_href = html.escape(step_href, quote=True)
                summary_html += f'<br><a href="{safe_href}">open interactive reasoning-step complex page</a>'
            if tree_href:
                safe_tree_href = html.escape(tree_href, quote=True)
                summary_html += f' · <a href="{safe_tree_href}">open simplex tree</a>'
        items.append(
            {
                "title": title,
                "summary": summary_html,
                "compact_summary": compact_summary,
                "svg": _simplicial_object_svg(obj if isinstance(obj, dict) else {}, max_vertices=160),
                "plot": _simplicial_plot_payload(obj if isinstance(obj, dict) else {}),
                "complexity": complexity,
                "filtration_min": filtration_min,
                "filtration_max": filtration_max,
                "multiparameter_hint": (
                    "Slider filters the scalar radius/filtration attached to displayed simplices. "
                    "Three-parameter persistence grades are exported in the JSON audit payload."
                ),
            }
        )
    return items


def _simplicial_plot_payload(obj: dict[str, object], max_vertices: int = 220) -> dict[str, object]:
    obj = _gudhi_canonical_complex(obj)
    simplices = obj.get("simplices", []) if isinstance(obj, dict) else []
    vertices = [s for s in simplices if isinstance(s, dict) and int(s.get("dimension", -1)) == 0]
    edges = [s for s in simplices if isinstance(s, dict) and int(s.get("dimension", -1)) == 1]
    triangles = [s for s in simplices if isinstance(s, dict) and int(s.get("dimension", -1)) == 2]
    labels = [str((s.get("simplex") or [f"v{idx}"])[0]) for idx, s in enumerate(vertices)]
    source_label_count = len(labels)
    if len(labels) > max_vertices:
        # Keep deterministic order but prefer actual model/GUDHI vertices with lower filtration first.
        ordered = sorted(
            enumerate(vertices),
            key=lambda row: (float(row[1].get("filtration", 0.0) or 0.0), str((row[1].get("simplex") or [""])[0])),
        )
        keep_idx = {idx for idx, _row in ordered[:max_vertices]}
        vertices = [row for idx, row in enumerate(vertices) if idx in keep_idx]
        labels = [str((s.get("simplex") or [f"v{idx}"])[0]) for idx, s in enumerate(vertices)]
    visible = set(labels)
    coords3, _projected, layout_kind = _simplicial_pca3_radius_layout(labels, vertices, edges, width=760, height=560)
    vertex_by_label = {str((row.get("simplex") or [""])[0]): row for row in vertices}
    threshold_values = _display_thresholds(obj, max_steps=96)
    radius_vertices_enter_at_zero = _is_radius_filtration_complex(obj)
    payload_vertices: list[dict[str, object]] = []
    for idx, label in enumerate(labels):
        row = vertex_by_label.get(label, {})
        x, y, z = coords3.get(label, (0.0, 0.0, 0.0))
        filt = 0.0 if radius_vertices_enter_at_zero else float(row.get("filtration", 0.0) or 0.0)
        payload_vertices.append(
            {
                "label": label,
                "short_label": _short_complex_vertex_label(label, row, idx),
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "size": 7.0 + 4.0 * max(0.0, min(1.0, float(z))),
                "filtration": filt,
                "type": str(row.get("type", "vertex")),
                "hover": _vertex_readable_summary(row, include_output=True),
            }
        )
    payload_edges: list[dict[str, object]] = []
    for edge in edges:
        simplex = [str(v) for v in (edge.get("simplex") or [])[:2]]
        if len(simplex) < 2 or simplex[0] not in visible or simplex[1] not in visible:
            continue
        payload_edges.append(
            {
                "a": simplex[0],
                "b": simplex[1],
                "filtration": float(edge.get("filtration", 0.0) or 0.0),
                "type": str(edge.get("type", "edge")),
            }
        )
    payload_directed_edges: list[dict[str, object]] = []
    decoding_overlay = obj.get("decoding_causal_overlay", {}) if isinstance(obj.get("decoding_causal_overlay"), dict) else {}
    for directed_edge in decoding_overlay.get("edges", []) if isinstance(decoding_overlay.get("edges"), list) else []:
        if not isinstance(directed_edge, dict):
            continue
        source = str(directed_edge.get("source", ""))
        target = str(directed_edge.get("target", ""))
        if source not in visible or target not in visible:
            continue
        try:
            edge_filtration = float(directed_edge.get("filtration", 0.0) or 0.0)
        except (TypeError, ValueError):
            edge_filtration = 0.0
        payload_directed_edges.append(
            {
                "a": source,
                "b": target,
                "filtration": edge_filtration,
                "type": str(directed_edge.get("edge_type", "decoding_order")),
                "role": str(directed_edge.get("role", "decoding_order")),
                "decoding_step": directed_edge.get("decoding_step"),
                "reasoning_level": directed_edge.get("reasoning_level"),
                "style": "dotted",
                "source_node_id": str(directed_edge.get("source_node_id", "")),
                "target_node_id": str(directed_edge.get("target_node_id", "")),
                "hover": (
                    f"dotted causal/decoding overlay<br>{html.escape(source)} -> {html.escape(target)}"
                    f"<br>role={html.escape(str(directed_edge.get('role', 'decoding_order')))}"
                    f"<br>decoding step={html.escape(str(directed_edge.get('decoding_step', '')))}"
                    f"<br>reasoning level={html.escape(str(directed_edge.get('reasoning_level', '')))}"
                ),
            }
        )
    payload_triangles: list[dict[str, object]] = []
    for tri in triangles[:1200]:
        simplex = [str(v) for v in (tri.get("simplex") or [])[:3]]
        if len(simplex) < 3 or any(label not in visible for label in simplex):
            continue
        payload_triangles.append(
            {
                "vertices": simplex,
                "filtration": float(tri.get("filtration", 0.0) or 0.0),
                "type": str(tri.get("type", "2-simplex")),
            }
        )
    return {
        "layout_kind": layout_kind,
        "vertices": payload_vertices,
        "edges": payload_edges,
        "triangles": payload_triangles,
        "thresholds": threshold_values,
        "filtration_min": min(threshold_values) if threshold_values else 0.0,
        "filtration_max": max(threshold_values) if threshold_values else 0.0,
        "vertex_count": len(payload_vertices),
        "edge_count": len(payload_edges),
        "triangle_count": len(payload_triangles),
        "directed_edge_count": len(payload_directed_edges),
        "directed_edges": payload_directed_edges,
        "source_vertex_count": sum(1 for row in simplices if isinstance(row, dict) and int(row.get("dimension", -1)) == 0),
        "source_edge_count": sum(1 for row in simplices if isinstance(row, dict) and int(row.get("dimension", -1)) == 1),
        "source_triangle_count": sum(1 for row in simplices if isinstance(row, dict) and int(row.get("dimension", -1)) == 2),
        "truncated_for_interactive_panel": source_label_count > len(payload_vertices),
        "vertices_enter_at_zero": radius_vertices_enter_at_zero,
    }


def _simplicial_object_svg(obj: dict[str, object], width: int = 380, height: int = 270, max_vertices: int = 160) -> str:
    simplices = obj.get("simplices", []) if isinstance(obj, dict) else []
    vertices = [s for s in simplices if isinstance(s, dict) and int(s.get("dimension", -1)) == 0]
    edges = [s for s in simplices if isinstance(s, dict) and int(s.get("dimension", -1)) == 1]
    triangles = [s for s in simplices if isinstance(s, dict) and int(s.get("dimension", -1)) == 2]
    labels = [str((s.get("simplex") or [f"v{idx}"])[0]) for idx, s in enumerate(vertices)]
    if not labels:
        summary = obj.get("summary", {}) if isinstance(obj, dict) and isinstance(obj.get("summary"), dict) else {}
        reason = obj.get("reason", "no vertices available") if isinstance(obj, dict) else "no vertices available"
        return (
            f"<svg class='pca-radius-filtered-complex unavailable-complex' viewBox='0 0 {width} {height}' role='img' aria-label='filtered simplicial object unavailable'>"
            "<rect width='100%' height='100%' rx='12' fill='#070a12'/>"
            "<rect x='8' y='8' width='364' height='222' rx='10' fill='#0f172a' stroke='rgba(251,113,133,0.28)'/>"
            "<text x='16' y='34' fill='#fecdd3' font-size='12' font-weight='650'>Filtered simplicial object unavailable</text>"
            f"<text x='16' y='56' fill='#cbd5e1' font-size='9'>{html.escape(str(reason))[:160]}</text>"
            f"<text x='16' y='74' fill='#94a3b8' font-size='8'>filtration_model={html.escape(str(summary.get('filtration_model', 'unknown')))}</text>"
            "</svg>"
        )
    visible_labels = labels[:max_vertices]
    visible = set(visible_labels)
    summary = obj.get("summary", {}) if isinstance(obj, dict) else {}
    thresholds = obj.get("thresholds", []) if isinstance(obj, dict) else []
    coords3, projected, layout_kind = _simplicial_pca3_radius_layout(visible_labels, vertices, edges, width=width, height=height)
    radius_vertices_enter_at_zero = _is_radius_filtration_complex(obj)

    def point(label: str) -> tuple[float, float] | None:
        return projected.get(str(label))

    def depth(label: str) -> float:
        return float(coords3.get(str(label), (0.0, 0.0, 0.0))[2])

    parts = [
        f"<svg class='pca-radius-filtered-complex' viewBox='0 0 {width} {height}' role='img' aria-label='3D PCA radius filtered simplicial object'>",
        "<rect width='100%' height='100%' rx='12' fill='#070a12'/>",
        "<defs>"
        "<filter id='glow'><feGaussianBlur stdDeviation='2.5' result='b'/><feMerge><feMergeNode in='b'/><feMergeNode in='SourceGraphic'/></feMerge></filter>"
        "<linearGradient id='complex-bg' x1='0' x2='1' y1='0' y2='1'><stop offset='0%' stop-color='#0f172a'/><stop offset='100%' stop-color='#020617'/></linearGradient>"
        "</defs>",
        "<rect x='8' y='8' width='364' height='222' rx='10' fill='url(#complex-bg)' stroke='rgba(125,211,252,0.16)'/>",
    ]
    for tri in triangles[:32]:
        simplex = tri.get("simplex", [])
        if any(str(v) not in visible for v in simplex):
            continue
        pts = [point(v) for v in simplex]
        if len(pts) == 3 and all(p is not None for p in pts):
            poly = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in pts if p is not None)
            filt = float(tri.get("filtration", 0.0) or 0.0)
            avg_z = float(np.mean([depth(v) for v in simplex]))
            parts.append(
                f"<polygon class='two-simplex pca-radius-face' data-filtration='{filt:.8f}' data-pca-z='{avg_z:.6f}' points='{poly}' "
                f"fill='rgba(94,234,212,0.08)' stroke='rgba(94,234,212,0.20)' stroke-width='0.8'>"
                f"<title>3D PCA face filtration={filt:.3f} mean_z={avg_z:.3f}</title></polygon>"
            )
    for edge in edges[:160]:
        simplex = edge.get("simplex", [])
        if len(simplex) < 2:
            continue
        if str(simplex[0]) not in visible or str(simplex[1]) not in visible:
            continue
        a = point(simplex[0]); b = point(simplex[1])
        if a is None or b is None:
            continue
        filt = float(edge.get("filtration", 0.0) or 0.0)
        color = _filtration_color(filt)
        edge_type = html.escape(str(edge.get("type", "edge"))[:30])
        avg_z = 0.5 * (depth(simplex[0]) + depth(simplex[1]))
        stroke_width = 0.9 + 1.0 * max(0.0, min(1.0, avg_z))
        parts.append(
            f"<line class='one-simplex pca-radius-edge' data-filtration='{filt:.8f}' data-pca-z='{avg_z:.6f}' x1='{a[0]:.1f}' y1='{a[1]:.1f}' x2='{b[0]:.1f}' y2='{b[1]:.1f}' "
            f"stroke='{color}' stroke-width='{stroke_width:.2f}' stroke-opacity='0.48'><title>{edge_type} radius filtration={filt:.3f} mean_z={avg_z:.3f}</title></line>"
        )
    vertex_by_label = {str((s.get("simplex") or [""])[0]): s for s in vertices}
    radius = 7.2 if len(visible_labels) <= 32 else 5.2
    font_size = 8 if len(visible_labels) <= 32 else 6
    for idx, label in enumerate(visible_labels):
        x, y = projected.get(label, (width / 2.0, height / 2.0))
        filt = 0.0
        vertex = vertex_by_label.get(label, {})
        if vertex:
            filt = 0.0 if radius_vertices_enter_at_zero else float(vertex.get("filtration", 0.0) or 0.0)
        color = _filtration_color(filt)
        z = depth(label)
        r = radius * (0.72 + 0.56 * max(0.0, min(1.0, z)))
        safe = html.escape(label[:18])
        vertex_type = html.escape(str(vertex.get("type", "vertex"))[:30]) if vertex else "vertex"
        parts.append(
            f"<circle class='zero-simplex pca-radius-node' data-filtration='{filt:.8f}' data-pca-z='{z:.6f}' cx='{x:.1f}' cy='{y:.1f}' r='{r:.1f}' fill='{color}' stroke='#e8eef8' "
            f"stroke-width='1.2' filter='url(#glow)'><title>{vertex_type} radius filtration={filt:.3f} PCA=({coords3.get(label, (0.0, 0.0, 0.0))[0]:.3f},{coords3.get(label, (0.0, 0.0, 0.0))[1]:.3f},{z:.3f})</title></circle>"
        )
        if len(visible_labels) <= 18:
            parts.append(f"<text x='{x:.1f}' y='{y + 18:.1f}' text-anchor='middle' fill='#dbe7f4' font-size='{font_size}'>{safe}</text>")
    parts.append("<text x='16' y='24' fill='#dbeafe' font-size='11' font-weight='650'>3D PCA radius-filtered simplicial complex</text>")
    parts.append(
        f"<text x='16' y='40' fill='#9fb3c8' font-size='9'>"
        f"dim0={summary.get('num_vertices', len(vertices))} dim1={summary.get('num_edges', len(edges))} dim2={summary.get('num_two_simplices', len(triangles))}</text>"
    )
    parts.append(
        f"<text x='16' y='54' fill='#9fb3c8' font-size='8'>layout={html.escape(layout_kind)} visible={len(visible_labels)}/{len(labels)}</text>"
    )
    tree = obj.get("simplex_tree", {}) if isinstance(obj, dict) and isinstance(obj.get("simplex_tree"), dict) else {}
    if tree:
        parts.append(
            f"<text x='16' y='66' fill='#99f6e4' font-size='8'>"
            f"filtration source={html.escape(str(tree.get('backend', 'json')))} dim={html.escape(str(tree.get('dimension', '?')))} simplices={html.escape(str(tree.get('num_simplices', '?')))}</text>"
        )
        trunc_y = 78
    else:
        trunc_y = 66
    if len(labels) > len(visible_labels):
        parts.append(
            f"<text x='16' y='{trunc_y}' fill='#fbbf24' font-size='8'>truncated {len(labels) - len(visible_labels)} vertices for legibility</text>"
        )
    parts.extend(_filtration_layer_svg(thresholds, width=width, y=238))
    parts.append(f"<text x='14' y='{height - 10}' fill='#7dd3fc' font-size='9'>thresholds: {html.escape(_json_clip(thresholds[:8], 116))}</text>")
    parts.append("</svg>")
    return "".join(parts)


def _simplicial_pca3_radius_layout(
    labels: list[str],
    vertices: list[dict[str, object]],
    edges: list[dict[str, object]],
    width: int,
    height: int,
) -> tuple[dict[str, tuple[float, float, float]], dict[str, tuple[float, float]], str]:
    if not labels:
        return {}, {}, "empty_3d_pca"
    vertex_by_label = {str((row.get("simplex") or [""])[0]): row for row in vertices if isinstance(row, dict)}
    features, feature_evidence = _vertex_metric_feature_matrix(labels, vertex_by_label)
    if features.shape[0] == 1:
        coords = np.asarray([[0.5, 0.5, 0.5]], dtype=float)
        stats = {"stress": 0.0, "corr": 1.0, "energy3": 1.0}
    else:
        target_distances = _target_simplicial_distance_matrix(labels, features, edges)
        coords, stats = _classical_mds3(target_distances)
        if coords.shape != (features.shape[0], 3) or not np.isfinite(coords).all():
            coords, projection_stats = _feature_pca3_without_synthetic_jitter(features, labels)
            stats = {
                "stress": 1.0,
                "corr": 0.0,
                "energy3": float(projection_stats.get("energy3", 0.0)),
                "projection_note": projection_stats.get("projection_note", "feature_pca3"),
            }
    mins = coords.min(axis=0, keepdims=True)
    spans = np.maximum(coords.max(axis=0, keepdims=True) - mins, 1e-8)
    unit = (coords - mins) / spans
    coords3 = {label: (float(unit[idx, 0]), float(unit[idx, 1]), float(unit[idx, 2])) for idx, label in enumerate(labels)}
    edge_pairs = _edge_pairs_for_labels(edges, set(labels))
    if edge_pairs:
        mean_radius = float(
            np.mean(
                [
                    np.linalg.norm(np.asarray(coords3[a], dtype=float) - np.asarray(coords3[b], dtype=float))
                    for a, b in edge_pairs
                    if a in coords3 and b in coords3
                ]
            )
        )
    else:
        mean_radius = 0.0
    projected: dict[str, tuple[float, float]] = {}
    for label, (x, y, z) in coords3.items():
        sx = 32.0 + x * (width - 104.0) + (z - 0.5) * 52.0
        sy = 52.0 + (1.0 - y) * (height - 108.0) - (z - 0.5) * 44.0
        projected[label] = (float(max(18.0, min(width - 18.0, sx))), float(max(28.0, min(height - 42.0, sy))))
    return (
        coords3,
        projected,
        "3d_pca_radius_projection "
        "method=classical_mds_pcoa "
        f"stress={float(stats.get('stress', 0.0)):.3f} "
        f"corr={float(stats.get('corr', 0.0)):.3f} "
        f"energy3={float(stats.get('energy3', 0.0)):.3f} "
        f"coordinate_evidence={feature_evidence['coordinate_evidence']} "
        f"safe_for_metric_claims={str(feature_evidence['safe_for_metric_claims']).lower()} "
        f"vector_rows={feature_evidence['vector_rows']} "
        f"display_metadata_rows={feature_evidence['display_metadata_rows']} "
        + (f"projection_note={stats.get('projection_note')} " if stats.get("projection_note") else "")
        + f"mean_radius={mean_radius:.3f}",
    )


def _vertex_metric_feature_matrix(labels: list[str], vertex_by_label: dict[str, dict[str, object]]) -> tuple[np.ndarray, dict[str, object]]:
    rows: list[list[float]] = []
    max_len = 0
    vector_rows = 0
    display_metadata_rows = 0
    for idx, label in enumerate(labels):
        vertex = vertex_by_label.get(label, {})
        vector = _coerce_vertex_vector(vertex)
        if not vector:
            vector = _vertex_display_layout_feature(label, vertex, idx, len(labels))
            display_metadata_rows += 1
        else:
            vector_rows += 1
        row = vector
        rows.append(row)
        max_len = max(max_len, len(row))
    padded = np.zeros((len(rows), max_len), dtype=float)
    for idx, row in enumerate(rows):
        if row:
            padded[idx, : len(row)] = np.asarray(row, dtype=float)
    means = np.nanmean(padded, axis=0, keepdims=True)
    padded = np.where(np.isfinite(padded), padded, means)
    scale = np.nanstd(padded, axis=0, keepdims=True)
    scale = np.where(scale > 1e-8, scale, 1.0)
    if vector_rows == len(labels) and labels:
        coordinate_evidence = "real_vertex_vectors"
        safe_for_metric_claims = True
    elif vector_rows:
        coordinate_evidence = "mixed_model_vectors_and_display_metadata_layout"
        safe_for_metric_claims = False
    else:
        coordinate_evidence = "display_only_vertex_metadata_layout"
        safe_for_metric_claims = False
    return (padded - means) / scale, {
        "coordinate_evidence": coordinate_evidence,
        "safe_for_metric_claims": safe_for_metric_claims,
        "vector_rows": vector_rows,
        "display_metadata_rows": display_metadata_rows,
    }


def _feature_pca3_without_synthetic_jitter(features: np.ndarray, labels: list[str]) -> tuple[np.ndarray, dict[str, object]]:
    if features.size == 0:
        return np.zeros((len(labels), 3), dtype=float), {"projection_note": "unavailable_no_vertex_vectors", "energy3": 0.0}
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(labels):
        return np.zeros((len(labels), 3), dtype=float), {"projection_note": "unavailable_invalid_vertex_vector_matrix", "energy3": 0.0}
    values = values - np.mean(values, axis=0, keepdims=True)
    try:
        _u, s, vt = np.linalg.svd(values, full_matrices=False)
        coords = values @ vt[: min(3, vt.shape[0])].T
    except np.linalg.LinAlgError:
        coords = np.zeros((len(labels), 3), dtype=float)
        s = np.asarray([], dtype=float)
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
    if not np.isfinite(coords).all() or float(np.ptp(coords, axis=0).sum()) <= 1e-12:
        coords = np.zeros((len(labels), 3), dtype=float)
        note = "degenerate_vertex_vector_projection"
    else:
        note = "feature_pca3_without_synthetic_jitter"
    energy = float(np.sum(s[:3] ** 2) / max(float(np.sum(s ** 2)), 1e-12)) if s.size else 0.0
    return coords[:, :3], {"projection_note": note, "energy3": energy}


def _coerce_vertex_vector(vertex: dict[str, object], max_dims: int = 128) -> list[float]:
    if not isinstance(vertex, dict):
        return []
    for key in ("embedding", "probability", "embedding_vector", "vector", "features", "feature", "coords", "coordinates", "pca"):
        value = vertex.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            return [float(x) for x in arr[:max_dims]]
    return []


def _target_simplicial_distance_matrix(labels: list[str], features: np.ndarray, edges: list[dict[str, object]]) -> np.ndarray:
    n = len(labels)
    base = _pairwise_euclidean(features)
    if n <= 1:
        return np.zeros((n, n), dtype=float)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    graph = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(graph, 0.0)
    has_metric_edges = False
    base_nonzero = base[base > 1e-8]
    default_edge_length = float(np.median(base_nonzero)) if base_nonzero.size else 1.0
    for edge in edges:
        simplex = edge.get("simplex", []) if isinstance(edge, dict) else []
        if len(simplex) < 2:
            continue
        a = label_to_idx.get(str(simplex[0]))
        b = label_to_idx.get(str(simplex[1]))
        if a is None or b is None or a == b:
            continue
        metric = _edge_metric_distance(edge, default_edge_length)
        graph[a, b] = min(graph[a, b], metric)
        graph[b, a] = min(graph[b, a], metric)
        has_metric_edges = True
    if not has_metric_edges:
        return base
    for k in range(n):
        graph = np.minimum(graph, graph[:, [k]] + graph[[k], :])
    finite = np.isfinite(graph)
    offdiag = ~np.eye(n, dtype=bool)
    finite_offdiag = finite & offdiag
    if not finite_offdiag.any():
        return base
    graph_nonzero = graph[finite_offdiag & (graph > 1e-8)]
    base_matching = base[finite_offdiag & (base > 1e-8)]
    if graph_nonzero.size and base_matching.size:
        graph = graph * (float(np.median(base_matching)) / max(float(np.median(graph_nonzero)), 1e-8))
    target = base.copy()
    target[finite_offdiag] = graph[finite_offdiag]
    target = 0.5 * (target + target.T)
    np.fill_diagonal(target, 0.0)
    return np.maximum(target, 0.0)


def _edge_metric_distance(edge: dict[str, object], default_edge_length: float) -> float:
    for key in ("distance", "metric_distance", "radius", "edge_length", "filtration"):
        value = edge.get(key) if isinstance(edge, dict) else None
        try:
            metric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(metric) and metric > 1e-8:
            return metric
    return max(default_edge_length, 1e-6)


def _pairwise_euclidean(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0, 0), dtype=float)
    diffs = values[:, None, :] - values[None, :, :]
    dist = np.sqrt(np.maximum(np.sum(diffs * diffs, axis=-1), 0.0))
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    return dist


def _classical_mds3(distances: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    distances = np.asarray(distances, dtype=float)
    n = distances.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=float), {"stress": 0.0, "corr": 1.0, "energy3": 1.0}
    distances = np.where(np.isfinite(distances), distances, 0.0)
    distances = np.maximum(0.0, 0.5 * (distances + distances.T))
    np.fill_diagonal(distances, 0.0)
    if n == 1:
        return np.zeros((1, 3), dtype=float), {"stress": 0.0, "corr": 1.0, "energy3": 1.0}
    j = np.eye(n) - np.ones((n, n), dtype=float) / n
    gram = -0.5 * j @ (distances * distances) @ j
    vals, vecs = np.linalg.eigh(0.5 * (gram + gram.T))
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    positive = np.maximum(vals[:3], 0.0)
    coords = vecs[:, :3] * np.sqrt(positive.reshape(1, -1))
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
    embedded = _pairwise_euclidean(coords)
    mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
    target = distances[mask]
    realized = embedded[mask]
    denom = float(np.dot(realized, realized))
    scale = float(np.dot(target, realized) / denom) if denom > 1e-12 else 1.0
    coords = coords * scale
    embedded = embedded * scale
    realized = embedded[mask]
    target_norm = max(float(np.dot(target, target)), 1e-12)
    stress = math.sqrt(float(np.dot(target - realized, target - realized)) / target_norm)
    if target.size >= 2 and float(np.std(target)) > 1e-12 and float(np.std(realized)) > 1e-12:
        corr = float(np.corrcoef(target, realized)[0, 1])
    else:
        corr = 1.0 if stress < 1e-8 else 0.0
    positive_vals = vals[vals > 1e-10]
    energy3 = float(np.sum(np.maximum(vals[:3], 0.0)) / max(float(np.sum(positive_vals)), 1e-12)) if positive_vals.size else 1.0
    return coords[:, :3], {"stress": stress, "corr": corr, "energy3": energy3}


def _vertex_display_layout_feature(label: str, vertex: dict[str, object], idx: int, total: int) -> list[float]:
    text = str(vertex.get("text", "")) if isinstance(vertex, dict) else ""
    filt = float(vertex.get("filtration", 0.0) or 0.0) if isinstance(vertex, dict) else 0.0
    weight = float(vertex.get("weight", 1.0) or 1.0) if isinstance(vertex, dict) else 1.0
    decoded = str(vertex.get("decoded_argmax", "")) if isinstance(vertex, dict) else ""
    input_text = str(vertex.get("input_text", "")) if isinstance(vertex, dict) else ""
    return [
        filt,
        math.log1p(abs(weight)),
        len(text) / 512.0,
        len(decoded) / 2048.0,
        len(input_text) / 4096.0,
    ]


def _edge_pairs_for_labels(edges: list[dict[str, object]], labels: set[str]) -> list[tuple[str, str]]:
    pairs = []
    for edge in edges:
        simplex = edge.get("simplex", []) if isinstance(edge, dict) else []
        if len(simplex) >= 2:
            a, b = str(simplex[0]), str(simplex[1])
            if a in labels and b in labels and a != b:
                pairs.append((a, b))
    return pairs


def _simplicial_layout(
    labels: list[str],
    edges: list[dict[str, object]],
    width: int,
    height: int,
) -> tuple[dict[str, tuple[float, float]], str]:
    if not labels:
        return {}, "empty"
    edge_pairs = []
    label_set = set(labels)
    for edge in edges:
        simplex = edge.get("simplex", []) if isinstance(edge, dict) else []
        if len(simplex) >= 2:
            a, b = str(simplex[0]), str(simplex[1])
            if a in label_set and b in label_set and a != b:
                edge_pairs.append((a, b))
    try:
        coords, kind = _dag_layer_layout(labels, edge_pairs)
        if coords:
            return _normalize_layout(coords, width, height), kind
    except Exception:
        pass
    try:
        import networkx as nx  # type: ignore

        graph = nx.Graph()
        graph.add_nodes_from(labels)
        graph.add_edges_from(edge_pairs)
        if graph.number_of_edges() > 0:
            pos = nx.spring_layout(graph, seed=17, iterations=80, weight=None)
            return _normalize_layout({str(k): (float(v[0]), float(v[1])) for k, v in pos.items()}, width, height), "spring_1_skeleton"
    except Exception:
        pass
    radius = max(min(width, height) * 0.30, 42.0)
    cx, cy = width / 2.0, height / 2.0 - 4.0
    coords = {}
    for idx, label in enumerate(labels):
        angle = -np.pi / 2 + 2 * np.pi * idx / max(len(labels), 1)
        coords[label] = (cx + radius * np.cos(angle), cy + radius * np.sin(angle))
    return coords, "radial_layout_no_metric_geometry"


def _dag_layer_layout(labels: list[str], edge_pairs: list[tuple[str, str]]) -> tuple[dict[str, tuple[float, float]], str]:
    if not edge_pairs:
        return {}, ""
    outgoing: dict[str, list[str]] = defaultdict(list)
    indegree = {label: 0 for label in labels}
    for a, b in edge_pairs:
        outgoing[a].append(b)
        indegree[b] = indegree.get(b, 0) + 1
        indegree.setdefault(a, 0)
    roots = [label for label in labels if indegree.get(label, 0) == 0]
    if not roots:
        return {}, ""
    depth = {label: 0 for label in roots}
    queue: deque[str] = deque(roots)
    seen_count = 0
    while queue:
        node = queue.popleft()
        seen_count += 1
        for child in outgoing.get(node, []):
            indegree[child] -= 1
            depth[child] = max(depth.get(child, 0), depth[node] + 1)
            if indegree[child] == 0:
                queue.append(child)
    if seen_count < max(2, int(0.75 * len(labels))):
        return {}, ""
    layers: dict[int, list[str]] = defaultdict(list)
    for label in labels:
        layers[int(depth.get(label, 0))].append(label)
    coords = {}
    layer_keys = sorted(layers)
    max_layer = max(layer_keys) if layer_keys else 1
    for layer in layer_keys:
        rows = layers[layer]
        for idx, label in enumerate(rows):
            x = layer / max(max_layer, 1)
            y = 0.5 if len(rows) == 1 else idx / (len(rows) - 1)
            coords[label] = (x, y)
    edge_count = len(edge_pairs)
    max_layer_width = max(len(rows) for rows in layers.values())
    path_like = edge_count >= len(labels) - 1 and max_layer_width <= max(3, int(math.sqrt(len(labels)) + 1))
    if path_like and len(labels) > 36 and max_layer_width <= 2:
        ordered_labels = [label for layer in layer_keys for label in layers[layer]]
        return _wrapped_path_layout(ordered_labels), "wrapped_topological_path"
    return coords, "topological_path_dag" if path_like else "topological_dag_layers"


def _wrapped_path_layout(labels: list[str]) -> dict[str, tuple[float, float]]:
    if not labels:
        return {}
    cols = max(8, min(18, int(math.ceil(math.sqrt(len(labels) * 2.2)))))
    rows = int(math.ceil(len(labels) / cols))
    coords: dict[str, tuple[float, float]] = {}
    for idx, label in enumerate(labels):
        row = idx // cols
        col = idx % cols
        if row % 2 == 1:
            col = cols - 1 - col
        x = 0.5 if cols <= 1 else col / (cols - 1)
        y = 0.5 if rows <= 1 else row / (rows - 1)
        coords[label] = (x, y)
    return coords


def _normalize_layout(coords: dict[str, tuple[float, float]], width: int, height: int) -> dict[str, tuple[float, float]]:
    xs = np.asarray([xy[0] for xy in coords.values()], dtype=float)
    ys = np.asarray([xy[1] for xy in coords.values()], dtype=float)
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    pad_x, pad_top, pad_bottom = 28.0, 72.0, 48.0
    usable_w = max(width - 2 * pad_x, 1.0)
    usable_h = max(height - pad_top - pad_bottom, 1.0)
    x_span = max(max_x - min_x, 1e-9)
    y_span = max(max_y - min_y, 1e-9)
    out = {}
    for label, (x, y) in coords.items():
        nx = pad_x + ((float(x) - min_x) / x_span if x_span > 1e-9 else 0.5) * usable_w
        ny = pad_top + ((float(y) - min_y) / y_span if y_span > 1e-9 else 0.5) * usable_h
        out[label] = (nx, ny)
    return out


def _filtration_layer_svg(thresholds: object, width: int, y: int) -> list[str]:
    raw_thresholds = thresholds if isinstance(thresholds, (list, tuple)) else []
    values = [float(v) for v in raw_thresholds if isinstance(v, (int, float))]
    if not values:
        values = [0.0]
    values = sorted(values[:14])
    x0 = 14.0
    bar_width = max(width - 28.0, 1.0)
    parts = [
        f"<g class='filtration-layer' aria-label='filtration layers'>",
        f"<line x1='{x0:.1f}' y1='{y:.1f}' x2='{x0 + bar_width:.1f}' y2='{y:.1f}' stroke='rgba(148,163,184,0.34)' stroke-width='2'/>",
    ]
    max_value = max(max(values), 1e-9)
    for idx, value in enumerate(values):
        frac = 0.0 if max_value <= 1e-9 else max(0.0, min(1.0, value / max_value))
        x = x0 + frac * bar_width
        color = _filtration_color(frac)
        height = 10 + 3 * (idx % 3)
        parts.append(
            f"<rect x='{x - 2.0:.1f}' y='{y - height:.1f}' width='4.0' height='{height:.1f}' rx='1.4' "
            f"fill='{color}' opacity='0.92'><title>filtration threshold {value:.3f}</title></rect>"
        )
    parts.append("<text x='14' y='256' fill='#9fb3c8' font-size='8'>filtration layers</text>")
    parts.append("</g>")
    return parts


def _filtration_color(value: float) -> str:
    palette = ["#38bdf8", "#5eead4", "#a3e635", "#facc15", "#fb7185"]
    idx = int(np.clip(round(value * (len(palette) - 1)), 0, len(palette) - 1))
    return palette[idx]


def _nll_surface_trace(
    x_values: np.ndarray,
    y_values: np.ndarray,
    nll_values: np.ndarray,
    z_values: np.ndarray | None = None,
    mode: str = "floor",
    name: str = "Smoothed NLL surface",
    grid_size: int = 32,
) -> tuple[go.BaseTraceType | None, dict[str, object]]:
    x = np.asarray(x_values, dtype=float).reshape(-1)
    y = np.asarray(y_values, dtype=float).reshape(-1)
    nll = np.asarray(nll_values, dtype=float).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(nll)
    x = x[finite]; y = y[finite]; nll = nll[finite]
    if z_values is not None:
        z_arr = np.asarray(z_values, dtype=float).reshape(-1)[finite]
    else:
        z_arr = nll
    z_arr = z_arr[np.isfinite(z_arr)]
    if x.size == 0 or y.size == 0 or nll.size == 0:
        return None, {"available": False, "reason": "no finite NLL points"}
    if mode in {"nll_height", "embedding_height"}:
        point_z = nll if mode == "nll_height" else (z_arr if z_arr.size == x.size else nll)
        mesh, mesh_meta = _smooth_anchored_surface(
            x,
            y,
            point_z,
            nll,
            name=name,
            mode=mode,
        )
        if mesh is not None:
            return mesh, mesh_meta

    x_span = float(np.ptp(x))
    y_span = float(np.ptp(y))
    pad_x = max(x_span * 0.08, 0.25)
    pad_y = max(y_span * 0.08, 0.25)
    if x_span < 1e-9:
        pad_x = 1.0
    if y_span < 1e-9:
        pad_y = 1.0
    xi = np.linspace(float(x.min() - pad_x), float(x.max() + pad_x), grid_size)
    yi = np.linspace(float(y.min() - pad_y), float(y.max() + pad_y), grid_size)
    grid_x, grid_y = np.meshgrid(xi, yi)
    field = _smooth_idw_field(x, y, nll, grid_x, grid_y)
    if mode == "nll_height":
        surface_z = field
        opacity = 0.44
    else:
        z_reference = z_arr if z_arr.size else nll
        z_floor = float(np.nanmin(z_reference) - max(float(np.nanstd(z_reference)), 1e-3) * 0.35 - 1e-3)
        surface_z = np.full_like(field, z_floor)
        opacity = 0.36
    trace = go.Surface(
        x=grid_x,
        y=grid_y,
        z=surface_z,
        surfacecolor=field,
        colorscale="Plasma",
        opacity=opacity,
        showscale=False,
        name=name,
        hovertemplate="PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>smoothed NLL=%{surfacecolor:.4f}<extra></extra>",
        contours=dict(z=dict(show=False)),
    )
    return trace, {
        "available": True,
        "mode": mode,
        "grid_size": grid_size,
        "point_count": int(x.size),
        "nll_min": float(np.nanmin(nll)),
        "nll_max": float(np.nanmax(nll)),
        "nll_mean": float(np.nanmean(nll)),
        "smoothing": "inverse_distance_weighted_three_pass_neighbor_average",
        "max_point_residual": None,
    }


def _project_points_to_nll_surface_z(
    x_values: np.ndarray,
    y_values: np.ndarray,
    raw_z_values: np.ndarray,
    decimals: int = 10,
) -> tuple[np.ndarray, dict[str, object]]:
    """Project marker z values onto the same single-valued surface used for display."""

    x = np.asarray(x_values, dtype=float).reshape(-1)
    y = np.asarray(y_values, dtype=float).reshape(-1)
    raw_z = np.asarray(raw_z_values, dtype=float).reshape(-1)
    projected = raw_z.copy()
    groups: dict[tuple[float, float], list[int]] = {}
    for idx, (px, py, pz) in enumerate(zip(x, y, raw_z)):
        if not (np.isfinite(px) and np.isfinite(py) and np.isfinite(pz)):
            continue
        groups.setdefault((round(float(px), decimals), round(float(py), decimals)), []).append(idx)
    duplicate_groups = 0
    max_multiplicity = 1
    max_raw_z_spread = 0.0
    for members in groups.values():
        if len(members) <= 1:
            continue
        duplicate_groups += 1
        max_multiplicity = max(max_multiplicity, len(members))
        values = raw_z[np.asarray(members, dtype=int)]
        surface_z = float(np.nanmean(values))
        max_raw_z_spread = max(max_raw_z_spread, float(np.nanmax(values) - np.nanmin(values)))
        projected[np.asarray(members, dtype=int)] = surface_z
    residual = np.abs(projected - raw_z)
    residual = residual[np.isfinite(residual)]
    return projected, {
        "surface_z_policy": "single-valued display surface; duplicate PCA coordinates share the mean centered-scaled NLL surface height",
        "duplicate_xy_group_count": int(duplicate_groups),
        "max_duplicate_xy_multiplicity": int(max_multiplicity),
        "max_duplicate_xy_raw_centered_z_spread": float(max_raw_z_spread),
        "max_raw_to_surface_z_delta": float(np.nanmax(residual)) if residual.size else 0.0,
    }


def _nll_triangulated_surface_trace(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    nll_values: np.ndarray,
    name: str,
) -> tuple[go.BaseTraceType | None, dict[str, object]]:
    x = np.asarray(x_values, dtype=float).reshape(-1)
    y = np.asarray(y_values, dtype=float).reshape(-1)
    z = np.asarray(z_values, dtype=float).reshape(-1)
    nll = np.asarray(nll_values, dtype=float).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(nll)
    x = x[finite]
    y = y[finite]
    z = z[finite]
    nll = nll[finite]
    if x.size == 0:
        return None, {"available": False, "reason": "no finite trajectory NLL points"}
    source_count = int(np.asarray(x_values).reshape(-1).size)
    x, y, z, nll, duplicate_meta = _collapse_duplicate_xy_for_surface(x, y, z, nll)
    trace, meta = _smooth_anchored_surface(x, y, z, nll, name=name, mode="nll_height")
    meta.update(duplicate_meta)
    meta["source_point_count_before_duplicate_collapse"] = source_count
    meta["unique_xy_point_count"] = int(x.size)
    meta["duplicate_xy_points_removed"] = int(max(source_count - x.size, 0))
    return trace, meta


def _nll_local_interpolating_sheet_trace(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    nll_values: np.ndarray,
    name: str,
) -> tuple[go.BaseTraceType | None, dict[str, object]]:
    x = np.asarray(x_values, dtype=float).reshape(-1)
    y = np.asarray(y_values, dtype=float).reshape(-1)
    z = np.asarray(z_values, dtype=float).reshape(-1)
    nll = np.asarray(nll_values, dtype=float).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(nll)
    x = x[finite]
    y = y[finite]
    z = z[finite]
    nll = nll[finite]
    if x.size < 3 or _xy_rank(x, y) < 2:
        return None, {"available": False, "reason": "insufficient non-collinear NLL anchors for local interpolating sheet"}
    source_count = int(x.size)
    x, y, z, nll, duplicate_meta = _collapse_duplicate_xy_for_surface(x, y, z, nll)
    if x.size < 3 or _xy_rank(x, y) < 2:
        return None, {"available": False, "reason": "duplicate collapse left insufficient local-sheet anchors"}
    grid_x, grid_y, z_grid, nll_grid, residual, support_meta = _supported_local_idw_grid(
        x,
        y,
        z,
        nll,
        grid_size=88,
    )
    finite_z = z_grid[np.isfinite(z_grid)]
    if finite_z.size == 0:
        return None, {"available": False, "reason": "local interpolating support grid is empty"}
    trace = go.Surface(
        x=grid_x,
        y=grid_y,
        z=z_grid,
        surfacecolor=nll_grid,
        colorscale=[
            [0.0, "#0f172a"],
            [0.18, "#1d4ed8"],
            [0.42, "#06b6d4"],
            [0.68, "#bef264"],
            [1.0, "#f97316"],
        ],
        opacity=0.58,
        showscale=False,
        name=name,
        hovertemplate=(
            "local interpolating NLL sheet<br>"
            "PC1=%{x:.3f}<br>"
            "PC2=%{y:.3f}<br>"
            "projected surface z=%{z:.4f}<br>"
            "local raw NLL=%{surfacecolor:.6f}<br>"
            "visible support is restricted to neighborhoods of model-evaluated GoT states<extra></extra>"
        ),
        contours=dict(z=dict(show=True, color="rgba(219,234,254,0.22)", width=1)),
        lighting=dict(ambient=0.80, diffuse=0.32, specular=0.08, roughness=0.88),
        showlegend=True,
    )
    meta = {
        "available": True,
        "surface_kind": "local_interpolating_nll_sheet",
        "source_point_count_before_duplicate_collapse": source_count,
        "point_count": source_count,
        "unique_xy_point_count": int(x.size),
        "duplicate_xy_points_removed": int(max(source_count - x.size, 0)),
        "max_point_residual": float(residual),
        "touches_points": bool(float(residual) <= 1e-8),
        "interpolation": "Observed-state inverse-distance interpolation through model-evaluated GoT state anchors.",
        "support_policy": support_meta.get("support_policy", "union of local observed-state neighborhoods"),
        "support_radius": float(support_meta.get("support_radius", 0.0)),
        "masked_fraction": float(support_meta.get("masked_fraction", 0.0)),
        "provenance": "computed from observed model-evaluated GoT state NLL anchors only; it is not a dense model forward pass over latent space",
    }
    meta.update(
        {
            "duplicate_xy_group_count": int(duplicate_meta.get("duplicate_xy_group_count", 0)),
            "max_duplicate_xy_multiplicity": int(duplicate_meta.get("max_duplicate_xy_multiplicity", 1)),
            "max_duplicate_xy_centered_z_spread": float(duplicate_meta.get("max_duplicate_xy_centered_z_spread", 0.0)),
            "max_duplicate_xy_raw_nll_spread": float(duplicate_meta.get("max_duplicate_xy_raw_nll_spread", 0.0)),
        }
    )
    return trace, meta


def _nll_support_footprint_trace(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    nll_values: np.ndarray,
    name: str,
) -> tuple[go.BaseTraceType | None, dict[str, object]]:
    x = np.asarray(x_values, dtype=float).reshape(-1)
    y = np.asarray(y_values, dtype=float).reshape(-1)
    z = np.asarray(z_values, dtype=float).reshape(-1)
    nll = np.asarray(nll_values, dtype=float).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(nll)
    x = x[finite]
    y = y[finite]
    z = z[finite]
    nll = nll[finite]
    if x.size < 3 or _xy_rank(x, y) < 2:
        return None, {"available": False, "reason": "insufficient non-collinear NLL anchors"}
    source_count = int(x.size)
    x, y, z, nll, duplicate_meta = _collapse_duplicate_xy_for_surface(x, y, z, nll)
    grid_x, grid_y, z_grid, nll_grid, _, support_meta = _supported_local_idw_grid(x, y, z, nll, grid_size=72)
    finite_z = z_grid[np.isfinite(z_grid)]
    if finite_z.size == 0:
        return None, {"available": False, "reason": "support grid is empty"}
    z_range = float(np.nanmax(z) - np.nanmin(z)) if z.size else 0.0
    floor_z = float(np.nanmin(z) - max(z_range * 0.22, 0.42))
    footprint_z = np.where(np.isfinite(z_grid), floor_z, np.nan)
    trace = go.Surface(
        x=grid_x,
        y=grid_y,
        z=footprint_z,
        surfacecolor=nll_grid,
        colorscale="Plasma",
        opacity=0.14,
        showscale=False,
        name=name,
        hovertemplate=(
            "local support footprint<br>"
            "PC1=%{x:.3f}<br>"
            "PC2=%{y:.3f}<br>"
            "projected z=%{z:.3f}<br>"
            "local smoothed raw NLL=%{surfacecolor:.6f}<br>"
            "visible only near observed reasoning states<extra></extra>"
        ),
        contours=dict(z=dict(show=False)),
        lighting=dict(ambient=0.76, diffuse=0.34, specular=0.08, roughness=0.88),
        showlegend=True,
    )
    meta = {
        "available": True,
        "surface_kind": "projected_local_support_footprint",
        "source_point_count_before_duplicate_collapse": source_count,
        "unique_xy_point_count": int(x.size),
        "floor_z": floor_z,
        "purpose": "Shows the area around actual PCA anchors where the local NLL field is evaluated; not an additional model prediction layer.",
        "masked_fraction": float(support_meta.get("masked_fraction", 0.0)),
        "support_radius": float(support_meta.get("support_radius", 0.0)),
        "support_policy": support_meta.get("support_policy", "union of local sample neighborhoods"),
    }
    meta.update(
        {
            "duplicate_xy_group_count": int(duplicate_meta.get("duplicate_xy_group_count", 0)),
            "max_duplicate_xy_multiplicity": int(duplicate_meta.get("max_duplicate_xy_multiplicity", 1)),
        }
    )
    return trace, meta


def _nll_visual_scale(nll_values: np.ndarray, target_range: float = 4.0) -> float:
    values = np.asarray(nll_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    raw_range = float(np.nanmax(values) - np.nanmin(values))
    if raw_range <= 1e-12:
        return 1.0
    return float(np.clip(target_range / raw_range, 1.0, 50.0))


def _collapse_duplicate_xy_for_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    nll: np.ndarray,
    decimals: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    groups: dict[tuple[float, float], list[int]] = {}
    for idx, (px, py) in enumerate(zip(x, y)):
        groups.setdefault((round(float(px), decimals), round(float(py), decimals)), []).append(idx)
    ux: list[float] = []
    uy: list[float] = []
    uz: list[float] = []
    unll: list[float] = []
    counts: list[int] = []
    z_spreads: list[float] = []
    nll_spreads: list[float] = []
    for members in groups.values():
        member_idx = np.asarray(members, dtype=int)
        ux.append(float(np.nanmean(x[member_idx])))
        uy.append(float(np.nanmean(y[member_idx])))
        uz.append(float(np.nanmean(z[member_idx])))
        unll.append(float(np.nanmean(nll[member_idx])))
        counts.append(int(member_idx.size))
        z_spreads.append(float(np.nanmax(z[member_idx]) - np.nanmin(z[member_idx])) if member_idx.size else 0.0)
        nll_spreads.append(float(np.nanmax(nll[member_idx]) - np.nanmin(nll[member_idx])) if member_idx.size else 0.0)
    duplicate_groups = [count for count in counts if count > 1]
    return (
        np.asarray(ux, dtype=float),
        np.asarray(uy, dtype=float),
        np.asarray(uz, dtype=float),
        np.asarray(unll, dtype=float),
        {
            "duplicate_xy_group_count": int(len(duplicate_groups)),
            "max_duplicate_xy_multiplicity": int(max(duplicate_groups) if duplicate_groups else 1),
            "max_duplicate_xy_centered_z_spread": float(max(z_spreads) if z_spreads else 0.0),
            "max_duplicate_xy_raw_nll_spread": float(max(nll_spreads) if nll_spreads else 0.0),
            "duplicate_xy_surface_policy": "surface uses mean centered NLL/raw NLL at duplicate PCA coordinates; all original states remain visible as anchor markers",
        },
    )


def _nll_anchor_trace(x: np.ndarray, y: np.ndarray, z: np.ndarray, nll: np.ndarray, name: str) -> go.Scatter3d:
    return go.Scatter3d(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        z=np.asarray(z, dtype=float),
        mode="markers",
        marker=dict(
            size=4,
            color=np.asarray(nll, dtype=float),
            colorscale="Plasma",
            showscale=False,
            symbol="diamond",
            line=dict(width=2, color="#f8fafc"),
        ),
        name=name,
        customdata=np.asarray(nll, dtype=float),
        hovertemplate=(
            "surface anchor<br>"
            "PC1=%{x:.3f}<br>"
            "PC2=%{y:.3f}<br>"
            "projected surface z=%{z:.4f}<br>"
            "raw NLL=%{customdata:.6f}<extra></extra>"
        ),
        showlegend=False,
    )


def _smooth_anchored_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    nll: np.ndarray,
    name: str,
    mode: str,
) -> tuple[go.BaseTraceType | None, dict[str, object]]:
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(nll)
    x = x[finite]
    y = y[finite]
    z = z[finite]
    nll = nll[finite]
    if x.size == 0:
        return None, {"available": False, "reason": "no finite interpolation points", "mode": mode}
    if x.size < 3 or _xy_rank(x, y) < 2:
        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines+markers",
            marker=dict(size=4, color=nll, colorscale="Plasma", showscale=False),
            line=dict(color="rgba(251,191,36,0.42)", width=5),
            name=name,
            hovertemplate="PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>centered scaled NLL=%{z:.4f}<extra></extra>",
        )
        return trace, {
            "available": True,
            "mode": mode,
            "surface_kind": "degenerate_interpolating_polyline",
            "point_count": int(x.size),
            "nll_min": float(np.nanmin(nll)),
            "nll_max": float(np.nanmax(nll)),
            "nll_mean": float(np.nanmean(nll)),
            "max_point_residual": 0.0,
            "touches_points": True,
        }
    if x.size == 3:
        order = _angle_order(x, y)
        trace = go.Mesh3d(
            x=x[order],
            y=y[order],
            z=z[order],
            i=[0],
            j=[1],
            k=[2],
            intensity=nll[order],
            colorscale="Plasma",
            opacity=0.55,
            showscale=False,
            name=name,
            hovertemplate="PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>projected surface z=%{z:.4f}<br>raw NLL=%{intensity:.6f}<extra></extra>",
        )
        return trace, {
            "available": True,
            "mode": mode,
            "surface_kind": "sparse_exact_triangular_nll_mesh",
            "actual_landscape_layer": False,
            "sparse_observed_anchor_layer": True,
            "dense_model_evaluated_field": False,
            "interpolation": "Exact triangular interpolant through the three observed PCA states",
            "point_count": int(x.size),
            "nll_min": float(np.nanmin(nll)),
            "nll_max": float(np.nanmax(nll)),
            "nll_mean": float(np.nanmean(nll)),
            "max_point_residual": 0.0,
            "touches_points": True,
        }
    _, _, _, _, residual, support_meta = _supported_local_idw_grid(x, y, z, nll)
    faces = _triangulate_xy_faces(x, y)
    mesh_kwargs: dict[str, object] = {}
    if faces:
        mesh_kwargs.update(
            {
                "i": [int(face[0]) for face in faces],
                "j": [int(face[1]) for face in faces],
                "k": [int(face[2]) for face in faces],
            }
        )
    else:
        mesh_kwargs["alphahull"] = 0
    surface = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        intensity=nll,
        colorscale="Plasma",
        opacity=0.52,
        showscale=False,
        name=name,
        hovertemplate=(
            "sparse observed-state NLL anchor mesh<br>"
            "PC1=%{x:.3f}<br>"
            "PC2=%{y:.3f}<br>"
            "projected surface z=%{z:.4f}<br>"
            "raw NLL=%{intensity:.6f}<br>"
            "piecewise-linear triangles use only observed model-evaluated GoT states<extra></extra>"
        ),
        flatshading=True,
        lighting=dict(ambient=0.82, diffuse=0.26, specular=0.05, roughness=0.92),
        **mesh_kwargs,
    )
    return surface, {
        "available": True,
        "mode": mode,
        "surface_kind": "sparse_observed_state_nll_anchor_mesh",
        "actual_landscape_layer": False,
        "sparse_observed_anchor_layer": True,
        "dense_model_evaluated_field": False,
        "interpolation": "Exact piecewise-linear triangulation through observed PCA states; no synthetic anchor values are introduced in this layer",
        "truthfulness_warning": "This is a sparse observed-state anchor mesh, not a dense model-evaluated NLL/fitness field.",
        "provenance": "computed only from observed model-evaluated GoT state embeddings and their measured raw NLL values",
        "point_count": int(x.size),
        "hull_masked_fraction": float(support_meta.get("masked_fraction", 0.0)),
        "support_radius": float(support_meta.get("support_radius", 0.0)),
        "support_policy": support_meta.get("support_policy", "union of local observed-state neighborhoods"),
        "value_clipping": "none; lifted mesh uses observed anchor values exactly",
        "nll_min": float(np.nanmin(nll)),
        "nll_max": float(np.nanmax(nll)),
        "nll_mean": float(np.nanmean(nll)),
        "smooth_surface_max_point_residual": float(residual),
        "max_point_residual": 0.0,
        "touches_points": True,
        "exact_anchor_layer": True,
        "triangulated_face_count": int(len(faces)),
    }


def _smooth_exact_rbf_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    nll: np.ndarray,
    grid_size: int = 52,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    x_span = float(np.ptp(x))
    y_span = float(np.ptp(y))
    pad_x = max(x_span * 0.25, 0.30)
    pad_y = max(y_span * 0.25, 0.30)
    xi = _axis_with_points(float(x.min() - pad_x), float(x.max() + pad_x), x, grid_size)
    yi = _axis_with_points(float(y.min() - pad_y), float(y.max() + pad_y), y, grid_size)
    grid_x, grid_y = np.meshgrid(xi, yi)
    points = np.column_stack([x, y])
    z_model = _fit_gaussian_rbf(points, z)
    nll_model = _fit_gaussian_rbf(points, nll)
    grid_points = np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)])
    z_grid = _eval_gaussian_rbf(z_model, grid_points).reshape(grid_x.shape)
    nll_grid = _eval_gaussian_rbf(nll_model, grid_points).reshape(grid_x.shape)
    z_at_points = _eval_gaussian_rbf(z_model, points)
    residual = float(np.nanmax(np.abs(z_at_points - z))) if z.size else 0.0
    return grid_x, grid_y, z_grid, nll_grid, residual


def _supported_local_idw_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    nll: np.ndarray,
    grid_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, dict[str, object]]:
    x_span = float(np.ptp(x))
    y_span = float(np.ptp(y))
    span = max(x_span, y_span, 1e-9)
    pad_x = x_span * 0.14 if x_span > 1e-9 else span * 0.2 + 0.05
    pad_y = y_span * 0.14 if y_span > 1e-9 else span * 0.2 + 0.05
    xi = _axis_with_points(float(x.min() - pad_x), float(x.max() + pad_x), x, grid_size)
    yi = _axis_with_points(float(y.min() - pad_y), float(y.max() + pad_y), y, grid_size)
    grid_x, grid_y = np.meshgrid(xi, yi)
    points = np.column_stack([x, y])
    grid_points = np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)])
    positive = _positive_pairwise_distances(points)
    if positive.size:
        support_radius = float(max(np.quantile(positive, 0.25) * 2.8, np.median(positive) * 1.15, span * 0.10))
        sigma = float(max(np.quantile(positive, 0.35), support_radius / 2.35, 1e-6))
    else:
        support_radius = float(span * 0.32 + 1e-6)
        sigma = float(max(support_radius / 2.0, 1e-6))
    z_flat, min_dist = _local_idw_interpolate(points, z, grid_points, sigma=sigma)
    nll_flat, _ = _local_idw_interpolate(points, nll, grid_points, sigma=sigma)
    support_mask = min_dist <= support_radius
    z_grid = z_flat.reshape(grid_x.shape)
    nll_grid = nll_flat.reshape(grid_x.shape)
    support_grid = support_mask.reshape(grid_x.shape)
    z_grid = np.where(support_grid, z_grid, np.nan)
    nll_grid = np.where(support_grid, nll_grid, np.nan)
    residual = 0.0
    if points.size:
        z_at_points, _ = _local_idw_interpolate(points, z, points, sigma=sigma)
        residual = float(np.nanmax(np.abs(z_at_points - z))) if z.size else 0.0
    return grid_x, grid_y, z_grid, nll_grid, residual, {
        "masked_fraction": float(1.0 - np.mean(support_mask)) if support_mask.size else 0.0,
        "support_radius": support_radius,
        "idw_sigma": sigma,
        "support_policy": "grid cells are visible only when their nearest observed PCA state is within the learned local-neighborhood radius",
    }


def _positive_pairwise_distances(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 2:
        return np.asarray([], dtype=float)
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    return dist[dist > 1e-10]


def _triangulate_xy_faces(x: np.ndarray, y: np.ndarray) -> list[tuple[int, int, int]]:
    points = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    if points.shape[0] < 3 or _xy_rank(points[:, 0], points[:, 1]) < 2:
        return []
    try:
        from scipy.spatial import Delaunay  # type: ignore

        tri = Delaunay(points)
        faces = []
        for simplex in tri.simplices:
            a, b, c = [int(v) for v in simplex]
            if len({a, b, c}) == 3:
                faces.append((a, b, c))
        return faces
    except Exception:
        order = _angle_order(points[:, 0], points[:, 1])
        if order.size < 3:
            return []
        anchor = int(order[0])
        faces = []
        for pos in range(1, int(order.size) - 1):
            a, b, c = anchor, int(order[pos]), int(order[pos + 1])
            if len({a, b, c}) == 3:
                faces.append((a, b, c))
        return faces


def _local_idw_interpolate(
    points: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
    sigma: float,
    power: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    diff = query[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    min_dist = np.min(dist, axis=1) if dist.size else np.full(query.shape[0], np.inf)
    exact = dist <= 1e-12
    out = np.empty(query.shape[0], dtype=float)
    if np.any(exact):
        exact_rows = np.where(np.any(exact, axis=1))[0]
        for row in exact_rows:
            out[row] = float(np.mean(values[exact[row]]))
    non_exact_mask = ~np.any(exact, axis=1)
    if np.any(non_exact_mask):
        d = dist[non_exact_mask]
        gaussian = np.exp(-(d * d) / (2.0 * sigma * sigma))
        weights = gaussian / np.maximum(d, 1e-9) ** power
        weights_sum = np.sum(weights, axis=1)
        out[non_exact_mask] = (weights @ values) / np.maximum(weights_sum, 1e-12)
    return out, min_dist


def _clip_interpolation_grid(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return grid
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    span = max(hi - lo, 1e-9)
    pad = max(span * 0.08, 1e-6)
    return np.clip(grid, lo - pad, hi + pad)


def _axis_with_points(lo: float, hi: float, points: np.ndarray, grid_size: int) -> np.ndarray:
    base = np.linspace(lo, hi, grid_size)
    merged = np.concatenate([base, np.asarray(points, dtype=float).reshape(-1)])
    return np.asarray(sorted({round(float(v), 12) for v in merged}), dtype=float)


def _fit_gaussian_rbf(points: np.ndarray, values: np.ndarray) -> dict[str, np.ndarray | float]:
    diff = points[:, None, :] - points[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    positive = dist2[dist2 > 1e-14]
    scale = float(np.sqrt(np.median(positive))) if positive.size else 1.0
    epsilon = max(scale, 1e-6)
    kernel = np.exp(-dist2 / (2.0 * epsilon * epsilon))
    kernel += np.eye(kernel.shape[0]) * 1e-10
    try:
        weights = np.linalg.solve(kernel, values)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(kernel, values, rcond=None)[0]
    return {"points": points, "weights": weights, "epsilon": epsilon}


def _eval_gaussian_rbf(model: dict[str, np.ndarray | float], query: np.ndarray) -> np.ndarray:
    points = np.asarray(model["points"], dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    epsilon = float(model["epsilon"])
    diff = query[:, None, :] - points[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    kernel = np.exp(-dist2 / (2.0 * epsilon * epsilon))
    return kernel @ weights


def _xy_rank(x: np.ndarray, y: np.ndarray) -> int:
    centered = np.column_stack([x - np.mean(x), y - np.mean(y)])
    if centered.shape[0] == 0:
        return 0
    return int(np.linalg.matrix_rank(centered, tol=1e-8))


def _angle_order(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    cx = float(np.mean(x))
    cy = float(np.mean(y))
    return np.argsort(np.arctan2(y - cy, x - cx))


def _convex_hull(points: np.ndarray) -> np.ndarray:
    unique = sorted({(round(float(px), 12), round(float(py), 12)) for px, py in np.asarray(points, dtype=float)})
    if len(unique) <= 2:
        return np.asarray(unique, dtype=float)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1e-12:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1e-12:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _points_in_convex_polygon(points: np.ndarray, polygon: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    if polygon.shape[0] < 3:
        return np.ones(points.shape[0], dtype=bool)
    out = np.ones(points.shape[0], dtype=bool)
    orientation = 1.0
    signed_area = 0.0
    for idx in range(polygon.shape[0]):
        x1, y1 = polygon[idx]
        x2, y2 = polygon[(idx + 1) % polygon.shape[0]]
        signed_area += x1 * y2 - x2 * y1
    if signed_area < 0:
        orientation = -1.0
    for idx in range(polygon.shape[0]):
        a = polygon[idx]
        b = polygon[(idx + 1) % polygon.shape[0]]
        edge = b - a
        rel = points - a
        cross = edge[0] * rel[:, 1] - edge[1] * rel[:, 0]
        out &= orientation * cross >= -tol
    return out


def _smooth_idw_field(x: np.ndarray, y: np.ndarray, values: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    dx = grid_x[..., None] - x[None, None, :]
    dy = grid_y[..., None] - y[None, None, :]
    dist2 = dx * dx + dy * dy
    scale = max(float(np.nanmedian(dist2[dist2 > 0.0])) if np.any(dist2 > 0.0) else 1.0, 1e-6)
    weights = 1.0 / (dist2 + 0.015 * scale)
    field = (weights * values[None, None, :]).sum(axis=-1) / weights.sum(axis=-1).clip(min=1e-12)
    for _ in range(3):
        padded = np.pad(field, 1, mode="edge")
        field = (
            4.0 * padded[1:-1, 1:-1]
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + 0.5 * (padded[:-2, :-2] + padded[:-2, 2:] + padded[2:, :-2] + padded[2:, 2:])
        ) / 10.0
    return field


def _candidate_hover(row: dict[str, object]) -> str:
    filtered = row.get("filtered_simplicial_object") if isinstance(row, dict) else None
    summary = filtered.get("summary", {}) if isinstance(filtered, dict) else {}
    topology = row.get("topological_algebra") if isinstance(row, dict) else None
    betti = topology.get("chain_complex", {}).get("homology", {}).get("betti") if isinstance(topology, dict) else None
    graphcg = row.get("graphcg_projection") if isinstance(row, dict) else None
    top_dirs = graphcg.get("top_directions", []) if isinstance(graphcg, dict) else []
    path = row.get("path", []) if isinstance(row, dict) else []
    simplex_sample = filtered.get("simplices", [])[:4] if isinstance(filtered, dict) else []
    intervals = topology.get("persistence", {}).get("intervals", [])[:4] if isinstance(topology, dict) else []
    rank_sample = topology.get("multiparameter_persistence", {}).get("rank_invariant_samples", [])[:3] if isinstance(topology, dict) else []
    completeness = row.get("reasoning_step_completeness") if isinstance(row, dict) else None
    complete_line = ""
    if isinstance(completeness, dict):
        if completeness.get("complete"):
            complete_line = "<br><b>reasoning-step data</b>: complete model-derived payload"
        else:
            complete_line = f"<br><b>reasoning-step data</b>: incomplete missing={_json_clip(completeness.get('missing', []), 420)}"
    return (
        f"<b>{row.get('record_id')}</b>"
        f"<br>level={row.get('level')} path={path}"
        f"<br>score={float(row.get('score', 0.0)):.4f} nll={float(row.get('nll', 0.0)):.4f}"
        f"<br>graph tokens={row.get('graph_tokens')} margin={float(row.get('margin_mean', 0.0)):.4f}"
        + complete_line
        + (f"<br><b>model input</b>: {_html_clip(row.get('input_text', ''), 900)}" if row.get("input_text") else "")
        + (f"<br><b>model argmax output</b>: {_html_clip(row.get('decoded_argmax', ''), 900)}" if row.get("decoded_argmax") else "")
        + f"<br><b>filtered complex</b>: V={summary.get('num_vertices')} E={summary.get('num_edges')} T={summary.get('num_two_simplices')}"
        f"<br>simplex preview={_json_clip(simplex_sample, 700)}"
        + (f"<br><b>Betti</b>: {betti}" if betti else "")
        + (f"<br><b>PH preview</b>: {_json_clip(intervals, 500)}" if intervals else "")
        + (f"<br><b>multi-rank preview</b>: {_json_clip(rank_sample, 500)}" if rank_sample else "")
        + (f"<br><b>GraphCG top dirs</b>: {top_dirs[:3]}" if top_dirs else "")
    )


def _json_clip(value: object, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False)
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _short_label(value: object, limit: int = 32) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: max(limit - 3, 0)] + "..."


def _html_clip(value: object, limit: int) -> str:
    text = str(value)
    escaped = html.escape(text)
    return escaped if len(escaped) <= limit else escaped[: limit - 3] + "..."


def _decode_shifted_bytes(ids: object) -> str:
    if torch.is_tensor(ids):
        values = ids.detach().cpu().reshape(-1).tolist()
    elif isinstance(ids, np.ndarray):
        values = ids.reshape(-1).tolist()
    else:
        values = list(ids) if isinstance(ids, (list, tuple)) else []
    raw = bytearray()
    for value in values:
        try:
            token = int(value)
        except Exception:
            continue
        if token <= 0:
            continue
        raw.append(max(0, min(255, token - 1)))
    return bytes(raw).decode("utf-8", "ignore")
