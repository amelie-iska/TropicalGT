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
    p3 = output_dir / "reasoning_trajectory_3d.html"; _write_plotly_dark_html(p3, fig3d, f"TropicalGT-I validation graph-state PCA sample{title_suffix}", panel_items, show_filtration_slider=True)
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
    p2 = output_dir / "reasoning_trajectory_pca_nll.html"; _write_plotly_dark_html(p2, fig2, f"TropicalGT-I validation PCA with NLL height{title_suffix}", panel_items, show_filtration_slider=True)
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
        if not isinstance(level_radius, dict) and isinstance(trajectory_growth, list):
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
        if isinstance(level_radius, dict):
            level_radius_path = output_dir / "trajectory_level_radius_bifiltration.json"
            level_radius_path.write_text(json.dumps(level_radius, indent=2), encoding="utf-8")
            paths["trajectory_level_radius_bifiltration"] = str(level_radius_path)

    memory = result.get("analogical_memory_retrieval") if isinstance(result, dict) else None
    if isinstance(memory, dict):
        memory_path = output_dir / "analogical_memory_retrieval.json"
        memory_path.write_text(json.dumps(memory, indent=2), encoding="utf-8")
        paths["analogical_memory_retrieval"] = str(memory_path)

    if render_html:
        if isinstance(scaling, dict):
            paths.update(write_got_trajectory_visualization(scaling, output_dir))
            paths.update(write_graphcg_trajectory_visualization(scaling, output_dir))
        paths.update(write_tropical_support_heatmap(result, output_dir))
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
            paths["trajectory_level_radius_bifiltration_3d"] = write_two_parameter_bifiltration_visualization(
                output_dir / "trajectory_persistence" / "two_parameter_bifiltration.html",
                level_radius,
                title="Trajectory 2-parameter persistence over F2[x_level,x_radius]",
            )
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
    nll_surface, nll_surface_meta = _nll_triangulated_surface_trace(
        pca[:, 0],
        pca[:, 1],
        nll_plot_z,
        nll_values,
        name="Observed-state NLL anchor mesh",
    )
    nll_surface_meta.update(
        {
            "z_axis": "projected_nll_fitness_energy",
            "z_axis_center_raw_nll": nll_center,
            "z_axis_scale": nll_plot_scale,
            "z_axis_label": f"observed-state NLL anchor z; raw centered NLL x {nll_plot_scale:g} retained separately",
            "raw_nll_range": float(np.nanmax(nll_values) - np.nanmin(nll_values)) if nll_values.size else 0.0,
            "exact_anchor_scope": "observed model-evaluated GoT states only",
            "actual_landscape_scope": "unavailable: this artifact shows only a sparse observed-state anchor mesh, not a dense model-evaluated landscape",
            "sparse_observed_anchor_layer": True,
            "dense_model_evaluated_field": False,
            "truthfulness_warning": "The surface is only a piecewise-linear mesh through sampled model GoT states; it must not be read as a dense latent-space NLL field.",
            "local_interpolation_anchor_scope": "observed model-evaluated GoT states only",
            "model_state_anchor_count": int(len(candidates)),
            "rendered_microsteps_are_nll_surface_anchors": False,
            "rendered_microsteps_policy": "disabled; only model-evaluated GoT states are plotted as trajectory points",
            "surface_contact_contract": "every rendered GoT state marker and trajectory edge endpoint uses plot.z/plot.z_surface read from the displayed NLL energy surface",
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
        fig.add_trace(nll_surface)
    fig.add_trace(_nll_anchor_trace(pca[:, 0], pca[:, 1], nll_plot_z, nll_values, name="model-evaluated GoT NLL anchors"))
    for idx, row in enumerate(candidates):
        parent = row.get("parent")
        if isinstance(parent, str) and parent in id_to_idx:
            j = id_to_idx[parent]
            action = _edge_action_label(row)
            raw_delta = float(nll_values[idx] - nll_values[j])
            improvement_label = "improved" if raw_delta < 0 else ("flat" if abs(raw_delta) <= 1e-12 else "regressed")
            chain = [
                {"x": float(pca[j, 0]), "y": float(pca[j, 1]), "z": float(nll_plot_z[j]), "label": str(parent)},
                *microsteps_by_candidate.get(idx, []),
                {"x": float(pca[idx, 0]), "y": float(pca[idx, 1]), "z": float(nll_plot_z[idx]), "label": ids[idx]},
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
                        f"surface-contact=endpoint z values are read from the displayed NLL mesh<br>"
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
            z=nll_plot_z,
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
        title="Graph-of-thought branching trajectory with observed NLL anchors",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="projected NLL / fitness energy",
            aspectmode="cube",
            camera=dict(eye=dict(x=1.52, y=-1.72, z=1.18)),
        ),
    )
    unique_ratio = float(pca_report.get("unique_embedding_ratio_rounded8", 1.0))
    if unique_ratio < 0.8:
        fig.add_annotation(
            text=(
                "embedding-state collapse diagnostic: "
                f"{int(pca_report.get('unique_embeddings_rounded8', len(candidates)))}/{len(candidates)} unique graph_state vectors; "
                f"max multiplicity={int(pca_report.get('max_embedding_multiplicity_rounded8', 1))}<br>"
                "coordinates are actual PCA; duplicate states are aggregated for the surface and retained as original anchor markers"
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
                f"{raw_nll_range:.6g}; z-axis shows centered NLL scaled by {nll_plot_scale:.1f}.<br>"
                "Visible layers: observed model-evaluated GoT-state anchors and their sparse exact anchor mesh; no dense latent-space NLL field, microstep, or surrogate NLL anchors."
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
    _write_plotly_dark_html(path, fig, "Graph-of-thought trajectory PCA in embedding space", panel_items, show_filtration_slider=True)
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
                    "z": float(nll_plot_z[idx]),
                    "z_surface": float(nll_plot_z[idx]),
                    "z_centered_scaled_nll": float(nll_plot_z[idx]),
                    "raw_centered_scaled_nll": float(raw_nll_plot_z[idx]),
                    "raw_nll": float(nll_values[idx]),
                    "touches_nll_surface": True,
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
        "sample_count": sample_count,
        "anchor_count": int(points.shape[0]),
        "sigma": sigma,
        "kernel": "isotropic Gaussian in 3D PCA coordinates",
        "local_nll_rule": "kernel-weighted mean of measured NLL at actual GoT states",
        "render_contract": "Gaussian cloud points are not model states; they visualize local NLL density around actual embedding vectors, while only large labeled markers are model states",
        "nll_min": float(np.min(nll)),
        "nll_max": float(np.max(nll)),
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
    cloud_points = cloud["points"]
    nearest = cloud["nearest"]
    fig.add_trace(
        go.Scatter3d(
            x=cloud_points[:, 0],
            y=cloud_points[:, 1],
            z=cloud_points[:, 2],
            mode="markers",
            marker=dict(
                size=2.25,
                color=cloud["local_nll"],
                colorscale="Plasma",
                opacity=0.18,
                showscale=True,
                colorbar=dict(title="local raw NLL", x=1.03, y=0.48, len=0.62, thickness=16),
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
            name="NLL density cloud around actual GoT embeddings",
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
                size=9,
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
    fig.update_layout(
        template="plotly_dark",
        title=(
            "3D PCA NLL density cloud around graph-of-thought embeddings"
            "<br><sup>Gaussian cloud points are not model states. They are visualization-only local mass around actual model graph_state vectors; "
            "color is kernel-weighted measured NLL, and labeled markers are the only model states.</sup>"
        ),
        scene=dict(
            xaxis_title="PC1(graph_state)",
            yaxis_title="PC2(graph_state)",
            zaxis_title="PC3(graph_state)",
            aspectmode="cube",
            camera=dict(eye=dict(x=1.55, y=-1.65, z=1.08)),
        ),
        margin=dict(t=108, r=72, b=36, l=36),
        legend=dict(orientation="h", x=0.02, y=1.0, xanchor="left", yanchor="bottom"),
    )
    fig.add_annotation(
        text=(
            "actual PCA distance corr="
            f"{float(pca_report.get('pairwise_distance_correlation', 0.0)):.3f}; "
            f"stress={float(pca_report.get('normalized_stress', 0.0)):.3f}; "
            f"density sigma={float(cloud_meta.get('sigma', 0.0)):.4g}"
        ),
        x=0,
        y=1.04,
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
    payload = {
        "available": True,
        "density_cloud": cloud_meta,
        "nodes": [
            {
                "record_id": ids[idx],
                "parent": candidates[idx].get("parent"),
                "level": int(inferred_levels[idx]),
                "path": candidates[idx].get("path", []),
                "pca": {"pc1": float(pca[idx, 0]), "pc2": float(pca[idx, 1]), "pc3": float(pca[idx, 2])},
                "nll": float(nll_values[idx]),
            }
            for idx in range(len(candidates))
        ],
        "edges": [
            {"source": row.get("parent"), "target": ids[idx], "action": _edge_action_label(row), "nll_delta": float(nll_values[idx] - nll_values[id_to_idx[row.get("parent")]])}
            for idx, row in enumerate(candidates)
            if isinstance(row.get("parent"), str) and row.get("parent") in id_to_idx
        ],
    }
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"got_nll_density_cloud_pca_3d": str(path), "got_nll_density_cloud_payload": str(payload_path)}


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
            "Full trajectory complex; slider filters radius/simplicial edges induced from the same embeddings "
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
            title="Full graph-of-thought trajectory probability SimplexTree face-coface poset",
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
        _write_dark_empty(probability_tree_path, message)
        payload["probability_filtered_simplicial_object"] = unavailable
    result["got_full_trajectory_complex_jensen_shannon"] = str(probability_path)
    result["got_full_trajectory_simplex_tree_3d_jensen_shannon"] = str(probability_tree_path)
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
    if int(summary.get("num_edges", 0) or 0) <= 0:
        return False
    return any(
        _probability_feature_vector(simplex) is not None
        for simplex in obj.get("simplices", [])
        if isinstance(simplex, dict) and int(simplex.get("dimension", -1)) == 0
    )


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
        _write_simplex_tree_3d_map(
            tree_path,
            obj,
            title=f"Reasoning step GUDHI SimplexTree face-coface poset: q{idx}",
            subtitle=f"record_id={record_id}; level={row.get('level')}; path={row.get('path', [])}",
        )
        rows.append(
            {
                "index": idx,
                "record_id": record_id,
                "level": int(row.get("level", 0) or 0),
                "path": row.get("path", []),
                "file": file_name,
                "simplex_tree_file": tree_file_name,
                "summary": obj_summary.get("summary", {}),
                "simplex_tree": obj_summary.get("simplex_tree", {}),
                "graph_token_direction_overlay": obj.get("graph_token_direction_overlay", {}),
                "decoding_causal_overlay": obj.get("decoding_causal_overlay", {}),
            }
        )
    index_path = directory / "index.html"
    _write_reasoning_step_complex_index(index_path, rows)
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps({"steps": rows}, indent=2), encoding="utf-8")
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
    summary = obj.get("summary", {})
    simplex_tree = obj.get("simplex_tree", {}) if isinstance(obj.get("simplex_tree"), dict) else {}
    backend_label = simplex_tree.get("backend", "json")
    fig.update_layout(
        template="plotly_dark",
        title=(
            f"{title}<br><sup>{html.escape(subtitle)} | filtration backend={html.escape(str(backend_label))} "
            f"| {html.escape(layout_kind)} | V={summary.get('num_vertices', len(vertices))}, "
            f"E={summary.get('num_edges', len(edges))}, T={summary.get('num_two_simplices', len(triangles))}</sup>"
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


def _write_simplex_tree_3d_map(path: Path, obj: dict[str, object], title: str, subtitle: str = "") -> None:
    obj = _gudhi_canonical_complex(obj)
    simplices = [row for row in obj.get("simplices", []) if isinstance(row, dict) and row.get("simplex")]
    if not simplices:
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
    by_dim: dict[int, list[tuple[str, ...]]] = defaultdict(list)
    for key, row in simplex_rows.items():
        by_dim[int(row.get("dimension", len(key) - 1))].append(key)
    positions: dict[tuple[str, ...], tuple[float, float, float]] = {}
    for dim, keys in by_dim.items():
        ordered = sorted(keys, key=lambda key: (float(simplex_rows[key].get("filtration", 0.0) or 0.0), _simplex_key(key)))
        denom = max(len(ordered) - 1, 1)
        for idx, key in enumerate(ordered):
            filt = float(simplex_rows[key].get("filtration", 0.0) or 0.0)
            positions[key] = (filt, float(dim), float(idx / denom))
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    edge_hover: list[str | None] = []
    for key, row in simplex_rows.items():
        if len(key) <= 1:
            continue
        for face in combinations(key, len(key) - 1):
            face_key = tuple(sorted(face))
            if face_key not in positions:
                continue
            ax, ay, az = positions[face_key]
            bx, by, bz = positions[key]
            label = (
                f"<b>simplex-tree inclusion</b><br>"
                f"face={html.escape(_simplex_key(face_key))}<br>"
                f"coface={html.escape(_simplex_key(key))}<br>"
                f"coface filtration={float(row.get('filtration', 0.0) or 0.0):.6g}"
            )
            edge_x.extend([ax, bx, None])
            edge_y.extend([ay, by, None])
            edge_z.extend([az, bz, None])
            edge_hover.extend([label, label, None])
    node_keys = list(simplex_rows)
    node_x = [positions[key][0] for key in node_keys]
    node_y = [positions[key][1] for key in node_keys]
    node_z = [positions[key][2] for key in node_keys]
    node_dim = [int(simplex_rows[key].get("dimension", len(key) - 1)) for key in node_keys]
    node_hover = []
    for key in node_keys:
        row = simplex_rows[key]
        node_hover.append(
            f"<b>{html.escape(_simplex_key(key))}</b>"
            f"<br>dimension={int(row.get('dimension', len(key) - 1))}"
            f"<br>filtration={float(row.get('filtration', 0.0) or 0.0):.6g}"
            f"<br>type={html.escape(str(row.get('type', 'simplex')))}"
            f"<br>source={html.escape(str(row.get('filtration_source', 'gudhi.SimplexTree')))}"
            + (f"<br>embedding distance={float(row.get('embedding_distance')):.6g}" if isinstance(row.get("embedding_distance"), (int, float)) else "")
            + (f"<br>reasoning transition={bool(row.get('reasoning_transition'))}" if row.get("reasoning_transition") is not None else "")
            + (f"<br>{_vertex_readable_summary(row, include_output=True)}" if int(row.get("dimension", -1)) == 0 else "")
        )
    fig = go.Figure()
    link_count = len(edge_x) // 3
    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(width=1.0, color="rgba(125,211,252,0.18)"),
            hovertext=edge_hover,
            hoverinfo="text",
            name=f"face-to-coface links ({link_count})",
            visible="legendonly" if link_count > 360 else True,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers",
            marker=dict(
                size=[9 if dim == 0 else 7 if dim == 1 else 5 for dim in node_dim],
                color=node_dim,
                colorscale=[[0, "#5eead4"], [0.5, "#60a5fa"], [1.0, "#facc15"]],
                showscale=True,
                colorbar=dict(title="dim", x=1.04, y=0.74, len=0.42, thickness=14),
                line=dict(color="#e8eef8", width=0.8),
            ),
            hovertext=node_hover,
            hoverinfo="text",
            name="simplices",
        )
    )
    summary = obj.get("summary", {}) if isinstance(obj.get("summary"), dict) else {}
    tree = obj.get("simplex_tree", {}) if isinstance(obj.get("simplex_tree"), dict) else {}
    fig.update_layout(
        template="plotly_dark",
        title=(
            f"{title}<br><sup>{html.escape(subtitle)} | backend={html.escape(str(tree.get('backend', 'json')))} "
            f"| displayed={len(node_keys)}/{int(tree.get('num_simplices', len(simplex_rows)) or len(simplex_rows))} "
            f"| face-coface poset view, not a literal trie layout "
            f"| V={summary.get('num_vertices', 0)}, E={summary.get('num_edges', 0)}, T={summary.get('num_two_simplices', 0)}"
            + (" | truncated for browser performance" if truncated else "")
            + "</sup>"
        ),
        scene=dict(
            xaxis_title="filtration value",
            yaxis_title="simplex dimension",
            zaxis_title="face-coface ordering coordinate",
            aspectmode="manual",
            aspectratio=dict(x=1.45, y=0.75, z=1.0),
            camera=dict(eye=dict(x=1.45, y=-1.75, z=1.15)),
        ),
        legend=dict(orientation="h", x=0.02, y=1.03, xanchor="left", yanchor="bottom", font=dict(size=10)),
        margin=dict(t=118, l=0, r=96, b=24),
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
        for simplex_ints, filtration in tree.get_filtration():
            simplex = [int_to_label[int(vertex)] for vertex in simplex_ints]
            key = tuple(sorted(simplex))
            base = dict(metadata.get(key, {}))
            base.update(
                {
                    "simplex": simplex,
                    "dimension": len(simplex) - 1,
                    "filtration": float(filtration),
                    "gudhi_simplex_tree": True,
                    "type": base.get("type", f"gudhi_dim_{len(simplex) - 1}"),
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
            }
        )
        return {
            **obj,
            "summary": summary,
            "thresholds": sorted(thresholds),
            "simplices": canonical,
            "simplex_tree": {
                "backend": "gudhi.SimplexTree",
                "num_vertices": int(tree.num_vertices()),
                "num_simplices": int(tree.num_simplices()),
                "dimension": int(tree.dimension()),
                "filtration_non_decreasing": True,
            },
        }
    except Exception as exc:
        serialized_obj = dict(obj)
        serialized_obj["simplex_tree"] = {
            "backend": "unavailable_gudhi_simplex_tree",
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
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
    edges = [
        s for s in obj.get("simplices", [])
        if isinstance(s, dict)
        and int(s.get("dimension", -1)) == 1
        and float(s.get("filtration", 0.0) or 0.0) <= threshold + 1e-12
        and all(str(v) in visible for v in (s.get("simplex") or [])[:2])
    ]
    triangles = [
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
    for overlay_edge in overlay.get("edges", []) if isinstance(overlay.get("edges"), list) else []:
        if not isinstance(overlay_edge, dict):
            continue
        source = str(overlay_edge.get("source", ""))
        target = str(overlay_edge.get("target", ""))
        if source not in coords3 or target not in coords3:
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
    for directed_edge in direction_overlay.get("edges", []) if isinstance(direction_overlay.get("edges"), list) else []:
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
    for directed_edge in decoding_overlay.get("edges", []) if isinstance(decoding_overlay.get("edges"), list) else []:
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
            name="radius/simplicial edges induced from the same embeddings",
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
            name="2-simplices",
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


def _write_reasoning_step_complex_index(path: Path, rows: list[dict[str, object]]) -> None:
    body = "\n".join(
        "<tr>"
        f"<td>{int(row['index'])}</td>"
        f"<td><a href='{html.escape(str(row['file']))}'>{html.escape(str(row['record_id']))}</a></td>"
        f"<td><a href='{html.escape(str(row.get('simplex_tree_file', '')))}'>simplex tree</a></td>"
        f"<td>{int(row.get('level', 0))}</td>"
        f"<td>{html.escape(_json_clip(row.get('path', []), 96))}</td>"
        f"<td>{html.escape(_json_clip(row.get('summary', {}), 140))}</td>"
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
  </style>
</head>
<body>
  <main>
    <h1>Reasoning step filtered simplicial complex maps</h1>
    <p>Each row opens a separate 3D PCoA/MDS radius-filtered complex for one observed model-evaluated graph-of-thought state.  These are deliberately separate from the embedding-space trajectory map.</p>
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
    z = np.zeros((n, len(support_indices)), dtype=float)
    assignment_z = np.zeros((n, len(support_indices)), dtype=float)
    selected_margin_matrix = np.full((n, len(support_indices)), np.nan, dtype=float)
    hover_grid: list[list[str]] = []
    query_labels = []
    for row_idx, token in enumerate(tokens):
        active = active_support_index(token) if isinstance(token, dict) else -1
        margin = float(token.get("margin", 0.0) or 0.0)
        query_labels.append(_support_token_label(row_idx, token))
        hover_row = []
        for col_idx, support_idx in enumerate(support_indices):
            support = tokens[support_idx] if 0 <= support_idx < n else {}
            selected = active == support_idx
            if selected:
                assignment_z[row_idx, col_idx] = 1.0
                z[row_idx, col_idx] = margin
                selected_margin_matrix[row_idx, col_idx] = margin
            hover_row.append(
                f"query={html.escape(_support_token_label(row_idx, token, long=True))}<br>"
                f"candidate support={html.escape(_support_token_label(support_idx, support, long=True))}<br>"
                f"selected={str(selected).lower()}<br>"
                f"selected-support margin={margin:.5f}" + ("" if selected else " (shown only on the selected support column)") + "<br>"
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
    margin_values = np.asarray([float(token.get("margin", 0.0) or 0.0) for token in tokens], dtype=float)
    finite_margins = margin_values[np.isfinite(margin_values)]
    margin_summary = {
        "min": float(np.min(finite_margins)) if finite_margins.size else 0.0,
        "max": float(np.max(finite_margins)) if finite_margins.size else 0.0,
        "mean": float(np.mean(finite_margins)) if finite_margins.size else 0.0,
        "std": float(np.std(finite_margins)) if finite_margins.size else 0.0,
        "p05": float(np.quantile(finite_margins, 0.05)) if finite_margins.size else 0.0,
        "p50": float(np.quantile(finite_margins, 0.50)) if finite_margins.size else 0.0,
        "p95": float(np.quantile(finite_margins, 0.95)) if finite_margins.size else 0.0,
    }
    support_flow_edges = []
    for row_idx, token in enumerate(tokens):
        active = active_support_index(token) if isinstance(token, dict) else -1
        support = tokens[active] if 0 <= active < n else {}
        support_flow_edges.append(
            {
                "query_index": int(row_idx),
                "query_label": _support_token_label(row_idx, token),
                "support_index": int(active),
                "support_label": _support_token_label(active, support) if 0 <= active < n else "invalid",
                "margin": float(token.get("margin", 0.0) or 0.0),
                "query_kind": str(token.get("kind", "?")),
                "support_kind": str(support.get("kind", "?")) if isinstance(support, dict) else "?",
            }
        )
    collapse_rate = float(counts.max() / max(float(n), 1.0)) if counts.size else 0.0
    support_probs = counts / max(float(counts.sum()), 1.0)
    support_entropy = float(-np.sum(support_probs * np.log2(np.maximum(support_probs, 1e-12)))) if counts.size else 0.0
    effective_supports = float(2.0 ** support_entropy) if counts.size else 0.0
    collapse_like_layout = bool(len(support_indices) == 1 or collapse_rate >= 0.70)
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
        "margin_summary": margin_summary,
        "layout_mode": "collapse_diagnostic" if collapse_like_layout else "observed_support_matrix",
        "raw_token_labels_truncated": True,
        "interpretation": "The heatmap is an assignment matrix: yellow cells mean the model selected that support token. Confidence lives in the separate selected-margin profile and distribution.",
    }
    payload_path.write_text(
        json.dumps(
            {
                "metrics": support_metrics,
                "query_labels": query_labels,
                "support_labels": support_labels,
                "assignment_matrix": assignment_z.tolist(),
                "selected_margin_matrix": [[None if not np.isfinite(value) else float(value) for value in row] for row in selected_margin_matrix.tolist()],
                "margin_matrix": z.tolist(),
                "tokens": tokens,
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
            ("mean margin", f"{margin_summary['mean']:.4f}"),
            ("margin range", f"{margin_summary['min']:.4f} to {margin_summary['max']:.4f}"),
            ("margin std", f"{margin_summary['std']:.4f}"),
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
                f"<br><sup>top-support collapse rate={collapse_rate:.3f}; top support {html.escape(top_support_label)} captures {100.0 * collapse_rate:.1f}% of tokens. Yellow cells are selected-support assignments; margins are plotted separately.</sup>"
            ),
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
        rows=3,
        cols=1,
        specs=[[{"type": "heatmap"}], [{"type": "bar"}], [{"type": "scatter"}]],
        row_heights=[0.48, 0.24, 0.28],
        vertical_spacing=0.09,
        subplot_titles=(
            "Active-support assignment matrix",
            "Support frequency and mean selected margin",
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
            marker=dict(color=mean_margins, colorscale="Turbo", line=dict(color="#e8eef8", width=0.8)),
            text=[f"n={int(c)}<br>m={m:.3f}" for c, m in zip(counts.tolist(), mean_margins)],
            textposition="outside",
            hovertext=[f"support={html.escape(label)}<br>count={int(c)}<br>mean margin={m:.4f}" for label, c, m in zip(support_labels, counts.tolist(), mean_margins)],
            hoverinfo="text",
            name="support frequency",
            showlegend=False,
        ),
        row=2,
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
        row=3,
        col=1,
    )
    fig.update_layout(
        template="plotly_dark",
        title=(
            "Tropical active-support audit: observed supports only"
            f"<br><sup>observed supports={len(support_indices)}/{n}; top-support collapse rate={collapse_rate:.3f}; effective={effective_supports:.2f}; entropy={support_entropy:.3f} bits. Yellow cells mark selected support assignments only.</sup>"
        ),
        height=max(1040, min(1660, 700 + 12 * n)),
        margin=dict(t=150, l=112, r=190, b=124),
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
    fig.update_yaxes(title_text="query count", automargin=True, row=2, col=1)
    fig.update_xaxes(title_text="graph-token index", automargin=True, row=3, col=1)
    fig.update_yaxes(title_text="active-support margin", automargin=True, row=3, col=1)
    _write_plotly_dark_html(path, fig, "Tropical active-support heatmap")
    return {"tropical_support_heatmap": str(path), "tropical_support_payload": str(payload_path)}


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
    return {
        "persistence_barcode": str(barcode),
        "persistence_module_betti": str(module_path),
        "persistence_representations": str(representations_path),
        "persistence_landscapes": str(landscapes_path),
    }



def _m2_style_report_from_bifiltration(bifiltration: Mapping[str, Any]) -> Dict[str, Any]:
    free = bifiltration.get("free_resolution") if isinstance(bifiltration, Mapping) else None
    if not isinstance(free, Mapping):
        return {}
    m2 = free.get("chain_presentation_diagnostics")
    if not isinstance(m2, Mapping):
        m2 = free.get("macaulay2_style")
    return dict(m2) if isinstance(m2, Mapping) else {}


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
    staircase = m2.get("staircase") if isinstance(m2, Mapping) else {}
    staircase = staircase if isinstance(staircase, Mapping) else {}
    selected = staircase.get("selected_real_resolution")
    if isinstance(selected, Mapping) and selected.get("available"):
        return dict(selected)
    positive = staircase.get("positive_radius_event_ideal_resolution")
    if isinstance(positive, Mapping) and positive.get("available"):
        return dict(positive)
    full = staircase.get("minimal_ideal_resolution")
    if isinstance(full, Mapping) and full.get("available"):
        return dict(full)
    return {}


def _m2_betti_columns(m2: Mapping[str, Any]) -> Tuple[List[str], List[List[str]]]:
    resolution = _m2_selected_staircase_resolution(m2)
    rows = list((resolution.get("betti_table_rows") if resolution else m2.get("betti_table_rows", [])) or [])
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
    modules = list((resolution.get("free_modules") if resolution else m2.get("free_modules", [])) or [])

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
        return ["module", "rank", "display"], [["unavailable"], ["0"], ["no computed free-chain presentation"]]
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
    differentials = list((resolution.get("differentials") if resolution else m2.get("differentials", [])) or [])[:max_rows]
    if not differentials:
        return ["map", "shape", "rank", "matrix preview"], [["unavailable"], [""], [""], ["no differential matrices computed"]]
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


def _m2_certificate_columns(m2: Mapping[str, Any], bifiltration: Mapping[str, Any]) -> Tuple[List[str], List[List[str]]]:
    cert = m2.get("chain_complex_certificate") if isinstance(m2, Mapping) else {}
    cert = cert if isinstance(cert, Mapping) else {}
    be = bifiltration.get("buchsbaum_eisenbud") if isinstance(bifiltration, Mapping) else {}
    be = be if isinstance(be, Mapping) else {}
    rank_inv = bifiltration.get("rank_invariant") if isinstance(bifiltration, Mapping) else {}
    rank_inv = rank_inv if isinstance(rank_inv, Mapping) else {}
    resolution = _m2_selected_staircase_resolution(m2)
    res_be = resolution.get("buchsbaum_eisenbud_diagnostics", {}) if isinstance(resolution.get("buchsbaum_eisenbud_diagnostics"), Mapping) else {}
    items = [
        ("ring", resolution.get("ring", m2.get("ring", bifiltration.get("module_ring", "F2[x_level,x_radius]")))),
        ("field", m2.get("field", "F2")),
        ("chain object", "finite chain presentation; not a free resolution"),
        ("scoped real resolution", bool(resolution)),
        ("resolution scope", resolution.get("scope", "unavailable")),
        ("object resolved", resolution.get("object_resolved", "unavailable")),
        ("minimality certified", res_be.get("minimality_certified", False)),
        ("exactness certified", res_be.get("exactness_certified", False)),
        ("not full persistence-module resolution", resolution.get("not_full_persistence_module_resolution", True)),
        ("derived equivalence certified", cert.get("derived_equivalence_certified", False)),
        ("Buchsbaum-Eisenbud exactness", be.get("passes_exactness_necessary_checks", False)),
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
    rows = list((bifiltration.get("fiber_rows") or bifiltration.get("fiber_rank_profile") or []) if isinstance(bifiltration, Mapping) else [])
    m2 = _m2_style_report_from_bifiltration(bifiltration)
    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "scene"}, {"type": "xy"}],
            [{"type": "table"}, {"type": "table"}],
            [{"type": "table"}, {"type": "table"}],
        ],
        subplot_titles=(
            "2-parameter module fibers on the (level, radius) lattice",
            "Miller-Sturmfels staircase / monomial generator diagram",
            "Multigraded chain-rank table (not a resolution)",
            "Free chain modules over S = F2[x_level,x_radius]",
            "Differential matrices d_i: F_i -> F_{i-1}",
            "Certificates, Fitting/Buchsbaum-Eisenbud diagnostics",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    if rows:
        levels = sorted({int(r.get("level", 0) or 0) for r in rows})
        radius_grades = sorted({int(r.get("radius_grade", r.get("radius", 0)) or 0) for r in rows})
        beta0_lookup = {(int(r.get("level", 0) or 0), int(r.get("radius_grade", r.get("radius", 0)) or 0)): int(r.get("beta", {}).get("0", r.get("beta", {}).get(0, 0)) or 0) for r in rows if isinstance(r.get("beta", {}), Mapping)}
        beta1_lookup = {(int(r.get("level", 0) or 0), int(r.get("radius_grade", r.get("radius", 0)) or 0)): int(r.get("beta", {}).get("1", r.get("beta", {}).get(1, 0)) or 0) for r in rows if isinstance(r.get("beta", {}), Mapping)}
        for label, lookup, color in (("H0 fiber rank", beta0_lookup, "#4fe3d3"), ("H1 fiber rank", beta1_lookup, "#78a7ff")):
            xs: List[int] = []
            ys: List[int] = []
            zs: List[int] = []
            hover: List[str] = []
            for lvl in levels:
                for rg in radius_grades:
                    if (lvl, rg) not in lookup:
                        continue
                    xs.append(lvl)
                    ys.append(rg)
                    zs.append(lookup[(lvl, rg)])
                    hover.append(f"level={lvl}<br>radius grade={rg}<br>{label}={lookup[(lvl, rg)]}")
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="markers+lines",
                    name=label,
                    marker=dict(size=4, color=color, opacity=0.92),
                    line=dict(color=color, width=4),
                    text=hover,
                    hovertemplate="%{text}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        edge_x: List[Any] = []
        edge_y: List[Any] = []
        edge_z: List[Any] = []
        for lvl in levels:
            for rg in radius_grades:
                z0 = beta0_lookup.get((lvl, rg))
                if z0 is None:
                    continue
                for nxt in ((lvl + 1, rg), (lvl, rg + 1)):
                    z1 = beta0_lookup.get(nxt)
                    if z1 is None:
                        continue
                    edge_x.extend([lvl, nxt[0], None])
                    edge_y.extend([rg, nxt[1], None])
                    edge_z.extend([z0, z1, None])
        if edge_x:
            fig.add_trace(
                go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode="lines",
                    name="module structure maps",
                    line=dict(color="rgba(255,210,70,0.45)", width=2),
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

    staircase = _m2_staircase_trace_payload(m2)
    if staircase["gen_x"]:
        fig.add_trace(
            go.Scatter(
                x=staircase["gen_x"],
                y=staircase["gen_y"],
                mode="markers",
                name="monomial generators",
                marker=dict(color="#5eead4", size=12, line=dict(color="#e8f2ff", width=1)),
                text=staircase["gen_label"],
                hovertemplate="generator %{text}<br>x-degree=%{x}<br>y-degree=%{y}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    if staircase["anti_x"]:
        order = sorted(range(len(staircase["anti_x"])), key=lambda i: (staircase["anti_x"][i], -staircase["anti_y"][i]))
        fig.add_trace(
            go.Scatter(
                x=[staircase["anti_x"][i] for i in order],
                y=[staircase["anti_y"][i] for i in order],
                mode="markers+lines",
                name="minimal antichain / staircase",
                marker=dict(color="#ffd54a", size=13, symbol="diamond", line=dict(color="#e8f2ff", width=1)),
                line=dict(color="#ffd54a", width=3, shape="hv"),
                text=[staircase["anti_label"][i] for i in order],
                hovertemplate="staircase generator %{text}<br>x-degree=%{x}<br>y-degree=%{y}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    if staircase["syz_x"]:
        fig.add_trace(
            go.Scatter(
                x=staircase["syz_x"],
                y=staircase["syz_y"],
                mode="markers",
                name="lcm first syzygies",
                marker=dict(color="#ff6b8a", size=11, symbol="x"),
                text=staircase["syz_label"],
                hovertemplate="first syzygy %{text}<br>lcm x-degree=%{x}<br>lcm y-degree=%{y}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    headers, columns = _m2_betti_columns(m2)
    fig.add_trace(_table_trace(headers, columns), row=2, col=1)
    headers, columns = _m2_free_module_columns(m2)
    fig.add_trace(_table_trace(headers, columns), row=2, col=2)
    headers, columns = _m2_differential_columns(m2)
    fig.add_trace(_table_trace(headers, columns), row=3, col=1)
    headers, columns = _m2_certificate_columns(m2, bifiltration)
    fig.add_trace(_table_trace(headers, columns), row=3, col=2)

    notes = bifiltration.get("notes", []) if isinstance(bifiltration, Mapping) else []
    subtitle = " ".join(str(n) for n in notes[:2])
    if subtitle:
        subtitle = _json_clip(subtitle, 260)
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"{html.escape(title, quote=False)}<br><sup>{html.escape(subtitle, quote=False)}</sup>",
            x=0.02,
        ),
        height=1260,
        margin=dict(l=52, r=42, t=132, b=62),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_scenes(
        xaxis_title="x_level grade",
        yaxis_title="x_radius grade",
        zaxis_title="module fiber rank",
        bgcolor="#050914",
        xaxis=dict(gridcolor="#315c86"),
        yaxis=dict(gridcolor="#315c86"),
        zaxis=dict(gridcolor="#315c86"),
    )
    fig.update_xaxes(title_text="x_level", row=1, col=2, gridcolor="#203d5e")
    fig.update_yaxes(title_text="x_radius", row=1, col=2, gridcolor="#203d5e")
    _write_plotly_dark_html(path, fig, title)
    return str(path)

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


def _write_growth_persistence_landscapes(path: Path, topology: dict[str, object], growth: list[object], title_prefix: str = "") -> None:
    rows = _trajectory_growth_rows(topology, growth)
    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "scene"}], [{"type": "heatmap"}]],
        subplot_titles=(
            "GUDHI persistence landscape lambda_k(t) curves by growth level",
            "First available lambda_1(t) image by growth level",
        ),
        row_heights=[0.64, 0.36],
        vertical_spacing=0.13,
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
            grid = _as_float_list(landscape.get("grid", []))
            values = landscape.get("values", [])
            if not grid or not isinstance(values, list):
                continue
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
            xaxis_title="filtration t",
            yaxis_title="trajectory growth level",
            zaxis_title="GUDHI persistence landscape value",
            aspectmode="manual",
            aspectratio=dict(x=1.15, y=0.78, z=0.86),
            camera=dict(eye=dict(x=1.55, y=-1.75, z=1.18)),
        ),
        showlegend=False,
        height=1260,
        margin=dict(t=150, l=82, r=118, b=96),
    )
    fig.update_xaxes(title_text="filtration t", row=2, col=1)
    fig.update_yaxes(title_text="growth level / homology dimension", row=2, col=1)
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


def _free_resolution_line(topology: dict[str, object]) -> str:
    modules = _free_resolution_modules(topology)
    ranks = [f"F{int(row.get('homological_degree', 0))}:{int(row.get('rank', row.get('rank_upper_bound', 0)))}" for row in modules[:6]]
    ca = topology.get("commutative_algebra", {}) if isinstance(topology, dict) else {}
    chain_report = ca.get("multiparameter_chain_presentation_diagnostics") if isinstance(ca.get("multiparameter_chain_presentation_diagnostics"), dict) else None
    if chain_report is None:
        chain_report = ca.get("multiparameter_free_resolution_proxy", {}) if isinstance(ca.get("multiparameter_free_resolution_proxy"), dict) else {}
    ring = chain_report.get("ring", "F2[x_filtration,x_dimension,x_position]")
    minimal = chain_report.get("minimal_free_resolution", {}) if isinstance(chain_report.get("minimal_free_resolution"), dict) else {}
    if minimal.get("available"):
        resolution = minimal.get("resolution", {}) if isinstance(minimal.get("resolution"), dict) else {}
        modules = resolution.get("free_modules", []) if isinstance(resolution.get("free_modules"), list) else []
        module_line = ", ".join(str(row.get("display", row.get("name", "F_i"))) for row in modules[:4] if isinstance(row, dict))
        return (
            f"scoped real staircase resolution over {resolution.get('ring', ring)}: {module_line}; "
            "ambient chain presentation is still not a full persistence-module free resolution"
        )
    return f"real free resolution unavailable; chain-presentation diagnostics over {ring}: " + (", ".join(ranks) if ranks else "no displayed chain modules")


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
    ca = topology.get("commutative_algebra", {}) if isinstance(topology, dict) else {}
    chain_report = ca.get("multiparameter_chain_presentation_diagnostics") if isinstance(ca.get("multiparameter_chain_presentation_diagnostics"), dict) else None
    if chain_report is None:
        chain_report = ca.get("multiparameter_free_resolution_proxy", {}) if isinstance(ca.get("multiparameter_free_resolution_proxy"), dict) else {}
    modules = chain_report.get("free_chain_modules", []) if isinstance(chain_report, dict) else []
    if modules:
        return [row for row in modules if isinstance(row, dict)]
    taylor = ca.get("taylor_resolution_upper_bound", {}) if isinstance(ca.get("taylor_resolution_upper_bound"), dict) else {}
    ranks = taylor.get("ranks", []) if isinstance(taylor, dict) else []
    return [row for row in ranks if isinstance(row, dict)]


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
    basis_source_counts = {basis: int(basis_sources.count(basis)) for basis in sorted(set(basis_sources))}
    projection_basis = sorted(set(basis_sources))[0] if len(set(basis_sources)) == 1 else "mixed"
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
    payload_path.write_text(
        json.dumps(
            {
                "available": True,
                "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
                "display_count": int(display_count),
                "displayed_direction_indices": [int(idx) for idx in top_idx.tolist()],
                "display_policy": "all_model_graphcg_directions_no_sampling",
                "full_rank_direction_count": int(direction_count),
                "active_rank_nonzero_mean_abs": active_rank,
                "mean_abs_min": float(np.min(mean_abs)) if mean_abs.size else 0.0,
                "mean_abs_max": float(np.max(mean_abs)) if mean_abs.size else 0.0,
                "mean_abs_p90": active_floor,
                "mean_abs": [float(v) for v in mean_abs.tolist()],
                "signed_mean": [float(v) for v in signed_mean.tolist()],
                "candidate_labels_compact": compact_labels,
                "candidate_labels": labels,
                "candidate_activity_mean_abs": [float(v) for v in candidate_mean_abs.tolist()],
                "candidate_peak_abs": [float(v) for v in candidate_peak_abs.tolist()],
                "candidate_effective_direction_count": [float(v) for v in candidate_effective_dirs.tolist()],
                "candidate_active_direction_count": [int(v) for v in candidate_active_counts.tolist()],
                "direction_activity_sorted": [float(v) for v in sorted_activity.tolist()],
                "direction_signed_mean_sorted": [float(v) for v in sorted_signed.tolist()],
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
    graphcg_height = int(max(1380, 940 + 32 * len(compact_labels)))
    fig = make_subplots(
        rows=4,
        cols=1,
        specs=[[{"type": "heatmap"}], [{"type": "scatter"}], [{"type": "scatter"}], [{"type": "scatter"}]],
        row_heights=[0.46, 0.18, 0.18, 0.18],
        vertical_spacing=0.09,
        subplot_titles=(
            "Readable full-rank heatmap: every model GraphCG direction",
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
        row=2,
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
        row=3,
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
        row=4,
        col=1,
    )
    fig.update_layout(
        template="plotly_dark",
        title=(
            "GraphCG full-rank direction audit"
            f"<br><sup>heatmap shows all {direction_count} model-derived directions; basis={html.escape(projection_basis)}; active nonzero rank={active_rank}; exact ids in hover/payload.</sup>"
        ),
        margin=dict(t=150, l=118, r=96, b=120),
        height=graphcg_height,
    )
    x_labels = [f"d{int(idx)}" for idx in top_idx]
    x_step = max(1, int(math.ceil(len(x_labels) / max(visible_direction_tick_label_limit, 1))))
    x_tickvals = [label for pos, label in enumerate(x_labels) if pos % x_step == 0 or pos == len(x_labels) - 1]
    y_step = max(1, int(math.ceil(len(labels) / max(visible_candidate_tick_label_limit, 1))))
    y_tickvals = [label for pos, label in enumerate(compact_labels) if pos % y_step == 0 or pos == len(compact_labels) - 1]
    fig.update_xaxes(title_text="GraphCG direction (bounded visible ticks; hover for exact direction)", tickangle=-35, tickmode="array", tickvals=x_tickvals, ticktext=x_tickvals, tickfont=dict(size=9), row=1, col=1)
    fig.update_yaxes(title_text="GoT state index (hover for path)", tickmode="array", tickvals=y_tickvals, ticktext=y_tickvals, row=1, col=1, tickfont=dict(size=10))
    fig.update_xaxes(title_text="direction rank by activity", row=2, col=1)
    fig.update_yaxes(title_text="mean absolute cosine", row=2, col=1)
    fig.update_xaxes(title_text="GoT candidate index", row=3, col=1)
    fig.update_yaxes(title_text="candidate mean |cos|", row=3, col=1)
    fig.update_xaxes(title_text="direction mean |cos|", row=4, col=1)
    fig.update_yaxes(title_text="direction signed mean", row=4, col=1)
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
        _write_dark_empty(path, reason)
        map_path.write_text(
            json.dumps(
                {
                    "available": False,
                    "reason": "no_non_self_model_memory",
                    "maps": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"analogical_memory_retrieval_html": str(path), "analogical_simplicial_maps": str(map_path)}
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
        _write_dark_empty(path, reason)
        map_path.write_text(
            json.dumps(
                {
                    "available": False,
                    "reason": "missing_model_probability_query_complex",
                    "maps": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"analogical_memory_retrieval_html": str(path), "analogical_simplicial_maps": str(map_path)}

    enriched = [
        row
        for row in enriched
        if _has_real_probability_filtration(row.get("trajectory_probability_filtered_simplicial_object"))
    ]
    if not enriched:
        reason = "No model probability filtered codomain trajectory complex was available; analogical maps are not rendered without model probabilities."
        _write_dark_empty(path, reason)
        map_path.write_text(
            json.dumps(
                {
                    "available": False,
                    "reason": "missing_model_probability_codomain_complex",
                    "query_complex_source": query_complex_source,
                    "maps": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"analogical_memory_retrieval_html": str(path), "analogical_simplicial_maps": str(map_path)}

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
    _write_analogical_topk_index(index_path, pair_pages, map_reports)

    map_path.write_text(
        json.dumps(
            {
                "available": True,
                "query_summary": query_complex.get("summary", {}) if isinstance(query_complex, dict) else {},
                "query_complex_source": query_complex_source,
                "query_derived_signature": query_topology.get("derived_equivalence_signature", {}) if isinstance(query_topology, dict) else {},
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
    sim_map = _simplicial_map_between_complexes(query_complex, mem_complex)
    derived_comparison = _derived_invariant_comparison(query_topology, mem_topology, sim)
    sim_map["derived_invariant_comparison"] = derived_comparison
    sim_map["algebraic_realization_certificate"] = _analogical_realization_certificate(sim, sim_map, derived_comparison)
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
        f"<br>chain-presentation diagnostic similarity={float(sim.get('chain_presentation_similarity', sim.get('free_resolution_similarity', 0.0))):.4f}"
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
            "<br><sup>filtered-complex certificate from model-probability Jensen-Shannon assignment; gold=preserved 1-simplices, rose=vertex-only correspondences; slider filters domain/codomain/certificate edges</sup>"
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
    status = "filtered correspondence certificate passed" if bool(sim_map.get("is_filtered_simplicial_map")) else "filtered correspondence certificate failed"
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
        ("retrieval weights", html.escape(str(sim.get('retrieval_weights', {})))),
        ("PH similarity", f"{float(sim.get('persistent_homology_similarity', 0.0)):.4f}"),
        ("chain-presentation similarity", f"{float(sim.get('chain_presentation_similarity', sim.get('free_resolution_similarity', 0.0))):.4f}"),
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
        f"<br>chain-presentation diagnostic similarity={float(sim.get('chain_presentation_similarity', sim.get('free_resolution_similarity', 0.0))):.4f}"
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
            f"<br>chain-presentation diagnostic similarity={float(sim.get('chain_presentation_similarity', sim.get('free_resolution_similarity', 0.0))):.4f}"
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


def _write_analogical_topk_index(path: Path, pair_pages: list[dict[str, object]], map_reports: list[dict[str, object]]) -> None:
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
            f"<td>{float(report.get('chain_presentation_similarity', report.get('free_resolution_similarity', 0.0))):.4f}</td>"
            f"<td>{float(report.get('commutative_algebra_similarity', 0.0)):.4f}</td>"
            f"<td>{float(report.get('persistence_landscape_l2_similarity', 0.0)):.4f}</td>"
            f"<td>{float(report.get('persistence_landscape_cosine', 0.0)):.4f}</td>"
            f"<td>{float(report.get('persistence_vector_aggregate_similarity', 0.0)):.4f}</td>"
            f"<td>{html.escape(str(report.get('persistence_vector_component_summary', report.get('persistence_vector_methods', ''))))}</td>"
            f"<td>{float(report.get('derived_algebraic_similarity', 0.0)):.4f}</td>"
            f"<td>{float(report.get('derived_signature_similarity', 0.0)):.4f}</td>"
            f"<td>{int(report.get('simplex_tree_map_preserved', 0))}/{int(report.get('simplex_tree_map_checked', 0))} = {float(report.get('simplex_tree_map_preservation_rate', 0.0)):.4f}</td>"
            f"<td>{float(report.get('edge_preservation_rate', 0.0)):.4f}</td>"
            "</tr>"
        )
    body = "\n".join(rows) or "<tr><td colspan='16'>No retrieved memories.</td></tr>"
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
  </style>
</head>
<body>
  <main>
	    <h1>Analogical top-k probability correspondences</h1>
	    <p>Each row opens one query-to-memory vertex assignment with a finite filtered-complex certificate. The NLL/fitness landscape and the GUDHI persistence landscape are different objects: this table reports the persistence-landscape vector plus the wider vectorized GUDHI family (Landscape, BettiCurve, Silhouette, Entropy, PersistenceLengths, TopologicalVector, PersistenceImage). These are real cached vectors; the cosine/L2 comparisons are differentiable with respect to those vectors, while this HTML does not claim autograd through GUDHI diagram vectorization. Unavailable vectors stay zero rather than being fabricated. Edge, face, and filtration preservation can fail and are reported on the rank page.</p>
	    <table>
	      <thead><tr><th>rank</th><th>correspondence</th><th>retrieval</th><th>landscape contrib.</th><th>vector contrib.</th><th>PH</th><th>chain pres.</th><th>comm. alg.</th><th>persistence-landscape L2 sim</th><th>persistence-landscape cosine</th><th>vector aggregate</th><th>vector methods</th><th>derived/algebraic</th><th>coarse signature</th><th>simplex-tree map</th><th>edge certificate</th></tr></thead>
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
            f"<br>chain-presentation diagnostic similarity={float(sim.get('chain_presentation_similarity', sim.get('free_resolution_similarity', 0.0))):.4f}"
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
    sig_sim = _cosine_similarity(q_sig, m_sig)
    free_sim = _cosine_similarity(q_free, m_free)
    ph_sim = _cosine_similarity(q_ph, m_ph)
    ca_sim = _cosine_similarity(q_ca, m_ca)
    required_components_available = bool(
        all(
            vec.size > 0 and float(np.linalg.norm(vec)) > 1e-12
            for vec in (q_sig, m_sig, q_free, m_free, q_ph, m_ph, q_ca, m_ca)
        )
    )
    derived_algebraic = min(sig_sim, free_sim, ph_sim, ca_sim) if required_components_available else 0.0
    return {
        "retrieval_score": float(row.get("retrieval_score", 0.0)),
        "base_retrieval_score": float(row.get("base_retrieval_score", 0.0)),
        "persistence_landscape_score_contribution": float(row.get("persistence_landscape_score_contribution", 0.0)),
        "persistence_vector_score_contribution": float(row.get("persistence_vector_score_contribution", 0.0)),
        "retrieval_score_components": row.get("retrieval_score_components", {}) if isinstance(row.get("retrieval_score_components"), dict) else {},
        "retrieval_weights": retrieval_weights,
        "embedding_similarity": float(row.get("embedding_similarity", 0.0)),
        "signature_similarity": float(row.get("signature_similarity", 0.0)),
        "derived_signature_similarity": float(sig_sim),
        "chain_presentation_similarity": float(free_sim),
        "free_resolution_similarity": float(free_sim),  # deprecated alias: this is a chain-presentation diagnostic unless a CAS certificate is attached.
        "persistent_homology_similarity": float(ph_sim),
        "commutative_algebra_similarity": float(ca_sim),
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
    return {
        "comparison_kind": "finite_F2xy_persistence_module_and_chain_presentation_invariant_comparison",
        "field": "F2",
        "module_category": "finite grid presentation over F2[x_level,x_radius]",
        "derived_category": "bounded finite-chain invariant comparison; no derived equivalence or free-resolution claim without a CAS certificate",
        "tolerance": float(tol),
        "derived_algebraic_similarity": float((sim or {}).get("derived_algebraic_similarity", 0.0)),
        "derived_signature_cosine": float((sim or {}).get("derived_signature_similarity", 0.0)),
        "derived_equivalence_claim": "compatible_finite_invariant_witness" if finite_match else "not_certified",
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
        "query_betti_vector": q_betti,
        "memory_betti_vector": m_betti,
        "signature_l2_distance": float(np.linalg.norm(q_signature - m_signature)),
        "free_rank_l2_distance": float(np.linalg.norm(q_free - m_free)),
        "persistence_l2_distance": float(np.linalg.norm(q_ph - m_ph)),
        "commutative_algebra_l2_distance": float(np.linalg.norm(q_ca - m_ca)),
        "persistence_landscape_l2_distance": float(np.linalg.norm(q_landscape_padded - m_landscape_padded)) if q_landscape_padded.size or m_landscape_padded.size else 0.0,
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
    real_resolution_certified = bool(
        derived_comparison.get("real_free_resolution_certified")
        or derived_comparison.get("certificate_attached")
        or derived_comparison.get("cas_free_resolution_certificate")
    )
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
        "chain_map_note": "A filtration-preserving simplicial map induces a chain map and hence a morphism of the associated F2[x,y] persistence modules.",
        "finite_invariants_match": derived_ok,
        "real_free_resolution_certified": real_resolution_certified,
        "filtered_simplex_tree_map": tree_ok,
        "simplex_tree_map_preservation_rate": simplex_tree_rate,
    }


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
    return {
        "source": "gudhi.SimplexTree finite simplex enumeration",
        "ring": "F2[x_level,x_radius]",
        "map_kind": "vertex_probability_assignment_extended_to_simplex_tree",
        "checked_simplices": int(checked),
        "preserved_simplices": int(preserved),
        "missing_codomain_simplices": int(missing),
        "preservation_rate": float(preserved / checked) if checked else 0.0,
        "positive_filtration_distortion_summary": _numeric_summary(positive_distortions),
        "dimension_counts": dim_counts,
        "rows": rows,
        "interpretation": "A filtered simplicial map on the displayed simplex trees induces a chain map and hence a morphism of the associated F2[x,y] persistence modules; failure here blocks geometric realization of any derived analogy.",
    }


def _empty_simplicial_map_report(query_vertices: list[dict[str, object]], memory_vertices: list[dict[str, object]], reason: str) -> dict[str, object]:
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



def _commutative_algebra_numeric_vector(topology: dict[str, object], length: int = 16) -> np.ndarray:
    ca = topology.get("commutative_algebra", {}) if isinstance(topology, dict) else {}
    values: list[float] = []
    chain_keys = (
        "two_parameter_chain_presentation_diagnostics",
        "multiparameter_chain_presentation_diagnostics",
    )
    legacy_keys = ("two_parameter_free_resolution", "multiparameter_free_resolution_proxy")
    keys = chain_keys if any(isinstance(ca.get(key), dict) for key in chain_keys) else legacy_keys
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
        "graphcg_loss",
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
        "graphcg_direction_norm_mean",
        "graphcg_direction_norm_min",
        "graphcg_direction_norm_max",
        "graphcg_direction_gram_offdiag_mean_abs",
        "graphcg_direction_gram_offdiag_max_abs",
        "graphcg_direction_covariance_mean_abs",
        "graphcg_direction_gram_condition_proxy",
        "graphcg_full_rank",
        "graphcg_direction_effective_rank",
        "graphcg_direction_numerical_rank",
        "graphcg_direction_rank_target",
        "graphcg_direction_singular_min",
        "graphcg_direction_singular_max",
        "graphcg_direction_svd_condition_proxy",
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
        "gpu_mem_mb",
    ]
    fig = go.Figure()
    for name in metrics:
        values = [row.get(name) for row in history]
        if any(v is not None for v in values):
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=[float(v) if v is not None else None for v in values],
                    mode="lines+markers",
                    name=name,
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


def _write_plotly_dark_html(path: Path, fig: go.Figure, title: str, panel_items: list[dict[str, object]] | None = None, show_filtration_slider: bool = False) -> None:
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
    panel_html = (
        f"""
    <aside class="panel" aria-live="polite">
      <h1 id="simplicial-title">{html.escape(initial["title"])}</h1>
      <div class="summary" id="simplicial-summary">{initial["summary"]}</div>
      {controls_html}
      <div class="simplicial-object-plot" id="simplicial-plot" aria-label="interactive selected filtered simplicial complex"></div>
      <details class="static-preview">
	        <summary>Static SVG fallback preview from the same filtered-complex payload</summary>
	        <div class="simplicial-object-panel" id="simplicial-svg">{initial["svg"]}</div>
	      </details>
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
	    .webgl-fallback {{
	      margin: 16px;
	      padding: 14px;
	      border: 1px solid rgba(251, 191, 36, 0.35);
	      border-radius: 8px;
	      background: rgba(30, 41, 59, 0.94);
	      color: var(--ink);
	    }}
	    .webgl-fallback h2 {{
	      margin: 0 0 6px;
	      font-size: 14px;
	      line-height: 1.25;
	    }}
	    .webgl-fallback p {{
	      margin: 0 0 10px;
	      color: var(--muted);
	      font-size: 12px;
	      line-height: 1.45;
	    }}
	    .webgl-fallback .simplicial-object-panel {{ margin-top: 8px; }}
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
      panelSvg.innerHTML = item.svg || "";
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
	    function staticFallbackMarkup(item, reason) {{
	      const svg = item && item.svg ? item.svg : "<div class='simplicial-object-panel'>No static filtered-complex SVG preview is available for this selected object.</div>";
	      const title = item && item.title ? item.title : "selected filtered simplicial object";
	      return `<div class="webgl-fallback"><h2>Static filtered-complex preview</h2><p>${{reason}} This preview is generated from the same serialized simplicial object payload as the interactive 3D panel.</p><div class="summary">${{title}}</div><div class="simplicial-object-panel">${{svg}}</div></div>`;
	    }}
	    function renderPanelStaticFallback(item, reason) {{
	      if (!panelPlot) return;
	      panelPlot.innerHTML = staticFallbackMarkup(item, reason);
	    }}
	    function promoteMainStaticFallback() {{
	      const chartEl = document.getElementById("chart");
	      if (!chartEl || !simplicialPanels.length || chartEl.querySelector(".webgl-fallback.main-fallback")) return;
	      if (!webglUnsupportedText(chartEl)) return;
	      const item = simplicialPanels[activePanelIndex] || simplicialPanels[initialPanelIndex] || simplicialPanels[0];
	      const fallback = document.createElement("div");
	      fallback.className = "webgl-fallback main-fallback";
	      fallback.innerHTML = `<h2>WebGL unavailable: static complex preview shown</h2><p>The browser could not create a WebGL context for the 3D Plotly view. The data were still loaded; the static preview below comes from the selected real filtered-complex payload.</p>${{staticFallbackMarkup(item, "Interactive WebGL rendering is unavailable in this browser context.")}}`;
	      chartEl.prepend(fallback);
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
	            if (webglUnsupportedText(selectedGraph)) renderPanelStaticFallback(item, "Interactive WebGL rendering is unavailable in this browser context.");
	          }}, 120);
	        }}).catch((err) => {{
	          renderPanelStaticFallback(item, `Could not render the interactive 3D panel: ${{err && err.message ? err.message : err}}.`);
	        }});
	      }} catch (err) {{
	        renderPanelStaticFallback(item, `Could not render the interactive 3D panel: ${{err && err.message ? err.message : err}}.`);
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
	    window.setTimeout(promoteMainStaticFallback, 900);
	    window.setTimeout(promoteMainStaticFallback, 1800);
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
    features = _vertex_metric_feature_matrix(labels, vertex_by_label)
    if features.shape[0] == 1:
        coords = np.asarray([[0.5, 0.5, 0.5]], dtype=float)
        stats = {"stress": 0.0, "corr": 1.0, "energy3": 1.0}
    else:
        target_distances = _target_simplicial_distance_matrix(labels, features, edges)
        coords, stats = _classical_mds3(target_distances)
        if coords.shape != (features.shape[0], 3) or not np.isfinite(coords).all():
            coords, projection_stats = _feature_pca3_without_synthetic_jitter(features, labels)
            stats = {"stress": 1.0, "corr": 0.0, "energy3": float(projection_stats.get("energy3", 0.0)), "projection_note": projection_stats.get("projection_note", "feature_pca3")}
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
        + (f"projection_note={stats.get('projection_note')} " if stats.get("projection_note") else "")
        + f"mean_radius={mean_radius:.3f}",
    )


def _vertex_metric_feature_matrix(labels: list[str], vertex_by_label: dict[str, dict[str, object]]) -> np.ndarray:
    rows: list[list[float]] = []
    max_len = 0
    for idx, label in enumerate(labels):
        vertex = vertex_by_label.get(label, {})
        vector = _coerce_vertex_vector(vertex)
        if not vector:
            vector = _vertex_numeric_feature(label, vertex, idx, len(labels))
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
    return (padded - means) / scale


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


def _vertex_numeric_feature(label: str, vertex: dict[str, object], idx: int, total: int) -> list[float]:
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
