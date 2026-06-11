from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import hashlib
import html
import json
import math

import numpy as np
import torch
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import encode_bytes
from .diagnostics import per_record_nll, record_diagnostics
from .simplicial import build_filtered_simplicial_object, build_reasoning_trajectory_complex


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
    filtered_objects = [build_filtered_simplicial_object(r) for r in records]
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
    return states, nll, hover, filtered_objects, diagnostics, io_rows


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
    states, nll, hover, filtered_objects, diagnostics, io_rows = collect_states(
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
    if states.shape[0] < 2:
        states = np.concatenate([states, states], axis=0)
        nll = np.concatenate([nll, nll])
        hover = hover + hover
        filtered_objects = filtered_objects + filtered_objects
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
    fig3d.update_layout(title="TropicalGT-I validation graph-state PCA sample", scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
    panel_items = _simplicial_panel_items(filtered_objects, hover)
    p3 = output_dir / "reasoning_trajectory_3d.html"; _write_plotly_dark_html(p3, fig3d, "TropicalGT-I validation graph-state PCA sample", panel_items, show_filtration_slider=True)
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
    fig2.update_layout(title="TropicalGT-I validation PCA with NLL height", scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="NLL"))
    p2 = output_dir / "reasoning_trajectory_pca_nll.html"; _write_plotly_dark_html(p2, fig2, "TropicalGT-I validation PCA with NLL height", panel_items, show_filtration_slider=True)
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
                "nll_surface": {"embedding_height": surface_meta, "nll_height": surface_nll_meta},
                "model_io": io_rows,
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
        if isinstance(memory, dict):
            query_context = {}
            if isinstance(scaling, dict):
                best = scaling.get("best") if isinstance(scaling.get("best"), dict) else {}
                query_context = {
                    "label": "query trajectory",
                    "embedding": best.get("embedding", []) if isinstance(best, dict) else [],
                    "filtered_simplicial_object": scaling.get("trajectory_filtered_simplicial_object")
                    or (best.get("filtered_simplicial_object") if isinstance(best, dict) else {}),
                    "topological_algebra": scaling.get("trajectory_topological_algebra")
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
    nll_values = np.asarray([float(row.get("nll", row.get("score", 0.0)) or 0.0) for row in candidates], dtype=float)
    nll_center = float(np.nanmedian(nll_values)) if nll_values.size else 0.0
    nll_plot_scale = _nll_visual_scale(nll_values)
    nll_plot_z = (nll_values - nll_center) * nll_plot_scale
    hover = [_candidate_hover(row) for row in candidates]
    candidate_objects = [
        row.get("filtered_simplicial_object") if isinstance(row.get("filtered_simplicial_object"), dict) else {}
        for row in candidates
    ]
    panel_objects = list(candidate_objects)
    panel_hover = list(hover)
    microstep_entries = _got_microstep_entries(
        candidates,
        ids,
        id_to_idx,
        pca,
        nll_plot_z,
        candidate_objects,
        panel_objects,
        panel_hover,
    )
    microsteps_by_candidate: dict[int, list[dict[str, object]]] = {}
    for entry in microstep_entries:
        microsteps_by_candidate.setdefault(int(entry["candidate_index"]), []).append(entry)
    fig = go.Figure()
    nll_surface, nll_surface_meta = _nll_triangulated_surface_trace(
        pca[:, 0],
        pca[:, 1],
        nll_plot_z,
        nll_values,
        name="Sample-supported local centered NLL surface with exact GoT anchors",
    )
    nll_surface_meta.update(
        {
            "z_axis": "centered_scaled_nll",
            "z_axis_center_raw_nll": nll_center,
            "z_axis_scale": nll_plot_scale,
            "z_axis_label": f"centered NLL x {nll_plot_scale:g}",
            "raw_nll_range": float(np.nanmax(nll_values) - np.nanmin(nll_values)) if nll_values.size else 0.0,
        }
    )
    if nll_surface is not None:
        fig.add_trace(nll_surface)
    fig.add_trace(_nll_anchor_trace(pca[:, 0], pca[:, 1], nll_plot_z, nll_values, name="NLL surface anchors"))
    for idx, row in enumerate(candidates):
        parent = row.get("parent")
        if isinstance(parent, str) and parent in id_to_idx:
            j = id_to_idx[parent]
            action = _edge_action_label(row)
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
                    hovertext=f"{parent} -> {ids[idx]}<br>action={html.escape(action)}<br>microsteps={len(chain) - 2}",
                    hoverinfo="text",
                )
            )
    if microstep_entries:
        fig.add_trace(
            go.Scatter3d(
                x=[float(entry["x"]) for entry in microstep_entries],
                y=[float(entry["y"]) for entry in microstep_entries],
                z=[float(entry["z"]) for entry in microstep_entries],
                mode="markers+text",
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
                colorbar=dict(title="NLL"),
                line=dict(width=1.5, color="#e8eef8"),
            ),
            text=[_state_plot_label(row, idx, int(inferred_levels[idx])) for idx, row in enumerate(candidates)],
            textposition="top center",
            hovertext=[
                text
                + f"<br><b>embedding/PCA multiplicity</b>: {int(pca_multiplicity[idx])} candidate(s) at this rounded coordinate"
                + f"<br><b>raw embedding unique ratio</b>: {float(pca_report.get('unique_embedding_ratio_rounded8', 1.0)):.3f}"
                for idx, text in enumerate(hover)
            ],
            hoverinfo="text",
            customdata=np.arange(len(candidates), dtype=int),
            name="GoT state",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Graph-of-thought branching trajectory with local NLL landscape",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title=f"centered NLL x {nll_plot_scale:g}",
            aspectmode="cube",
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
                    "z_centered_scaled_nll": float(nll_plot_z[idx]),
                    "raw_nll": float(nll_values[idx]),
                },
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
            customdata=np.arange(len(candidates), dtype=int),
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
    _write_plotly_dark_html(path, fig, "Graph-of-thought embedding-space trajectory map")
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
        "edges": [
            {"source": row.get("parent"), "target": ids[idx], "action": _edge_action_label(row)}
            for idx, row in enumerate(candidates)
            if row.get("parent") is not None
        ],
    }
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"got_embedding_map_3d": str(path), "got_embedding_map_payloads": str(payload_path)}


def _write_full_trajectory_complex_map(scaling_report: dict[str, object], output_dir: Path) -> dict[str, str]:
    candidates = [row for row in scaling_report.get("candidates", []) if isinstance(row, dict)]
    obj = scaling_report.get("trajectory_filtered_simplicial_object")
    if candidates and (not isinstance(obj, dict) or not _complex_has_model_io(obj)):
        obj = build_reasoning_trajectory_complex(candidates)
    if not isinstance(obj, dict):
        return {}
    obj = _gudhi_canonical_complex(obj)
    path = output_dir / "got_full_trajectory_complex.html"
    payload_path = output_dir / "got_full_trajectory_complex_payload.json"
    title = "Full graph-of-thought trajectory filtered simplicial complex"
    _write_complex_slider_map(path, obj, title=title, subtitle="Full trajectory complex; slider filters simplices by scalar radius/filtration.")
    payload_path.write_text(json.dumps({"filtered_simplicial_object": obj}, indent=2), encoding="utf-8")
    return {"got_full_trajectory_complex": str(path), "got_full_trajectory_complex_payload": str(payload_path)}


def _complex_has_model_io(obj: dict[str, object]) -> bool:
    for simplex in obj.get("simplices", []) if isinstance(obj, dict) else []:
        if not isinstance(simplex, dict) or int(simplex.get("dimension", -1)) != 0:
            continue
        if simplex.get("input_text") or simplex.get("decoded_argmax") or simplex.get("target_text"):
            return True
    return False


def _write_reasoning_step_complex_maps(candidates: list[dict[str, object]], output_dir: Path) -> dict[str, str]:
    directory = output_dir / "reasoning_step_complex_maps"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, row in enumerate(candidates):
        obj = row.get("filtered_simplicial_object")
        if not isinstance(obj, dict):
            continue
        obj_summary = _gudhi_canonical_complex(obj)
        record_id = str(row.get("record_id", f"step-{idx}"))
        file_name = f"reasoning_step_{idx:03d}.html"
        path = directory / file_name
        _write_complex_slider_map(
            path,
            obj,
            title=f"Reasoning step filtered simplicial complex map: q{idx}",
            subtitle=f"record_id={record_id}; level={row.get('level')}; path={row.get('path', [])}",
        )
        rows.append(
            {
                "index": idx,
                "record_id": record_id,
                "level": int(row.get("level", 0) or 0),
                "path": row.get("path", []),
                "file": file_name,
                "summary": obj_summary.get("summary", {}),
                "simplex_tree": obj_summary.get("simplex_tree", {}),
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
    initial = thresholds[-1] if thresholds else float("inf")
    base_traces = _complex_slider_traces(obj, coords3, threshold=initial, panel_index_by_label=panel_index_by_label)
    fig = go.Figure(data=base_traces)
    frames = []
    for threshold in thresholds:
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
        fig.update_layout(
            sliders=[
                {
                    "active": len(steps) - 1,
                    "currentvalue": {"prefix": "radius/filtration <= ", "font": {"color": "#dbeafe"}},
                    "pad": {"t": 44},
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
                            "args": [None, {"frame": {"duration": 260, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}],
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
        legend=dict(itemsizing="constant"),
    )
    _write_plotly_dark_html(
        path,
        fig,
        title,
        _simplicial_panel_items(panel_objects, panel_hovers),
        show_filtration_slider=True,
    )


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
        fallback = dict(obj)
        fallback["simplex_tree"] = {
            "backend": "json-fallback",
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        return fallback


def _complex_slider_traces(
    obj: dict[str, object],
    coords3: dict[str, tuple[float, float, float]],
    threshold: float,
    panel_index_by_label: dict[str, int] | None = None,
) -> list[go.Scatter3d | go.Mesh3d]:
    vertices = [s for s in obj.get("simplices", []) if isinstance(s, dict) and int(s.get("dimension", -1)) == 0 and float(s.get("filtration", 0.0) or 0.0) <= threshold + 1e-12]
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
    traces: list[go.Scatter3d | go.Mesh3d] = [
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(width=4, color="rgba(125,211,252,0.66)"),
            hovertext=edge_hover,
            hoverinfo="text",
            name="1-simplices",
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
                colorbar=dict(title="filtration"),
                line=dict(color="#e8eef8", width=1),
            ),
            text=[label[:16] for label in vertex_labels],
            textposition="top center",
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
            color="rgba(94,234,212,0.22)",
            opacity=0.36,
            name="2-simplices",
            hoverinfo="skip",
            showscale=False,
        )
    )
    return traces


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
        f"<td>{int(row.get('level', 0))}</td>"
        f"<td>{html.escape(_json_clip(row.get('path', []), 96))}</td>"
        f"<td>{html.escape(_json_clip(row.get('summary', {}), 140))}</td>"
        "</tr>"
        for row in rows
    ) or "<tr><td colspan='5'>No reasoning-step complexes were generated.</td></tr>"
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
    <p>Each row opens a separate 3D PCoA/MDS radius-filtered complex for one sampled graph-of-thought state.  These are deliberately separate from the embedding-space trajectory map.</p>
    <table>
      <thead><tr><th>index</th><th>complex map</th><th>level</th><th>path</th><th>summary</th></tr></thead>
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
        return {"tropical_support_heatmap": str(path)}
    n = len(tokens)
    support_indices = []
    for token in tokens:
        active = int(token.get("active_support_index", -1))
        if 0 <= active < n and active not in support_indices:
            support_indices.append(active)
    support_indices = sorted(support_indices, key=lambda idx: (-sum(1 for token in tokens if int(token.get("active_support_index", -1)) == idx), idx))
    if not support_indices:
        support_indices = [0]
    z = np.zeros((n, len(support_indices)), dtype=float)
    hover_grid: list[list[str]] = []
    query_labels = []
    for row_idx, token in enumerate(tokens):
        active = int(token.get("active_support_index", -1))
        margin = float(token.get("margin", 0.0) or 0.0)
        query_labels.append(_support_token_label(row_idx, token))
        hover_row = []
        for col_idx, support_idx in enumerate(support_indices):
            support = tokens[support_idx] if 0 <= support_idx < n else {}
            if active == support_idx:
                z[row_idx, col_idx] = margin if margin != 0.0 else 1.0
            hover_row.append(
                f"query={html.escape(_support_token_label(row_idx, token, long=True))}<br>"
                f"active support={html.escape(_support_token_label(support_idx, support, long=True))}<br>"
                f"value={z[row_idx, col_idx]:.4f}<br>"
                f"query text={_html_clip(token.get('text', ''), 360)}<br>"
                f"support text={_html_clip(support.get('text', ''), 360)}"
            )
        hover_grid.append(hover_row)
    support_labels = [_support_token_label(idx, tokens[idx]) for idx in support_indices]
    counts = np.asarray([sum(1 for token in tokens if int(token.get("active_support_index", -1)) == idx) for idx in support_indices], dtype=float)
    mean_margins = []
    for idx in support_indices:
        vals = [float(token.get("margin", 0.0) or 0.0) for token in tokens if int(token.get("active_support_index", -1)) == idx]
        mean_margins.append(float(np.mean(vals)) if vals else 0.0)
    collapse_rate = float(counts.max() / max(float(n), 1.0)) if counts.size else 0.0
    support_probs = counts / max(float(counts.sum()), 1.0)
    support_entropy = float(-np.sum(support_probs * np.log2(np.maximum(support_probs, 1e-12)))) if counts.size else 0.0
    effective_supports = float(2.0 ** support_entropy) if counts.size else 0.0
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "heatmap"}, {"type": "bar"}]],
        column_widths=[0.68, 0.32],
        horizontal_spacing=0.08,
        subplot_titles=("Observed active-support margins", "Support frequency and mean margin"),
    )
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=support_labels,
            y=query_labels,
            colorscale="Viridis",
            colorbar=dict(title="margin", x=0.64),
            customdata=hover_grid,
            hovertemplate="%{customdata}<extra></extra>",
            zmin=float(np.nanmin(z)) if z.size and np.isfinite(z).any() else None,
            zmax=float(np.nanmax(z)) if z.size and np.isfinite(z).any() else None,
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
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        template="plotly_dark",
        title=(
            "Tropical active-support audit: observed supports only "
            f"<br><sup>unique supports={len(support_indices)}/{n}, effective supports={effective_supports:.2f}, "
            f"entropy={support_entropy:.3f} bits, top-support collapse rate={collapse_rate:.3f}; "
            "uniform blocks mean active-support collapse or nearly constant margins</sup>"
        ),
    )
    fig.update_xaxes(title_text="active support token", tickangle=55, row=1, col=1)
    fig.update_yaxes(title_text="query token", row=1, col=1)
    fig.update_xaxes(title_text="support token", tickangle=55, row=1, col=2)
    fig.update_yaxes(title_text="query count", row=1, col=2)
    _write_plotly_dark_html(path, fig, "Tropical active-support heatmap")
    return {"tropical_support_heatmap": str(path)}


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
    if growth:
        _write_growth_persistence_barcode(barcode, topology, growth, title_prefix=title_prefix)
        _write_growth_persistence_module(module_path, topology, growth, title_prefix=title_prefix)
        return {"persistence_barcode": str(barcode), "persistence_module_betti": str(module_path)}

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
    return {"persistence_barcode": str(barcode), "persistence_module_betti": str(module_path)}


def _write_growth_persistence_barcode(path: Path, topology: dict[str, object], growth: list[object], title_prefix: str = "") -> None:
    rows = _trajectory_growth_rows(topology, growth)
    fig = go.Figure()
    colors = {0: "#55d6be", 1: "#7aa2ff", 2: "#fbbf24", 3: "#fb7185"}
    panel_objects = []
    panel_hover = []
    backend_counts: dict[str, int] = defaultdict(int)
    synthetic_count = 0
    interval_count = 0
    max_death = 1.0
    for row in rows:
        topo = row.get("topological_algebra", {})
        backend_counts[_persistence_backend_label(topo if isinstance(topo, dict) else {})] += len(row.get("intervals", []))
        for interval in row["intervals"]:
            interval_count += 1
            if interval.get("synthetic"):
                synthetic_count += 1
            death = interval.get("display_death")
            if isinstance(death, (int, float)) and math.isfinite(float(death)):
                max_death = max(max_death, float(death))
    y_cursor = 0
    tick_vals: list[int] = []
    tick_text: list[str] = []
    for row_idx, row in enumerate(rows):
        level = int(row["level"])
        obj = row.get("filtered_simplicial_object", {})
        topo = row.get("topological_algebra", {})
        panel_objects.append(obj if isinstance(obj, dict) else {})
        panel_hover.append(_topology_growth_hover(level, obj if isinstance(obj, dict) else {}, topo if isinstance(topo, dict) else {}))
        for interval_idx, interval in enumerate(row["intervals"]):
            dim = int(interval.get("dimension", 0))
            birth = float(interval.get("birth", 0.0))
            death = float(interval.get("display_death", max_death))
            true_death = "inf" if interval.get("infinite") else f"{float(interval.get('death', death)):.4g}"
            source = interval.get("source") or _persistence_backend_label(topo if isinstance(topo, dict) else {})
            synthetic = interval.get("synthetic")
            hover = (
                f"<b>trajectory growth level {level}</b>"
                f"<br>H{dim} interval [{birth:.4g}, {true_death}]"
                f"<br>source={html.escape(str(source))}"
                + (f"<br>synthetic fallback={html.escape(str(synthetic))}" if synthetic else "")
                + f"<br>complex: {_summary_line(obj if isinstance(obj, dict) else {})}"
                f"<br>{_derived_signature_line(topo if isinstance(topo, dict) else {})}"
                f"<br>{_free_resolution_line(topo if isinstance(topo, dict) else {})}"
            )
            y = y_cursor
            tick_vals.append(y)
            tick_text.append(f"L{level} H{dim} #{interval_idx}")
            y_cursor += 1
            fig.add_trace(
                go.Scatter(
                    x=[birth, death],
                    y=[y, y],
                    mode="lines+markers",
                    line=dict(width=8, color=colors.get(dim, "#cbd5e1")),
                    marker=dict(size=5, color=colors.get(dim, "#cbd5e1")),
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
    if not rows:
        fig.add_annotation(text="No trajectory growth topology available.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    fig.update_layout(
        template="plotly_dark",
        title=(
            f"{title_prefix}persistent homology growth barcode"
            "<br><sup>standard interval view; "
            f"intervals={interval_count}, synthetic fallback={synthetic_count}; "
            f"backends={html.escape(_json_clip(dict(backend_counts), 160))}; "
            "hover shows GUDHI/topology provenance and free-resolution summary</sup>"
        ),
        xaxis_title="filtration birth/death",
        yaxis=dict(title="interval", tickmode="array", tickvals=tick_vals, ticktext=tick_text, autorange="reversed"),
        legend=dict(itemsizing="constant"),
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
            "Free-resolution proxy ranks by trajectory level",
        ),
        horizontal_spacing=0.08,
    )
    panel_objects = []
    panel_hover = []
    betti_cells: list[dict[str, object]] = []
    free_cells: list[dict[str, object]] = []
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
        fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                marker=dict(color=y, colorscale="Turbo", line=dict(color="#e8eef8", width=0.7)),
                hovertext=[str(cell["hover"]) for cell in free_cells],
                hoverinfo="text",
                name="free-resolution proxy",
            ),
            row=1,
            col=2,
        )
    if not rows:
        fig.add_annotation(text="No trajectory growth persistence module available.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    fig.update_layout(
        template="plotly_dark",
        title=(
            f"{title_prefix}multiparameter persistence and free-resolution growth"
            "<br><sup>2D matrix/bar view replaces decorative 3D spikes; hover opens the corresponding filtered complex panel</sup>"
        ),
    )
    fig.update_xaxes(title_text="homology dimension / filtration", tickangle=55, row=1, col=1)
    fig.update_yaxes(title_text="trajectory growth level", row=1, col=1)
    fig.update_xaxes(title_text="level and free module", tickangle=55, row=1, col=2)
    fig.update_yaxes(title_text="free rank", row=1, col=2)
    _write_plotly_dark_html(
        path,
        fig,
        f"{title_prefix}multiparameter persistence and free-resolution growth",
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
        obj = item.get("filtered_simplicial_object") if isinstance(item.get("filtered_simplicial_object"), dict) else {}
        intervals, _ = _prepare_barcode_intervals(_intervals_or_synthetic_h0(topo), epsilon=0.0)
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
        intervals, _ = _prepare_barcode_intervals(_intervals_or_synthetic_h0(topology), epsilon=0.0)
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


def _intervals_or_synthetic_h0(topology: dict[str, object]) -> list[dict[str, object]]:
    intervals = topology.get("persistence", {}).get("intervals", []) if isinstance(topology.get("persistence"), dict) else []
    if intervals:
        return [row for row in intervals if isinstance(row, dict)]
    states = topology.get("persistence_module", {}).get("states", []) if isinstance(topology.get("persistence_module"), dict) else []
    if not states:
        return []
    first = states[0] if isinstance(states[0], dict) else {}
    betti0 = int(first.get("betti", {}).get("0", 0)) if isinstance(first.get("betti"), dict) else 0
    if betti0 <= 0:
        return []
    birth = float(first.get("threshold", 0.0) or 0.0)
    return [{"dimension": 0, "birth": birth, "death": None, "infinite": True, "synthetic": "module_beta0"}]


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
    proxy = ca.get("multiparameter_free_resolution_proxy", {}) if isinstance(ca.get("multiparameter_free_resolution_proxy"), dict) else {}
    ring = proxy.get("ring", "F2[x_filtration,x_dimension,x_position]")
    return f"free-resolution proxy over {ring}: " + (", ".join(ranks) if ranks else "no displayed free modules")


def _free_resolution_modules(topology: dict[str, object]) -> list[dict[str, object]]:
    ca = topology.get("commutative_algebra", {}) if isinstance(topology, dict) else {}
    proxy = ca.get("multiparameter_free_resolution_proxy", {}) if isinstance(ca.get("multiparameter_free_resolution_proxy"), dict) else {}
    modules = proxy.get("free_chain_modules", []) if isinstance(proxy, dict) else []
    if modules:
        return [row for row in modules if isinstance(row, dict)]
    taylor = ca.get("taylor_resolution_upper_bound", {}) if isinstance(ca.get("taylor_resolution_upper_bound"), dict) else {}
    ranks = taylor.get("ranks", []) if isinstance(taylor, dict) else []
    return [row for row in ranks if isinstance(row, dict)]


def write_graphcg_trajectory_visualization(scaling_report: dict[str, object], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    path = output_dir / "graphcg_direction_cosines.html"
    candidates = [row for row in scaling_report.get("candidates", []) if isinstance(row, dict)]
    matrices = []
    labels = []
    hover_rows = []
    for idx, row in enumerate(candidates):
        proj = row.get("graphcg_projection")
        if isinstance(proj, dict) and proj.get("all_direction_cosines") is not None:
            matrices.append([float(v) for v in proj["all_direction_cosines"]])
            labels.append(_candidate_axis_label(row, idx))
            hover_rows.append(
                f"<b>{html.escape(_candidate_axis_label(row, idx, long=True))}</b>"
                f"<br>level={row.get('level')} path={html.escape(_json_clip(row.get('path', []), 180))}"
                f"<br>NLL={float(row.get('nll', 0.0) or 0.0):.4f} score={float(row.get('score', 0.0) or 0.0):.4f}"
            )
    if not matrices:
        _write_dark_empty(path, "No GraphCG projection diagnostics available.")
        return {"graphcg_direction_cosines": str(path)}
    matrix = np.asarray(matrices, dtype=float)
    mean_abs = np.mean(np.abs(matrix), axis=0)
    direction_count = int(matrix.shape[1])
    display_count = min(48, direction_count)
    top_idx = np.argsort(mean_abs)[::-1][:display_count]
    top_idx = top_idx[np.argsort(top_idx)]
    z = matrix[:, top_idx]
    custom = [
        [
            hover_rows[row_idx]
            + f"<br>direction={int(direction_idx)}"
            + f"<br>cosine={z[row_idx, col_idx]:.5f}"
            + f"<br>mean |cos| for direction={mean_abs[int(direction_idx)]:.5f}"
            for col_idx, direction_idx in enumerate(top_idx)
        ]
        for row_idx in range(z.shape[0])
    ]
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "heatmap"}, {"type": "scatter"}]],
        column_widths=[0.72, 0.28],
        horizontal_spacing=0.08,
        subplot_titles=("Top active GraphCG directions across GoT states", "Full-rank direction activity spectrum"),
    )
    fig.add_trace(
        go.Heatmap(
            z=z,
            y=labels,
            x=[f"d{int(idx)}" for idx in top_idx],
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="cosine", x=0.68),
            customdata=custom,
            hovertemplate="%{customdata}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    sorted_activity = np.sort(mean_abs)[::-1]
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
        row=1,
        col=2,
    )
    fig.update_layout(
        template="plotly_dark",
        title=(
            "GraphCG full-rank direction audit"
            f"<br><sup>displaying top {display_count}/{direction_count} directions by mean absolute cosine; full-rank spectrum retained</sup>"
        ),
    )
    fig.update_xaxes(title_text="GraphCG direction", tickangle=45, row=1, col=1)
    fig.update_yaxes(title_text="GoT state", row=1, col=1)
    fig.update_xaxes(title_text="direction rank by activity", row=1, col=2)
    fig.update_yaxes(title_text="mean absolute cosine", row=1, col=2)
    _write_plotly_dark_html(path, fig, "GraphCG direction cosines along GoT candidates")
    return {"graphcg_direction_cosines": str(path)}


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
        _write_dark_empty(path, "No analogical memories retrieved.")
        map_path.write_text(json.dumps({"query": {}, "maps": []}, indent=2), encoding="utf-8")
        return {"analogical_memory_retrieval_html": str(path), "analogical_simplicial_maps": str(map_path)}
    bank_records = _load_memory_bank_records(memory.get("bank_path", ""))
    enriched = [_enrich_memory_row(row, bank_records) for row in rows]
    query = query_context if isinstance(query_context, dict) else {}
    query_complex = query.get("filtered_simplicial_object") if isinstance(query.get("filtered_simplicial_object"), dict) else {}
    query_topology = query.get("topological_algebra") if isinstance(query.get("topological_algebra"), dict) else {}
    if not query_complex and enriched:
        query_complex = enriched[0].get("filtered_simplicial_object", {}) if isinstance(enriched[0].get("filtered_simplicial_object"), dict) else {}
    if not query_topology and enriched:
        query_topology = enriched[0].get("topological_algebra", {}) if isinstance(enriched[0].get("topological_algebra"), dict) else {}

    pair_pages: list[dict[str, object]] = []
    map_reports: list[dict[str, object]] = []
    for idx, row in enumerate(enriched):
        pair_path = path if idx == 0 else output_dir / f"analogical_memory_map_{idx + 1:02d}.html"
        fig, panel_items, map_report = _analogical_pair_figure(row, idx, query_complex, query_topology)
        _write_plotly_dark_html(
            pair_path,
            fig,
            f"Analogical simplicial map retrieval rank {idx + 1}",
            panel_items,
            show_filtration_slider=True,
        )
        map_report["pair_page"] = str(pair_path)
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
                "query_summary": query_complex.get("summary", {}) if isinstance(query_complex, dict) else {},
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
    mem_complex = (
        row.get("trajectory_filtered_simplicial_object")
        if isinstance(row.get("trajectory_filtered_simplicial_object"), dict)
        else row.get("filtered_simplicial_object", {})
    )
    mem_complex = _gudhi_canonical_complex(mem_complex if isinstance(mem_complex, dict) else {})
    mem_topology = row.get("topological_algebra", {}) if isinstance(row.get("topological_algebra"), dict) else {}
    mem_layout = _complex_3d_layout(mem_complex, slab=slab)
    sim = _topological_similarity_summary(query_topology, mem_topology, row)
    sim_map = _simplicial_map_between_complexes(query_complex, mem_complex)
    map_report = {
        "memory_id": row.get("memory_id"),
        "record_id": row.get("record_id"),
        "rank": idx + 1,
        "domain_complex_summary": query_complex.get("summary", {}) if isinstance(query_complex, dict) else {},
        "codomain_complex_summary": mem_complex.get("summary", {}) if isinstance(mem_complex, dict) else {},
        "domain_simplex_tree": query_complex.get("simplex_tree", {}) if isinstance(query_complex, dict) else {},
        "codomain_simplex_tree": mem_complex.get("simplex_tree", {}) if isinstance(mem_complex, dict) else {},
        "codomain_complex_source": "trajectory_filtered_simplicial_object" if isinstance(row.get("trajectory_filtered_simplicial_object"), dict) else "filtered_simplicial_object",
        **sim,
        **sim_map,
    }
    panel_hover = (
        f"<b>codomain memory {idx + 1}</b>: {html.escape(str(row.get('memory_id', idx)))}"
        f"<br>retrieval={float(row.get('retrieval_score', 0.0)):.4f}"
        f"<br>persistent homology similarity={sim['persistent_homology_similarity']:.4f}"
        f"<br>free-resolution similarity={sim['free_resolution_similarity']:.4f}"
        f"<br>derived signature similarity={sim['derived_signature_similarity']:.4f}"
        f"<br>simplicial edge preservation={sim_map['edge_preservation_rate']:.4f}"
        f"<br>{_summary_line(mem_complex)}"
        f"<br>{_derived_signature_line(mem_topology)}"
        f"<br>{_free_resolution_line(mem_topology)}"
    )
    panel_items.extend(_simplicial_panel_items([mem_complex], [panel_hover]))

    thresholds = _combined_display_thresholds(query_complex, mem_complex)
    initial = thresholds[-1] if thresholds else 1.0
    data = _analogical_pair_traces(query_complex, query_layout, mem_complex, mem_layout, sim_map, row, sim, idx, panel_idx, initial)
    fig = go.Figure(data=data)
    frames = []
    for threshold in thresholds:
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
                    "active": len(steps) - 1,
                    "currentvalue": {"prefix": "domain/codomain filtration <= ", "font": {"color": "#dbeafe"}},
                    "pad": {"t": 42},
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
    fig.update_layout(
        template="plotly_dark",
        title=(
            f"Analogical reasoning memory retrieval as simplicial maps: rank {idx + 1} "
            "trajectory-complex map from query domain to retrieved codomain"
            "<br><sup>binary filtered-complex map; slider filters domain, codomain, and visible map edges; hover shows vertex-level model output and topology provenance</sup>"
        ),
        scene=dict(
            xaxis=dict(title="domain / codomain embedding slabs", range=[-0.78, slab + 0.78]),
            yaxis=dict(title="PCoA/MDS-2", range=[-1.25, 1.25]),
            zaxis=dict(title="PCoA/MDS-3", range=[-1.25, 1.25]),
            camera=dict(eye=dict(x=1.45, y=1.25, z=0.85), center=dict(x=0.02, y=0.0, z=0.0)),
        ),
        legend=dict(itemsizing="constant"),
    )
    return fig, panel_items, map_report


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
    traces.append(_simplicial_map_trace(query_complex, mem_complex, query_layout, mem_layout, sim_map, panel_idx, row, sim, threshold))
    marker = dict(size=11, color=_memory_color(idx), showscale=False, line=dict(width=1.2, color="#e8eef8"))
    panel_hover = (
        f"<b>rank {idx + 1} retrieved memory</b>"
        f"<br>memory_id={html.escape(str(row.get('memory_id', idx)))}"
        f"<br>retrieval={float(row.get('retrieval_score', 0.0)):.4f}"
        f"<br>PH similarity={sim['persistent_homology_similarity']:.4f}"
        f"<br>free-resolution similarity={sim['free_resolution_similarity']:.4f}"
        f"<br>derived similarity={sim['derived_signature_similarity']:.4f}"
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
            text=[label[:14] for label in labels],
            textposition="top center",
            name=name,
            hovertext=vertex_hover,
            hoverinfo="text",
            customdata=[panel_idx] * len(labels),
        ),
    ]


def _complex_vertex_hover(label: str, vertex_row: dict[str, object], prefix: str) -> str:
    return f"<b>{html.escape(prefix)}</b><br>" + _vertex_readable_summary({**vertex_row, "simplex": [label]}, include_output=True)


def _simplicial_map_trace(
    query_obj: dict[str, object],
    memory_obj: dict[str, object],
    query_layout: dict[str, tuple[float, float, float]],
    memory_layout: dict[str, tuple[float, float, float]],
    sim_map: dict[str, object],
    panel_idx: int,
    row: dict[str, object],
    sim: dict[str, float],
    threshold: float,
) -> go.Scatter3d:
    q_vertex = _vertex_by_label(query_obj)
    m_vertex = _vertex_by_label(memory_obj)
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
        q_filtration = float(q_vertex.get(q, {}).get("filtration", 0.0) or 0.0)
        m_filtration = float(m_vertex.get(m, {}).get("filtration", 0.0) or 0.0)
        if max(q_filtration, m_filtration) > threshold + 1e-12:
            continue
        qx, qy, qz = query_layout[q]
        mx, my, mz = memory_layout[m]
        q_summary = _vertex_readable_summary({**q_vertex.get(q, {}), "simplex": [q]}, include_output=True)
        m_summary = _vertex_readable_summary({**m_vertex.get(m, {}), "simplex": [m]}, include_output=True)
        text = (
            f"<b>simplicial map candidate</b>"
            f"<br>{html.escape(q)} -> {html.escape(m)}"
            f"<br>vertex score={float(mapping.get('score', 0.0)):.4f}"
            f"<br>edge preservation={float(sim_map.get('edge_preservation_rate', 0.0)):.4f}"
            f"<br>2-simplex preservation={float(sim_map.get('two_simplex_preservation_rate', 0.0)):.4f}"
            f"<br>PH similarity={sim['persistent_homology_similarity']:.4f}"
            f"<br>free-resolution similarity={sim['free_resolution_similarity']:.4f}"
            f"<br>derived similarity={sim['derived_signature_similarity']:.4f}"
            f"<br><br><b>domain vertex</b><br>{q_summary}"
            f"<br><br><b>codomain vertex</b><br>{m_summary}"
        )
        xs.extend([qx, mx, None])
        ys.extend([qy, my, None])
        zs.extend([qz, mz, None])
        hover.extend([text, text, None])
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color="rgba(251,191,36,0.58)", width=3),
        name=f"simplicial map to {row.get('memory_id')}",
        hovertext=hover,
        hoverinfo="text",
        customdata=[panel_idx if value is not None else None for value in xs],
    )


def _write_analogical_topk_index(path: Path, pair_pages: list[dict[str, object]], map_reports: list[dict[str, object]]) -> None:
    rows = []
    for page, report in zip(pair_pages, map_reports, strict=False):
        rel = html.escape(Path(str(page.get("path", ""))).name)
        rows.append(
            "<tr>"
            f"<td>{int(page.get('rank', 0))}</td>"
            f"<td><a href='{rel}'>{html.escape(str(page.get('memory_id', 'memory')))}</a></td>"
            f"<td>{float(page.get('retrieval_score', 0.0)):.4f}</td>"
            f"<td>{float(report.get('persistent_homology_similarity', 0.0)):.4f}</td>"
            f"<td>{float(report.get('free_resolution_similarity', 0.0)):.4f}</td>"
            f"<td>{float(report.get('derived_signature_similarity', 0.0)):.4f}</td>"
            f"<td>{float(report.get('edge_preservation_rate', 0.0)):.4f}</td>"
            "</tr>"
        )
    body = "\n".join(rows) or "<tr><td colspan='7'>No retrieved memories.</td></tr>"
    path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Analogical top-k simplicial maps</title>
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
    <h1>Analogical top-k retrieval as separate filtered simplicial maps</h1>
    <p>Each row opens one binary map from the query filtered simplicial complex to exactly one retrieved memory complex. Top-k remains available without overlaying several codomains into a single map.</p>
    <table>
      <thead><tr><th>rank</th><th>binary map</th><th>retrieval</th><th>PH</th><th>free res.</th><th>derived</th><th>edge map</th></tr></thead>
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
            f"<b>simplicial map candidate</b>"
            f"<br>{html.escape(q)} -> {html.escape(m)}"
            f"<br>vertex score={float(mapping.get('score', 0.0)):.4f}"
            f"<br>edge preservation={float(sim_map.get('edge_preservation_rate', 0.0)):.4f}"
            f"<br>2-simplex preservation={float(sim_map.get('two_simplex_preservation_rate', 0.0)):.4f}"
            f"<br>PH similarity={sim['persistent_homology_similarity']:.4f}"
            f"<br>free-resolution similarity={sim['free_resolution_similarity']:.4f}"
            f"<br>derived similarity={sim['derived_signature_similarity']:.4f}"
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
                name=f"simplicial map to {row.get('memory_id')}",
                hovertext=hover,
                hoverinfo="text",
                customdata=[panel_idx if value is not None else None for value in xs],
            )
        )


def _topological_similarity_summary(query_topology: dict[str, object], memory_topology: dict[str, object], row: dict[str, object]) -> dict[str, float]:
    q_sig = _signature_numeric_vector(query_topology)
    m_sig = _signature_numeric_vector(memory_topology)
    q_free = _free_rank_vector(query_topology)
    m_free = _free_rank_vector(memory_topology)
    q_ph = _persistence_numeric_vector(query_topology)
    m_ph = _persistence_numeric_vector(memory_topology)
    return {
        "retrieval_score": float(row.get("retrieval_score", 0.0)),
        "embedding_similarity": float(row.get("embedding_similarity", 0.0)),
        "signature_similarity": float(row.get("signature_similarity", 0.0)),
        "derived_signature_similarity": _cosine_similarity(q_sig, m_sig),
        "free_resolution_similarity": _cosine_similarity(q_free, m_free),
        "persistent_homology_similarity": _cosine_similarity(q_ph, m_ph),
    }


def _simplicial_map_between_complexes(query_obj: dict[str, object], memory_obj: dict[str, object], max_vertices: int = 54) -> dict[str, object]:
    query_vertices = _complex_vertex_records(query_obj)[:max_vertices]
    memory_vertices = _complex_vertex_records(memory_obj)[:max_vertices]
    if not query_vertices or not memory_vertices:
        return {
            "vertex_map": [],
            "displayed_domain_vertices": len(query_vertices),
            "displayed_codomain_vertices": len(memory_vertices),
            "edge_preservation_rate": 0.0,
            "two_simplex_preservation_rate": 0.0,
            "is_simplicial_on_displayed_skeleton": False,
        }
    used: set[str] = set()
    vertex_map = []
    for q in query_vertices:
        scored = sorted(
            ((_vertex_match_score(q, m), m) for m in memory_vertices),
            key=lambda item: item[0],
            reverse=True,
        )
        chosen = None
        for score, candidate in scored:
            if candidate["label"] not in used:
                chosen = (score, candidate)
                break
        if chosen is None:
            chosen = scored[0]
        score, m = chosen
        used.add(m["label"])
        vertex_map.append({"query_vertex": q["label"], "memory_vertex": m["label"], "score": float(score)})
    mapping = {row["query_vertex"]: row["memory_vertex"] for row in vertex_map}
    memory_edges = {frozenset(pair) for pair in _complex_edge_pairs(memory_obj)}
    query_edges = _complex_edge_pairs(query_obj)
    preserved_edges = 0
    checked_edges = 0
    for a, b in query_edges:
        if a not in mapping or b not in mapping:
            continue
        checked_edges += 1
        ma, mb = mapping[a], mapping[b]
        if ma == mb or frozenset((ma, mb)) in memory_edges:
            preserved_edges += 1
    memory_triangles = {frozenset(simplex) for simplex in _complex_simplices(memory_obj, 2)}
    query_triangles = _complex_simplices(query_obj, 2)
    preserved_triangles = 0
    checked_triangles = 0
    for simplex in query_triangles:
        if any(v not in mapping for v in simplex):
            continue
        checked_triangles += 1
        mapped = frozenset(mapping[v] for v in simplex)
        if len(mapped) <= 2 or mapped in memory_triangles:
            preserved_triangles += 1
    edge_rate = preserved_edges / checked_edges if checked_edges else 1.0
    triangle_rate = preserved_triangles / checked_triangles if checked_triangles else 1.0
    return {
        "vertex_map": vertex_map,
        "displayed_domain_vertices": len(query_vertices),
        "displayed_codomain_vertices": len(memory_vertices),
        "checked_edges": checked_edges,
        "preserved_edges": preserved_edges,
        "edge_preservation_rate": float(edge_rate),
        "checked_two_simplices": checked_triangles,
        "preserved_two_simplices": preserved_triangles,
        "two_simplex_preservation_rate": float(triangle_rate),
        "is_simplicial_on_displayed_skeleton": bool(edge_rate >= 0.999 and triangle_rate >= 0.999),
    }


def _complex_vertex_records(obj: dict[str, object]) -> list[dict[str, object]]:
    simplices = obj.get("simplices", []) if isinstance(obj, dict) else []
    vertices = []
    for idx, simplex in enumerate(simplices):
        if not isinstance(simplex, dict) or int(simplex.get("dimension", -1)) != 0:
            continue
        label = _simplex_label(simplex, fallback=f"v{idx}")
        vertices.append(
            {
                "label": label,
                "type": str(simplex.get("type", "vertex")),
                "text": str(simplex.get("text", "")),
                "filtration": float(simplex.get("filtration", 0.0) or 0.0),
                "embedding": simplex.get("embedding", []),
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


def _simplex_label(simplex: dict[str, object], fallback: str) -> str:
    raw = simplex.get("simplex", [])
    if isinstance(raw, list) and raw:
        return str(raw[0])
    return fallback


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
    links = "\n".join(
        f'<li><a href="{Path(path).name}"><span>{name}</span><code>{Path(path).name}</code></a></li>'
        for name, path in sorted(paths.items())
    )
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


def _write_plotly_dark_html(path: Path, fig: go.Figure, title: str, panel_items: list[dict[str, str]] | None = None, show_filtration_slider: bool = False) -> None:
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
    chart = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displaylogo": False, "responsive": True})
    items = panel_items or []
    initial = items[0] if items else {"title": "Filtered simplicial object", "svg": "", "summary": "Hover a reasoning node to render its complex."}
    controls_html = (
        """<div class=\"filtration-controls\" id=\"filtration-controls\">
        <label><span>Filtration radius</span><strong id=\"filtration-value\">all</strong></label>
        <input id=\"filtration-slider\" type=\"range\" min=\"0\" max=\"1\" step=\"0.001\" value=\"1\" aria-label=\"filtered simplicial complex radius\">
        <div class=\"hint\" id=\"filtration-hint\">Shows simplices with scalar filtration at or below the selected radius. Multiparameter summaries remain in the JSON payload.</div>
      </div>"""
        if show_filtration_slider
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
    .chart {{
      min-width: 0;
      min-height: 100vh;
      border-right: 1px solid var(--edge);
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
      max-height: 168px;
      overflow: hidden;
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
  <main class="layout">
    <section class="chart" id="chart">{chart}</section>
    <aside class="panel" aria-live="polite">
      <h1 id="simplicial-title">{html.escape(initial["title"])}</h1>
      <div class="summary" id="simplicial-summary">{initial["summary"]}</div>
      {controls_html}
      <div class="simplicial-object-panel" id="simplicial-svg">{initial["svg"]}</div>
    </aside>
  </main>
  <div class="hover-simplicial-card" id="hover-simplicial-card" role="tooltip" aria-label="hovered filtered simplicial object">
    <h2 id="hover-simplicial-title">{html.escape(initial["title"])}</h2>
    <div class="hover-summary" id="hover-simplicial-summary">{initial["summary"]}</div>
    <div id="hover-simplicial-svg">{initial["svg"]}</div>
  </div>
  <script>
    const simplicialPanels = {json.dumps(items)};
    const panelTitle = document.getElementById("simplicial-title");
    const panelSummary = document.getElementById("simplicial-summary");
    const panelSvg = document.getElementById("simplicial-svg");
    const hoverCard = document.getElementById("hover-simplicial-card");
    const hoverTitle = document.getElementById("hover-simplicial-title");
    const hoverSummary = document.getElementById("hover-simplicial-summary");
    const hoverSvg = document.getElementById("hover-simplicial-svg");
    const filtrationSlider = document.getElementById("filtration-slider");
    const filtrationValue = document.getElementById("filtration-value");
    const filtrationHint = document.getElementById("filtration-hint");
    let activePanelIndex = 0;
    function setPanel(index) {{
      const item = simplicialPanels[index];
      if (!item) return;
      activePanelIndex = index;
      panelTitle.textContent = item.title || "Filtered simplicial object";
      panelSummary.innerHTML = item.summary || "";
      panelSvg.innerHTML = item.svg || "";
      configureFiltrationSlider(item, panelSvg);
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
      filtrationSlider.value = hasRange ? String(max) : "1";
      filtrationSlider.disabled = !hasRange;
      filtrationHint.textContent = item.multiparameter_hint || "Shows simplices with scalar filtration at or below the selected radius. Multiparameter summaries remain in the JSON payload.";
      applyFiltrationThreshold(root, hasRange ? max : Infinity);
      filtrationValue.textContent = hasRange ? max.toFixed(3) : "all";
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
      }});
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
    if (plot && simplicialPanels.length) {{
      setPanel(0);
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
      plot.on("plotly_unhover", hideHoverCard);
      plot.on("plotly_relayout", hideHoverCard);
    }}
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def _write_dark_empty(path: Path, message: str) -> None:
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<style>:root{color-scheme:dark;background:#090b12;color:#e8eef8}"
        "body{margin:0;background:#090b12;color:#e8eef8;font-family:Inter,ui-sans-serif,system-ui;padding:24px}</style>"
        f"</head><body><p>{html.escape(message)}</p></body></html>",
        encoding="utf-8",
    )


def _simplicial_panel_items(objects: list[dict[str, object]], hover: list[str]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
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
        summary_html = compact_summary
        if idx < len(hover):
            summary_html += f"<br>{hover[idx]}"
        items.append(
            {
                "title": title,
                "summary": summary_html,
                "compact_summary": compact_summary,
                "svg": _simplicial_object_svg(obj if isinstance(obj, dict) else {}),
                "filtration_min": filtration_min,
                "filtration_max": filtration_max,
                "multiparameter_hint": (
                    "Slider filters the scalar radius/filtration attached to displayed simplices. "
                    "Three-parameter persistence grades are exported in the JSON audit payload."
                ),
            }
        )
    return items


def _simplicial_object_svg(obj: dict[str, object], width: int = 380, height: int = 270, max_vertices: int = 64) -> str:
    simplices = obj.get("simplices", []) if isinstance(obj, dict) else []
    vertices = [s for s in simplices if isinstance(s, dict) and int(s.get("dimension", -1)) == 0]
    edges = [s for s in simplices if isinstance(s, dict) and int(s.get("dimension", -1)) == 1]
    triangles = [s for s in simplices if isinstance(s, dict) and int(s.get("dimension", -1)) == 2]
    labels = [str((s.get("simplex") or [f"v{idx}"])[0]) for idx, s in enumerate(vertices)]
    if not labels:
        labels = ["empty"]
    visible_labels = labels[:max_vertices]
    visible = set(visible_labels)
    summary = obj.get("summary", {}) if isinstance(obj, dict) else {}
    thresholds = obj.get("thresholds", []) if isinstance(obj, dict) else []
    coords3, projected, layout_kind = _simplicial_pca3_radius_layout(visible_labels, vertices, edges, width=width, height=height)

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
                f"fill='rgba(94,234,212,0.16)' stroke='rgba(94,234,212,0.46)' stroke-width='1.2'>"
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
        stroke_width = 1.5 + 1.6 * max(0.0, min(1.0, avg_z))
        parts.append(
            f"<line class='one-simplex pca-radius-edge' data-filtration='{filt:.8f}' data-pca-z='{avg_z:.6f}' x1='{a[0]:.1f}' y1='{a[1]:.1f}' x2='{b[0]:.1f}' y2='{b[1]:.1f}' "
            f"stroke='{color}' stroke-width='{stroke_width:.2f}' stroke-opacity='0.82'><title>{edge_type} radius filtration={filt:.3f} mean_z={avg_z:.3f}</title></line>"
        )
    vertex_by_label = {str((s.get("simplex") or [""])[0]): s for s in vertices}
    radius = 7.2 if len(visible_labels) <= 32 else 5.2
    font_size = 8 if len(visible_labels) <= 32 else 6
    for idx, label in enumerate(visible_labels):
        x, y = projected.get(label, (width / 2.0, height / 2.0))
        filt = 0.0
        vertex = vertex_by_label.get(label, {})
        if vertex:
            filt = float(vertex.get("filtration", 0.0) or 0.0)
        color = _filtration_color(filt)
        z = depth(label)
        r = radius * (0.72 + 0.56 * max(0.0, min(1.0, z)))
        safe = html.escape(label[:18])
        vertex_type = html.escape(str(vertex.get("type", "vertex"))[:30]) if vertex else "vertex"
        parts.append(
            f"<circle class='zero-simplex pca-radius-node' data-filtration='{filt:.8f}' data-pca-z='{z:.6f}' cx='{x:.1f}' cy='{y:.1f}' r='{r:.1f}' fill='{color}' stroke='#e8eef8' "
            f"stroke-width='1.2' filter='url(#glow)'><title>{vertex_type} radius filtration={filt:.3f} PCA=({coords3.get(label, (0.0, 0.0, 0.0))[0]:.3f},{coords3.get(label, (0.0, 0.0, 0.0))[1]:.3f},{z:.3f})</title></circle>"
        )
        if len(visible_labels) <= 48:
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
        if coords.shape != (features.shape[0], 3) or not np.isfinite(coords).all() or float(np.abs(coords).sum()) <= 1e-12:
            coords, fallback_stats = _feature_pca3_with_jitter(features, labels)
            stats = {"stress": 1.0, "corr": 0.0, "energy3": float(fallback_stats.get("energy3", 0.0)), "fallback": fallback_stats.get("fallback", "feature_pca3")}
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
        + (f"fallback={stats.get('fallback')} " if stats.get("fallback") else "")
        + f"mean_radius={mean_radius:.3f}",
    )


def _vertex_metric_feature_matrix(labels: list[str], vertex_by_label: dict[str, dict[str, object]]) -> np.ndarray:
    rows: list[list[float]] = []
    max_len = 0
    for idx, label in enumerate(labels):
        vertex = vertex_by_label.get(label, {})
        vector = _coerce_vertex_vector(vertex)
        fallback = _vertex_pca_feature(label, vertex, idx, len(labels))
        row = vector + fallback
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


def _feature_pca3_with_jitter(features: np.ndarray, labels: list[str]) -> tuple[np.ndarray, dict[str, object]]:
    if features.size == 0:
        features = np.asarray([_hashed_text_feature_vector(label, bins=16) for label in labels], dtype=float)
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(labels):
        values = np.asarray([_hashed_text_feature_vector(label, bins=16) for label in labels], dtype=float)
    jitter = np.asarray([_hashed_text_feature_vector(f"layout::{label}", bins=min(16, max(values.shape[1], 1))) for label in labels], dtype=float)
    if jitter.shape[1] < values.shape[1]:
        jitter = np.pad(jitter, ((0, 0), (0, values.shape[1] - jitter.shape[1])))
    elif jitter.shape[1] > values.shape[1]:
        jitter = jitter[:, : values.shape[1]]
    values = values + 1e-3 * jitter
    values = values - np.mean(values, axis=0, keepdims=True)
    try:
        _u, s, vt = np.linalg.svd(values, full_matrices=False)
        coords = values @ vt[: min(3, vt.shape[0])].T
    except np.linalg.LinAlgError:
        coords = jitter[:, : min(3, jitter.shape[1])]
        s = np.asarray([], dtype=float)
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
    if not np.isfinite(coords).all() or float(np.ptp(coords, axis=0).sum()) <= 1e-12:
        coords = np.asarray([_hashed_text_feature_vector(f"coords::{label}", bins=3) for label in labels], dtype=float)
    energy = float(np.sum(s[:3] ** 2) / max(float(np.sum(s ** 2)), 1e-12)) if s.size else 0.0
    return coords[:, :3], {"fallback": "feature_pca3_from_vertex_embeddings_or_semantic_hash", "energy3": energy}


def _coerce_vertex_vector(vertex: dict[str, object], max_dims: int = 128) -> list[float]:
    if not isinstance(vertex, dict):
        return []
    for key in ("embedding", "embedding_vector", "vector", "features", "feature", "coords", "coordinates", "pca"):
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


def _vertex_pca_feature(label: str, vertex: dict[str, object], idx: int, total: int) -> list[float]:
    text = str(vertex.get("text", "")) if isinstance(vertex, dict) else ""
    kind = str(vertex.get("type", "vertex")) if isinstance(vertex, dict) else "vertex"
    filt = float(vertex.get("filtration", 0.0) or 0.0) if isinstance(vertex, dict) else 0.0
    weight = float(vertex.get("weight", 1.0) or 1.0) if isinstance(vertex, dict) else 1.0
    decoded = str(vertex.get("decoded_argmax", "")) if isinstance(vertex, dict) else ""
    input_text = str(vertex.get("input_text", "")) if isinstance(vertex, dict) else ""
    path = _json_clip(vertex.get("path", []), 256) if isinstance(vertex, dict) else ""
    semantic = _hashed_text_feature_vector(f"{label}\n{kind}\n{text}\n{decoded}\n{input_text}\n{path}", bins=48)
    type_digest = hashlib.sha1(kind.encode("utf-8", "ignore")).digest()
    type_bits = [int(byte) / 255.0 for byte in type_digest[:6]]
    return [
        filt,
        math.log1p(abs(weight)),
        len(text) / 512.0,
        len(decoded) / 2048.0,
        len(input_text) / 4096.0,
        *type_bits,
        *semantic,
    ]


def _hashed_text_feature_vector(text: str, bins: int = 48) -> list[float]:
    bins = max(int(bins), 8)
    vec = np.zeros(bins, dtype=float)
    clean = " ".join(str(text or "").lower().split())
    if not clean:
        return vec.tolist()
    tokens = clean.split()
    for token in tokens[:512]:
        digest = hashlib.blake2b(token.encode("utf-8", "ignore"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % bins
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[bucket] += sign
    compact = clean[:2048]
    for idx in range(max(len(compact) - 2, 0)):
        tri = compact[idx : idx + 3]
        digest = hashlib.blake2b(tri.encode("utf-8", "ignore"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % bins
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[bucket] += 0.25 * sign
    norm = float(np.linalg.norm(vec))
    if norm > 1e-12:
        vec = vec / norm
    return [float(v) for v in vec]


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
    return coords, "circular_fallback"


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


def _nll_visual_scale(nll_values: np.ndarray, target_range: float = 4.0) -> float:
    values = np.asarray(nll_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1000.0
    raw_range = float(np.nanmax(values) - np.nanmin(values))
    if raw_range <= 1e-12:
        return 1000.0
    return float(np.clip(target_range / raw_range, 1000.0, 10000.0))


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
            "centered scaled NLL=%{z:.4f}<br>"
            "raw NLL=%{customdata:.6f}<extra></extra>"
        ),
        showlegend=True,
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
            hovertemplate="PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>centered scaled NLL=%{z:.4f}<br>raw NLL=%{intensity:.6f}<extra></extra>",
        )
        return trace, {
            "available": True,
            "mode": mode,
            "surface_kind": "sparse_exact_triangular_nll_mesh",
            "interpolation": "Exact triangular interpolant through the three observed PCA states",
            "point_count": int(x.size),
            "nll_min": float(np.nanmin(nll)),
            "nll_max": float(np.nanmax(nll)),
            "nll_mean": float(np.nanmean(nll)),
            "max_point_residual": 0.0,
            "touches_points": True,
        }
    grid_x, grid_y, z_grid, nll_grid, residual, support_meta = _supported_local_idw_grid(x, y, z, nll)
    surface = go.Surface(
        x=grid_x,
        y=grid_y,
        z=z_grid,
        surfacecolor=nll_grid,
        colorscale="Plasma",
        opacity=0.64,
        showscale=False,
        name=name,
        hovertemplate=(
            "PC1=%{x:.3f}<br>"
            "PC2=%{y:.3f}<br>"
            "centered scaled NLL=%{z:.4f}<br>"
            "local smoothed raw NLL=%{surfacecolor:.6f}<br>"
            "surface shown only inside sample-supported neighborhoods<extra></extra>"
        ),
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="#f8fafc", project_z=True, width=2)),
    )
    return surface, {
        "available": True,
        "mode": mode,
        "surface_kind": "sample_supported_local_idw_surface",
        "interpolation": "Modified Shepard/local IDW interpolation on a PCA grid augmented with sample coordinates; cells outside the union of sample-supported neighborhoods are masked",
        "point_count": int(x.size),
        "grid_shape": [int(grid_x.shape[0]), int(grid_x.shape[1])],
        "hull_masked_fraction": float(support_meta.get("masked_fraction", 0.0)),
        "support_radius": float(support_meta.get("support_radius", 0.0)),
        "support_policy": support_meta.get("support_policy", "union of local sample neighborhoods"),
        "value_clipping": "none; IDW is a convex weighted average of observed values",
        "nll_min": float(np.nanmin(nll)),
        "nll_max": float(np.nanmax(nll)),
        "nll_mean": float(np.nanmean(nll)),
        "smooth_surface_max_point_residual": float(residual),
        "max_point_residual": 0.0,
        "touches_points": True,
        "exact_anchor_layer": True,
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
    return (
        f"<b>{row.get('record_id')}</b>"
        f"<br>level={row.get('level')} path={path}"
        f"<br>score={float(row.get('score', 0.0)):.4f} nll={float(row.get('nll', 0.0)):.4f}"
        f"<br>graph tokens={row.get('graph_tokens')} margin={float(row.get('margin_mean', 0.0)):.4f}"
        + (f"<br><b>model input</b>: {_html_clip(row.get('input_text', ''), 900)}" if row.get("input_text") else "")
        + (f"<br><b>model argmax output</b>: {_html_clip(row.get('decoded_argmax', ''), 900)}" if row.get("decoded_argmax") else "")
        + f"<br><b>filtered complex</b>: V={summary.get('num_vertices')} E={summary.get('num_edges')} T={summary.get('num_two_simplices')}"
        f"<br>simplex sample={_json_clip(simplex_sample, 700)}"
        + (f"<br><b>Betti</b>: {betti}" if betti else "")
        + (f"<br><b>PH sample</b>: {_json_clip(intervals, 500)}" if intervals else "")
        + (f"<br><b>multi-rank sample</b>: {_json_clip(rank_sample, 500)}" if rank_sample else "")
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
