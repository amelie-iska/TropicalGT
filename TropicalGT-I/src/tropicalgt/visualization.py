from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
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
from .simplicial import build_filtered_simplicial_object


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
    p3 = output_dir / "reasoning_trajectory_3d.html"; _write_plotly_dark_html(p3, fig3d, "TropicalGT-I validation graph-state PCA sample", panel_items)
    surface_nll, surface_nll_meta = _nll_surface_trace(pca[:, 0], pca[:, 1], nll, z_values=nll, mode="nll_height", name="Interpolating NLL surface through reasoning points")
    fig2 = go.Figure()
    if surface_nll is not None:
        fig2.add_trace(surface_nll)
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
    p2 = output_dir / "reasoning_trajectory_pca_nll.html"; _write_plotly_dark_html(p2, fig2, "TropicalGT-I validation PCA with NLL height", panel_items)
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
    pca = _pca3(embeddings)
    ids = [str(row.get("record_id", idx)) for idx, row in enumerate(candidates)]
    id_to_idx = {rid: idx for idx, rid in enumerate(ids)}
    inferred_levels = _infer_candidate_levels(candidates, ids)
    scores = [float(row.get("score", 0.0)) for row in candidates]
    nll_values = np.asarray([float(row.get("nll", row.get("score", 0.0)) or 0.0) for row in candidates], dtype=float)
    hover = [_candidate_hover(row) for row in candidates]
    fig = go.Figure()
    nll_surface, nll_surface_meta = _nll_surface_trace(
        pca[:, 0],
        pca[:, 1],
        nll_values,
        z_values=nll_values,
        mode="nll_height",
        name="Interpolating trajectory NLL surface through GoT nodes",
    )
    if nll_surface is not None:
        fig.add_trace(nll_surface)
    for idx, row in enumerate(candidates):
        parent = row.get("parent")
        if isinstance(parent, str) and parent in id_to_idx:
            j = id_to_idx[parent]
            action = _edge_action_label(row)
            fig.add_trace(
                go.Scatter3d(
                    x=[pca[j, 0], pca[idx, 0]],
                    y=[pca[j, 1], pca[idx, 1]],
                    z=[nll_values[j], nll_values[idx]],
                    mode="lines",
                    line=dict(color=_action_color(action), width=5),
                    showlegend=False,
                    hovertext=f"{parent} -> {ids[idx]}<br>action={html.escape(action)}",
                    hoverinfo="text",
                )
            )
    fig.add_trace(
        go.Scatter3d(
            x=pca[:, 0],
            y=pca[:, 1],
            z=nll_values,
            mode="markers+text",
            marker=dict(
                size=9,
                color=nll_values,
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(title="NLL"),
                line=dict(width=1.5, color="#e8eef8"),
            ),
            text=[f"L{inferred_levels[idx]}:{_last_action(row)}" for idx, row in enumerate(candidates)],
            textposition="top center",
            hovertext=hover,
            hoverinfo="text",
            customdata=np.arange(len(candidates), dtype=int),
            name="GoT state",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Graph-of-thought branching trajectory with NLL height",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="NLL"),
    )
    panel_items = _simplicial_panel_items(
        [row.get("filtered_simplicial_object") if isinstance(row.get("filtered_simplicial_object"), dict) else {} for row in candidates],
        hover,
    )
    _write_plotly_dark_html(path, fig, "Graph-of-thought trajectory PCA in embedding space", panel_items)
    payload = {
        "nodes": [
            {
                "record_id": ids[idx],
                "parent": candidates[idx].get("parent"),
                "path": candidates[idx].get("path", []),
                "pca": {"pc1": float(pca[idx, 0]), "pc2": float(pca[idx, 1]), "pc3": float(pca[idx, 2])},
                "plot": {"x": float(pca[idx, 0]), "y": float(pca[idx, 1]), "z_nll": float(nll_values[idx])},
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
        "filtered_simplicial_objects": [
            candidates[idx].get("filtered_simplicial_object")
            if isinstance(candidates[idx].get("filtered_simplicial_object"), dict)
            else {}
            for idx in range(len(candidates))
        ],
        "nll_surface": nll_surface_meta,
    }
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"got_trajectory_3d": str(path), "got_payloads": str(payload_path)}


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
    z = np.zeros((n, n), dtype=float)
    labels = []
    for row_idx, token in enumerate(tokens):
        active = int(token.get("active_support_index", -1))
        margin = float(token.get("margin", 0.0))
        if 0 <= active < n:
            z[row_idx, active] = margin if margin != 0.0 else 1.0
        labels.append(f"{row_idx}:{token.get('kind','?')}:{token.get('label','')}")
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="Viridis",
            colorbar=dict(title="margin on active support"),
            hovertemplate="query=%{y}<br>active=%{x}<br>value=%{z}<extra></extra>",
        )
    )
    fig.update_layout(template="plotly_dark", title="Tropical active-support heatmap", xaxis_title="active support token", yaxis_title="query token")
    _write_plotly_dark_html(path, fig, "Tropical active-support heatmap")
    return {"tropical_support_heatmap": str(path)}


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
    max_death = 1.0
    for row in rows:
        for interval in row["intervals"]:
            death = interval.get("display_death")
            if isinstance(death, (int, float)) and math.isfinite(float(death)):
                max_death = max(max_death, float(death))
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
            z = float(dim) + 0.08 * interval_idx
            true_death = "inf" if interval.get("infinite") else f"{float(interval.get('death', death)):.4g}"
            hover = (
                f"<b>trajectory growth level {level}</b>"
                f"<br>H{dim} interval [{birth:.4g}, {true_death}]"
                f"<br>complex: {_summary_line(obj if isinstance(obj, dict) else {})}"
                f"<br>{_derived_signature_line(topo if isinstance(topo, dict) else {})}"
                f"<br>{_free_resolution_line(topo if isinstance(topo, dict) else {})}"
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[birth, death],
                    y=[level, level],
                    z=[z, z],
                    mode="lines+markers",
                    line=dict(width=8, color=colors.get(dim, "#cbd5e1")),
                    marker=dict(size=5, color=colors.get(dim, "#cbd5e1")),
                    name=f"L{level} H{dim}",
                    hovertext=[hover, hover],
                    hoverinfo="text",
                    customdata=[row_idx, row_idx],
                    showlegend=interval_idx == 0,
                )
            )
            if interval.get("infinite"):
                fig.add_trace(
                    go.Scatter3d(
                        x=[death],
                        y=[level],
                        z=[z],
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
        title=f"{title_prefix}persistent homology growth barcode",
        scene=dict(
            xaxis_title="filtration birth/death",
            yaxis_title="trajectory growth level",
            zaxis_title="homology dimension",
        ),
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
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=(
            "Persistence module Betti ranks across trajectory growth",
            "Multigraded free-chain / free-resolution proxy ranks",
        ),
        horizontal_spacing=0.02,
    )
    panel_objects = []
    panel_hover = []
    for row_idx, row in enumerate(rows):
        level = int(row["level"])
        obj = row.get("filtered_simplicial_object", {})
        topo = row.get("topological_algebra", {})
        panel_objects.append(obj if isinstance(obj, dict) else {})
        panel_hover.append(_topology_growth_hover(level, obj if isinstance(obj, dict) else {}, topo if isinstance(topo, dict) else {}))
        states = row.get("states", [])
        for dim, color in [(0, "#55d6be"), (1, "#7aa2ff"), (2, "#fbbf24"), (3, "#fb7185")]:
            xs = [float(state.get("threshold", 0.0)) for state in states if isinstance(state, dict)]
            zs = [int(state.get("betti", {}).get(str(dim), 0)) for state in states if isinstance(state, dict)]
            if xs and any(value != 0 for value in zs):
                ys = [level] * len(xs)
                hover = [
                    f"<b>trajectory level {level}</b><br>beta_{dim}={z}<br>filtration={x:.4g}<br>{_summary_line(obj if isinstance(obj, dict) else {})}"
                    for x, z in zip(xs, zs)
                ]
                fig.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="lines+markers",
                        line=dict(width=4, color=color),
                        marker=dict(size=4, color=color),
                        name=f"L{level} beta_{dim}",
                        hovertext=hover,
                        hoverinfo="text",
                        customdata=[row_idx] * len(xs),
                    ),
                    row=1,
                    col=1,
                )
        free_modules = _free_resolution_modules(topo if isinstance(topo, dict) else {})
        for module in free_modules:
            degree = int(module.get("homological_degree", 0))
            rank = int(module.get("rank", module.get("rank_upper_bound", 0)))
            hover = (
                f"<b>trajectory level {level}</b>"
                f"<br>homological degree={degree}"
                f"<br>free rank={rank}"
                f"<br>{_free_resolution_line(topo if isinstance(topo, dict) else {})}"
                f"<br>{_derived_signature_line(topo if isinstance(topo, dict) else {})}"
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[degree],
                    y=[level],
                    z=[rank],
                    mode="markers+text",
                    marker=dict(size=max(5, min(18, 4 + math.log1p(max(rank, 0)))), color=[rank], colorscale="Viridis", showscale=False),
                    text=[f"F{degree}"],
                    textposition="top center",
                    name=f"L{level} F{degree}",
                    hovertext=hover,
                    hoverinfo="text",
                    customdata=[row_idx],
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
    if not rows:
        fig.add_annotation(text="No trajectory growth persistence module available.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    fig.update_layout(template="plotly_dark", title=f"{title_prefix}multiparameter persistence and free-resolution growth")
    fig.update_scenes(xaxis_title="filtration", yaxis_title="growth level", zaxis_title="Betti rank", row=1, col=1)
    fig.update_scenes(xaxis_title="homological degree", yaxis_title="growth level", zaxis_title="free rank", row=1, col=2)
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
    return (
        f"V={summary.get('num_vertices', 0)} E={summary.get('num_edges', 0)} "
        f"T={summary.get('num_two_simplices', 0)} thresholds={summary.get('num_thresholds', 0)}"
    )


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
    for row in candidates:
        proj = row.get("graphcg_projection")
        if isinstance(proj, dict) and proj.get("all_direction_cosines") is not None:
            matrices.append([float(v) for v in proj["all_direction_cosines"]])
            labels.append(str(row.get("record_id", len(labels))))
    if not matrices:
        _write_dark_empty(path, "No GraphCG projection diagnostics available.")
        return {"graphcg_direction_cosines": str(path)}
    fig = go.Figure(
        data=go.Heatmap(
            z=np.asarray(matrices, dtype=float),
            y=labels,
            x=[f"dir_{idx}" for idx in range(len(matrices[0]))],
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="cosine"),
        )
    )
    fig.update_layout(template="plotly_dark", title="GraphCG direction cosines along GoT candidates", xaxis_title="GraphCG direction", yaxis_title="candidate state")
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

    fig = go.Figure()
    panel_items: list[dict[str, object]] = []
    map_reports: list[dict[str, object]] = []
    query_layout = _complex_3d_layout(query_complex, slab=0.0)
    query_hover = (
        "<b>query trajectory filtered complex</b>"
        f"<br>{_summary_line(query_complex)}"
        f"<br>{_derived_signature_line(query_topology)}"
        f"<br>{_free_resolution_line(query_topology)}"
    )
    panel_items.extend(_simplicial_panel_items([query_complex], [query_hover]))
    _add_complex_3d_traces(
        fig,
        query_complex,
        query_layout,
        panel_idx=0,
        name="query trajectory complex",
        color="#55d6be",
        hover_prefix="query trajectory",
    )
    for idx, row in enumerate(enriched):
        panel_idx = idx + 1
        slab = float(idx + 1) * 2.4
        mem_complex = row.get("filtered_simplicial_object", {}) if isinstance(row.get("filtered_simplicial_object"), dict) else {}
        mem_topology = row.get("topological_algebra", {}) if isinstance(row.get("topological_algebra"), dict) else {}
        mem_layout = _complex_3d_layout(mem_complex, slab=slab)
        sim = _topological_similarity_summary(query_topology, mem_topology, row)
        sim_map = _simplicial_map_between_complexes(query_complex, mem_complex)
        map_reports.append({"memory_id": row.get("memory_id"), "record_id": row.get("record_id"), **sim, **sim_map})
        panel_hover = (
            f"<b>retrieved memory {idx + 1}</b>: {html.escape(str(row.get('memory_id', idx)))}"
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
        _add_complex_3d_traces(
            fig,
            mem_complex,
            mem_layout,
            panel_idx=panel_idx,
            name=f"memory {idx + 1} complex",
            color=_memory_color(idx),
            hover_prefix=f"memory {idx + 1}",
        )
        _add_simplicial_map_traces(fig, query_layout, mem_layout, sim_map, panel_idx, row, sim)
        marker = dict(size=11, color=float(row.get("retrieval_score", 0.0)), colorscale="Viridis", showscale=idx == 0)
        if idx == 0:
            marker["colorbar"] = dict(title="retrieval")
        fig.add_trace(
            go.Scatter3d(
                x=[slab],
                y=[-1.28],
                z=[1.25],
                mode="markers+text",
                marker=marker,
                text=[f"memory {idx + 1}"],
                textposition="top center",
                hovertext=panel_hover,
                hoverinfo="text",
                customdata=[panel_idx],
                name=f"memory {idx + 1} invariants",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        title="Analogical reasoning memory retrieval as simplicial maps between filtered complexes",
        scene=dict(
            xaxis_title="query / retrieved memory slab",
            yaxis_title="filtered-complex layout x",
            zaxis_title="filtered-complex layout y",
        ),
        legend=dict(itemsizing="constant"),
    )
    _write_plotly_dark_html(
        path,
        fig,
        "Analogical simplicial map retrieval",
        panel_items,
    )
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
    return {"analogical_memory_retrieval_html": str(path), "analogical_simplicial_maps": str(map_path)}


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
    if not isinstance(enriched.get("filtered_simplicial_object"), dict) and isinstance(bank_row.get("filtered_simplicial_object"), dict):
        enriched["filtered_simplicial_object"] = bank_row["filtered_simplicial_object"]
    if not isinstance(enriched.get("topological_algebra"), dict) and isinstance(bank_row.get("topological_algebra"), dict):
        enriched["topological_algebra"] = bank_row["topological_algebra"]
    if "derived_signature" not in enriched and isinstance(enriched.get("topological_algebra"), dict):
        enriched["derived_signature"] = enriched["topological_algebra"].get("derived_equivalence_signature", {})
    return enriched


def _complex_3d_layout(obj: dict[str, object], slab: float, max_vertices: int = 54) -> dict[str, tuple[float, float, float]]:
    vertices = _complex_vertex_records(obj)[:max_vertices]
    labels = [row["label"] for row in vertices]
    edges = [
        {"simplex": list(pair)}
        for pair in _complex_edge_pairs(obj)
        if pair[0] in set(labels) and pair[1] in set(labels)
    ]
    coords2d, _ = _simplicial_layout(labels, edges, width=360, height=260)
    coords: dict[str, tuple[float, float, float]] = {}
    for label in labels:
        x2, y2 = coords2d.get(label, (180.0, 130.0))
        coords[label] = (slab, (x2 - 180.0) / 150.0, (130.0 - y2) / 110.0)
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
        return {"vertex_map": [], "edge_preservation_rate": 0.0, "two_simplex_preservation_rate": 0.0, "is_simplicial_on_displayed_skeleton": False}
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
            }
        )
    return vertices


def _vertex_by_label(obj: dict[str, object]) -> dict[str, dict[str, object]]:
    return {row["label"]: row for row in _complex_vertex_records(obj)}


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
    if query.get("label") == memory.get("label"):
        score += 4.0
    if query.get("type") == memory.get("type"):
        score += 2.0
    q_text = set(str(query.get("text", "")).lower().split())
    m_text = set(str(memory.get("text", "")).lower().split())
    if q_text or m_text:
        score += len(q_text & m_text) / max(len(q_text | m_text), 1)
    score += 1.0 / (1.0 + abs(float(query.get("filtration", 0.0)) - float(memory.get("filtration", 0.0))))
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
        "graphcg_effective_rank",
        "graphcg_numerical_rank",
        "graphcg_rank_target",
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
    directions = model.graphcg.directions.detach().cpu().float().numpy()
    if directions.size == 0:
        return {}
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    normalized = directions / np.maximum(norms, 1e-8)
    gram = normalized @ normalized.T
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    rank_margin = float(getattr(model.graphcg, "full_rank_margin", 0.05))
    numerical_rank = int(np.sum(singular_values > rank_margin))
    rank_target = min(normalized.shape)

    gram_path = output_dir / "graphcg_direction_gram.html"
    fig = go.Figure(
        data=go.Heatmap(
            z=gram,
            x=[f"dir_{idx}" for idx in range(gram.shape[0])],
            y=[f"dir_{idx}" for idx in range(gram.shape[0])],
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="cosine"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title=f"GraphCG direction Gram matrix (rank {numerical_rank}/{rank_target})",
    )
    _write_plotly_dark_html(gram_path, fig, f"GraphCG direction Gram matrix (rank {numerical_rank}/{rank_target})")

    pca_path = output_dir / "graphcg_direction_pca.html"
    pca = _pca3(directions)
    fig2 = go.Figure()
    for idx, point in enumerate(pca):
        fig2.add_trace(
            go.Scatter3d(
                x=[0, point[0]],
                y=[0, point[1]],
                z=[0, point[2]],
                mode="lines+markers",
                line=dict(width=5),
                marker=dict(size=[2, 7]),
                name=f"dir_{idx}",
                hovertext=f"direction {idx}<br>norm={float(norms[idx,0]):.4f}",
                hoverinfo="text",
            )
        )
    fig2.update_layout(
        template="plotly_dark",
        title="GraphCG steering directions in PCA",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
    )
    _write_plotly_dark_html(pca_path, fig2, "GraphCG steering directions in PCA")

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
        title="GraphCG full-rank singular spectrum",
        xaxis_title="singular direction",
        yaxis_title="singular value",
    )
    _write_plotly_dark_html(sv_path, fig3, "GraphCG full-rank singular spectrum")
    return {
        "graphcg_direction_gram": str(gram_path),
        "graphcg_direction_pca": str(pca_path),
        "graphcg_direction_singular_values": str(sv_path),
    }


def _pca3(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        values = values.reshape((len(values), -1))
    if values.shape[0] < 2:
        return np.pad(values[:, :1], ((0, 0), (0, 2)), constant_values=0.0)
    n_components = min(3, values.shape[0], values.shape[1])
    coords = PCA(n_components=n_components).fit_transform(values)
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])), constant_values=0.0)
    return coords


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


def _write_plotly_dark_html(path: Path, fig: go.Figure, title: str, panel_items: list[dict[str, str]] | None = None) -> None:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#090b12",
        plot_bgcolor="#090b12",
        font=dict(color="#e8eef8"),
        margin=dict(l=0, r=0, b=0, t=44),
    )
    chart = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displaylogo": False, "responsive": True})
    items = panel_items or []
    initial = items[0] if items else {"title": "Filtered simplicial object", "svg": "", "summary": "Hover a reasoning node to render its complex."}
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
      width: min(380px, calc(100vw - 24px));
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
      max-height: 58px;
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
      <div class="filtration-controls" id="filtration-controls">
        <label><span>Filtration radius</span><strong id="filtration-value">all</strong></label>
        <input id="filtration-slider" type="range" min="0" max="1" step="0.001" value="1" aria-label="filtered simplicial complex radius">
        <div class="hint" id="filtration-hint">Shows simplices with scalar filtration at or below the selected radius. Multiparameter summaries remain in the JSON payload.</div>
      </div>
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
    coords, layout_kind = _simplicial_layout(visible_labels, edges, width=width, height=height)

    def point(label: str) -> tuple[float, float] | None:
        return coords.get(str(label))

    parts = [
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='rendered filtered simplicial object'>",
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
            parts.append(
                f"<polygon class='two-simplex' data-filtration='{filt:.8f}' points='{poly}' "
                f"fill='rgba(94,234,212,0.16)' stroke='rgba(94,234,212,0.46)' stroke-width='1.2'>"
                f"<title>2-simplex filtration={filt:.3f}</title></polygon>"
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
        parts.append(
            f"<line class='one-simplex' data-filtration='{filt:.8f}' x1='{a[0]:.1f}' y1='{a[1]:.1f}' x2='{b[0]:.1f}' y2='{b[1]:.1f}' "
            f"stroke='{color}' stroke-width='2.2' stroke-opacity='0.82'><title>{edge_type} filtration={filt:.3f}</title></line>"
        )
    vertex_by_label = {str((s.get("simplex") or [""])[0]): s for s in vertices}
    radius = 7.2 if len(visible_labels) <= 32 else 5.2
    font_size = 8 if len(visible_labels) <= 32 else 6
    for idx, label in enumerate(visible_labels):
        x, y = coords[label]
        filt = 0.0
        vertex = vertex_by_label.get(label, {})
        if vertex:
            filt = float(vertex.get("filtration", 0.0) or 0.0)
        color = _filtration_color(filt)
        safe = html.escape(label[:18])
        vertex_type = html.escape(str(vertex.get("type", "vertex"))[:30]) if vertex else "vertex"
        parts.append(
            f"<circle class='zero-simplex' data-filtration='{filt:.8f}' cx='{x:.1f}' cy='{y:.1f}' r='{radius:.1f}' fill='{color}' stroke='#e8eef8' "
            f"stroke-width='1.2' filter='url(#glow)'><title>{vertex_type} filtration={filt:.3f}</title></circle>"
        )
        if len(visible_labels) <= 48:
            parts.append(f"<text x='{x:.1f}' y='{y + 18:.1f}' text-anchor='middle' fill='#dbe7f4' font-size='{font_size}'>{safe}</text>")
    parts.append("<text x='16' y='24' fill='#dbeafe' font-size='11' font-weight='650'>filtered simplicial complex</text>")
    parts.append(
        f"<text x='16' y='40' fill='#9fb3c8' font-size='9'>"
        f"dim0={summary.get('num_vertices', len(vertices))} dim1={summary.get('num_edges', len(edges))} dim2={summary.get('num_two_simplices', len(triangles))}</text>"
    )
    parts.append(
        f"<text x='16' y='54' fill='#9fb3c8' font-size='8'>layout={html.escape(layout_kind)} visible={len(visible_labels)}/{len(labels)}</text>"
    )
    if len(labels) > len(visible_labels):
        parts.append(
            f"<text x='16' y='66' fill='#fbbf24' font-size='8'>truncated {len(labels) - len(visible_labels)} vertices for legibility</text>"
        )
    parts.extend(_filtration_layer_svg(thresholds, width=width, y=238))
    parts.append(f"<text x='14' y='{height - 10}' fill='#7dd3fc' font-size='9'>thresholds: {html.escape(_json_clip(thresholds[:8], 116))}</text>")
    parts.append("</svg>")
    return "".join(parts)


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
            hovertemplate="PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>surface z=%{z:.4f}<extra></extra>",
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
            hovertemplate="PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>surface z=%{z:.4f}<br>NLL=%{intensity:.4f}<extra></extra>",
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
    grid_x, grid_y, z_grid, nll_grid, residual = _smooth_exact_rbf_grid(x, y, z, nll)
    hull = _convex_hull(np.column_stack([x, y]))
    masked_fraction = 0.0
    if hull.shape[0] >= 3:
        mask = _points_in_convex_polygon(np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)]), hull).reshape(grid_x.shape)
        masked_fraction = 1.0 - float(mask.sum()) / float(mask.size)
        z_grid = np.where(mask, z_grid, np.nan)
        nll_grid = np.where(mask, nll_grid, np.nan)
    z_grid = _clip_interpolation_grid(z_grid, z)
    nll_grid = _clip_interpolation_grid(nll_grid, nll)
    surface = go.Surface(
        x=grid_x,
        y=grid_y,
        z=z_grid,
        surfacecolor=nll_grid,
        colorscale="Plasma",
        opacity=0.46,
        showscale=False,
        name=name,
        hovertemplate="PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>surface z=%{z:.4f}<br>smoothed NLL=%{surfacecolor:.4f}<extra></extra>",
        contours=dict(z=dict(show=False)),
    )
    return surface, {
        "available": True,
        "mode": mode,
        "surface_kind": "smooth_exact_rbf_surface",
        "interpolation": "Gaussian radial-basis exact interpolation on grid augmented with sample coordinates and masked to the observed PCA convex hull",
        "point_count": int(x.size),
        "grid_shape": [int(grid_x.shape[0]), int(grid_x.shape[1])],
        "hull_masked_fraction": float(masked_fraction),
        "value_clipping": "display grid clipped to observed value range; anchor residual is computed before clipping",
        "nll_min": float(np.nanmin(nll)),
        "nll_max": float(np.nanmax(nll)),
        "nll_mean": float(np.nanmean(nll)),
        "max_point_residual": float(residual),
        "touches_points": bool(residual <= 1e-5),
    }


def _smooth_exact_rbf_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    nll: np.ndarray,
    grid_size: int = 36,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    x_span = float(np.ptp(x))
    y_span = float(np.ptp(y))
    pad_x = max(x_span * 0.08, 0.20)
    pad_y = max(y_span * 0.08, 0.20)
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
