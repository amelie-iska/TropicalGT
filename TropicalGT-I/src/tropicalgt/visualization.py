from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch
from sklearn.decomposition import PCA
import plotly.graph_objects as go

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
            + "<br><b>Filtered simplicial object</b>"
            + f"<br>0-simplices: {summary['num_vertices']}"
            + f"<br>1-simplices: {summary['num_edges']}"
            + f"<br>2-simplices: {summary['num_two_simplices']}"
            + f"<br>filtration thresholds: {summary['num_thresholds']}"
            + (f"<br><b>F2 Betti</b>: {betti}" if betti else "")
        )
    return states, nll, hover, filtered_objects, diagnostics


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
    states, nll, hover, filtered_objects, diagnostics = collect_states(
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
    fig3d = go.Figure(data=[go.Scatter3d(x=pca[:,0], y=pca[:,1], z=pca[:,2], mode="markers+lines", marker=dict(size=6, color=nll, colorscale="Viridis"), text=hover, hoverinfo="text")])
    fig3d.update_layout(title="TropicalGT-I 3D PCA reasoning trajectory", scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
    p3 = output_dir / "reasoning_trajectory_3d.html"; fig3d.write_html(p3)
    fig2 = go.Figure(data=[go.Scatter3d(x=pca[:,0], y=pca[:,1], z=nll, mode="markers+lines", marker=dict(size=6, color=nll, colorscale="Plasma"), text=hover, hoverinfo="text")])
    fig2.update_layout(title="TropicalGT-I PCA with NLL height", scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="NLL"))
    p2 = output_dir / "reasoning_trajectory_pca_nll.html"; fig2.write_html(p2)
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
            }
        )
    payload.write_text(
        json.dumps(
            {
                "hover": hover,
                "points": points,
                "filtered_simplicial_objects": filtered_objects,
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
            paths.update({f"trajectory_{key}": value for key, value in write_persistence_visualizations(trajectory_topology, output_dir / "trajectory_persistence").items()})
        if isinstance(memory, dict):
            paths.update(write_analogical_memory_visualization(memory, output_dir))
        paths["dashboard"] = str(_write_inference_dashboard(paths, output_dir))
    return paths


def write_got_trajectory_visualization(scaling_report: dict[str, object], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    candidates = [row for row in scaling_report.get("candidates", []) if isinstance(row, dict) and row.get("embedding") is not None]
    path = output_dir / "got_trajectory_pca_3d.html"
    payload_path = output_dir / "got_trajectory_payloads.json"
    if not candidates:
        path.write_text("<html><body><p>No graph-of-thought candidate embeddings available.</p></body></html>", encoding="utf-8")
        payload_path.write_text(json.dumps({"nodes": [], "edges": []}, indent=2), encoding="utf-8")
        return {"got_trajectory_3d": str(path), "got_payloads": str(payload_path)}

    embeddings = np.asarray([row["embedding"] for row in candidates], dtype=float)
    pca = _pca3(embeddings)
    ids = [str(row.get("record_id", idx)) for idx, row in enumerate(candidates)]
    id_to_idx = {rid: idx for idx, rid in enumerate(ids)}
    scores = [float(row.get("score", 0.0)) for row in candidates]
    hover = [_candidate_hover(row) for row in candidates]
    fig = go.Figure()
    for idx, row in enumerate(candidates):
        parent = row.get("parent")
        if isinstance(parent, str) and parent in id_to_idx:
            j = id_to_idx[parent]
            fig.add_trace(
                go.Scatter3d(
                    x=[pca[j, 0], pca[idx, 0]],
                    y=[pca[j, 1], pca[idx, 1]],
                    z=[pca[j, 2], pca[idx, 2]],
                    mode="lines",
                    line=dict(color="rgba(140,160,180,0.45)", width=3),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    fig.add_trace(
        go.Scatter3d(
            x=pca[:, 0],
            y=pca[:, 1],
            z=pca[:, 2],
            mode="markers+text",
            marker=dict(size=8, color=scores, colorscale="Turbo", showscale=True, colorbar=dict(title="score")),
            text=[f"L{row.get('level', 0)}" for row in candidates],
            textposition="top center",
            hovertext=hover,
            hoverinfo="text",
            name="GoT state",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Graph-of-thought trajectory PCA in embedding space",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
    )
    fig.write_html(path)
    payload = {
        "nodes": [
            {
                "record_id": ids[idx],
                "parent": candidates[idx].get("parent"),
                "path": candidates[idx].get("path", []),
                "pca": {"pc1": float(pca[idx, 0]), "pc2": float(pca[idx, 1]), "pc3": float(pca[idx, 2])},
                "score": scores[idx],
                "nll": candidates[idx].get("nll"),
                "filtered_simplicial_object": candidates[idx].get("filtered_simplicial_object"),
                "topological_algebra": candidates[idx].get("topological_algebra"),
            }
            for idx in range(len(candidates))
        ],
        "edges": [
            {"source": row.get("parent"), "target": ids[idx], "action_path": row.get("path", [])}
            for idx, row in enumerate(candidates)
            if row.get("parent") is not None
        ],
    }
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"got_trajectory_3d": str(path), "got_payloads": str(payload_path)}


def write_tropical_support_heatmap(result: dict[str, object], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    path = output_dir / "tropical_support_heatmap.html"
    trace = result.get("graph_token_trace", {}) if isinstance(result, dict) else {}
    tokens = trace.get("tokens", []) if isinstance(trace, dict) else []
    if not tokens:
        path.write_text("<html><body><p>No graph-token trace available.</p></body></html>", encoding="utf-8")
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
    fig.write_html(path)
    return {"tropical_support_heatmap": str(path)}


def write_persistence_visualizations(topology: dict[str, object], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    barcode = output_dir / "persistence_barcode.html"
    module_path = output_dir / "persistence_module_betti.html"
    intervals = topology.get("persistence", {}).get("intervals", []) if isinstance(topology.get("persistence"), dict) else []
    fig = go.Figure()
    if intervals:
        for idx, interval in enumerate(intervals):
            birth = float(interval.get("birth", 0.0))
            death = 1.05 if interval.get("death") is None else float(interval.get("death", birth))
            fig.add_trace(
                go.Scatter(
                    x=[birth, death],
                    y=[idx, idx],
                    mode="lines",
                    line=dict(width=6),
                    name=f"H{interval.get('dimension', '?')}",
                    hovertext=f"H{interval.get('dimension')} [{birth:.3f}, {'inf' if interval.get('death') is None else f'{death:.3f}'}]",
                    hoverinfo="text",
                )
            )
    fig.update_layout(template="plotly_dark", title="Persistent homology barcode", xaxis_title="filtration", yaxis_title="class")
    fig.write_html(barcode)

    states = topology.get("persistence_module", {}).get("states", []) if isinstance(topology.get("persistence_module"), dict) else []
    fig2 = go.Figure()
    for dim in range(4):
        xs = [state["threshold"] for state in states]
        ys = [int(state.get("betti", {}).get(str(dim), 0)) for state in states]
        if any(value != 0 for value in ys):
            fig2.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=f"beta_{dim}"))
    fig2.update_layout(template="plotly_dark", title="Persistence module Betti rank profile", xaxis_title="filtration", yaxis_title="Betti rank")
    fig2.write_html(module_path)
    return {"persistence_barcode": str(barcode), "persistence_module_betti": str(module_path)}


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
        path.write_text("<html><body><p>No GraphCG projection diagnostics available.</p></body></html>", encoding="utf-8")
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
    fig.write_html(path)
    return {"graphcg_direction_cosines": str(path)}


def write_analogical_memory_visualization(memory: dict[str, object], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    path = output_dir / "analogical_memory_retrieval.html"
    rows = [row for row in memory.get("retrieved", []) if isinstance(row, dict)]
    if not rows:
        path.write_text("<html><body><p>No analogical memories retrieved.</p></body></html>", encoding="utf-8")
        return {"analogical_memory_retrieval_html": str(path)}
    labels = [str(row.get("memory_id", idx)) for idx, row in enumerate(rows)]
    fig = go.Figure()
    for key, color in [
        ("retrieval_score", "#55d6be"),
        ("embedding_similarity", "#7aa2ff"),
        ("signature_similarity", "#ffb86b"),
        ("quality_score", "#ff6b9a"),
    ]:
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[float(row.get(key, 0.0)) for row in rows],
                name=key,
                marker_color=color,
                hovertext=[_json_clip(row, 1100) for row in rows],
                hoverinfo="text+y",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        title="Analogical reasoning memory retrieval",
        xaxis_title="memory",
        yaxis_title="score",
        barmode="group",
    )
    fig.write_html(path)
    return {"analogical_memory_retrieval_html": str(path)}


def _write_inference_dashboard(paths: dict[str, str], output_dir: Path) -> Path:
    dash = output_dir / "inference_audit.html"
    links = "\n".join(f'<li><a href="{Path(path).name}">{name}</a></li>' for name, path in sorted(paths.items()))
    dash.write_text(
        f"<html><body style='background:#090b12;color:#eef;font-family:sans-serif'><h1>TropicalGT-I Inference Audit</h1><ul>{links}</ul></body></html>",
        encoding="utf-8",
    )
    return dash


def write_metric_visualizations(history: list[dict[str, float]], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_path = output_dir / "training_metrics.html"
    if not history:
        metric_path.write_text("<html><body><p>No training metrics recorded.</p></body></html>", encoding="utf-8")
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
    fig.write_html(metric_path)
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
    fig.update_layout(template="plotly_dark", title="GraphCG direction Gram matrix")
    fig.write_html(gram_path)

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
    fig2.write_html(pca_path)
    return {"graphcg_direction_gram": str(gram_path), "graphcg_direction_pca": str(pca_path)}


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
        f"<br><b>filtered complex</b>: V={summary.get('num_vertices')} E={summary.get('num_edges')} T={summary.get('num_two_simplices')}"
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
