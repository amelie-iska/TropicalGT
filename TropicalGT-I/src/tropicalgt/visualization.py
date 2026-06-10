from __future__ import annotations

from pathlib import Path
import html
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
    surface_pc3, surface_meta = _nll_surface_trace(pca[:, 0], pca[:, 1], nll, z_values=pca[:, 2], mode="floor", name="Smoothed NLL floor")
    fig3d = go.Figure()
    if surface_pc3 is not None:
        fig3d.add_trace(surface_pc3)
    fig3d.add_trace(
        go.Scatter3d(
            x=pca[:, 0],
            y=pca[:, 1],
            z=pca[:, 2],
            mode="markers+lines",
            marker=dict(size=6, color=nll, colorscale="Viridis", showscale=True, colorbar=dict(title="NLL")),
            text=hover,
            hoverinfo="text",
            name="reasoning state",
        )
    )
    node_indices = np.arange(len(pca), dtype=int)
    fig3d.data[-1].customdata = node_indices
    fig3d.update_layout(title="TropicalGT-I 3D PCA reasoning trajectory", scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
    panel_items = _simplicial_panel_items(filtered_objects, hover)
    p3 = output_dir / "reasoning_trajectory_3d.html"; _write_plotly_dark_html(p3, fig3d, "TropicalGT-I 3D PCA reasoning trajectory", panel_items)
    surface_nll, surface_nll_meta = _nll_surface_trace(pca[:, 0], pca[:, 1], nll, z_values=nll, mode="nll_height", name="Smoothed NLL surface")
    fig2 = go.Figure()
    if surface_nll is not None:
        fig2.add_trace(surface_nll)
    fig2.add_trace(
        go.Scatter3d(
            x=pca[:, 0],
            y=pca[:, 1],
            z=nll,
            mode="markers+lines",
            marker=dict(size=6, color=nll, colorscale="Plasma", showscale=True, colorbar=dict(title="NLL")),
            text=hover,
            hoverinfo="text",
            name="reasoning state",
        )
    )
    fig2.data[-1].customdata = node_indices
    fig2.update_layout(title="TropicalGT-I PCA with NLL height", scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="NLL"))
    p2 = output_dir / "reasoning_trajectory_pca_nll.html"; _write_plotly_dark_html(p2, fig2, "TropicalGT-I PCA with NLL height", panel_items)
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
                "nll_surface": {"pca_floor": surface_meta, "nll_height": surface_nll_meta},
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
        _write_dark_empty(path, "No graph-of-thought candidate embeddings available.")
        payload_path.write_text(json.dumps({"nodes": [], "edges": []}, indent=2), encoding="utf-8")
        return {"got_trajectory_3d": str(path), "got_payloads": str(payload_path)}

    embeddings = np.asarray([row["embedding"] for row in candidates], dtype=float)
    pca = _pca3(embeddings)
    ids = [str(row.get("record_id", idx)) for idx, row in enumerate(candidates)]
    id_to_idx = {rid: idx for idx, rid in enumerate(ids)}
    scores = [float(row.get("score", 0.0)) for row in candidates]
    nll_values = np.asarray([float(row.get("nll", row.get("score", 0.0)) or 0.0) for row in candidates], dtype=float)
    hover = [_candidate_hover(row) for row in candidates]
    fig = go.Figure()
    nll_surface, nll_surface_meta = _nll_surface_trace(pca[:, 0], pca[:, 1], nll_values, z_values=pca[:, 2], mode="floor", name="Smoothed trajectory NLL surface")
    if nll_surface is not None:
        fig.add_trace(nll_surface)
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
            marker=dict(size=8, color=nll_values, colorscale="Plasma", showscale=True, colorbar=dict(title="NLL")),
            text=[f"L{row.get('level', 0)}" for row in candidates],
            textposition="top center",
            hovertext=hover,
            hoverinfo="text",
            customdata=np.arange(len(candidates), dtype=int),
            name="GoT state",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Graph-of-thought trajectory PCA in embedding space",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
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
                "score": scores[idx],
                "nll": float(nll_values[idx]),
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
    path = output_dir / "tropical_support_heatmap.html"
    trace = result.get("graph_token_trace", {}) if isinstance(result, dict) else {}
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
    _write_plotly_dark_html(barcode, fig, "Persistent homology barcode")

    states = topology.get("persistence_module", {}).get("states", []) if isinstance(topology.get("persistence_module"), dict) else []
    fig2 = go.Figure()
    for dim in range(4):
        xs = [state["threshold"] for state in states]
        ys = [int(state.get("betti", {}).get(str(dim), 0)) for state in states]
        if any(value != 0 for value in ys):
            fig2.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=f"beta_{dim}"))
    fig2.update_layout(template="plotly_dark", title="Persistence module Betti rank profile", xaxis_title="filtration", yaxis_title="Betti rank")
    _write_plotly_dark_html(module_path, fig2, "Persistence module Betti rank profile")
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


def write_analogical_memory_visualization(memory: dict[str, object], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    path = output_dir / "analogical_memory_retrieval.html"
    rows = [row for row in memory.get("retrieved", []) if isinstance(row, dict)]
    if not rows:
        _write_dark_empty(path, "No analogical memories retrieved.")
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
    _write_plotly_dark_html(path, fig, "Analogical reasoning memory retrieval")
    return {"analogical_memory_retrieval_html": str(path)}


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
    .simplicial-object-panel {{
      border: 1px solid rgba(94, 234, 212, 0.2);
      background: #070a12;
      border-radius: 8px;
      padding: 10px;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03), 0 18px 36px rgba(0,0,0,0.24);
    }}
    .simplicial-object-panel svg {{ width: 100%; height: auto; display: block; }}
    @media (max-width: 900px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .chart {{ min-height: 68vh; border-right: 0; border-bottom: 1px solid var(--edge); }}
      .panel {{ min-height: auto; }}
    }}
  </style>
</head>
<body>
  <main class="layout">
    <section class="chart" id="chart">{chart}</section>
    <aside class="panel" aria-live="polite">
      <h1 id="simplicial-title">{html.escape(initial["title"])}</h1>
      <div class="summary" id="simplicial-summary">{initial["summary"]}</div>
      <div class="simplicial-object-panel" id="simplicial-svg">{initial["svg"]}</div>
    </aside>
  </main>
  <script>
    const simplicialPanels = {json.dumps(items)};
    const panelTitle = document.getElementById("simplicial-title");
    const panelSummary = document.getElementById("simplicial-summary");
    const panelSvg = document.getElementById("simplicial-svg");
    function setPanel(index) {{
      const item = simplicialPanels[index];
      if (!item) return;
      panelTitle.textContent = item.title || "Filtered simplicial object";
      panelSummary.innerHTML = item.summary || "";
      panelSvg.innerHTML = item.svg || "";
    }}
    const plot = document.querySelector("#chart .plotly-graph-div");
    if (plot && simplicialPanels.length) {{
      plot.on("plotly_hover", (event) => {{
        const point = (event.points || []).find((p) => p.customdata !== undefined && p.customdata !== null);
        if (!point) return;
        const raw = Array.isArray(point.customdata) ? point.customdata[0] : point.customdata;
        const idx = Number(raw);
        if (Number.isFinite(idx)) setPanel(idx);
      }});
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
        summary_html = (
            f"V={summary.get('num_vertices', 0)} | E={summary.get('num_edges', 0)} | "
            f"T={summary.get('num_two_simplices', 0)} | thresholds={summary.get('num_thresholds', 0)}"
        )
        if idx < len(hover):
            summary_html += f"<br>{hover[idx]}"
        items.append({"title": title, "summary": summary_html, "svg": _simplicial_object_svg(obj if isinstance(obj, dict) else {})})
    return items


def _simplicial_object_svg(obj: dict[str, object], width: int = 380, height: int = 270) -> str:
    simplices = obj.get("simplices", []) if isinstance(obj, dict) else []
    vertices = [s for s in simplices if isinstance(s, dict) and int(s.get("dimension", -1)) == 0]
    edges = [s for s in simplices if isinstance(s, dict) and int(s.get("dimension", -1)) == 1]
    triangles = [s for s in simplices if isinstance(s, dict) and int(s.get("dimension", -1)) == 2]
    labels = [str((s.get("simplex") or [f"v{idx}"])[0]) for idx, s in enumerate(vertices)]
    if not labels:
        labels = ["empty"]
    labels = labels[:18]
    cx, cy = width / 2.0, height / 2.0 + 4.0
    radius = max(min(width, height) * 0.34, 48.0)
    coords: dict[str, tuple[float, float]] = {}
    for idx, label in enumerate(labels):
        angle = -np.pi / 2 + 2 * np.pi * idx / max(len(labels), 1)
        coords[label] = (cx + radius * np.cos(angle), cy + radius * np.sin(angle))

    def point(label: str) -> tuple[float, float] | None:
        return coords.get(str(label))

    parts = [
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='rendered filtered simplicial object'>",
        "<rect width='100%' height='100%' rx='12' fill='#070a12'/>",
        "<defs><filter id='glow'><feGaussianBlur stdDeviation='2.5' result='b'/><feMerge><feMergeNode in='b'/><feMergeNode in='SourceGraphic'/></feMerge></filter></defs>",
    ]
    for tri in triangles[:18]:
        simplex = tri.get("simplex", [])
        pts = [point(v) for v in simplex]
        if len(pts) == 3 and all(p is not None for p in pts):
            poly = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in pts if p is not None)
            parts.append(f"<polygon points='{poly}' fill='rgba(94,234,212,0.13)' stroke='rgba(94,234,212,0.38)' stroke-width='1.2'/>")
    for edge in edges[:48]:
        simplex = edge.get("simplex", [])
        if len(simplex) < 2:
            continue
        a = point(simplex[0]); b = point(simplex[1])
        if a is None or b is None:
            continue
        filt = float(edge.get("filtration", 0.0) or 0.0)
        color = _filtration_color(filt)
        parts.append(f"<line x1='{a[0]:.1f}' y1='{a[1]:.1f}' x2='{b[0]:.1f}' y2='{b[1]:.1f}' stroke='{color}' stroke-width='2.2' stroke-opacity='0.82'/>")
    for idx, label in enumerate(labels):
        x, y = coords[label]
        filt = 0.0
        if idx < len(vertices):
            filt = float(vertices[idx].get("filtration", 0.0) or 0.0)
        color = _filtration_color(filt)
        safe = html.escape(label[:18])
        parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='8.5' fill='{color}' stroke='#e8eef8' stroke-width='1.2' filter='url(#glow)'/>")
        parts.append(f"<text x='{x:.1f}' y='{y + 22:.1f}' text-anchor='middle' fill='#dbe7f4' font-size='9'>{safe}</text>")
    thresholds = obj.get("thresholds", []) if isinstance(obj, dict) else []
    parts.append("<text x='14' y='22' fill='#9fb3c8' font-size='11'>filtered simplicial complex</text>")
    parts.append(f"<text x='14' y='{height - 14}' fill='#7dd3fc' font-size='10'>thresholds: {html.escape(_json_clip(thresholds[:8], 120))}</text>")
    parts.append("</svg>")
    return "".join(parts)


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
) -> tuple[go.Surface | None, dict[str, object]]:
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
    }


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
