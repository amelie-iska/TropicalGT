from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch
from sklearn.decomposition import PCA
import plotly.graph_objects as go

from .data import encode_bytes
from .simplicial import build_filtered_simplicial_object


def collect_states(model, dataset, tokenizer, seq_len: int, device: torch.device, limit: int = 8):
    records = [dataset[i] for i in range(min(limit, len(dataset)))]
    xs, ys = zip(*(encode_bytes(r.text, seq_len) for r in records))
    graph_batch = tokenizer.batch_encode(records)
    with torch.no_grad():
        out = model(torch.stack(xs).to(device), graph_batch, torch.stack(ys).to(device))
    states = out["graph_state"].detach().cpu().numpy()
    nll_value = float(out.get("nll", torch.tensor(0.0)).detach().cpu())
    nll = np.full((len(records),), nll_value)
    filtered_objects = [build_filtered_simplicial_object(r) for r in records]
    hover = []
    base_hover = graph_batch.hover_payloads or [r.to_hover_html() for r in records]
    for html, obj in zip(base_hover, filtered_objects):
        summary = obj["summary"]
        hover.append(
            html
            + "<br><b>Filtered simplicial object</b>"
            + f"<br>0-simplices: {summary['num_vertices']}"
            + f"<br>1-simplices: {summary['num_edges']}"
            + f"<br>2-simplices: {summary['num_two_simplices']}"
            + f"<br>filtration thresholds: {summary['num_thresholds']}"
        )
    return states, nll, hover, filtered_objects


def write_reasoning_visualizations(model, dataset, tokenizer, seq_len: int, device: torch.device, output_dir: str | Path, limit: int = 8) -> dict[str, str]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    states, nll, hover, filtered_objects = collect_states(model, dataset, tokenizer, seq_len, device, limit)
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
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"pca_3d": str(p3), "pca_nll": str(p2), "payloads": str(payload)}


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
        "graphcg_loss",
        "margin_mean",
        "support_entropy",
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
