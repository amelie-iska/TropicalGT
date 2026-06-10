from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch
from sklearn.decomposition import PCA
import plotly.graph_objects as go

from .data import encode_bytes


def collect_states(model, dataset, tokenizer, seq_len: int, device: torch.device, limit: int = 8):
    records = [dataset[i] for i in range(min(limit, len(dataset)))]
    xs, ys = zip(*(encode_bytes(r.text, seq_len) for r in records))
    graph_batch = tokenizer.batch_encode(records)
    with torch.no_grad():
        out = model(torch.stack(xs).to(device), graph_batch, torch.stack(ys).to(device))
    states = out["graph_state"].detach().cpu().numpy()
    nll_value = float(out.get("nll", torch.tensor(0.0)).detach().cpu())
    nll = np.full((len(records),), nll_value)
    hover = graph_batch.hover_payloads or [r.to_hover_html() for r in records]
    return states, nll, hover


def write_reasoning_visualizations(model, dataset, tokenizer, seq_len: int, device: torch.device, output_dir: str | Path, limit: int = 8) -> dict[str, str]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    states, nll, hover = collect_states(model, dataset, tokenizer, seq_len, device, limit)
    if states.shape[0] < 2:
        states = np.concatenate([states, states], axis=0)
        nll = np.concatenate([nll, nll])
        hover = hover + hover
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
    payload.write_text(json.dumps({"hover": hover}, indent=2), encoding="utf-8")
    return {"pca_3d": str(p3), "pca_nll": str(p2), "payloads": str(payload)}
