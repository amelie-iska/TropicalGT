from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import torch

from .records import GraphRecord, GraphTokenBatch


@dataclass
class TokenGTTokenizer:
    max_nodes: int = 64
    max_edges: int = 128
    node_id_dim: int = 16
    feature_dim: int = 48
    graph_token: bool = True

    def encode(self, record: GraphRecord) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, str]:
        graph = record.graph_json or {"nodes": [], "edges": []}
        raw_nodes = list(graph.get("nodes", []))[: self.max_nodes]
        node_ids = [str(n.get("id", i)) for i, n in enumerate(raw_nodes)]
        node_lookup = {nid: i for i, nid in enumerate(node_ids)}
        token_features: list[list[float]] = []
        type_ids: list[int] = []
        endpoints: list[list[int]] = []
        if self.graph_token:
            token_features.append(self._feature("graph", record.text, 0, 0, len(raw_nodes)))
            type_ids.append(2)
            endpoints.append([-1, -1])
        for i, node in enumerate(raw_nodes):
            token_features.append(self._feature(str(node.get("type", "node")), str(node.get("text", "")), i, i, len(raw_nodes)))
            type_ids.append(0)
            endpoints.append([i, i])
        edge_count = 0
        for edge in list(graph.get("edges", []))[: self.max_edges]:
            s = node_lookup.get(str(edge.get("source", edge.get("src", ""))), -1)
            t = node_lookup.get(str(edge.get("target", edge.get("dst", ""))), -1)
            if s < 0 or t < 0:
                continue
            token_features.append(self._feature(str(edge.get("type", "edge")), str(edge.get("text", "")), s, t, len(raw_nodes)))
            type_ids.append(1)
            endpoints.append([s, t])
            edge_count += 1
        if not token_features:
            token_features.append(self._feature("graph", record.text, 0, 0, 1))
            type_ids.append(2)
            endpoints.append([-1, -1])
        return (
            torch.tensor(token_features, dtype=torch.float32),
            torch.tensor(type_ids, dtype=torch.long),
            torch.tensor(endpoints, dtype=torch.long),
            len(raw_nodes),
            edge_count,
            record.to_hover_html(),
        )

    def batch_encode(self, records: Sequence[GraphRecord]) -> GraphTokenBatch:
        encoded = [self.encode(r) for r in records]
        max_len = max(item[0].shape[0] for item in encoded)
        feat_dim = encoded[0][0].shape[1]
        bsz = len(records)
        feats = torch.zeros(bsz, max_len, feat_dim, dtype=torch.float32)
        type_ids = torch.full((bsz, max_len), -1, dtype=torch.long)
        endpoints = torch.full((bsz, max_len, 2), -1, dtype=torch.long)
        mask = torch.zeros(bsz, max_len, dtype=torch.bool)
        counts = torch.zeros(bsz, dtype=torch.long)
        node_counts = torch.zeros(bsz, dtype=torch.long)
        edge_counts = torch.zeros(bsz, dtype=torch.long)
        hovers: list[str] = []
        for i, (f, t, e, n_count, edge_count, hover) in enumerate(encoded):
            n = f.shape[0]
            feats[i, :n] = f
            type_ids[i, :n] = t
            endpoints[i, :n] = e
            mask[i, :n] = True
            counts[i] = n
            node_counts[i] = n_count
            edge_counts[i] = edge_count
            hovers.append(hover)
        return GraphTokenBatch(feats, type_ids, endpoints, mask, counts, node_counts, edge_counts, hovers)

    def _feature(self, kind: str, text: str, a: int, b: int, n_nodes: int) -> list[float]:
        vec = [0.0] * self.feature_dim
        vec[0] = _hash01(kind)
        vec[1] = _hash01(text[:256])
        vec[2] = min(len(text), 2048) / 2048.0
        vec[3] = (a + 1) / max(n_nodes, 1)
        vec[4] = (b + 1) / max(n_nodes, 1)
        first = 5
        second = first + self.node_id_dim
        if first + self.node_id_dim <= self.feature_dim:
            vec[first + (a % self.node_id_dim)] = 1.0 if a >= 0 else 0.0
        if second + self.node_id_dim <= self.feature_dim:
            vec[second + (b % self.node_id_dim)] = 1.0 if b >= 0 else 0.0
        return vec


def _hash01(text: str) -> float:
    digest = hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(2**64 - 1)
