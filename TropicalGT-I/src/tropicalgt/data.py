from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .records import GraphRecord

PAD_ID = 0
VOCAB_SIZE = 257


class ParquetGraphDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", limit: int | None = None) -> None:
        self.root = Path(root)
        self.split = split
        split_dir = self.root / split
        files = sorted(split_dir.glob("*.parquet")) if split_dir.exists() else sorted(self.root.glob(f"{split}*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files for split={split} under {self.root}")
        frames = []
        remaining = limit
        for file in files:
            df = pd.read_parquet(file)
            if remaining is not None:
                df = df.head(remaining)
                remaining -= len(df)
            frames.append(df)
            if remaining is not None and remaining <= 0:
                break
        self.frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> GraphRecord:
        return GraphRecord.from_mapping(self.frame.iloc[index].to_dict(), index=index)


class FixtureGraphDataset(Dataset):
    def __init__(self, size: int = 8) -> None:
        self.records = []
        for i in range(size):
            graph = {"nodes": [{"id": "p", "type": "problem", "text": f"add {i} and {i+1}"}, {"id": "s", "type": "reasoning_step", "text": f"{i}+{i+1}={2*i+1}"}, {"id": "a", "type": "answer", "text": str(2*i+1)}], "edges": [{"source": "p", "target": "s", "type": "depends_on"}, {"source": "s", "target": "a", "type": "supports_answer"}]}
            self.records.append(GraphRecord(f"fixture-{i}", f"Question: add {i} and {i+1}\nAnswer: {2*i+1}", answer=str(2*i+1), graph_json=graph))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> GraphRecord:
        return self.records[index]


def encode_bytes(text: str, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [b + 1 for b in text.encode("utf-8", "ignore")[: seq_len + 1]]
    if len(ids) < 2:
        ids = ids + [1] * (2 - len(ids))
    x = ids[:-1][:seq_len]
    y = ids[1:][:seq_len]
    x += [PAD_ID] * (seq_len - len(x))
    y += [PAD_ID] * (seq_len - len(y))
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def make_dataset(root: str | Path | None, split: str, limit: int | None = None, fixture_size: int = 8) -> Dataset:
    if root:
        try:
            return ParquetGraphDataset(root, split=split, limit=limit)
        except Exception:
            if split != "train":
                raise
    return FixtureGraphDataset(size=fixture_size if limit is None else min(limit, fixture_size))
