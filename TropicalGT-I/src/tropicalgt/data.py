from __future__ import annotations

import bisect
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

from .records import GraphRecord

PAD_ID = 0
VOCAB_SIZE = 257
DEFAULT_PARQUET_COLUMNS = (
    "record_id",
    "id",
    "dataset",
    "question",
    "answer",
    "solution",
    "reasoning",
    "metadata",
    "metadata_json",
    "text",
    "graph_json",
    "estimated_tokens",
)


@dataclass(frozen=True)
class ParquetChunk:
    path: Path
    row_group: int
    rows: int
    size_bytes: int


class ParquetGraphDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        limit: int | None = None,
        cache_shards: int = 2,
        columns: Iterable[str] = DEFAULT_PARQUET_COLUMNS,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.cache_shards = max(int(cache_shards), 0)
        self.columns = tuple(columns)
        files = discover_parquet_files(self.root, split)
        if not files:
            raise FileNotFoundError(f"No parquet files for split={split} under {self.root}")
        self.chunks: list[ParquetChunk] = []
        self._cumulative_rows: list[int] = []
        self._columns_by_file: dict[Path, tuple[str, ...]] = {}
        remaining = limit
        for file in files:
            parquet = pq.ParquetFile(file)
            available = tuple(column for column in self.columns if column in parquet.schema_arrow.names)
            self._columns_by_file[file] = available
            for row_group in range(parquet.num_row_groups):
                rows = int(parquet.metadata.row_group(row_group).num_rows)
                if remaining is not None:
                    rows = min(rows, remaining)
                    remaining -= rows
                if rows > 0:
                    size_bytes = int(parquet.metadata.row_group(row_group).total_byte_size)
                    self.chunks.append(ParquetChunk(file, row_group, rows, size_bytes))
                    self._cumulative_rows.append((self._cumulative_rows[-1] if self._cumulative_rows else 0) + rows)
                if remaining is not None and remaining <= 0:
                    break
            if remaining is not None and remaining <= 0:
                break
        if not self.chunks:
            raise ValueError(f"No rows available for split={split} under {self.root}")
        self._cache: OrderedDict[tuple[Path, int], Any] = OrderedDict()

    def __len__(self) -> int:
        return self._cumulative_rows[-1] if self._cumulative_rows else 0

    def __getitem__(self, index: int) -> GraphRecord:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        chunk_index = bisect.bisect_right(self._cumulative_rows, index)
        previous = self._cumulative_rows[chunk_index - 1] if chunk_index else 0
        chunk = self.chunks[chunk_index]
        local_index = index - previous
        frame = self._load_chunk(chunk)
        return GraphRecord.from_mapping(frame.iloc[local_index].to_dict(), index=index)

    def manifest(self) -> dict[str, Any]:
        files = {chunk.path for chunk in self.chunks}
        return {
            "root": str(self.root),
            "split": self.split,
            "files": len(files),
            "row_groups": len(self.chunks),
            "rows": len(self),
            "bytes": sum(path.stat().st_size for path in files),
            "cache_shards": self.cache_shards,
            "columns": sorted({column for columns in self._columns_by_file.values() for column in columns}),
            "chunks": [
                {"path": str(chunk.path), "row_group": chunk.row_group, "rows": chunk.rows, "bytes": chunk.size_bytes}
                for chunk in self.chunks
            ],
        }

    def chunk_bounds(self) -> list[tuple[int, int]]:
        bounds: list[tuple[int, int]] = []
        previous = 0
        for end in self._cumulative_rows:
            bounds.append((previous, end))
            previous = end
        return bounds

    def _load_chunk(self, chunk: ParquetChunk):
        key = (chunk.path, chunk.row_group)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        frame = pq.ParquetFile(chunk.path).read_row_group(chunk.row_group, columns=self._columns_by_file[chunk.path]).to_pandas()
        if self.cache_shards > 0:
            self._cache[key] = frame
            while len(self._cache) > self.cache_shards:
                self._cache.popitem(last=False)
        return frame


class ChunkShuffleSampler(Sampler[int]):
    """Shuffle parquet row groups while preserving cache-friendly reads."""

    def __init__(
        self,
        dataset: ParquetGraphDataset,
        seed: int = 0,
        shuffle_rows: bool = False,
        epoch: int = 0,
    ) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.shuffle_rows = bool(shuffle_rows)
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.dataset.chunks)))
        rng.shuffle(order)
        bounds = self.dataset.chunk_bounds()
        for chunk_index in order:
            start, end = bounds[chunk_index]
            if self.shuffle_rows:
                indices = list(range(start, end))
                rng.shuffle(indices)
                yield from indices
            else:
                yield from range(start, end)

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "shuffle_rows": self.shuffle_rows,
            "epoch": self.epoch,
            "chunks": len(self.dataset.chunks),
            "rows": len(self.dataset),
        }


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


def discover_parquet_files(root: str | Path, split: str) -> list[Path]:
    root_path = Path(root)
    split_dir = root_path / split
    if split_dir.exists():
        return sorted(split_dir.glob("*.parquet"))
    return sorted(root_path.glob(f"{split}*.parquet"))


def parquet_num_rows(path: str | Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def parquet_manifest(
    root: str | Path | None,
    splits: Iterable[str] = ("train", "validation", "test"),
    include_shards: bool = True,
) -> dict[str, Any]:
    if not root:
        return {"root": None, "splits": {}}
    root_path = Path(root)
    manifest: dict[str, Any] = {"root": str(root_path), "splits": {}}
    for split in splits:
        files = discover_parquet_files(root_path, split)
        rows = 0
        bytes_total = 0
        shards = []
        for file in files:
            row_count = parquet_num_rows(file)
            size = file.stat().st_size
            rows += row_count
            bytes_total += size
            shards.append({"path": str(file), "rows": row_count, "bytes": size})
        split_manifest: dict[str, Any] = {"files": len(files), "rows": rows, "bytes": bytes_total}
        if include_shards:
            split_manifest["shards"] = shards
        manifest["splits"][split] = split_manifest
    return manifest


def make_dataset(
    root: str | Path | None,
    split: str,
    limit: int | None = None,
    fixture_size: int = 8,
    require_data: bool = False,
    cache_shards: int = 2,
) -> Dataset:
    if root:
        try:
            return ParquetGraphDataset(root, split=split, limit=limit, cache_shards=cache_shards)
        except Exception as exc:
            if require_data or split != "train":
                raise
            print(f"Warning: falling back to fixture data for split={split}: {exc}")
    return FixtureGraphDataset(size=fixture_size if limit is None else min(limit, fixture_size))
