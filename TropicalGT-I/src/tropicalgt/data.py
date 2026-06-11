from __future__ import annotations

import bisect
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
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


@dataclass(frozen=True)
class ParameterGolfShard:
    path: Path
    tokens: int
    windows: int
    start_window: int


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


class ParameterGolfBinGraphDataset(Dataset):
    """OpenAI Parameter Golf token shards as graph-structured records.

    Each sampled token window is represented as a sequential DAG. Pure-byte
    tokenizers decode losslessly to bytes; SentencePiece or unknown tokenizers
    fall back to textual token identifiers while preserving graph structure.
    """

    header_ints: int = 256
    magic: int = 20240520
    version: int = 1

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        limit: int | None = None,
        window_tokens: int = 1025,
        stride: int | None = None,
        tokenizer_path: str | Path | None = None,
        max_graph_chunks: int = 64,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.window_tokens = max(int(window_tokens), 2)
        self.stride = max(int(stride or window_tokens), 1)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else None
        self.max_graph_chunks = max(int(max_graph_chunks), 1)
        self._decoder = _ParameterGolfDecoder(self.tokenizer_path)
        files = discover_parameter_golf_bin_files(self.root, split)
        if not files:
            raise FileNotFoundError(f"No Parameter Golf .bin files for split={split} under {self.root}")
        self.shards: list[ParameterGolfShard] = []
        total = 0
        remaining = limit
        for file in files:
            tokens = _parameter_golf_token_count(file)
            if tokens < 2:
                continue
            windows = max(1, 1 + max(tokens - self.window_tokens, 0) // self.stride)
            if remaining is not None:
                windows = min(windows, remaining)
                remaining -= windows
            if windows > 0:
                self.shards.append(ParameterGolfShard(file, tokens, windows, total))
                total += windows
            if remaining is not None and remaining <= 0:
                break
        if not self.shards:
            raise ValueError(f"No Parameter Golf windows available for split={split} under {self.root}")
        self._cumulative_windows = [shard.start_window + shard.windows for shard in self.shards]
        self._maps: OrderedDict[Path, np.memmap] = OrderedDict()

    def __len__(self) -> int:
        return self._cumulative_windows[-1] if self._cumulative_windows else 0

    def __getitem__(self, index: int) -> GraphRecord:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._cumulative_windows, index)
        shard = self.shards[shard_index]
        local_window = index - shard.start_window
        start = min(local_window * self.stride, max(shard.tokens - 2, 0))
        stop = min(start + self.window_tokens, shard.tokens)
        token_ids = np.asarray(self._memmap(shard.path)[start:stop], dtype=np.uint16)
        text = self._decoder.decode(token_ids)
        graph = parameter_golf_window_graph(token_ids, text, max_chunks=self.max_graph_chunks)
        return GraphRecord.from_mapping(
            {
                "record_id": f"parameter-golf:{self.split}:{shard.path.stem}:{local_window}",
                "dataset": "openai_parameter_golf",
                "text": text,
                "question": text,
                "graph_json": graph,
                "metadata": {
                    "source": "parameter_golf_bin",
                    "shard": str(shard.path),
                    "window_index": local_window,
                    "token_start": int(start),
                    "token_stop": int(stop),
                    "tokenizer_kind": self._decoder.kind,
                },
            }
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "split": self.split,
            "files": len(self.shards),
            "windows": len(self),
            "window_tokens": self.window_tokens,
            "stride": self.stride,
            "tokenizer_path": str(self.tokenizer_path) if self.tokenizer_path else "",
            "tokenizer_kind": self._decoder.kind,
            "shards": [
                {"path": str(shard.path), "tokens": shard.tokens, "windows": shard.windows}
                for shard in self.shards
            ],
        }

    def _memmap(self, path: Path) -> np.memmap:
        cached = self._maps.get(path)
        if cached is not None:
            self._maps.move_to_end(path)
            return cached
        tokens = _parameter_golf_token_count(path)
        mmap = np.memmap(path, dtype="<u2", mode="r", offset=self.header_ints * np.dtype("<i4").itemsize, shape=(tokens,))
        self._maps[path] = mmap
        while len(self._maps) > 2:
            self._maps.popitem(last=False)
        return mmap


class HybridGraphDataset(Dataset):
    """Weighted mixture of already graph-structured datasets."""

    def __init__(self, sources: list[tuple[str, Dataset, float]], seed: int = 0, length: int | None = None) -> None:
        if not sources:
            raise ValueError("HybridGraphDataset requires at least one source")
        self.sources = [(name, dataset, max(float(weight), 0.0)) for name, dataset, weight in sources if len(dataset) > 0]
        if not self.sources:
            raise ValueError("HybridGraphDataset sources are empty")
        total_weight = sum(weight for _, _, weight in self.sources)
        if total_weight <= 0:
            total_weight = float(len(self.sources))
            self.sources = [(name, dataset, 1.0) for name, dataset, _ in self.sources]
        cumulative = []
        running = 0.0
        for _name, _dataset, weight in self.sources:
            running += weight / total_weight
            cumulative.append(running)
        self._cumulative_weights = cumulative
        self.seed = int(seed)
        self._length = int(length) if length is not None else max(len(dataset) for _, dataset, _ in self.sources)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> GraphRecord:
        rng = random.Random(self.seed + int(index))
        pick = rng.random()
        source_index = bisect.bisect_left(self._cumulative_weights, pick)
        source_index = min(source_index, len(self.sources) - 1)
        name, dataset, _weight = self.sources[source_index]
        source_offset = rng.randrange(len(dataset))
        record = dataset[source_offset]
        metadata = dict(record.metadata or {})
        metadata["hybrid_source"] = name
        metadata["hybrid_source_index"] = source_offset
        return GraphRecord(
            record_id=record.record_id,
            text=record.text,
            question=record.question,
            answer=record.answer,
            reasoning=record.reasoning,
            metadata=metadata,
            graph_json=record.graph_json,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "kind": "hybrid",
            "seed": self.seed,
            "length": len(self),
            "sources": [
                {
                    "name": name,
                    "weight": weight,
                    "rows": len(dataset),
                    "manifest": dataset.manifest() if hasattr(dataset, "manifest") else {"type": type(dataset).__name__},
                }
                for name, dataset, weight in self.sources
            ],
        }


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
            self.records.append(
                GraphRecord.from_mapping(
                    {
                        "record_id": f"fixture-{i}",
                        "question": f"add {i} and {i+1}",
                        "answer": str(2 * i + 1),
                        "text": f"Question: add {i} and {i+1}\nAnswer: {2 * i + 1}",
                        "graph_json": graph,
                    }
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> GraphRecord:
        return self.records[index]


def encode_bytes(text: str, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [b + 1 for b in text.encode("utf-8", "ignore")[: seq_len + 1]]
    return encode_byte_ids(ids, seq_len)


def encode_byte_ids(ids: list[int], seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [int(token) for token in ids[: seq_len + 1]]
    if len(ids) < 2:
        ids = ids + [1] * (2 - len(ids))
    x = ids[:-1][:seq_len]
    y = ids[1:][:seq_len]
    x += [PAD_ID] * (seq_len - len(x))
    y += [PAD_ID] * (seq_len - len(y))
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def encode_record_bytes(record: GraphRecord, seq_len: int, graph_autoregressive: bool = False, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    text = record.autoregressive_text(seed=seed) if graph_autoregressive else record.text
    return encode_bytes(text, seq_len)


class _ParameterGolfDecoder:
    def __init__(self, tokenizer_path: Path | None = None) -> None:
        self.tokenizer_path = tokenizer_path
        self.kind = "token_ids"
        self.byte_offset = 4
        self._sp = None
        if tokenizer_path and tokenizer_path.is_file():
            if tokenizer_path.suffix == ".json":
                try:
                    payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
                    if payload.get("tokenizer_type") == "pure_byte":
                        cfg = payload.get("config", {})
                        self.kind = "pure_byte"
                        self.byte_offset = int(cfg.get("byte_offset", 4))
                except Exception:
                    self.kind = "token_ids"
            elif tokenizer_path.suffix == ".model":
                try:
                    import sentencepiece as spm

                    self._sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
                    self.kind = "sentencepiece"
                except Exception:
                    self.kind = "token_ids"

    def decode(self, token_ids: np.ndarray) -> str:
        if self.kind == "pure_byte":
            byte_values = np.clip(token_ids.astype(np.int32) - self.byte_offset, 0, 255).astype(np.uint8)
            return bytes(byte_values.tolist()).decode("utf-8", "replace")
        if self.kind == "sentencepiece" and self._sp is not None:
            try:
                return self._sp.decode([int(token) for token in token_ids.tolist()])
            except Exception:
                pass
        return " ".join(f"tok_{int(token)}" for token in token_ids.tolist())


def parameter_golf_window_graph(token_ids: np.ndarray, text: str, max_chunks: int = 64) -> dict[str, Any]:
    token_list = [int(token) for token in token_ids.tolist()]
    chunk_count = max(1, min(int(max_chunks), len(token_list)))
    chunk_size = max(1, (len(token_list) + chunk_count - 1) // chunk_count)
    nodes: list[dict[str, Any]] = [
        {
            "id": "pg_window",
            "type": "token_window",
            "text": text[:512],
            "token_count": len(token_list),
        }
    ]
    edges: list[dict[str, Any]] = []
    previous = "pg_window"
    for idx, start in enumerate(range(0, len(token_list), chunk_size)):
        chunk_tokens = token_list[start : start + chunk_size]
        node_id = f"pg_chunk_{idx:03d}"
        chunk_text = _token_chunk_text(text, idx, chunk_count)
        nodes.append(
            {
                "id": node_id,
                "type": "sequence_chunk",
                "text": chunk_text,
                "position": idx,
                "token_start": start,
                "token_stop": start + len(chunk_tokens),
                "token_ids": chunk_tokens[:64],
            }
        )
        edges.append(
            {
                "source": previous,
                "target": node_id,
                "type": "next_token_window_chunk" if previous != "pg_window" else "starts_token_window",
                "causal": True,
            }
        )
        previous = node_id
    return {"nodes": nodes, "edges": edges}


def discover_parameter_golf_bin_files(root: str | Path, split: str) -> list[Path]:
    root_path = Path(root)
    if split == "validation":
        patterns = ["fineweb_val_*.bin", "validation*.bin", "val*.bin"]
    else:
        patterns = [f"fineweb_{split}_*.bin", f"{split}*.bin"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(root_path.glob(pattern)))
    return sorted(dict.fromkeys(files))


def parameter_golf_manifest(root: str | Path | None, splits: Iterable[str] = ("train", "validation")) -> dict[str, Any]:
    if not root:
        return {"root": None, "splits": {}}
    root_path = Path(root)
    manifest: dict[str, Any] = {"root": str(root_path), "splits": {}}
    for split in splits:
        shards = []
        tokens = 0
        bytes_total = 0
        for file in discover_parameter_golf_bin_files(root_path, split):
            token_count = _parameter_golf_token_count(file)
            size = file.stat().st_size
            tokens += token_count
            bytes_total += size
            shards.append({"path": str(file), "tokens": token_count, "bytes": size})
        manifest["splits"][split] = {"files": len(shards), "tokens": tokens, "bytes": bytes_total, "shards": shards}
    return manifest


def discover_parquet_files(root: str | Path, split: str) -> list[Path]:
    root_path = Path(root)
    split_dir = root_path / split
    if split_dir.exists():
        return sorted(split_dir.glob("*.parquet"))
    return sorted(root_path.glob(f"{split}*.parquet"))


def _parameter_golf_token_count(path: str | Path) -> int:
    path = Path(path)
    header = np.fromfile(path, dtype="<i4", count=ParameterGolfBinGraphDataset.header_ints)
    if header.size != ParameterGolfBinGraphDataset.header_ints or int(header[0]) != ParameterGolfBinGraphDataset.magic or int(header[1]) != ParameterGolfBinGraphDataset.version:
        raise ValueError(f"Unexpected Parameter Golf shard header for {path}")
    expected_size = ParameterGolfBinGraphDataset.header_ints * np.dtype("<i4").itemsize + int(header[2]) * np.dtype("<u2").itemsize
    if path.stat().st_size < expected_size:
        raise ValueError(f"Truncated Parameter Golf shard {path}: expected at least {expected_size} bytes")
    return int(header[2])


def _token_chunk_text(text: str, idx: int, chunk_count: int) -> str:
    clean = text or ""
    if not clean:
        return ""
    chunk_chars = max(1, (len(clean) + max(chunk_count, 1) - 1) // max(chunk_count, 1))
    return clean[idx * chunk_chars : (idx + 1) * chunk_chars]


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


def make_dataset_from_config(cfg: dict[str, Any], split: str) -> Dataset:
    split_limit = cfg.get("train_limit") if split == "train" else cfg.get("val_limit", cfg.get("train_limit"))
    hybrid_cfg = cfg.get("hybrid_data", {})
    if isinstance(hybrid_cfg, dict) and hybrid_cfg.get("enabled"):
        sources: list[tuple[str, Dataset, float]] = []
        for source in hybrid_cfg.get("sources", []):
            if not isinstance(source, dict):
                continue
            name = str(source.get("name") or source.get("kind") or f"source_{len(sources)}")
            weight = float(source.get("weight", 1.0))
            required = bool(source.get("required", False))
            try:
                dataset = _make_dataset_source(cfg, source, split, split_limit)
            except Exception as exc:
                if required:
                    raise
                print(f"Warning: skipping optional hybrid source {name} for split={split}: {exc}")
                continue
            sources.append((name, dataset, weight))
        if sources:
            length = hybrid_cfg.get("length")
            if split == "train":
                length = hybrid_cfg.get("train_length", length)
            else:
                length = hybrid_cfg.get("eval_length", length)
            return HybridGraphDataset(sources, seed=int(hybrid_cfg.get("seed", cfg.get("seed", 0))), length=length)
        if cfg.get("require_data", False):
            raise FileNotFoundError("No hybrid data sources resolved and require_data=true")
    return make_dataset(
        cfg.get("data_root"),
        split,
        limit=split_limit,
        fixture_size=int(cfg.get("fixture_size", 8)),
        require_data=bool(cfg.get("require_data", bool(cfg.get("data_root")))),
        cache_shards=int(cfg.get("cache_shards", 2)),
    )


def dataset_manifest(dataset: Dataset, fallback_root: str | Path | None = None, splits: Iterable[str] = ("train", "validation")) -> dict[str, Any]:
    if hasattr(dataset, "manifest"):
        return dataset.manifest()
    return parquet_manifest(fallback_root, splits, include_shards=False) if fallback_root else {"type": type(dataset).__name__, "rows": len(dataset)}


def _make_dataset_source(cfg: dict[str, Any], source: dict[str, Any], split: str, split_limit: int | None) -> Dataset:
    kind = str(source.get("kind", "parquet"))
    limit = source.get(f"{split}_limit", source.get("limit", split_limit))
    if kind in {"parquet", "hf_parquet", "graph_parquet"}:
        root = source.get("root", cfg.get("data_root"))
        return ParquetGraphDataset(
            root,
            split=source.get("split", split),
            limit=limit,
            cache_shards=int(source.get("cache_shards", cfg.get("cache_shards", 2))),
        )
    if kind in {"parameter_golf_bin", "oai_parameter_golf", "openai_parameter_golf"}:
        return ParameterGolfBinGraphDataset(
            source["root"],
            split=source.get("split", split),
            limit=limit,
            window_tokens=int(source.get("window_tokens", cfg.get("seq_len", 1024) + 1)),
            stride=source.get("stride"),
            tokenizer_path=source.get("tokenizer_path"),
            max_graph_chunks=int(source.get("max_graph_chunks", 64)),
        )
    raise ValueError(f"Unsupported dataset source kind: {kind}")
