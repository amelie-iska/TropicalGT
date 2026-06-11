#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import zlib

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "TropicalGT-I" / "src"
PARAMETER_GOLF_ROOT = ROOT / "external" / "parameter-golf"
for path in (SRC, PARAMETER_GOLF_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_gpt import quantize_state_dict_int8  # noqa: E402
from tropicalgt.model import TropicalGTConfig, TropicalGTModel  # noqa: E402

CODE_FILES = ("train_gpt.py", "tropicalgt_tokengt_adapter.py")
COMPETITION_EXCLUDED_PREFIXES = ("gfn.", "graphcg.", "memory.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate stripped Parameter-Golf export size for a TropicalGT config")
    parser.add_argument("--config", type=Path, default=ROOT / "TropicalGT-I" / "configs" / "train.json")
    parser.add_argument("--cap-bytes", type=int, default=16_000_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    model_cfg = dict(cfg.get("model", {}))
    model = TropicalGTModel(TropicalGTConfig(**model_cfg))
    param_count = sum(int(param.numel()) for param in model.parameters())
    full_state = model.state_dict()
    competition_state, stripped = _competition_state_dict(full_state)
    competition_param_count = sum(int(tensor.numel()) for tensor in competition_state.values())
    stripped_param_count = sum(int(tensor.numel()) for tensor in stripped.values())
    artifact_bytes = _compressed_quantized_bytes(competition_state)
    code_sizes = {name: int((PARAMETER_GOLF_ROOT / name).stat().st_size) for name in CODE_FILES}
    code_bytes = int(sum(code_sizes.values()))
    total = int(artifact_bytes + code_bytes)
    report = {
        "config": str(args.config),
        "cap_bytes": int(args.cap_bytes),
        "within_cap": bool(total <= args.cap_bytes),
        "cap_margin_bytes": int(args.cap_bytes - total),
        "cap_utilization": float(total / max(args.cap_bytes, 1)),
        "artifact_bytes": int(artifact_bytes),
        "code_bytes": code_bytes,
        "code_sizes": code_sizes,
        "total_competition_bytes": total,
        "parameter_count": param_count,
        "competition_parameter_count": competition_param_count,
        "stripped_training_only_parameter_count": stripped_param_count,
        "stripped_training_only_prefixes": list(COMPETITION_EXCLUDED_PREFIXES),
        "stripped_training_only_keys": sorted(stripped),
        "included_files": [*CODE_FILES, "final_model.int8.ptz", "manifest.json"],
        "model": model_cfg,
        "estimator": "same competition-state filtering, int8 state-dict quantization, and zlib compression as external/parameter-golf/scripts/export_tropicalgt_parameter_golf.py",
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if not report["within_cap"]:
        raise SystemExit(f"estimated export exceeds cap by {-report['cap_margin_bytes']} bytes")


def _compressed_quantized_bytes(state_dict: dict[str, torch.Tensor]) -> int:
    quant_obj, _stats = quantize_state_dict_int8(state_dict)
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    return len(zlib.compress(buf.getvalue(), level=9))


def _competition_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    kept: dict[str, torch.Tensor] = {}
    stripped: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.startswith(COMPETITION_EXCLUDED_PREFIXES):
            stripped[key] = tensor
        else:
            kept[key] = tensor
    return kept, stripped


if __name__ == "__main__":
    main()
