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
    artifact_bytes = _compressed_quantized_bytes(model.state_dict())
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
        "included_files": [*CODE_FILES, "final_model.int8.ptz", "manifest.json"],
        "model": model_cfg,
        "estimator": "same int8 state-dict quantization plus zlib compression as external/parameter-golf/scripts/export_tropicalgt_parameter_golf.py",
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


if __name__ == "__main__":
    main()
