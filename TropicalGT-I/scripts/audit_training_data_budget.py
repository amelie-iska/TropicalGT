#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tropicalgt.data import dataset_budget_report, make_dataset_from_config, validate_dataset_budget


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit TropicalGT-I training data source and token-slot coverage")
    parser.add_argument("--config", default=str(ROOT / "configs" / "train.json"))
    parser.add_argument("--split", default="train", choices=("train", "validation"))
    parser.add_argument("--output", default="")
    parser.add_argument("--min-available-token-slots", type=int, default=None)
    parser.add_argument("--min-training-token-slots", type=int, default=None)
    parser.add_argument("--no-enforce-config", action="store_true")
    args = parser.parse_args()

    report = build_report(
        Path(args.config),
        split=args.split,
        min_available_token_slots=args.min_available_token_slots,
        min_training_token_slots=args.min_training_token_slots,
        enforce_config=not args.no_enforce_config,
    )
    text = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if report["errors"]:
        raise SystemExit(1)


def build_report(
    config_path: Path,
    *,
    split: str,
    min_available_token_slots: int | None,
    min_training_token_slots: int | None,
    enforce_config: bool,
) -> dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    ds = make_dataset_from_config(cfg, split)
    max_steps = int(cfg.get("max_steps", 0)) if split == "train" else 0
    budget = dataset_budget_report(
        ds,
        seq_len=int(cfg.get("seq_len", 1024)),
        batch_size=int(cfg.get("batch_size", 0)) if split == "train" else 0,
        max_steps=max_steps,
    )
    config_available = cfg.get("min_available_train_token_slots") if enforce_config and split == "train" else None
    config_training = cfg.get("min_training_token_slots") if enforce_config and split == "train" else None
    errors = validate_dataset_budget(
        budget,
        min_available_token_slots=_max_threshold(config_available, min_available_token_slots),
        min_training_token_slots=_max_threshold(config_training, min_training_token_slots),
        required_sources=cfg.get("required_hybrid_sources", ()) if enforce_config else (),
        source_requirements=cfg.get("hybrid_source_requirements", {}) if enforce_config and split == "train" else {},
    )
    return {"config": str(config_path), "split": split, "status": "ready" if not errors else "blocked", "errors": errors, "budget": budget}


def _max_threshold(*values: int | None) -> int | None:
    present = [int(value) for value in values if value is not None]
    return max(present) if present else None


if __name__ == "__main__":
    main()
