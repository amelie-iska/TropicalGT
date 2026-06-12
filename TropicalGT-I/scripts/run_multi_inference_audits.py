#!/usr/bin/env python3
"""Run multiple TropicalGT-I inference audits and build a sample-first browser bundle.

This is intentionally a coordinator around infer_tropicalgt_i.py: each sample is
an independent inference run with its own audit directory, then the top-level
browser_index.html lets reviewers choose the sample first and inspect that
sample's generated GoT, topology, memory, GraphCG, and tropical-support pages.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_sample_browser_index import build_index


DEFAULT_PROMPTS = (
    "Question: add 2 and 3\nAnswer:",
    "Question: if a path has vertices A B C, how many directed edges connect consecutive vertices?\nAnswer:",
    "Question: name one invariant that can summarize holes in a filtered simplicial complex.\nAnswer:",
)


def _read_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    if args.prompt_file:
        path = Path(args.prompt_file)
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            data = json.loads(raw)
            if not isinstance(data, list):
                raise SystemExit(f"prompt JSON must be a list: {path}")
            prompts.extend(str(row) for row in data)
        else:
            blocks = [block.strip() for block in raw.split("\n---\n")]
            prompts.extend(block for block in blocks if block)
    prompts.extend(args.prompt or [])
    if not prompts:
        prompts.extend(DEFAULT_PROMPTS)
    return prompts[: max(1, int(args.samples))]


def _command(args: argparse.Namespace, prompt: str, sample_dir: Path, index: int) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "infer_tropicalgt_i.py"),
        "--config",
        str(args.config),
        "--checkpoint",
        str(args.checkpoint),
        "--prompt",
        prompt,
        "--audit-all",
        "--audit-output-dir",
        str(sample_dir),
        "--output",
        str(sample_dir / "inference_result.json"),
        "--scale-depth",
        str(args.scale_depth),
        "--scale-width",
        str(args.scale_width),
        "--scale-branch-factor",
        str(args.scale_branch_factor),
        "--scale-sampling-seed",
        str(int(args.seed) + index),
        "--audit-max-simplices",
        str(args.audit_max_simplices),
    ]
    if args.audit_ph_backend:
        cmd.extend(["--audit-ph-backend", args.audit_ph_backend])
    if args.scale_stochastic_actions:
        cmd.append("--scale-stochastic-actions")
    if args.scale_sampling_temperature is not None:
        cmd.extend(["--scale-sampling-temperature", str(args.scale_sampling_temperature)])
    if args.scale_sampling_exploration is not None:
        cmd.extend(["--scale-sampling-exploration", str(args.scale_sampling_exploration)])
    if args.memory_bank:
        cmd.extend(["--memory-bank", str(args.memory_bank), "--memory-retrieve-top-k", str(args.memory_retrieve_top_k)])
        if args.memory_save:
            cmd.append("--memory-save")
    if args.no_memory_retrieve:
        cmd.extend(["--memory-retrieve-top-k", "0"])
    return cmd


def _write_sample_metadata(sample_dir: Path, index: int, prompt: str, cmd: list[str], returncode: int) -> None:
    payload: dict[str, Any] = {
        "sample_index": index,
        "prompt": prompt,
        "command": cmd,
        "returncode": returncode,
    }
    (sample_dir / "sample_run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _clean_output_root(output_root: Path) -> None:
    for child in output_root.iterdir():
        if child.name.startswith("sample_"):
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        elif child.name in {"manifest.json", "browser_index.html", "codex_browser_index.html"}:
            child.unlink()
        elif child.name == "browser_memory":
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)


def _configure_default_memory_bank(args: argparse.Namespace, output_root: Path) -> None:
    if args.memory_bank or args.no_memory_retrieve:
        return
    if args.memory_save or int(args.memory_retrieve_top_k) > 0:
        args.memory_bank = str(output_root / "browser_memory" / "reasoning_memory.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "tropicalgt_i_train.latest.pt")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "multi_inference_audit" / "latest")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--prompt", action="append", help="Prompt for one inference sample. May be repeated.")
    parser.add_argument("--prompt-file", default="", help="Text prompts separated by a line containing --- or a JSON list.")
    parser.add_argument("--scale-depth", type=int, default=3)
    parser.add_argument("--scale-width", type=int, default=4)
    parser.add_argument("--scale-branch-factor", type=int, default=3)
    parser.add_argument("--scale-stochastic-actions", dest="scale_stochastic_actions", action="store_true", default=True)
    parser.add_argument("--no-scale-stochastic-actions", dest="scale_stochastic_actions", action="store_false")
    parser.add_argument("--scale-sampling-temperature", type=float, default=2.0)
    parser.add_argument("--scale-sampling-exploration", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--audit-max-simplices", type=int, default=1024)
    parser.add_argument("--audit-ph-backend", default="auto", choices=["auto", "gudhi", "ripser", "none"])
    parser.add_argument("--memory-bank", default="")
    parser.add_argument("--memory-retrieve-top-k", type=int, default=5)
    parser.add_argument("--memory-save", action="store_true", help="Append each sample inference trajectory to the memory bank for later samples.")
    parser.add_argument("--no-memory-retrieve", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--title", default="TropicalGT-I Multi-Run Sample Audit")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root.exists() and not args.skip_existing:
        _clean_output_root(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _configure_default_memory_bank(args, output_root)
    prompts = _read_prompts(args)
    manifest: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        sample_dir = output_root / f"sample_{index:03d}"
        done_marker = sample_dir / "inference_scaling_tree.json"
        if args.skip_existing and done_marker.exists():
            manifest.append({"sample_index": index, "prompt": prompt, "directory": str(sample_dir), "skipped": True})
            continue
        sample_dir.mkdir(parents=True, exist_ok=True)
        cmd = _command(args, prompt, sample_dir, index)
        log_path = sample_dir / "run.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(ROOT.parent), stdout=log, stderr=subprocess.STDOUT, text=True)
        _write_sample_metadata(sample_dir, index, prompt, cmd, proc.returncode)
        manifest.append({"sample_index": index, "prompt": prompt, "directory": str(sample_dir), "returncode": proc.returncode, "log": str(log_path)})
        if proc.returncode != 0:
            (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            raise SystemExit(proc.returncode)

    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    build_index(output_root, output_root / "browser_index.html", args.title)
    build_index(output_root, output_root / "codex_browser_index.html", args.title)
    print(output_root / "browser_index.html")
    print(output_root / "codex_browser_index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
