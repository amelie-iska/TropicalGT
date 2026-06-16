#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import time
from typing import Any

STEP_RE = re.compile(r"\|\s*(\d+)/(?:\d+)\s*\[")
FATAL_RE = re.compile(r"(?i)(traceback|out of memory|runtimeerror|fatal|nonfinite|cuda error|(?<![A-Za-z0-9_])[-+]?(?:nan|inf)(?![A-Za-z0-9_]))")


def latest_training_status(log_path: Path, *, tail_bytes: int = 500_000) -> dict[str, Any]:
    if not log_path.exists():
        return {"available": False, "reason": "log_not_found", "latest_step": None, "fatal": False}
    with log_path.open("rb") as handle:
        try:
            handle.seek(max(0, log_path.stat().st_size - int(tail_bytes)))
        except OSError:
            pass
        text = handle.read().decode("utf-8", errors="ignore")
    steps = [int(match.group(1)) for match in STEP_RE.finditer(text)]
    losses = re.findall(r"loss=([0-9.]+), nll=([0-9.]+)", text)
    return {
        "available": True,
        "latest_step": max(steps) if steps else None,
        "latest_loss_nll": list(losses[-1]) if losses else None,
        "fatal": bool(FATAL_RE.search(text)),
        "fatal_markers": sorted(set(match.group(1) for match in FATAL_RE.finditer(text)))[:12],
    }


def process_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def write_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def monitor_step_gate(
    *,
    pid: int,
    log_path: Path,
    target_step: int,
    record_path: Path,
    poll_seconds: float = 60.0,
    terminate_signal: signal.Signals = signal.SIGTERM,
    once: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    while True:
        status = latest_training_status(log_path)
        alive = process_alive(pid)
        payload = {
            "pid": int(pid),
            "pid_alive": bool(alive),
            "log_path": str(log_path),
            "target_step": int(target_step),
            "poll_seconds": float(poll_seconds),
            "checked_at": time.time(),
            "status": status,
            "action": "monitoring",
            "dry_run": bool(dry_run),
        }
        latest_step = status.get("latest_step")
        if isinstance(latest_step, int) and latest_step >= int(target_step):
            payload["action"] = "target_reached_terminate" if alive else "target_reached_process_already_stopped"
            if alive and not dry_run:
                os.kill(pid, terminate_signal)
            write_record(record_path, payload)
            return payload
        if status.get("fatal"):
            payload["action"] = "fatal_marker_observed_no_automatic_restart"
            write_record(record_path, payload)
            return payload
        if not alive:
            payload["action"] = "process_not_alive_before_target"
            write_record(record_path, payload)
            return payload
        write_record(record_path, payload)
        if once:
            return payload
        time.sleep(max(float(poll_seconds), 1.0))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Monitor a TropicalGT-I training log and terminate a PID when a step gate is reached.")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--target-step", type=int, default=5000)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--signal", default="TERM", choices=("TERM", "INT"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    sig = signal.SIGTERM if args.signal == "TERM" else signal.SIGINT
    result = monitor_step_gate(
        pid=args.pid,
        log_path=args.log,
        target_step=args.target_step,
        record_path=args.record,
        poll_seconds=args.poll_seconds,
        terminate_signal=sig,
        once=args.once,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
