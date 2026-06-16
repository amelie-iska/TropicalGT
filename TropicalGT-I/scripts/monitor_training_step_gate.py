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


def _required_path_statuses(required_paths: list[Path]) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for path in required_paths:
        row: dict[str, Any] = {"path": str(path), "available": False, "reason": "missing"}
        if not path.exists():
            statuses.append(row)
            continue
        try:
            if path.is_file():
                size = path.stat().st_size
                row["size_bytes"] = int(size)
                if size <= 0:
                    row["reason"] = "empty_file"
                    statuses.append(row)
                    continue
            row["available"] = True
            row["reason"] = "available"
        except OSError as exc:
            row["reason"] = f"stat_failed:{exc.__class__.__name__}"
        statuses.append(row)
    return statuses


def _missing_required_paths(required_paths: list[Path]) -> list[str]:
    return [row["path"] for row in _required_path_statuses(required_paths) if not row.get("available", False)]


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
    required_paths: list[Path] | None = None,
    grace_polls_after_target: int = 30,
    settle_polls_after_target: int = 0,
) -> dict[str, Any]:
    required_paths = list(required_paths or [])
    target_seen_polls = 0
    while True:
        status = latest_training_status(log_path)
        alive = process_alive(pid)
        required_path_statuses = _required_path_statuses(required_paths)
        missing_required_paths = [row["path"] for row in required_path_statuses if not row.get("available", False)]
        settle_polls = max(int(settle_polls_after_target), 0)
        required_wait_polls = max(target_seen_polls - settle_polls, 0)
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
            "required_paths": [str(path) for path in required_paths],
            "required_path_statuses": required_path_statuses,
            "missing_required_paths": missing_required_paths,
            "target_seen_polls": target_seen_polls,
            "required_wait_polls": required_wait_polls,
            "grace_polls_after_target": int(grace_polls_after_target),
            "settle_polls_after_target": settle_polls,
        }
        latest_step = status.get("latest_step")
        if isinstance(latest_step, int) and latest_step >= int(target_step):
            if alive and target_seen_polls < settle_polls:
                target_seen_polls += 1
                payload["target_seen_polls"] = target_seen_polls
                payload["action"] = "target_reached_settling"
                write_record(record_path, payload)
                if once:
                    return payload
                time.sleep(max(float(poll_seconds), 1.0))
                continue
            if alive and missing_required_paths and required_wait_polls < int(grace_polls_after_target):
                target_seen_polls += 1
                payload["target_seen_polls"] = target_seen_polls
                payload["required_wait_polls"] = max(target_seen_polls - settle_polls, 0)
                payload["action"] = "target_reached_waiting_for_required_paths"
                write_record(record_path, payload)
                if once:
                    return payload
                time.sleep(max(float(poll_seconds), 1.0))
                continue
            if missing_required_paths:
                payload["action"] = "target_reached_grace_expired_terminate" if alive else "target_reached_process_already_stopped"
            else:
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
    parser.add_argument("--require-path", action="append", type=Path, default=[], help="Artifact path that should exist before terminating after the target step. May be repeated.")
    parser.add_argument("--grace-polls-after-target", type=int, default=30, help="Polls to wait for required paths after the target step before terminating anyway.")
    parser.add_argument("--settle-polls-after-target", type=int, default=0, help="Polls to wait after first seeing the target step before evaluating termination.")
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
        required_paths=args.require_path,
        grace_polls_after_target=args.grace_polls_after_target,
        settle_polls_after_target=args.settle_polls_after_target,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
