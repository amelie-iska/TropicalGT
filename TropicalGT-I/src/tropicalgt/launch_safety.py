from __future__ import annotations

import json
import os
from typing import Any, Mapping

_TRUE_VALUES = {"1", "true", "yes", "y", "on", "cleared", "allow", "allowed"}


def _env_true(value: str | None) -> bool:
    return str(value or "").strip().lower() in _TRUE_VALUES


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def gpu_launch_safety_contract(
    *,
    allow_gpu_launch: bool = False,
    gpu_memory_budget_mb: int | None = None,
    gpu_clearance_note: str = "",
    environ: Mapping[str, str] | None = None,
    action: str = "training_launch",
) -> dict[str, Any]:
    """Return the explicit no-surprise GPU launch contract.

    This function intentionally does not query GPU state. It is a policy gate for
    launch/restart automation, so CPU-only review/report generation can continue
    while a user's separate GPU job is active.
    """

    env = os.environ if environ is None else environ
    env_allows = _env_true(env.get("TROPICALGT_ALLOW_GPU_LAUNCH"))
    env_budget = _positive_int(env.get("TROPICALGT_GPU_MEMORY_BUDGET_MB"))
    env_note = str(env.get("TROPICALGT_GPU_CLEARANCE_NOTE", "")).strip()
    budget = _positive_int(gpu_memory_budget_mb) or env_budget
    note = str(gpu_clearance_note or env_note).strip()
    explicit_clearance = bool(allow_gpu_launch or env_allows)
    budget_clearance = bool(budget is not None and note)
    blockers: list[str] = []
    if not explicit_clearance and not budget_clearance:
        blockers.append("missing_explicit_gpu_clearance_or_budget")
    if budget is not None and not note:
        blockers.append("gpu_memory_budget_requires_clearance_note")
    allowed = not blockers
    source = "cli_allow_gpu_launch" if allow_gpu_launch else "env_allow_gpu_launch" if env_allows else "declared_budget" if budget_clearance else "blocked"
    return {
        "schema_version": "tropicalgt.gpu_launch_safety_contract.v1",
        "action": action,
        "gpu_launch_allowed": allowed,
        "clearance_source": source,
        "explicit_clearance": explicit_clearance,
        "declared_gpu_memory_budget_mb": budget,
        "gpu_clearance_note": note,
        "blockers": blockers,
        "policy": (
            "Training/restart automation must not launch CUDA/GPU work unless the user has explicitly cleared GPU use "
            "or the caller provides a declared safe GPU memory budget with a clearance note. CPU-only report/review "
            "generation may proceed without this clearance. This gate does not inspect or reserve GPU memory."
        ),
    }


def enforce_gpu_launch_safety(**kwargs: Any) -> dict[str, Any]:
    contract = gpu_launch_safety_contract(**kwargs)
    if not contract.get("gpu_launch_allowed", False):
        raise RuntimeError("GPU launch blocked by TropicalGT-I safety contract: " + json.dumps(contract, sort_keys=True))
    return contract
