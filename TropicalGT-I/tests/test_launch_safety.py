from __future__ import annotations

import json

import pytest

from tropicalgt.launch_safety import enforce_gpu_launch_safety, gpu_launch_safety_contract


def test_gpu_launch_safety_blocks_training_without_clearance_or_budget():
    contract = gpu_launch_safety_contract(environ={}, action="unit_train_launch")

    assert contract["schema_version"] == "tropicalgt.gpu_launch_safety_contract.v1"
    assert contract["gpu_launch_allowed"] is False
    assert contract["clearance_source"] == "blocked"
    assert "missing_explicit_gpu_clearance_or_budget" in contract["blockers"]
    assert "CPU-only report/review generation may proceed" in contract["policy"]

    with pytest.raises(RuntimeError, match="GPU launch blocked") as exc:
        enforce_gpu_launch_safety(environ={}, action="unit_train_launch")
    payload = json.loads(str(exc.value).split(": ", 1)[1])
    assert payload["action"] == "unit_train_launch"
    assert payload["gpu_launch_allowed"] is False


def test_gpu_launch_safety_allows_explicit_clearance_flag():
    contract = gpu_launch_safety_contract(allow_gpu_launch=True, environ={}, action="unit_train_launch")

    assert contract["gpu_launch_allowed"] is True
    assert contract["clearance_source"] == "cli_allow_gpu_launch"
    assert contract["blockers"] == []


def test_gpu_launch_safety_allows_env_clearance():
    contract = gpu_launch_safety_contract(environ={"TROPICALGT_ALLOW_GPU_LAUNCH": "true"}, action="unit_train_launch")

    assert contract["gpu_launch_allowed"] is True
    assert contract["clearance_source"] == "env_allow_gpu_launch"


def test_gpu_launch_safety_budget_requires_note():
    blocked = gpu_launch_safety_contract(gpu_memory_budget_mb=1024, environ={}, action="unit_train_launch")
    allowed = gpu_launch_safety_contract(
        gpu_memory_budget_mb=1024,
        gpu_clearance_note="user cleared a 1GB budget for this run",
        environ={},
        action="unit_train_launch",
    )

    assert blocked["gpu_launch_allowed"] is False
    assert "gpu_memory_budget_requires_clearance_note" in blocked["blockers"]
    assert allowed["gpu_launch_allowed"] is True
    assert allowed["clearance_source"] == "declared_budget"
    assert allowed["declared_gpu_memory_budget_mb"] == 1024
    assert "user cleared" in allowed["gpu_clearance_note"]


def test_advanced_campaign_supervisor_imports_launch_safety():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_advanced_bpb_campaign.py"
    spec = importlib.util.spec_from_file_location("run_advanced_bpb_campaign_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    contract = module.gpu_launch_safety_contract(environ={}, action="advanced_bpb_campaign_train_launch")
    assert contract["gpu_launch_allowed"] is False
    assert "missing_explicit_gpu_clearance_or_budget" in contract["blockers"]
