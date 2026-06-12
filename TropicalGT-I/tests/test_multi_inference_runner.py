import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_multi_inference_audits.py"
    spec = importlib.util.spec_from_file_location("run_multi_inference_audits", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    base = {
        "config": Path("config.json"),
        "checkpoint": Path("checkpoint.pt"),
        "memory_bank": "",
        "memory_retrieve_top_k": 3,
        "memory_save": True,
        "no_memory_retrieve": False,
        "scale_depth": 3,
        "scale_width": 4,
        "scale_branch_factor": 3,
        "seed": 1729,
        "trace_limit": 64,
        "audit_max_simplices": 1024,
        "audit_ph_backend": "auto",
        "scale_stochastic_actions": True,
        "scale_sampling_temperature": 1.25,
        "scale_sampling_exploration": 0.45,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_multi_inference_runner_defaults_to_bundle_local_memory_bank(tmp_path):
    module = _load_runner_module()
    args = _args()
    output_root = tmp_path / "multi_sample_browser" / "latest"

    module._configure_default_memory_bank(args, output_root)
    command = module._command(args, "Question: test\nAnswer:", output_root / "sample_000", 0)

    expected = output_root / "browser_memory" / "reasoning_memory.jsonl"
    assert args.memory_bank == str(expected)
    memory_index = command.index("--memory-bank")
    assert command[memory_index + 1] == str(expected)
    assert "--memory-save" in command
    assert "--memory-retrieve-top-k" in command
    trace_index = command.index("--trace-limit")
    assert command[trace_index + 1] == "64"


def test_multi_inference_runner_keeps_explicit_memory_bank(tmp_path):
    module = _load_runner_module()
    explicit = tmp_path / "explicit.jsonl"
    args = _args(memory_bank=str(explicit))

    module._configure_default_memory_bank(args, tmp_path / "out")

    assert args.memory_bank == str(explicit)


def test_multi_inference_runner_full_preset_enforces_deep_wide_complete_budgets():
    module = _load_runner_module()
    args = _args(
        audit_preset="full",
        scale_depth=3,
        scale_width=4,
        scale_branch_factor=3,
        trace_limit=64,
        audit_max_simplices=1024,
        memory_retrieve_top_k=3,
    )

    module._apply_audit_preset(args)

    assert args.scale_depth >= 7
    assert args.scale_width >= 12
    assert args.scale_branch_factor >= 5
    assert args.trace_limit >= 2048
    assert args.trace_limit >= 8192
    assert args.audit_max_simplices >= 8192
    assert args.memory_retrieve_top_k >= 8
    assert args.require_complete_reasoning_steps is True
    assert args.scale_stochastic_actions is True
    assert args.scale_sampling_temperature == 1.25
    assert args.scale_sampling_exploration == 0.45


def test_multi_inference_runner_requires_complete_steps_for_browser_artifacts(tmp_path):
    module = _load_runner_module()
    args = _args()

    command = module._command(args, "Question: test\nAnswer:", tmp_path / "sample_000", 0)

    assert "--require-complete-reasoning-steps" in command
