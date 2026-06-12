from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_infer_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "infer_tropicalgt_i.py"
    spec = importlib.util.spec_from_file_location("infer_tropicalgt_i_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_output_dir_alone_does_not_enable_html():
    infer = _load_infer_module()

    assert (
        infer._resolve_render_html(
            audit_all=False,
            audit_output_dir="audit",
            audit_render_html=False,
            no_audit_render_html=False,
            interactive_artifacts=False,
        )
        is False
    )


def test_interactive_artifacts_flag_enables_html():
    infer = _load_infer_module()

    assert (
        infer._resolve_render_html(
            audit_all=False,
            audit_output_dir="audit",
            audit_render_html=False,
            no_audit_render_html=False,
            interactive_artifacts=True,
        )
        is True
    )


def test_no_audit_render_html_overrides_interactive_artifacts():
    infer = _load_infer_module()

    assert (
        infer._resolve_render_html(
            audit_all=True,
            audit_output_dir="audit",
            audit_render_html=True,
            no_audit_render_html=True,
            interactive_artifacts=True,
        )
        is False
    )


def test_audit_all_html_requires_complete_reasoning_steps():
    infer = _load_infer_module()

    assert (
        infer._resolve_require_complete_reasoning_steps(
            audit_all=True,
            render_html=True,
            require_complete_reasoning_steps=False,
        )
        is True
    )


def test_json_only_audit_does_not_require_complete_browser_steps():
    infer = _load_infer_module()

    assert (
        infer._resolve_require_complete_reasoning_steps(
            audit_all=False,
            render_html=False,
            require_complete_reasoning_steps=False,
        )
        is False
    )


def test_multi_inference_audit_driver_requests_interactive_artifacts(tmp_path: Path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_multi_inference_audits.py"
    spec = importlib.util.spec_from_file_location("run_multi_inference_audits_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(
        config=Path("config.json"),
        checkpoint=Path("checkpoint.pt"),
        scale_depth=3,
        scale_width=4,
        scale_branch_factor=3,
        seed=1729,
        trace_limit=64,
        audit_max_simplices=1024,
        audit_ph_backend="auto",
        scale_stochastic_actions=True,
        scale_sampling_temperature=2.0,
        scale_sampling_exploration=0.35,
        memory_bank="",
        memory_retrieve_top_k=0,
        memory_save=False,
        no_memory_retrieve=True,
    )
    cmd = module._command(args, "Question: test\nAnswer:", tmp_path / "sample_000", 0)
    assert "--audit-all" in cmd
    assert "--interactive-artifacts" in cmd
    assert "--require-complete-reasoning-steps" in cmd
    trace_index = cmd.index("--trace-limit")
    assert cmd[trace_index + 1] == "64"
