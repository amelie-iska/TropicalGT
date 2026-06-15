import importlib.util
import os
import sys
from pathlib import Path


def _load_compile_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "compile_tropicalgt_paper.py"
    spec = importlib.util.spec_from_file_location("compile_tropicalgt_paper", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_texinputs_for_preserves_defaults_and_adds_extra_roots(tmp_path):
    module = _load_compile_module()
    extra = tmp_path / "tex-extra"
    extra.mkdir()

    texinputs = module.texinputs_for([extra], inherited="/already/there")

    parts = texinputs.split(os.pathsep)
    assert parts[0].endswith("tex-extra/tex/latex//")
    assert parts[1].endswith("tex-extra/tex/generic//")
    assert "/already/there" in parts
    assert parts[-1] == ""


def test_compile_paper_dry_run_reports_command(tmp_path):
    module = _load_compile_module()
    tex = tmp_path / "paper.tex"
    tex.write_text("\\documentclass{article}\\begin{document}ok\\end{document}", encoding="utf-8")

    summary = module.compile_paper(
        tex=tex,
        build_dir=tmp_path / "build",
        passes=2,
        pdflatex=sys.executable,
        extras=[],
        copy_pdf=True,
        dry_run=True,
    )

    assert summary["dry_run"] is True
    assert summary["passes"] == 2
    assert summary["command"][0] == sys.executable
    assert str(tex.resolve()) in summary["command"]
    assert summary["copied_pdf"] is False
