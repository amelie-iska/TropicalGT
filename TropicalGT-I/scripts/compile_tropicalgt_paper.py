#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEX = ROOT / "assets" / "tropicalgt_neurips_research_paper.tex"
DEFAULT_BUILD_DIR = Path("/tmp/tropicalgt-paper-build")
DEFAULT_EXTRA_DIR = Path("/tmp/tropicalgt-tex-extra")


def split_extra_paths(value: str | None) -> list[Path]:
    if not value:
        return []
    return [Path(part).expanduser() for part in value.split(os.pathsep) if part]


def discover_tex_extras(cli_paths: list[Path] | None = None) -> list[Path]:
    extras: list[Path] = []
    extras.extend(path.expanduser() for path in (cli_paths or []))
    extras.extend(split_extra_paths(os.environ.get("TROPICALGT_TEX_EXTRA")))
    if DEFAULT_EXTRA_DIR.exists():
        extras.append(DEFAULT_EXTRA_DIR)

    seen: set[Path] = set()
    discovered: list[Path] = []
    for path in extras:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            discovered.append(path)
    return discovered


def texinputs_for(extras: list[Path], inherited: str = "") -> str:
    entries: list[str] = []
    for extra in extras:
        entries.append(str(extra / "tex" / "latex") + "//")
        entries.append(str(extra / "tex" / "generic") + "//")
    if inherited:
        entries.append(inherited)
    entries.append("")
    return os.pathsep.join(entries)


def pdflatex_command(pdflatex: str, tex: Path, build_dir: Path) -> list[str]:
    return [
        pdflatex,
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={build_dir}",
        str(tex),
    ]


def compile_paper(
    *,
    tex: Path,
    build_dir: Path,
    passes: int,
    pdflatex: str,
    extras: list[Path],
    copy_pdf: bool,
    dry_run: bool,
) -> dict[str, object]:
    tex = tex.resolve()
    build_dir = build_dir.resolve()
    if not tex.exists():
        raise FileNotFoundError(f"TeX source not found: {tex}")
    if shutil.which(pdflatex) is None:
        raise FileNotFoundError(f"pdflatex executable not found: {pdflatex}")
    if passes < 1:
        raise ValueError("passes must be >= 1")

    env = os.environ.copy()
    env["TEXINPUTS"] = texinputs_for(extras, env.get("TEXINPUTS", ""))
    command = pdflatex_command(pdflatex, tex, build_dir)
    output_pdf = build_dir / tex.with_suffix(".pdf").name
    target_pdf = tex.with_suffix(".pdf")
    summary: dict[str, object] = {
        "tex": str(tex),
        "build_dir": str(build_dir),
        "passes": passes,
        "pdflatex": pdflatex,
        "tex_extras": [str(path) for path in extras],
        "texinputs": env["TEXINPUTS"],
        "command": command,
        "output_pdf": str(output_pdf),
        "target_pdf": str(target_pdf),
        "copied_pdf": False,
        "dry_run": dry_run,
    }
    if dry_run:
        return summary

    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(passes):
        subprocess.run(command, cwd=tex.parent, env=env, check=True)
    if not output_pdf.exists():
        raise FileNotFoundError(f"Expected PDF was not written: {output_pdf}")
    if copy_pdf:
        shutil.copy2(output_pdf, target_pdf)
        summary["copied_pdf"] = True
        summary["target_pdf_bytes"] = target_pdf.stat().st_size
    summary["output_pdf_bytes"] = output_pdf.stat().st_size
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile the TropicalGT-I NeurIPS paper PDF with stable TeX extras discovery.")
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--pdflatex", default="pdflatex")
    parser.add_argument("--tex-extra", type=Path, action="append", default=[], help="Additional TeX tree root; may be repeated. Also honors TROPICALGT_TEX_EXTRA.")
    parser.add_argument("--no-copy", action="store_true", help="Leave the PDF in the build directory instead of replacing the tracked asset PDF.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command summary without running pdflatex.")
    args = parser.parse_args(argv)

    extras = discover_tex_extras(args.tex_extra)
    summary = compile_paper(
        tex=args.tex,
        build_dir=args.build_dir,
        passes=args.passes,
        pdflatex=args.pdflatex,
        extras=extras,
        copy_pdf=not args.no_copy,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2), file=sys.stderr)
        raise
