#!/usr/bin/env python3
"""Build a sample-first browser index for TropicalGT-I interactive audits.

The periodic audit renderer writes one directory per model-evaluated row/input. This
script keeps that structure visible in the browser: choose the sample first,
then inspect that sample's GoT trajectory, topology, memory, GraphCG, and
tropical support artifacts.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, NamedTuple


class ArtifactSpec(NamedTuple):
    label: str
    rel: str
    tag: str


PRIMARY_ARTIFACTS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec("GoT NLL landscape", "got_trajectory_pca_3d.html", "NLL"),
    ArtifactSpec("Embedding map", "got_embedding_map_3d.html", "PCA"),
    ArtifactSpec("Full radius complex", "got_full_trajectory_complex.html", "radius"),
    ArtifactSpec("Full radius simplex tree", "got_full_trajectory_simplex_tree_3d.html", "tree"),
    ArtifactSpec("Probability complex", "got_full_trajectory_complex_jensen_shannon.html", "prob"),
    ArtifactSpec(
        "Probability simplex tree",
        "got_full_trajectory_simplex_tree_3d_jensen_shannon.html",
        "tree",
    ),
    ArtifactSpec("Reasoning step index", "reasoning_step_complex_maps/index.html", "steps"),
    ArtifactSpec("Analogical many-map overview", "analogical_memory_retrieval.html", "many"),
    ArtifactSpec("Analogical top-k index", "analogical_memory_topk_index.html", "top-k"),
    ArtifactSpec("Persistence barcode", "trajectory_persistence/persistence_barcode.html", "bars"),
    ArtifactSpec("2-parameter F2[x,y] bifiltration", "trajectory_persistence/two_parameter_bifiltration.html", "2-param"),
    ArtifactSpec("Persistence Betti/free-resolution", "trajectory_persistence/persistence_module_betti.html", "betti"),
    ArtifactSpec("Persistence vector representations", "trajectory_persistence/persistence_representations.html", "vector"),
    ArtifactSpec("Persistence landscapes", "trajectory_persistence/persistence_landscapes.html", "land"),
    ArtifactSpec("GraphCG directions", "graphcg_direction_cosines.html", "rank"),
    ArtifactSpec("Tropical support", "tropical_support_heatmap.html", "support"),
    ArtifactSpec("Generated sample index", "browser_index.html", "raw"),
    ArtifactSpec("Generated audit dashboard", "inference_audit.html", "audit"),
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _short(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _fmt(value: float | None, digits: int = 5) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}g}"


def _sample_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    dirs = sorted({path.parent for path in root.rglob("inference_scaling_tree.json")})
    if (root / "inference_scaling_tree.json").exists():
        dirs = [root] + [path for path in dirs if path != root]
    return dirs


def _sample_label(root: Path, sample_dir: Path, index: int) -> str:
    if sample_dir == root:
        return "Sample 00"
    name = sample_dir.name.replace("_", " ").title()
    if name.lower().startswith("example "):
        suffix = name.split(" ", 1)[1]
        return f"Sample {int(suffix):02d}" if suffix.isdigit() else f"Sample {index:02d}"
    return f"Sample {index:02d}: {name}"


def _relative_src(root: Path, sample_dir: Path, target: Path) -> str:
    return target.relative_to(root).as_posix() if sample_dir != root else target.relative_to(sample_dir).as_posix()


def _relative_between(base: Path, target: Path) -> str:
    return os.path.relpath(target, start=base).replace(os.sep, "/")


def _artifact_row(root: Path, sample_dir: Path, target: Path, label: str, tag: str) -> dict[str, str]:
    return {
        "label": label,
        "src": _relative_src(root, sample_dir, target),
        "tag": tag,
    }


def _step_label(path: Path) -> str:
    stem = path.stem.replace("_simplex_tree", "")
    try:
        number = int(stem.rsplit("_", 1)[1])
    except Exception:
        return path.stem.replace("_", " ")
    suffix = "simplex tree" if path.stem.endswith("_simplex_tree") else "complex"
    return f"Reasoning step {number:03d} {suffix}"


def _artifact_rows(root: Path, sample_dir: Path) -> list[dict[str, str]]:
    rows = []
    seen: set[str] = set()
    for artifact in PRIMARY_ARTIFACTS:
        target = sample_dir / artifact.rel
        if not target.exists():
            continue
        row = _artifact_row(root, sample_dir, target, artifact.label, artifact.tag)
        rows.append(row)
        seen.add(row["src"])

    step_dir = sample_dir / "reasoning_step_complex_maps"
    if step_dir.exists():
        complex_targets = sorted(
            target
            for target in step_dir.glob("reasoning_step_*.html")
            if not target.stem.endswith("_simplex_tree")
        )
        tree_targets = sorted(step_dir.glob("reasoning_step_*_simplex_tree.html"))
        if complex_targets:
            rollup = step_dir / "complexes_catalog.html"
            _write_step_rollup(rollup, complex_targets, "All reasoning step complexes")
            row = _artifact_row(root, sample_dir, rollup, "All reasoning step complexes", "steps")
            if row["src"] not in seen:
                rows.append(row)
                seen.add(row["src"])
        if tree_targets:
            rollup = step_dir / "simplex_trees_catalog.html"
            _write_step_rollup(rollup, tree_targets, "All reasoning step simplex trees")
            row = _artifact_row(root, sample_dir, rollup, "All reasoning step simplex trees", "trees")
            if row["src"] not in seen:
                rows.append(row)
                seen.add(row["src"])

    for target in sorted(sample_dir.glob("analogical_memory_map_*.html")):
        try:
            number = int(target.stem.rsplit("_", 1)[1])
        except Exception:
            number = len([row for row in rows if row["src"].endswith(target.name)]) + 1
        row = _artifact_row(root, sample_dir, target, f"Analogical top-k map {number:02d}", "map")
        if row["src"] not in seen:
            rows.append(row)
            seen.add(row["src"])
    return rows


def _write_step_rollup(path: Path, targets: list[Path], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first = targets[0].name if targets else ""
    buttons = []
    for target in targets:
        label = _step_label(target)
        href = target.name
        buttons.append(
            f'<button class="artifact" data-src="{html.escape(href, quote=True)}" '
            f'data-label="{html.escape(label, quote=True)}">{html.escape(label)}</button>'
        )
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: dark; --bg:#070b12; --panel:#0d1728; --ink:#eaf2ff; --muted:#a9bfdf; --accent:#5eead4; --edge:rgba(130,170,220,.28); }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; min-height:100vh; background:#070b12; color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,sans-serif; }}
    .shell {{ display:grid; grid-template-columns:minmax(300px,390px) minmax(560px,1fr); min-height:100vh; }}
    aside {{ padding:18px; border-right:1px solid var(--edge); background:#0d1728; overflow:auto; }}
    main {{ min-width:0; display:grid; grid-template-rows:auto 1fr; }}
    h1 {{ margin:0 0 8px; font-size:22px; letter-spacing:0; }}
    p {{ margin:0 0 14px; color:var(--muted); line-height:1.45; font-size:13px; }}
    .artifact-list {{ display:grid; gap:8px; }}
    button.artifact {{ width:100%; text-align:left; border:1px solid rgba(130,170,220,.28); border-radius:6px; background:#10203a; color:var(--ink); padding:10px 11px; font:inherit; cursor:pointer; }}
    button.artifact:hover, button.artifact.active {{ border-color:rgba(94,234,212,.78); background:#143657; }}
    .topbar {{ padding:13px 16px; border-bottom:1px solid var(--edge); display:flex; justify-content:space-between; gap:12px; align-items:center; background:#070b12; }}
    #title {{ margin:0; font-size:15px; color:#dbeafe; }}
    a.open {{ color:var(--accent); text-decoration:none; border:1px solid rgba(94,234,212,.45); border-radius:6px; padding:7px 9px; }}
    iframe {{ width:100%; height:100%; min-height:820px; border:0; background:#05070d; }}
    @media (max-width:960px) {{ .shell {{ grid-template-columns:1fr; }} aside {{ border-right:0; border-bottom:1px solid var(--edge); }} iframe {{ min-height:760px; }} }}
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <h1>{html.escape(title)}</h1>
      <p>Single-page rollup for the generated per-step interactive artifacts. Each row opens the exact original page in the inspection frame.</p>
      <div class="artifact-list">{''.join(buttons)}</div>
    </aside>
    <main>
      <div class="topbar"><h2 id="title">{html.escape(_step_label(targets[0]) if targets else title)}</h2><a id="open" class="open" href="{html.escape(first, quote=True)}">open full page</a></div>
      <iframe id="frame" src="{html.escape(first, quote=True)}" title="{html.escape(title, quote=True)}"></iframe>
    </main>
  </div>
  <script>
    const buttons = Array.from(document.querySelectorAll("button.artifact"));
    const frame = document.getElementById("frame");
    const title = document.getElementById("title");
    const open = document.getElementById("open");
    function setPanel(button) {{
      const src = button.dataset.src;
      buttons.forEach((candidate) => candidate.classList.toggle("active", candidate === button));
      frame.src = src;
      title.textContent = button.dataset.label || button.textContent.trim();
      open.href = src;
    }}
    buttons.forEach((button) => button.addEventListener("click", () => setPanel(button)));
    if (buttons.length) setPanel(buttons[0]);
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def _artifact_family(path: Path) -> str:
    parts = set(path.parts)
    name = path.name
    if "reasoning_step_complex_maps" in parts:
        return "reasoning steps"
    if name.startswith("analogical_memory"):
        return "analogical memory"
    if "trajectory_persistence" in parts or name.startswith("persistence_"):
        return "persistence"
    if name.startswith("got_"):
        return "graph-of-thought"
    if name.startswith("graphcg_"):
        return "GraphCG"
    if name.startswith("tropical_"):
        return "tropical support"
    if name.endswith(".json"):
        return "payload"
    return "other"


def discover_interactive_artifacts(root: Path, samples: list[dict[str, Any]]) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    seen: set[str] = set()
    for sample in samples:
        sample_dir = root if sample["dir"] == "." else root / str(sample["dir"])
        for target in sorted(sample_dir.rglob("*")):
            if not target.is_file() or target.suffix.lower() not in {".html", ".json"}:
                continue
            rel = target.relative_to(root).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            artifacts.append(
                {
                    "sample": str(sample["label"]),
                    "family": _artifact_family(target.relative_to(sample_dir)),
                    "label": target.stem.replace("_", " "),
                    "src": rel,
                    "extension": target.suffix.lower().lstrip("."),
                    "tag": "html" if target.suffix.lower() == ".html" else "payload",
                    "exists": "yes",
                    "source": "generated audit artifact",
                }
            )
    return artifacts


def _write_interactive_catalog(root: Path, output: Path, artifacts: list[dict[str, str]], title: str) -> None:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['sample'])}</td>"
        f"<td>{html.escape(row['family'])}</td>"
        f"<td>{html.escape(row['label'])}</td>"
        f"<td><a href='{html.escape(row['src'], quote=True)}'>{html.escape(row['src'])}</a></td>"
        f"<td>{html.escape(row['extension'])}</td>"
        f"<td>{html.escape(row['tag'])}</td>"
        "</tr>"
        for row in artifacts
    ) or "<tr><td colspan='6'>No generated artifacts found.</td></tr>"
    output.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} Artifact Catalog</title>
  <style>
    :root {{ color-scheme: dark; --bg:#070b12; --panel:#0d1728; --ink:#eaf2ff; --muted:#a9bfdf; --accent:#5eead4; --edge:rgba(130,170,220,.28); }}
    body {{ margin:0; background:#070b12; color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,sans-serif; }}
    main {{ max-width:1280px; margin:0 auto; padding:28px 22px 42px; }}
    h1 {{ margin:0 0 8px; font-size:26px; letter-spacing:0; }}
    p {{ color:var(--muted); line-height:1.45; }}
    .controls {{ display:flex; gap:10px; flex-wrap:wrap; margin:16px 0; }}
    input, select {{ background:#0d1728; color:var(--ink); border:1px solid var(--edge); border-radius:6px; padding:9px 10px; }}
    table {{ width:100%; border-collapse:collapse; background:#0d1728; border:1px solid var(--edge); }}
    th, td {{ padding:9px 10px; border-bottom:1px solid rgba(130,170,220,.18); text-align:left; vertical-align:top; font-size:13px; }}
    th {{ color:#99f6e4; position:sticky; top:0; background:#0d1728; }}
    a {{ color:#7dd3fc; text-decoration:none; overflow-wrap:anywhere; }}
    a:hover {{ text-decoration:underline; }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(title)} Complete Interactive Artifact Catalog</h1>
    <p>One searchable catalog for every generated HTML page and JSON payload in this browser bundle. Paths are relative to <code>{html.escape(str(root))}</code>.</p>
    <div class="controls">
      <input id="search" type="search" placeholder="Search paths, samples, or families">
      <select id="family"><option value="">All families</option></select>
    </div>
    <table id="catalog">
      <thead><tr><th>sample</th><th>family</th><th>label</th><th>relative path</th><th>ext</th><th>tag</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </main>
  <script>
    const search = document.getElementById("search");
    const family = document.getElementById("family");
    const rows = Array.from(document.querySelectorAll("#catalog tbody tr"));
    const families = Array.from(new Set(rows.map((row) => row.children[1]?.textContent || "").filter(Boolean))).sort();
    for (const value of families) {{
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      family.appendChild(option);
    }}
    function applyFilters() {{
      const needle = search.value.toLowerCase();
      const selected = family.value;
      for (const row of rows) {{
        const text = row.textContent.toLowerCase();
        const rowFamily = row.children[1]?.textContent || "";
        row.style.display = (!needle || text.includes(needle)) && (!selected || rowFamily === selected) ? "" : "none";
      }}
    }}
    search.addEventListener("input", applyFilters);
    family.addEventListener("change", applyFilters);
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def _sample_summary(root: Path, sample_dir: Path, index: int) -> dict[str, Any]:
    scaling = _read_json(sample_dir / "inference_scaling_tree.json")
    got = _read_json(sample_dir / "got_trajectory_payloads.json")
    analogical = _read_json(sample_dir / "analogical_simplicial_maps.json")
    candidates = [row for row in scaling.get("candidates", []) if isinstance(row, dict)]
    root_candidate = candidates[0] if candidates else {}
    nlls = [_safe_float(row.get("nll")) for row in candidates]
    finite_nlls = [value for value in nlls if value is not None]
    levels = scaling.get("levels", [])
    level_count = len(levels) if isinstance(levels, list) else len({row.get("level") for row in candidates})
    progress = got.get("nll_progress", {}) if isinstance(got.get("nll_progress"), dict) else {}
    pca = got.get("embedding_pca_diagnostics", {}) if isinstance(got.get("embedding_pca_diagnostics"), dict) else {}
    maps = analogical.get("maps", []) if isinstance(analogical.get("maps"), list) else []
    failed_maps = 0
    warnings = []
    for row in maps:
        if not isinstance(row, dict):
            continue
        if row.get("warning"):
            warnings.append(str(row.get("warning")))
        failed_maps += int(row.get("failed_edge_count", 0) or 0) + int(row.get("failed_face_count", 0) or 0)

    artifacts = _artifact_rows(root, sample_dir)
    open_target = None
    for candidate_name in ("browser_index.html", "inference_audit.html", "got_trajectory_pca_3d.html"):
        candidate_path = sample_dir / candidate_name
        if candidate_path.exists():
            open_target = candidate_path
            break
    open_src = _relative_src(root, sample_dir, open_target) if open_target is not None else (artifacts[0]["src"] if artifacts else ".")
    record_id = str(root_candidate.get("record_id") or "")
    input_text = root_candidate.get("input_text") or root_candidate.get("input_preview") or ""
    target_text = root_candidate.get("target_text") or ""
    output_text = root_candidate.get("decoded_argmax") or root_candidate.get("decoded_preview") or ""

    return {
        "index": index,
        "label": _sample_label(root, sample_dir, index),
        "dir": "." if sample_dir == root else sample_dir.relative_to(root).as_posix(),
        "record_id": record_id,
        "record_short": record_id[:16] if record_id else "n/a",
        "candidate_count": len(candidates),
        "edge_count": max(0, len(candidates) - 1),
        "level_count": level_count,
        "nll_min": min(finite_nlls) if finite_nlls else None,
        "nll_mean": mean(finite_nlls) if finite_nlls else None,
        "nll_std": pstdev(finite_nlls) if len(finite_nlls) > 1 else 0.0 if finite_nlls else None,
        "improving_edge_fraction": _safe_float(progress.get("improving_edge_fraction")),
        "terminal_best_improvement": _safe_float(progress.get("terminal_best_improvement")),
        "pca_corr": _safe_float(pca.get("pairwise_distance_correlation")),
        "pca_stress": _safe_float(pca.get("normalized_stress")),
        "sample_source": str(scaling.get("sample_source") or scaling.get("source") or "periodic eval/inference sample"),
        "input_preview": _short(input_text, 260),
        "target_preview": _short(target_text, 220),
        "output_preview": _short(output_text, 220),
        "artifacts": artifacts,
        "open_src": open_src,
        "warning_count": len(warnings),
        "failed_simplicial_correspondences": failed_maps,
    }


def _json_attr(value: Any) -> str:
    return html.escape(json.dumps(value), quote=True)


def build_index(root: Path, output: Path, title: str) -> None:
    root = root.resolve()
    output = output.resolve()
    samples = [_sample_summary(root, sample_dir, idx) for idx, sample_dir in enumerate(_sample_dirs(root))]
    if not samples:
        raise SystemExit(f"no sample audit directories found under {root}")
    catalog_path = output.parent / "interactive_visualization_catalog.html"
    artifacts = discover_interactive_artifacts(root, samples)
    _write_interactive_catalog(root, catalog_path, artifacts, title)
    catalog_href = _relative_between(output.parent, catalog_path)
    first_artifact = samples[0]["artifacts"][0] if samples[0]["artifacts"] else {"src": "", "label": "No artifacts"}
    cards = []
    for sample in samples:
        artifact_buttons = []
        for artifact in sample["artifacts"]:
            artifact_buttons.append(
                f'<button class="artifact" data-sample="{sample["index"]}" data-src="{html.escape(artifact["src"], quote=True)}" '
                f'data-label="{html.escape(sample["label"] + " / " + artifact["label"], quote=True)}">'
                f'<span>{html.escape(artifact["label"])}</span><span class="tag">{html.escape(artifact["tag"])}</span></button>'
            )
        cards.append(
            f"""
            <section class="sample" data-sample="{sample['index']}">
              <div class="sample-head">
                <div>
                  <h2>{html.escape(sample['label'])}</h2>
                  <p class="record">record <code>{html.escape(sample['record_short'])}</code> · {html.escape(sample['dir'])}</p>
                </div>
                <a class="open-sample" href="{html.escape(sample['open_src'], quote=True)}">open sample</a>
              </div>
              <div class="metrics">
                <span>{sample['candidate_count']} states</span>
                <span>{sample['edge_count']} edges</span>
                <span>{sample['level_count']} levels</span>
                <span>{len(sample['artifacts'])} plots</span>
                <span>NLL min {_fmt(sample['nll_min'])}</span>
                <span>NLL std {_fmt(sample['nll_std'])}</span>
                <span>PCA corr {_fmt(sample['pca_corr'], 3)}</span>
              </div>
              <details>
                <summary>model input/output preview</summary>
                <dl>
                  <dt>Input</dt><dd>{html.escape(sample['input_preview'] or 'n/a')}</dd>
                  <dt>Target</dt><dd>{html.escape(sample['target_preview'] or 'n/a')}</dd>
                  <dt>Model output</dt><dd>{html.escape(sample['output_preview'] or 'n/a')}</dd>
                  <dt>NLL progress</dt><dd>improving edge fraction {_fmt(sample['improving_edge_fraction'])}; terminal best improvement {_fmt(sample['terminal_best_improvement'])}</dd>
                  <dt>Analogical map audit</dt><dd>{sample['warning_count']} warnings; {sample['failed_simplicial_correspondences']} failed displayed simplex correspondences</dd>
                </dl>
              </details>
              <div class="artifact-grid">{''.join(artifact_buttons)}</div>
            </section>
            """
        )
    payload = _json_attr(samples)
    output.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: dark; --bg:#070b12; --panel:#0d1728; --panel2:#101f36; --ink:#eaf2ff; --muted:#a9bfdf; --accent:#5eead4; --gold:#f6d365; --edge:rgba(130,170,220,.28); }}
    * {{ box-sizing: border-box; }}
    html, body {{ height: 100%; }}
    body {{ margin:0; background:linear-gradient(135deg,#05070d 0%,#07111f 55%,#0b1322 100%); color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    .shell {{ display:grid; grid-template-columns:minmax(380px,460px) minmax(780px,1fr); min-height:100vh; }}
    aside {{ position:sticky; top:0; height:100vh; overflow:auto; padding:18px; background:linear-gradient(180deg,rgba(13,23,40,.98),rgba(5,7,13,.98)); border-right:1px solid var(--edge); }}
    main {{ min-width:0; min-height:100vh; display:grid; grid-template-rows:auto minmax(900px,1fr); }}
    h1 {{ margin:0 0 8px; font-size:24px; line-height:1.12; letter-spacing:0; }}
    .summary {{ color:var(--muted); font-size:13px; line-height:1.45; margin:0 0 16px; }}
    .catalog-link {{ display:flex; align-items:center; justify-content:space-between; gap:12px; color:#071018; background:var(--accent); text-decoration:none; border-radius:6px; padding:10px 12px; font-weight:800; margin:0 0 16px; box-shadow:0 14px 30px rgba(94,234,212,.16); }}
    .catalog-link code {{ color:#071018; font-size:10px; white-space:nowrap; }}
    .sample {{ border:1px solid rgba(130,170,220,.22); border-radius:8px; background:rgba(16,31,54,.48); padding:12px; margin:0 0 14px; }}
    .sample.active {{ border-color:rgba(94,234,212,.82); box-shadow:0 0 0 1px rgba(94,234,212,.18) inset; }}
    .sample-head {{ display:flex; justify-content:space-between; gap:10px; align-items:start; }}
    h2 {{ margin:0 0 4px; font-size:14px; letter-spacing:0; }}
    .record {{ margin:0 0 9px; color:var(--muted); font-size:12px; }}
    .open-sample, .open {{ flex:0 0 auto; color:var(--accent); text-decoration:none; border:1px solid rgba(94,234,212,.48); border-radius:6px; padding:7px 9px; font-size:12px; }}
    .metrics {{ display:flex; flex-wrap:wrap; gap:6px; margin:8px 0; }}
    .metrics span {{ border:1px solid rgba(130,170,220,.2); background:rgba(7,17,31,.75); border-radius:999px; color:#d8e8ff; padding:4px 7px; font-size:11px; }}
    details {{ margin:8px 0; }}
    summary {{ color:#c7e2ff; cursor:pointer; font-size:12px; }}
    dl {{ margin:8px 0 0; display:grid; grid-template-columns:72px minmax(0,1fr); gap:6px 8px; font-size:12px; line-height:1.4; }}
    dt {{ color:var(--gold); }}
    dd {{ margin:0; color:#d8e8ff; overflow-wrap:anywhere; }}
    .artifact-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:7px; margin-top:9px; max-height:360px; overflow:auto; padding-right:2px; }}
    button.artifact {{ display:flex; justify-content:space-between; align-items:center; gap:8px; width:100%; border:1px solid rgba(130,170,220,.24); border-radius:6px; background:rgba(13,35,62,.9); color:var(--ink); padding:8px 9px; font:inherit; font-size:12px; cursor:pointer; text-align:left; }}
    button.artifact:hover, button.artifact.active {{ border-color:rgba(94,234,212,.78); background:rgba(19,52,83,.98); }}
    .tag {{ flex:0 0 auto; color:#071018; background:var(--accent); border-radius:999px; padding:2px 6px; font-size:9px; font-weight:800; }}
    .topbar {{ display:flex; align-items:center; justify-content:space-between; gap:18px; padding:14px 18px; border-bottom:1px solid var(--edge); background:rgba(7,11,18,.86); }}
    #title {{ margin:0; font-size:16px; line-height:1.2; color:#dbeafe; word-break:break-word; }}
    #path {{ color:var(--muted); font-size:12px; margin-top:3px; word-break:break-all; }}
    iframe {{ width:100%; height:100%; min-height:900px; border:0; background:#05070d; }}
    @media (max-width:1200px) {{
      .shell {{ grid-template-columns:1fr; }}
      aside {{ position:static; height:auto; border-right:0; border-bottom:1px solid var(--edge); }}
      main {{ min-height:920px; grid-template-rows:auto minmax(820px,1fr); }}
      iframe {{ min-height:820px; }}
    }}
    @media (max-width:720px) {{
      aside {{ padding:12px; }}
      .topbar {{ align-items:flex-start; flex-direction:column; gap:10px; padding:12px; }}
      .artifact-grid {{ grid-template-columns:1fr; max-height:440px; }}
      dl {{ grid-template-columns:1fr; }}
      main {{ min-height:820px; grid-template-rows:auto minmax(720px,1fr); }}
      iframe {{ min-height:720px; }}
    }}
  </style>
</head>
<body data-samples="{payload}">
  <div class="shell">
    <aside>
      <h1>{html.escape(title)}</h1>
      <p class="summary">Sample-first audit. Each card is one model-evaluated row/input with its own generated graph-of-thought reasoning trajectory and per-sample topology, memory, GraphCG, and tropical-support artifacts.</p>
      <a class="catalog-link" href="{html.escape(catalog_href, quote=True)}"><span>open complete artifact catalog</span><code>{len(artifacts)} artifacts</code></a>
      {''.join(cards)}
    </aside>
    <main>
      <div class="topbar">
        <div>
          <h2 id="title">{html.escape(samples[0]['label'] + ' / ' + first_artifact['label'])}</h2>
          <div id="path">{html.escape(first_artifact['src'])}</div>
        </div>
        <a id="open" class="open" href="{html.escape(first_artifact['src'], quote=True)}">open full page</a>
      </div>
      <iframe id="frame" src="{html.escape(first_artifact['src'], quote=True)}" title="TropicalGT-I sample audit panel"></iframe>
    </main>
  </div>
  <script>
    const buttons = Array.from(document.querySelectorAll("button.artifact"));
    const samples = Array.from(document.querySelectorAll(".sample"));
    const frame = document.getElementById("frame");
    const title = document.getElementById("title");
    const path = document.getElementById("path");
    const open = document.getElementById("open");
    function setPanel(button) {{
      const src = button.dataset.src;
      const sample = button.dataset.sample;
      buttons.forEach((candidate) => candidate.classList.toggle("active", candidate === button));
      samples.forEach((candidate) => candidate.classList.toggle("active", candidate.dataset.sample === sample));
      frame.src = src;
      title.textContent = button.dataset.label || button.textContent.trim();
      path.textContent = src;
      open.href = src;
    }}
    buttons.forEach((button) => button.addEventListener("click", () => setPanel(button)));
    if (buttons.length) setPanel(buttons[0]);
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_root", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="Primary sample-first browser index path. Defaults to browser_index.html.")
    parser.add_argument("--codex-output", type=Path, default=None, help="Optional secondary alias path. Defaults to codex_browser_index.html.")
    parser.add_argument("--title", default="TropicalGT-I Sample-Based Interactive Audit")
    args = parser.parse_args()
    output = args.output if args.output is not None else args.audit_root / "browser_index.html"
    codex_output = args.codex_output if args.codex_output is not None else args.audit_root / "codex_browser_index.html"
    build_index(args.audit_root, output, args.title)
    written = [output]
    if codex_output.resolve() != output.resolve():
        build_index(args.audit_root, codex_output, args.title)
        written.append(codex_output)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
