# TropicalGT-I Full-Dataset Training, Visualization, and Telemetry Repair Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch a fresh step-0 full-dataset TropicalGT-I run optimized for OpenAI Parameter-Golf BPB while repairing browser visualization truthfulness, interactive artifact indexing, W&B telemetry hygiene, inference artifact opt-in behavior, docs, tests, and QA.

**Architecture:** Training telemetry must be scalar-first and W&B-clean by default. Browser artifacts are local, optional review products with a single catalog page and provenance-backed payloads. All visualization layers must be built from the same real model data: graph-state embeddings, graph-of-thought trajectory records, model NLL/score values, and model probability vectors.

**Tech Stack:** Python, PyTorch, Plotly, GUDHI-backed filtered complexes, local HTML browser artifacts, pytest, W&B scalar metrics, hybrid HuggingFace + OpenAI Parameter-Golf datasets.

---

## Objective List

1. Start a new fresh step-0 full-dataset training run using both configured hybrid sources, with a strict >10B token-slot gate and adjusted hyperparameters focused on OpenAI Parameter-Golf BPB without dropping the HuggingFace reasoning corpus.
2. Keep the fresh run alive throughout implementation, monitor process/checkpoint/W&B scalar health, and never conflate the old resumed run with the new step-0 run.
3. Clean W&B so default training logs scalar losses, BPB, graph BPB, objective terms, regularizers, optimizer/system telemetry, validation metrics, and compact topology summary scalars only.
4. Make periodic interactive training artifacts optional and disabled by default; if enabled, write local manifests and HTML bundles, but do not upload HTML/media to W&B unless a separate explicit flag is enabled.
5. Make final training visualizations optional rather than unconditional.
6. Make inference browser artifacts explicit opt-in behavior; plain inference must be able to write JSON/text without heavyweight HTML.
7. Add one top-level interactive artifact catalog linked from the sample browser index; it must list all sample-level and nested per-step HTML/JSON artifacts.
8. Improve full trajectory complex views so the same embedded point cloud shows both true GoT parent-child path edges and induced radius/simplex topology with separate encodings, legends, hover text, and default visibility; QA bundles must include deeper and wider reasoning trajectory graphs than the current smoke depth/width.
9. Improve NLL/energy landscape semantics so it looks like a local model-energy field around embedding neighborhoods, not just a sparse triangular hull, while preserving exact surface contact for every trajectory point.
10. Keep probability/analogical maps based on model probability vectors and Jensen-Shannon distance, with certificate summaries and no identity-copy implication.
11. Redesign analogical reasoning and associated simplicial-map pages so they are readable, non-copy-looking, certificate-first, and visually honest about query/codomain geometry, assignment costs, preservation failures, and Jensen-Shannon matching.
12. Repair persistence landscape views so they come from actual GUDHI/persim landscape functions or explicit unavailable diagnostics, not norm-only or synthetic-looking summaries.
13. Repair tropical-support views so they do not render as giant block heatmaps with unreadable labels; show support collapse, active-support distribution, margin profile, and token provenance in a compact, inspectable layout.
14. Repair embedding-map trajectory views so they communicate the real sampled GoT path over model graph-state PCA, with branch/depth identity, parent-child transitions, and diagnostics for degenerate or collapsed embeddings.
15. Repair Euclidean and Jensen-Shannon simplex-tree views so they summarize actual SimplexTree inclusion/provenance in a readable hierarchy instead of a sprayed Hasse point cloud.
16. Add tests before production changes for telemetry gating, optional artifacts, catalog discovery, trajectory overlay metadata, NLL contact, analogical-map visual contracts, persistence-landscape provenance, tropical-support layout, simplex-tree readability, embedding-map trajectory metadata, and fake/fallback rejection.
17. Regenerate browser review outputs, inspect them in the in-app browser, capture screenshots, run extensive tests, update `README.md` and `TropicalGT-I/README.md`, commit, and push on the non-main branch.
18. Add explicit long-context full-trajectory audit budgets: per-step pages remain complete local reasoning-step objects, while full-trajectory audits use tropical ring-attention context budgets and all actual model output/model-derived GoT states, graph-token embeddings, internal vectors, probability vectors, and support/retrieval vectors that the run produced.
19. Generate dedicated full-audit browser bundles with deeper/wider GoT graphs, separate from smoke bundles. Smoke bundles may be small for quick regression checks, but full repair acceptance must use deeper/wider trajectories and complete per-step reasoning objects.

---

## Stage 0: Fresh Training Run Contract

**Files:**
- Create: `TropicalGT-I/configs/train_full_dataset_pg_bpb_step0.json`
- Read/verify: `TropicalGT-I/configs/train_full_dataset_active.json`
- Read/verify: `TropicalGT-I/outputs/train_full_dataset_active/train_report.json`
- Modify docs later: `README.md`, `TropicalGT-I/README.md`

- [ ] **Step 0.1: Snapshot previous scalar evidence**

Run:
```bash
python - <<'PY'
import json
from pathlib import Path
for path in [
    Path("TropicalGT-I/outputs/train_full_dataset_active/train_report.json"),
    Path("TropicalGT-I/outputs/train_full_dataset_active/periodic/manifest.jsonl"),
]:
    print(path, "exists=", path.exists())
    if path.exists() and path.suffix == ".json":
        data=json.loads(path.read_text())
        print(json.dumps(data.get("metrics", {}), indent=2)[:4000])
PY
```
Expected: previous metrics are visible if report exists; missing report is acceptable but must be recorded.

- [x] **Step 0.2: Create a fresh step-0 config**

Create `TropicalGT-I/configs/train_full_dataset_pg_bpb_step0.json` by copying the full-dataset active config and changing only the new run name and telemetry/artifact controls:
```json
{
  "run_name": "tropicalgt_i_pg_bpb_step0",
  "resume_from": "",
  "validation_every_steps": 500,
  "visualization_every_steps": 0,
  "periodic_interactive_artifacts_enabled": false,
  "final_interactive_artifacts_enabled": false,
  "wandb_log_interactive_artifacts": false,
  "wandb": {
    "log_interactive_artifacts": false,
    "html_artifact_limit": 0
  },
  "parameter_golf_bpb_focus": true,
  "graph_bpb_side_weight": 1.0,
  "model": {
    "gflownet_weight": 0.01,
    "graphcg_weight": 0.01,
    "certificate_weight": 0.0005,
    "sequence_tropical_weight": 0.0625
  }
}
```
Keep both hybrid sources, `min_available_train_token_slots`, `min_training_token_slots`, and `required_hybrid_sources` from the active full-dataset config.

- [x] **Step 0.3: Run the readiness/data-budget gate**

Run:
```bash
PYTHONPATH=TropicalGT-I/src python TropicalGT-I/scripts/audit_tropicalgt_i_readiness.py \
  --config TropicalGT-I/configs/train_full_dataset_pg_bpb_step0.json \
  --train-dry-run --require-cuda --check-wandb-key
```
Expected: PASS, both hybrid sources present, CUDA dry-run finite, >10B configured token slots.

- [x] **Step 0.4: Launch fresh step-0 training**

Run under a durable shell/session with no `--resume-from`:
```bash
PYTHONPATH=TropicalGT-I/src python TropicalGT-I/scripts/train_tropicalgt_i.py \
  --config TropicalGT-I/configs/train_full_dataset_pg_bpb_step0.json
```
Observed: W&B run name `tropicalgt_i_pg_bpb_step0`, run id `xcuqkmjf`, process PID `1070996`, first log starts from `0/2500000` and advances from step 1.

- [x] **Step 0.5: Correct the first launch to the actual full-dataset token budget and target VRAM**

The initial step-0 launch proved too conservative: it used a 10B token-slot
floor and a small batch, which did not consume the intended dataset budget or
the requested GPU memory. That run was stopped and replaced with the corrected
fresh run below.

Current active run:
```text
config: TropicalGT-I/configs/train_full_dataset_pg_bpb_step0_full24b_b44.json
run_name: tropicalgt_i_pg_bpb_step0_full24b_b44
wandb id: y4do5bu0
pid: 1086309
batch_size: 44
sequence length: 1024
max_steps: 537083
configured token slots: 24,198,811,648
audited available token slots: 24,198,796,288
dataset roots:
  - TropicalGT-I/data/toricgt/curated_hf_shards
  - external/oai-parameter-golf/data/datasets/fineweb10B_sp1024
GPU observation: RTX 4090 at 18,105 MiB used, training process at 18,082 MiB, 100% utilization
```

Readiness audit status: `ready`, `failed_gates=[]`. The corrected config uses
the full audited token-slot budget rather than a 10B floor.

- [ ] **Step 0.6: Keep liveness checks running through implementation**

At every major stage run:
```bash
pgrep -af "train_full_dataset_pg_bpb_step0_full24b_b44|tropicalgt_i_pg_bpb_step0_full24b_b44|train_tropicalgt_i.py"
ls -lt TropicalGT-I/checkpoints/tropicalgt_i_pg_bpb_step0_full24b_b44.latest.pt
```
Expected: process alive and checkpoint/log timestamps advance.

---

## Stage 1: W&B and Training Artifact Gating

**Files:**
- Modify: `TropicalGT-I/src/tropicalgt/run.py`
- Test: `TropicalGT-I/tests/test_training_metrics.py`

- [x] **Step 1.1: Write failing tests for scalar-only W&B defaults**

Add tests using a fake W&B object:
```python
class FakeWandbRun:
    def __init__(self):
        self.logged = []
    def log(self, payload, step=None):
        self.logged.append((payload, step))

def test_wandb_html_artifacts_disabled_by_default(tmp_path):
    from tropicalgt.run import _log_wandb_html_artifacts
    html_path = tmp_path / "plot.html"
    html_path.write_text("<html>plot</html>", encoding="utf-8")
    wb = FakeWandbRun()
    _log_wandb_html_artifacts(wb, {"plot": str(html_path)}, {}, prefix="periodic", step=1)
    assert wb.logged == []

def test_wandb_html_artifacts_require_explicit_opt_in(tmp_path):
    from tropicalgt.run import _log_wandb_html_artifacts
    html_path = tmp_path / "plot.html"
    html_path.write_text("<html>plot</html>", encoding="utf-8")
    wb = FakeWandbRun()
    _log_wandb_html_artifacts(
        wb,
        {"plot": str(html_path)},
        {"wandb": {"log_interactive_artifacts": True, "html_artifact_limit": 1}},
        prefix="periodic",
        step=1,
    )
    assert wb.logged and "periodic/plot" in wb.logged[0][0]
```
Run: `PYTHONPATH=TropicalGT-I/src pytest -q TropicalGT-I/tests/test_training_metrics.py::test_wandb_html_artifacts_disabled_by_default`
Expected: FAIL before implementation.

- [x] **Step 1.2: Implement explicit W&B media gate**

In `_log_wandb_html_artifacts`, return immediately unless W&B interactive
artifact upload is explicitly enabled. Preferred config is nested
`wandb.log_interactive_artifacts: true` plus positive
`wandb.html_artifact_limit`; old top-level aliases remain backward-compatible.
Keep scalar `wb.log(organize_wandb_metrics(...))` unchanged.

Follow-up hardening implemented after review:
- `wandb_html_artifact_limit` is `0` by default in full-dataset configs.
- `test_wandb_html_artifacts_need_positive_limit` verifies opt-in with limit
  `0` still uploads nothing.
- `test_periodic_got_visualization_requires_complete_steps_by_default`
  verifies periodic training GoT artifact generation passes
  `require_complete_reasoning_steps=True` by default.
- `audit_tropicalgt_i_readiness.py --render-visualizations` now also requires
  complete model-derived reasoning steps unless
  `--allow-incomplete-reasoning-visualizations` is explicitly passed.

- [x] **Step 1.3: Gate periodic visualization generation**

In `train()`, compute:
```python
periodic_interactive_artifacts_enabled = bool(cfg.get("periodic_interactive_artifacts_enabled", False))
visualization_every = int(cfg.get("visualization_every_steps", 0) or 0) if periodic_interactive_artifacts_enabled else 0
```
Validation must continue to run from `validation_every_steps`; only local interactive artifacts are disabled by default.

- [x] **Step 1.4: Gate final visualization generation**

Wrap final calls to `write_reasoning_visualizations`, `write_metric_visualizations`, `write_graphcg_training_visualizations`, and final `viz_got_scaling` audit behind `final_interactive_artifacts_enabled = bool(cfg.get("final_interactive_artifacts_enabled", False))`. Always write `train_report.json`.

---

## Stage 2: Inference Optional Artifact Contract

**Files:**
- Modify: `TropicalGT-I/scripts/infer_tropicalgt_i.py`
- Test: create `TropicalGT-I/tests/test_inference_artifact_options.py`

- [x] **Step 2.1: Write failing CLI/config tests**

Test argument parsing through a small helper if needed, or add a pure helper:
```python
def test_audit_output_dir_alone_does_not_enable_html():
    from infer_tropicalgt_i import _resolve_render_html
    assert _resolve_render_html(audit_all=False, audit_output_dir="out", audit_render_html=False, no_audit_render_html=False, interactive_artifacts=False) is False

def test_interactive_artifacts_flag_enables_html():
    from infer_tropicalgt_i import _resolve_render_html
    assert _resolve_render_html(audit_all=False, audit_output_dir="out", audit_render_html=False, no_audit_render_html=False, interactive_artifacts=True) is True
```
Expected: FAIL until helper/flag exists.

- [x] **Step 2.2: Add `--interactive-artifacts`**

Add parser flag `--interactive-artifacts`. Set `render_html` true only when `--audit-all`, `--audit-render-html`, or `--interactive-artifacts` is true and `--no-audit-render-html` is false. `--audit-output-dir` alone writes JSON artifacts but not HTML.

- [x] **Step 2.3: Keep multi-run browser tool explicit**

In `run_multi_inference_audits.py`, continue passing `--audit-all`; optionally add `--interactive-artifacts` for clarity after `infer_tropicalgt_i.py` supports it.

Implemented: the multi-sample browser/audit driver now passes
`--interactive-artifacts` explicitly. Test:
`TropicalGT-I/tests/test_inference_artifact_options.py::test_multi_inference_audit_driver_requests_interactive_artifacts`.

---

## Stage 3: Single Interactive Catalog and Step Rollups

**Files:**
- Modify: `TropicalGT-I/scripts/build_sample_browser_index.py`
- Test: create/extend `TropicalGT-I/tests/test_sample_browser_index.py`
- Validator: `TropicalGT-I/scripts/validate_interactive_audit_artifacts.py`

- [x] **Step 3.1: Write failing test for catalog file and link**

Create a temporary bundle with `sample_000/got_trajectory_pca_3d.html`, `sample_000/reasoning_step_complex_maps/reasoning_step_000.html`, and `sample_000/got_full_trajectory_complex_payload.json`. Assert `build_index()` writes `interactive_visualization_catalog.html` and that `browser_index.html` links to it.

- [x] **Step 3.2: Implement artifact discovery**

Add a function:
```python
def discover_interactive_artifacts(root: Path, samples: list[dict[str, Any]]) -> list[dict[str, str]]:
    ...
```
It must include `*.html`, key `*.json` payloads, nested `reasoning_step_complex_maps/*.html`, and `analogical_memory_map_*.html`, all as relative links.

- [x] **Step 3.3: Render catalog**

Catalog columns: sample, family, label, relative path, extension, tag, existence, source. Include a search input and family filter. Main index gets one prominent `open complete artifact catalog` link.

- [x] **Step 3.4: Collapse per-step links into two per-sample rollups**

The sample card must not list dozens of individual `Reasoning step NNN complex`
and `Reasoning step NNN simplex tree` buttons. Add one `All reasoning step
complexes` link and one `All reasoning step simplex trees` link per sample.
Each rollup page should list every step artifact and provide a single iframe
inspection surface, with exact per-step pages retained as drill-down links.

---

## Stage 4: Trajectory/Radius Visualization Semantics

**Files:**
- Modify: `TropicalGT-I/src/tropicalgt/visualization.py`
- Test: `TropicalGT-I/tests/test_simplicial_visualization.py`

- [x] **Step 4.1: Write failing test for trajectory overlay metadata**

Extend `test_got_trajectory_visualization_renders_simplicial_panel_and_nll_surface` to load `got_full_trajectory_complex_payload.json` and assert:
```python
payload["filtered_simplicial_object"]["trajectory_overlay"]["source"] == "graph_of_thought_parent_edges"
assert len(payload["filtered_simplicial_object"]["trajectory_overlay"]["edges"]) == 3
assert "GoT parent-child trajectory edges" in full_complex_html
assert "radius/simplicial edges induced from the same embeddings" in full_complex_html
```

- [x] **Step 4.2: Add trajectory overlay to canonical complex**

When writing the full trajectory complex, derive overlay edges from `candidates[*].parent` and attach them to the same object without changing the radius filtration:
```python
obj["trajectory_overlay"] = {
  "source": "graph_of_thought_parent_edges",
  "semantic_note": "Radius topology is induced from these trajectory embeddings; overlay highlights the sampled GoT path.",
  "edges": [...]
}
```

- [x] **Step 4.3: Render overlay as a distinct Plotly trace**

In `_complex_slider_traces` or `_write_complex_slider_map`, add a thick gold/cyan trace named `GoT parent-child trajectory edges`; radius 1-simplices remain thinner and may default to lower opacity.

Implemented and browser-verified. Current focused test:
```bash
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q \
  TropicalGT-I/tests/test_simplicial_visualization.py::test_got_trajectory_visualization_renders_simplicial_panel_and_nll_surface
```
Result: `1 passed`. Browser QA screenshots:
`/tmp/tropicalgt_full_complex_overlay_after_second_legend_fix.png`,
`/tmp/tropicalgt_browser_index_catalog_repair.png`,
`/tmp/tropicalgt_interactive_catalog_repair.png`,
`/tmp/tropicalgt_step_complex_rollup_repair.png`.

---

## Stage 5: NLL Local Landscape Repair

**Files:**
- Modify: `TropicalGT-I/src/tropicalgt/visualization.py`
- Test: `TropicalGT-I/tests/test_simplicial_visualization.py`

- [ ] **Step 5.1: Write failing test for neighborhood-surface metadata**

Assert `payload["nll_surface"]["local_embedding_neighborhood_surface"]["available"] is True` when enough candidates exist, and that `trajectory_point_surface_residual_max == 0.0`.

- [ ] **Step 5.2: Use real candidate anchors only**

Build local surface anchors from all generated candidate embeddings/NLL values in the scaling report. Do not invent NLL values. If insufficient anchors exist, emit an explicit diagnostic rather than a fake smooth surface.

- [ ] **Step 5.3: Keep exact contact**

For every GoT node, keep `plot.z == plot.z_surface == surface_projected_z_by_record_id[record_id]`. Validator must reject any mismatch.

---

## Stage 6: Analogical Reasoning and Simplicial-Map Visual Repair

**Files:**
- Modify: `TropicalGT-I/src/tropicalgt/visualization.py`
- Test: `TropicalGT-I/tests/test_simplicial_visualization.py`
- Validator: `TropicalGT-I/scripts/validate_interactive_audit_artifacts.py`

- [x] **Step 6.1: Write failing tests for analogical page contracts**

Extend analogical visualization tests to assert the HTML contains separate query/codomain panels, finite Jensen-Shannon assignment summaries, certificate pass/fail text, preserved and failed edge counts, and no language implying an exact simplicial map when the certificate fails:
```python
assert "query trajectory probability complex" in html
assert "retrieved memory probability complex" in html
assert "Jensen-Shannon assignment cost" in html
assert "certificate status" in html
assert "not an exact simplicial map" in html or report["simplicial_map_certificate"]["valid"] is True
```

- [ ] **Step 6.2: Redesign analogical map layout**

Use side-by-side or domain/codomain small multiples with a correspondence table. Query vertices, memory vertices, assignment edges, preserved simplex evidence, failed simplex evidence, and filtration distortion should be visually separate. The default page should not resemble duplicated trajectories; it should read as probability-vector matching between two independently generated trajectory complexes.

- [ ] **Step 6.3: Fail closed on bad analogical evidence**

If either side lacks model probability vectors, finite Jensen-Shannon distances, or GUDHI simplex-tree provenance, render an unavailable diagnostic and do not draw a pseudo-map.

---

## Stage 7: Persistence Landscape Provenance and Visual Repair

**Files:**
- Modify: `TropicalGT-I/src/tropicalgt/visualization.py`
- Test: `TropicalGT-I/tests/test_simplicial_visualization.py`
- Validator: `TropicalGT-I/scripts/validate_interactive_audit_artifacts.py`

- [x] **Step 7.1: Write failing tests for real landscape functions**

Extend persistence tests to assert landscape payload/HTML reports actual `lambda_k(t)` samples, backend provenance, interval counts, homology dimension, and unavailable diagnostics when no real landscape can be computed:
```python
assert "Actual GUDHI persistence landscape functions" in html
assert "lambda_1(t)" in html
assert "landscape_backend" in html
assert "not norm-only summaries" in html
```

- [ ] **Step 7.2: Repair rendering**

Render landscape curves as functions over filtration, separated by homology dimension and trajectory growth level. Do not show generic filled bands or canned-looking lines without interval provenance. If the backend cannot compute curves, render a dark unavailable page with the exact missing backend/data reason.

- [ ] **Step 7.3: Add validator checks**

Require persistence landscape pages in trajectory bundles to contain backend provenance, lambda labels, interval/growth counts, and no synthetic/fallback scientific claim text.

---

## Stage 8: Tropical Support, Embedding Map, and SimplexTree Visual Repair

**Files:**
- Modify: `TropicalGT-I/src/tropicalgt/visualization.py`
- Test: `TropicalGT-I/tests/test_simplicial_visualization.py`
- Validator: `TropicalGT-I/scripts/validate_interactive_audit_artifacts.py`

- [x] **Step 8.1: Write failing tests for tropical-support readability**

Extend tropical-support tests to require compact labels, bounded colorbars/margins, an explicit support-collapse diagnostic, and no giant unreadable token-label heatmap:
```python
assert "Tropical active-support audit" in html
assert "support collapse" in html
assert "active-support distribution" in html
assert "token margin profile" in html
assert "raw token labels truncated" in html
```

- [x] **Step 8.2: Redesign tropical support**

Replace the default large matrix with a readable multi-panel view: active-support histogram, margin-by-token line with abbreviated labels, support-flow summary, and a payload table for exact token text. If only one or two supports dominate, label it as model support collapse rather than making it look like a meaningful dense heatmap.

Implemented high-collapse routing to a compact diagnostic when a top support
dominates, with `layout_mode=collapse_diagnostic`, compact labels, exact token
text in hover/payload, and support-collapse language in the title. Tests:
`test_tropical_support_high_collapse_uses_compact_diagnostic`,
`test_tropical_support_heatmap_layout_keeps_legend_out_of_margin`,
`test_tropical_support_heatmap_does_not_fabricate_invalid_supports`. Browser
QA screenshot: `/tmp/tropicalgt_tropical_support_collapse_titles_fixed.png`.

- [ ] **Step 8.3: Write failing tests for embedding-map trajectory identity**

Assert the embedding map HTML/payload includes branch/depth metadata and actual parent-child transitions:
```python
assert "GoT parent-child trajectory" in embedding_map_html
assert "branch/depth" in embedding_map_html
assert payload["edges"][0]["source"]
assert payload["nodes"][0]["embedding_source"] == "model graph_state"
```

- [ ] **Step 8.4: Redesign embedding map**

Keep PCA from model graph-state embeddings, but add branch/depth styling, explicit parent-child arrows, chosen/best terminal path emphasis, and collapse diagnostics. If PCA stress or duplicate-coordinate ratio is poor, show that warning prominently and avoid overclaiming geometric separation.

- [x] **Step 8.5: Write failing tests for SimplexTree correctness and readability**

Assert all Euclidean and Jensen-Shannon simplex-tree pages, including
per-reasoning-step simplex trees, include dimension counts, backend provenance,
filtration-source certificates, compact hierarchy/table summaries, and default
hiding or aggregation for excessive face-to-coface links:
```python
assert "SimplexTree provenance summary" in html
assert "dimension counts" in html
assert "face-to-coface links hidden by default" in html
assert "open dense inclusion graph" in html
assert "GUDHI filtration certificate" in html
```

- [ ] **Step 8.6: Redesign simplex-tree pages**

Default to a readable summary: dimension-count bars, filtration histogram, sample of representative inclusions, and provenance/certificate table. Keep the dense 3D Hasse diagram available behind a button or legend-only trace, not as the default visual.

- [x] **Step 8.7: Replace static filtered-complex previews**

Persistence and dashboard side panels currently expose static SVG previews from
filtered-complex payloads. Replace default previews with embedded interactive
Plotly 3D PCA filtered simplicial objects using the same payload, with a
left-to-right radius/filtration slider from minimum to maximum. Static SVG may
remain only as an explicit unavailable fallback when valid vertices/edges are
absent.

Implemented in the shared Plotly dark HTML panel: the interactive selected
filtered-complex panel remains primary, while the static SVG is collapsed under
`Static SVG fallback preview from the same filtered-complex payload`. Focused
tests passed, and browser QA screenshot:
`/tmp/tropicalgt_full_complex_static_fallback_collapsed.png`.

- [x] **Step 8.8: Add per-step directionality**

Each reasoning-step complex page must represent a collection of model-derived
embedding vectors or probability vectors for that single GoT state. It must
show directed Graph-of-Thought relationships for that step over the same point
cloud, not anonymous unconnected radius points. Probability-vector versions
must label Jensen-Shannon filtration from model probabilities.

Implemented:
- `_attach_graph_token_direction_overlay(...)` reads each step's
  `graph_token_trace.tokens` and the model-derived 0-simplices'
  `token_index`, `token_kind`, `node_id`, `source`, and `target` metadata.
- Each graph edge token is rendered as `source-node -> edge-token ->
  target-node` on the same radius-filtered vector complex, with hover metadata
  for edge type, source/target node ids, active support, and model margin.
- Per-step manifest rows now include `graph_token_direction_overlay` so the
  catalog can audit whether a rendered step has directed graph-token evidence.
- Tests added:
  `test_graph_token_direction_overlay_uses_model_trace_edges` and the affected
  GoT trajectory visualization test. Broader focused suite passed: `20 passed`.
- Browser QA:
  `/tmp/tropicalgt_directed_overlay_step000.png` from a strict-smoke JSON
  rebuild, plus `/tmp/tropicalgt_fresh_directed_step004.png` from a fresh
  CPU-only checkpoint-backed inference run with 8 model-derived reasoning-step
  pages. The fresh run reports directed overlay counts
  `[4, 16, 16, 16, 28, 28, 28, 28]`.

- [x] **Step 8.9: Repair GraphCG readability**

GraphCG direction heatmaps must use readable abbreviated tick labels, hover
metadata for exact direction/state labels, bounded label counts, and a payload
table. If there are thousands of directions, the default heatmap remains a
rank/activity summary while full labels live in hover/table payloads.

Implemented visible direction tick limit metadata and bounded heatmap ticks
with exact direction ids retained in hover and payload. Test:
`TropicalGT-I/tests/test_simplicial_visualization.py::test_graphcg_visualization_preserves_projection_basis_certificate`.
Browser QA screenshot: `/tmp/tropicalgt_graphcg_bounded_ticks.png`.

Follow-up repair: the GraphCG heatmap now renders all model-derived directions
instead of a top-32 view; only visible tick labels are bounded. Browser QA
screenshot: `/tmp/tropicalgt_graphcg_all_observed_states.png`.

- [x] **Step 8.10: Repair analogical memory maps**

Analogical maps must show the actual query trajectory complex, retrieved memory
trajectory complex, model-probability Jensen-Shannon assignment, and
simplicial-map certificate. If the correspondence is not a simplicial map on
the displayed skeleton, the visual must lead with that as a failed/partial
certificate and distinguish preserved 1-simplices, vertex-only assignments,
failed edges, and failed 2-simplices instead of implying a clean analogy.

Implemented first repair pass:
- Reworded failed/no-memory cases and hover/trace names from overclaimed
  "simplicial map candidate" to "probability correspondence candidate" unless
  the finite filtered-complex certificate actually passes.
- Legend now distinguishes preserved `1`-simplex correspondences from
  vertex-only correspondences.
- Focused analogical tests passed:
  `test_analogical_memory_visualization_renders_simplicial_maps`,
  `test_analogical_memory_visualization_rejects_non_trajectory_probability_fallback`,
  and `test_analogical_memory_without_query_probabilities_is_unavailable_not_fallback`.

- [x] **Step 8.11: Encode long-context full-trajectory budgets**

Each reasoning step must be a complete reasoning step object, not a placeholder
vertex set: include the actual graph-state embedding, graph-token embeddings,
support margins/probabilities, action/proposal probabilities, logits-derived
probability vectors, retrieved-memory vectors when present, decoded output,
input/target context, parent-child GoT transition metadata, and any local
filtered simplicial object built from those model-derived vectors. Minimum
meaningful local geometry is `12-16` real vectors, with `32-64` preferred when
the model exposes enough per-step vectors. The full reasoning trajectory should
use the long context afforded by tropical ring attention and must include all
actual model outputs and model-derived vectors/probabilities produced by the
inference run: GoT states, graph-token embeddings, action/proposal vectors,
support vectors, retrieval vectors, graph-state embeddings, logits/probability
vectors, and Jensen-Shannon probability geometry. No scientific point cloud,
topology, GraphCG direction set, probability complex, or full-trajectory metric
may be randomly sampled, downsampled, top-k substituted, or synthesized for
browser convenience. If a browser view becomes crowded, use pagination, facet
pages, bounded tick labels, hover metadata, full payload tables,
progressive/virtual rendering, or hidden-by-default dense traces while
preserving every actual model-derived point and edge in the artifact payload
and/or exact drill-down page. Current smoke bundles with roughly `12` GoT
states and `7-37` graph tokens per state must be labeled as smoke diagnostics
rather than full long-context audits.

- [x] **Step 8.12: Add full-audit deep/wide inference preset**

Create a full browser-audit preset distinct from quick smoke QA. The smoke
bundle can stay small for regression speed, but the full acceptance bundle
should use a deeper and wider trajectory graph, for example depth `5-6`, width
`8-12`, branch factor `4-6`, deterministic branch expansion for acceptance
with stochastic action selection enabled for full browser QA unless a separate experiment explicitly asks for deterministic action selection,
and a long-context prompt budget backed by tropical ring attention. The preset must fail closed if
per-step payloads omit complete reasoning-step model data; it should not render
per-step complexes as if they are complete when only sparse graph-token traces
are present.

Implemented:
- `run_multi_inference_audits.py --audit-preset full` enforces minimum
  depth `5`, width `8`, branch factor `4`, trace limit `8192`, topology budget
  `8192`, and memory retrieval budget `8`.
- Full browser-audit commands pass `--require-complete-reasoning-steps`, and
  direct `infer_tropicalgt_i.py --audit-all` HTML rendering now requires the
  same complete-step contract by default.
- Full acceptance probes can pass `--no-scale-stochastic-actions` so every
  branch is selected deterministically from model action probabilities.
- `_public_candidate` now emits `reasoning_step_complete`,
  `reasoning_step_completeness`, and `reasoning_step_model_data`; a step cannot
  certify complete unless it has model NLL output, graph-state embedding,
  valid action probability vector, untruncated graph-token trace,
  Jensen-Shannon probability complex from model support probabilities,
  Euclidean embedding complex from model graph-token embeddings, GraphCG
  all-direction cosines, and directed graph context.
- Hover text now reports the complete/incomplete certificate when present and
  calls clipped hover snippets previews rather than samples.

Validation so far:
- Targeted tests passed: `TropicalGT-I/tests/test_inference_scaling_completeness.py`,
  `TropicalGT-I/tests/test_multi_inference_runner.py`,
  `TropicalGT-I/tests/test_inference_artifact_options.py`, and the affected
  GoT/GraphCG visualization tests (`11 passed`).
- Follow-up strict-gate tests passed:
  `TropicalGT-I/tests/test_inference_scaling_completeness.py`,
  `TropicalGT-I/tests/test_multi_inference_runner.py`, and
  `TropicalGT-I/tests/test_inference_artifact_options.py` (`14 passed`).
- Browser QA refreshed `/tmp/tropicalgt_got_nll_model_evaluated.png` and
  `/tmp/tropicalgt_graphcg_all_observed_states.png`.
- The obsolete pre-strict-gate one-sample full-audit deep/wide probe at
  `TropicalGT-I/outputs/multi_sample_browser/full_audit_deepwide_20260612_143525`
  was stopped because it had no emitted artifacts and did not enforce the
  complete-step gate.
- A strict complete-step smoke bundle completed at
  `TropicalGT-I/outputs/multi_sample_browser/strict_gate_smoke_20260612_151559`
  with `require_complete=true`, `3/3` complete candidates, zero incomplete
  records, model-evaluated NLL surface contact residual `0.0`, Euclidean
  trajectory complex, and Jensen-Shannon probability trajectory complex.
  Browser QA screenshots:
  `/tmp/tropicalgt_strict_gate_browser_index_wording_fixed.png`,
  `/tmp/tropicalgt_strict_gate_nll.png`,
  `/tmp/tropicalgt_strict_gate_full_complex.png`,
  `/tmp/tropicalgt_strict_gate_step_rollup.png`,
  `/tmp/tropicalgt_strict_gate_step_001.png`, and
  `/tmp/tropicalgt_strict_gate_graphcg.png`.
- A fresh CPU-only patched browser QA run completed at
  `TropicalGT-I/outputs/multi_sample_browser/directed_step_overlay_smoke_20260612_1536`
  without consuming GPU VRAM. It generated 8 reasoning-step pages with
  model-derived Jensen-Shannon complexes and directed graph-token overlays.
  Local browser screenshots:
  `/tmp/tropicalgt_fresh_directed_index.png` and
  `/tmp/tropicalgt_fresh_directed_step004.png`.

---

## Stage 9: Docs, Browser QA, Verification, Push

**Files:**
- Modify: `README.md`
- Modify: `TropicalGT-I/README.md`
- Possibly modify: `TropicalGT-I/scripts/validate_interactive_audit_artifacts.py`

- [ ] **Step 9.1: Update docs**

Document fresh step-0 Parameter-Golf BPB run config, scalar-only W&B default, opt-in local interactive artifacts, opt-in W&B media upload, inference artifact flags, single catalog, browser QA, and provenance rules.

- [ ] **Step 9.2: Regenerate browser bundle**

Run multi-sample audit using a checkpoint snapshot and open `browser_index.html`. Verify the catalog link, NLL page, embedding map, full complex page, Euclidean simplex tree, probability complex, JS simplex tree, analogical maps, persistence landscapes, GraphCG, and tropical support.

- [ ] **Step 9.3: Run extensive tests**

Run:
```bash
PYTHONPATH=TropicalGT-I/src pytest -q TropicalGT-I/tests
PYTHONPATH=TropicalGT-I/src python TropicalGT-I/scripts/audit_metric_provenance.py --fail-on-uncovered
PYTHONPATH=TropicalGT-I/src python TropicalGT-I/scripts/validate_interactive_audit_artifacts.py --audit-root TropicalGT-I/outputs/multi_sample_browser/latest --min-rows 3 --min-candidates 8 --min-depth 2
git diff --check
```

- [ ] **Step 9.4: Push**

Stage only intended files, exclude unrelated files, commit, push `tropicalgt-i-implementation`.


## Repair Cycle Update: 2026-06-12 20:18:23

- Patched `TropicalGT-I/src/tropicalgt/visualization.py` so the GoT NLL page no longer labels sparse triangulations as an actual dense landscape. The emitted contract is now `surface_kind=sparse_observed_state_nll_anchor_mesh`, `actual_landscape_layer=false`, `sparse_observed_anchor_layer=true`, and `dense_model_evaluated_field=false` unless a genuine dense model-evaluated field exists.
- Capped NLL display scaling to avoid tiny raw NLL ranges becoming visually absurd vertical walls while preserving exact point/surface contact in payload coordinates.
- Patched analogical map title/certificate text so preserved 1-simplices and vertex-only correspondences are readable even when a trace is empty/legend-only.
- Patched `validate_interactive_audit_artifacts.py` to require the repaired observed-NLL-anchor contract and the current full-rank GraphCG layout marker.
- Patched `run_multi_inference_audits.py` so `--audit-preset full` preserves caller sampling temperature/exploration and defaults to stochastic actions instead of silently forcing deterministic sampling.
- Focused tests passed: GoT NLL visualization, analogical memory visualization, and full-audit preset contract.
- Launched refreshed CPU-side model-backed browser QA from the active latest checkpoint with full audit depth/width/branch, stochastic actions, GUDHI, memory top-k 8, and no GPU contention with live training. Output root is recorded in `/tmp/tropicalgt_takeover_out.txt`.

### 2026-06-13 08:12:59 UTC Focused Artifact Contract Update

- [x] Require `F2[x_level,x_radius]` bifiltration HTML and JSON in the interactive artifact validator.
- [x] Require exact level/radius grade metadata, sorted min-to-max radius grades, rank-invariant samples with 2D grades, and uncertified-chain diagnostics to stay explicitly non-resolution unless a CAS certificate is present.
- [x] Add model-backed 3D PCA NLL density-cloud page with explicit disclosure that Gaussian cloud points are not model states.
- [ ] Replace the current dense bifiltration/free-chain visual layout with a clearer Miller-Sturmfels staircase plus Macaulay2-style tables once a real CAS-backed resolution path is available.
- [x] Add tests for persistence-landscape-weighted analogical retrieval once memory-quality threshold fixtures are stable.

## Vectorized Persistence Retrieval Update

Status: implemented and under test. Analogical memory retrieval now compares the full cached GUDHI vector-representation family when available, not only persistence landscapes. The comparison contract is:

- Source data: real `persistence_representations.methods` payloads computed from finite persistence intervals by GUDHI.
- Methods: Landscape vector, BettiCurve values, Silhouette values, Entropy vector, PersistenceLengths, TopologicalVector, and PersistenceImage flattened grids.
- Scoring: each shared query-memory method is concatenated by homology dimension, zero-padded only to align existing vector lengths, and compared by L2 similarity, cosine, and correlation. The retrieval aggregate is a weighted mean of vector similarities across available methods.
- Guardrail: if a method is absent for either query or memory it is reported unavailable and contributes no score. There is no fabricated topology vector.
- Differentiability statement: the vector comparisons are differentiable with respect to the cached vectors, but the current GUDHI vectorizers are NumPy/scikit-learn transforms rather than torch-native differentiable persistent-homology layers.
- Browser impact: analogical retrieval tables now show the vector topology aggregate, available method count, and method names alongside the landscape-only diagnostics.

This keeps both overloaded “landscape” meanings separate: persistence landscapes `lambda_k(t)` are topological vector features; NLL/fitness landscapes or density clouds are model-evaluation visualizations over projected embeddings.

