# TropicalGT-I Takeover Repair Log - 2026-06-12

This file is the handoff/takeover ledger for the current remote-only repair cycle on `/home/iska/Documents/amelie/bio/TropicalGT`.

## Remote State

- Branch: `tropicalgt-i-implementation`.
- Latest visible remote commit: `da2c9ca Repair provenance-backed browser audit visualizations`.
- Working tree is intentionally dirty with visualization, training, docs, config, and test edits from the previous agent.
- The visible browser QA surface is served from the remote artifact server through a local SSH tunnel at `http://127.0.0.1:8988/browser_index.html`.
- Current remote sample browser root: `TropicalGT-I/outputs/multi_sample_browser/latest`.
- Current validation artifact status for that root: failed only on analogical map provenance/certificate fields, not on every artifact class.

## Active Training Run

The full-dataset step-0 training run is alive and must stay alive unless it is deliberately replaced by another active run.

- Config: `TropicalGT-I/configs/train_full_dataset_pg_bpb_step0_full24b_b44.json`.
- Run name: `tropicalgt_i_pg_bpb_step0_full24b_b44`.
- W&B run id: `y4do5bu0`.
- PID observed during takeover: `1086309`.
- Shape: `batch_size=44`, `seq_len=1024`, `max_steps=537083`.
- Configured token slots: `24,198,811,648`.
- Audited available token slots: `24,198,796,288`.
- Data roots: `TropicalGT-I/data/toricgt/curated_hf_shards` and `external/oai-parameter-golf/data/datasets/fineweb10B_sp1024`.
- GPU observed during takeover: RTX 4090, about `21.6/24.6 GiB` used and active training process at high CPU/GPU utilization.
- Latest parsed progress at takeover: step `9040/537083`, training log `loss ~= 0.974`, `nll ~= 0.965`.
- Latest validation report checked: `periodic/step_00009000/validation_report.json`.
- Step 9000 validation scalars: `nll=0.913997`, `bpb=1.318619`, `text_bpb=1.318619`, `graph_conditioned_bpb_no_side_cost=1.159307`, `graph_bpb=16.879733`, `invalid_graph_rate=0.0`, `graph_json_fallback_records=0`, `graph_autoregressive_decoding_enabled=1.0`, `random_graph_ar_rate=0.0`.

## Alterations Already Made By The Previous Agent

### Planning and Review Files

- Added `planning/tropicalgt_i_full_dataset_visualization_telemetry_plan.md` with a staged repair program covering fresh full-dataset training, W&B gating, optional inference artifacts, artifact cataloging, trajectory/radius semantics, NLL surfaces, analogical maps, persistence landscapes, tropical support, embedding maps, simplex trees, full-audit budgets, tests, and browser QA.
- Added `planning/tropicalgt_i_photo_annotation_review.md` plus `planning/photo_annotation_review_assets/` with 21 screenshot-backed annotations and a consolidated checklist.
- Added `references/2303.07295v1.pdf` for meet-in-the-middle decoding context.

### Documentation

- Updated root `README.md` to mark `train_full_dataset_pg_bpb_step0_full24b_b44.json` as the active fresh full-dataset Parameter-Golf BPB run.
- Updated root `README.md` and `TropicalGT-I/README.md` to document scalar-first W&B behavior, local optional browser artifacts, current token-slot budget, current run shape, full-audit preset usage, and sample-based browser review.
- Clarified that `train_full_dataset_active.json` is now historical relative to the current BPB repair run.

### Training Configs

- Modified `TropicalGT-I/configs/train.json` and `TropicalGT-I/configs/train_full_dataset_active.json` to disable W&B HTML artifact uploads by default, increase GoT depth/width/branch defaults, and move inference scaling defaults toward deterministic full-preset behavior.
- Added full-dataset step-0 configs: `train_full_dataset_pg_bpb_step0.json`, `train_full_dataset_pg_bpb_step0_full24b.json`, `train_full_dataset_pg_bpb_step0_full24b_b32.json`, and `train_full_dataset_pg_bpb_step0_full24b_b44.json`.
- The active b44 config uses the full audited token budget and targets roughly 18-20+ GiB VRAM while leaving test/browser headroom.

### W&B and Training Runtime

- Added `_wandb_log_interactive_artifacts_enabled`, `_wandb_html_artifact_limit`, and `_interactive_viz_require_complete_steps` to `TropicalGT-I/src/tropicalgt/run.py`.
- Gated periodic interactive visualizations behind `periodic_interactive_artifacts_enabled` and made final interactive visualizations optional via `final_interactive_artifacts_enabled`.
- Changed W&B logging so scalar metrics remain organized and HTML artifacts upload only with explicit opt-in plus positive limit.
- Added stricter complete-reasoning-step requirements for periodic/final GoT audit rendering paths when interactive artifacts are requested.

### Inference and Scaling

- Added `--interactive-artifacts` and `--require-complete-reasoning-steps` to `TropicalGT-I/scripts/infer_tropicalgt_i.py`.
- Added helper contracts `_resolve_render_html` and `_resolve_require_complete_reasoning_steps` so `--audit-output-dir` alone no longer implies heavy HTML rendering.
- Added `FULL_AUDIT_MINIMUMS` and `--audit-preset full` to `TropicalGT-I/scripts/run_multi_inference_audits.py`.
- Full preset currently enforces at least depth `7`, width `12`, branch factor `5`, trace limit `8192`, topology budget `8192`, stochastic GoT sampling by default, and memory retrieval budget `8`. Explicit `--no-scale-stochastic-actions` remains available for deterministic QA.
- `TropicalGT-I/src/tropicalgt/scaling.py` now supports complete reasoning-step audits and can fail closed if browserable GoT states lack real model NLL, graph-state embeddings, action probability vectors, complete graph-token traces, JS probability complexes, Euclidean embedding complexes, GraphCG all-direction projections, or directed GoT parent context.
- Public scaling candidates now include `reasoning_step_structure`, `reasoning_step_complete`, `reasoning_step_completeness`, and `reasoning_step_model_data`.

### Browser Index and Catalog

- `TropicalGT-I/scripts/build_sample_browser_index.py` now discovers sample directories recursively by `inference_scaling_tree.json`.
- Added complete artifact catalog generation (`interactive_visualization_catalog.html`) over nested HTML and JSON payloads.
- Added per-step rollup pages for all reasoning step complexes and all reasoning step simplex trees, replacing dozens of repeated buttons on each sample card.
- Browser index is sample-first and currently lists the remote bundle as 3 samples / 186 artifacts.

### Visualization Changes Already Present

- Removed prior duplicate-point PCA fabrication for single-state visualizations; one-state inputs now render as degenerate diagnostics instead of synthetic duplicate clouds.
- Added trajectory overlays to full Euclidean and Jensen-Shannon trajectory complexes from actual GoT parent-child edges.
- Added graph-token direction overlays to per-step complexes using model graph-token traces: graph edge tokens are rendered as source-node -> edge-token -> target-node overlays.
- Added interactive selected filtered-complex panels with Plotly 3D slider surfaces and collapsed static SVG preview fallback for WebGL failures.
- Changed tropical support from a default giant uniform heatmap into a collapse diagnostic path when support entropy/collapse indicates a true model support collapse.
- Changed GraphCG visualizations to preserve full-rank direction data in payload/hover while bounding visible tick labels.
- Added first analogical-memory wording repair: failed/no-memory cases are described as probability correspondence candidates rather than overclaimed exact simplicial maps.
- Added persistence landscape pages with actual `lambda_k(t)` language and trajectory growth hooks where topology growth rows are present.

### Tests Added Or Expanded

- Added tests for W&B HTML artifact gating and complete-step periodic visualization requirements.
- Added tests for inference artifact option resolution and multi-inference driver flags.
- Added tests for sample-browser catalog discovery and per-step rollups.
- Added tests for GoT trajectory NLL/contact payloads, single-state PCA policy, graph-token direction overlays, GraphCG projection-basis certificate, tropical support collapse layout, and inference scaling completeness gates.

## Live Browser QA Findings At Takeover

The visible remote bundle at `http://127.0.0.1:8988/browser_index.html` is useful but still smoke-level: each sample has only 12 states and 4 levels, so it is not a full deep/wide acceptance artifact.

Current direct screenshot QA of `sample_000/got_trajectory_pca_3d.html` shows:

- The NLL view is still a sparse Delaunay triangle mesh with only observed GoT anchors.
- Payload reports `surface_kind=exact_delaunay_nll_mesh`, `point_count=12`, and `z_axis_scale=1000.0`.
- The surface contact contract is numerically satisfied (`trajectory_point_surface_residual_max=0.0`), but the visual over-reads as a smooth energy landscape even though it is only a sparse exact interpolation over observed states.
- Immediate repair: either use a genuine model-evaluated local grid/field or render this as an explicit sparse-anchor diagnostic/contour scaffold, not a decorative landscape.

Current validator failures for `TropicalGT-I/outputs/multi_sample_browser/latest` are all analogical-map specific:

- analogical maps are not recognized as derived from model probability vectors;
- missing Jensen-Shannon summaries;
- missing assignment-cost summaries;
- missing filtration-distortion summaries;
- missing preserved/failed edge evidence;
- missing preserved-edge vertex sets;
- identical preservation and probability-assignment diagnostics.

## Repair Program For This Takeover Cycle

1. Keep PID `1086309` training alive and recheck progress/checkpoint every major cycle.
2. Patch NLL rendering so sparse observed-state triangulations are labeled and drawn as sparse exact anchor meshes, not as smooth/local energy landscapes. Add tests that forbid `actual_landscape_layer=True` unless the data source is a genuine model-evaluated local field.
3. Patch analogical map reports/pages so each rank contains explicit model-probability JS assignment summaries, assignment cost, filtration distortion, preserved/failed edge counts and preserved-edge vertex sets, or an explicit unavailable diagnostic.
4. Patch simplex-tree pages to default to readable provenance summaries and expose dense Hasse plots as optional/detail traces, not the only view.
5. Patch GraphCG and tropical-support layouts if browser screenshots still show jammed labels after current code changes.
6. Generate a new remote browser bundle from the active latest checkpoint only after patches/tests pass enough to avoid publishing another misleading artifact.
7. Keep the Codex browser open on the remote tunnel and inspect screenshots after each regeneration.
8. Update this file and `planning/tropicalgt_i_full_dataset_visualization_telemetry_plan.md` at each repair milestone.
9. Push only on `tropicalgt-i-implementation` after tests, browser QA, docs/planning updates, and no secrets/data are staged.

## Non-Negotiable Truthfulness Contract

- Scientific/topological plots may use only model outputs, model embeddings, model probabilities, graph-token traces, or explicit unavailable diagnostics.
- Rendering fallbacks are allowed only for browser/WebGL display failures and must use the same serialized real payload, never substitute a metric or topology.
- Sparse observed-state interpolation is not a dense model landscape. It must be named and styled as sparse observed-anchor geometry unless a real model-evaluated local field is computed.
- Analogical maps must come from model probability vectors with Jensen-Shannon matching and finite filtered-complex certificates, or they must fail closed.


## Repair Cycle Update: 2026-06-12 20:18:23

- Patched `TropicalGT-I/src/tropicalgt/visualization.py` so the GoT NLL page no longer labels sparse triangulations as an actual dense landscape. The emitted contract is now `surface_kind=sparse_observed_state_nll_anchor_mesh`, `actual_landscape_layer=false`, `sparse_observed_anchor_layer=true`, and `dense_model_evaluated_field=false` unless a genuine dense model-evaluated field exists.
- Capped NLL display scaling to avoid tiny raw NLL ranges becoming visually absurd vertical walls while preserving exact point/surface contact in payload coordinates.
- Patched analogical map title/certificate text so preserved 1-simplices and vertex-only correspondences are readable even when a trace is empty/legend-only.
- Patched `validate_interactive_audit_artifacts.py` to require the repaired observed-NLL-anchor contract and the current full-rank GraphCG layout marker.
- Patched `run_multi_inference_audits.py` so `--audit-preset full` preserves caller sampling temperature/exploration and defaults to stochastic actions instead of silently forcing deterministic sampling.
- Focused tests passed: GoT NLL visualization, analogical memory visualization, and full-audit preset contract.
- Launched refreshed CPU-side model-backed browser QA from the active latest checkpoint with full audit depth/width/branch, stochastic actions, GUDHI, memory top-k 8, and no GPU contention with live training. Output root is recorded in `/tmp/tropicalgt_takeover_out.txt`.


## Repair Cycle Update: 2026-06-12 Memory Gate And Certificate Telemetry

- Added `AnalogicalMemoryQualityGate` in `TropicalGT-I/src/tropicalgt/memory.py`. Memory insertion is now explicitly quality-gated: configured thresholds can require model-probability Jensen-Shannon filtered complexes, enough probability vertices/simplices, topological algebra payloads, nonnegative NLL improvement, score/BPB/NLL/margin thresholds, and a minimum composite quality score.
- Added `memory_quality_gate_summary` so periodic audits can report candidates seen, eligible trajectories, rejected trajectories, and rejection reasons. Retrieval can still request many analogies, but the bank is curated; early training may correctly produce no retrievable memories.
- Updated training configs with `memory_quality_*` defaults requiring model-probability complexes, at least four probability vertices, at least three simplices, topological algebra, and nonnegative NLL improvement before storage.
- Expanded certificate telemetry. `certificate_loss` remains the raw negative log mass assigned to allowed support sets, but W&B now also receives allowed-mass mean/min, node/edge/graph agreement and loss, disallowed-support rate, graph-support rate, node->graph support rate, edge->graph support rate, and token-order support transition rate.
- Focused tests passed after fixes for `dimension=0` and `level=0` truthiness bugs in the memory gate: `test_metrics_and_memory.py` and `test_training_metrics.py` (`13 passed`).


## 2026-06-12 23:26:47 UTC - Visual Math Repair Plan and Run Reset

Added `planning/tropicalgt_i_visual_math_repair_plan_2026-06-12.md` after reviewing the latest browser annotations and the Miller-Sturmfels two-variable monomial ideal staircase section. New hard requirements:

- Reasoning-step complexes must be derived from each step's model outputs, embeddings, probabilities, and graph-token trace; identical pages are treated as a data-path defect until payload hashes prove otherwise.
- Radius filtrations must begin as disjoint vertex clouds and grow monotonically through GUDHI SimplexTree filtration values or model-probability Jensen-Shannon filtrations.
- Two-parameter persistence views must be `F2[x_level,x_radius]` lattice/module views with multiplication maps, rank grids, staircase/minimal-generator candidates, adjacent lcm syzygy candidates, and conservative free-resolution labels.
- Analogical retrieval must distinguish probability-JS correspondences from verified simplicial maps, and derived/algebraic similarity must fail closed when PH/free-resolution/rank-invariant checks fail.
- Simplex-tree plots must be connected trie/Hasse diagrams of actual simplices and immediate face-to-coface inclusions, not disconnected stripe plots.
- The current `b44` training run will be stopped and replaced by a fresh step-0 BPB-first `b48_v2` run with lower auxiliary pressure and deeper stochastic GoT audits.


### Added Macaulay2-style algebra figure requirement

The algebra repair target now includes Macaulay2/research-style Betti tables, multigraded free modules, sparse differential matrices over `F2[x_level,x_radius]`, staircase/lcm syzygy diagrams, and chain-map/derived-morphism diagrams between query and retrieved memory objects. These must be labeled as exact only when the computation certifies exactness/minimality; otherwise they are candidates/witnesses.

## Repair Cycle Update: Persistence-Landscape Vectors And Resolution Guardrails

- Added persistence-landscape vector comparisons to analogical memory retrieval. Retrieval now accepts query topology, reads real GUDHI `Landscape` vectors from query and memory topology payloads, and contributes an optional L2-similarity term only when both sides have actual vectors. Missing landscape vectors are reported unavailable/zero and are not fabricated.
- Added browser/top-k diagnostics for persistence-landscape L2 similarity, cosine, correlation, vector overlap dimension, and L2 distance. The analogical top-k table now has explicit persistence-landscape columns so derived/algebraic and coarse-signature columns are not shifted or mislabeled.
- Kept both overloaded “landscape” pages, but separated language: GUDHI persistence landscapes remain `lambda_k(t)` vectorized topology pages; GoT/NLL pages are labeled as observed NLL/fitness fields or anchor views, not persistence landscapes.
- Tightened the free-resolution truthfulness contract. Multigraded diagnostics are now canonicalized as `*_chain_presentation_diagnostics`; legacy `*_free_resolution*` keys remain as deprecated aliases for compatibility and carry `not_a_free_resolution=true`.
- Real/minimal free resolutions are now unavailable unless a CAS certificate is attached. Macaulay2/Sage/Singular are reported as candidate backends only; `multipers` is tracked separately for multiparameter persistence and is not treated as a certified minimal-free-resolution backend.
- Memory quality and compact payloads now distinguish `has_chain_presentation_diagnostics` from `has_real_free_resolution` so early training and browser artifacts do not count chain modules as true resolutions.
- Focused tests passed after the patch: `test_algebraic_persistence.py` and `test_simplicial_visualization.py` (`28 passed`).
- Regenerating the model-backed multi-sample browser bundle from the active checkpoint on CPU, leaving the live GPU training process untouched.

## 2026-06-12 persistence-landscape vector retrieval and label audit update

- Integrated the persistence-landscape vector comparison into analogical memory retrieval and inference using the real GUDHI `Landscape.vector` payloads already stored in topology reports.
- Aligned browser-side persistence-landscape similarity with the same dimension-aware retrieval helper so top-k scores, analogical map diagnostics, and rendered tables cannot diverge when homology dimensions have different vector lengths.
- Relabeled browser/table output from `free-res similarity` to `chain-presentation similarity` unless a future CAS backend attaches an explicit free-resolution certificate. The legacy JSON key remains only as a deprecated alias for compatibility.
- Kept the GUDHI persistence-landscape page as a separate artifact; renamed the GoT page link to `GoT observed NLL PCA anchors` so the two uses of “landscape” are not conflated.
- Focused tests passed after the patch: `test_algebraic_persistence.py`, `test_simplicial_visualization.py`, `test_interactive_artifact_validator.py`, and `test_metric_provenance.py`.


## 2026-06-13 07:37:49 UTC - Browser-Verified NLL Density Cloud And Artifact Contract

- Added `got_nll_density_cloud_pca_3d.html` as a separate GoT visualization from the sparse observed-NLL anchor page. The density cloud is computed deterministically from actual model-evaluated `graph_state` PCA anchors and measured raw NLL values. Gaussian cloud samples are explicitly marked as visualization-only local support, not additional model states; only the large labeled markers are real GoT states.
- Updated the sample browser index to expose `GoT NLL density cloud (3D PCA)` for each sample while keeping the GUDHI persistence-landscape artifact separate. This keeps the overloaded word "landscape" out of the GoT link label and preserves the real persistence-landscape page.
- Renamed SimplexTree visualization labels to `SimplexTree face-coface poset` and documented that the displayed 3D page is a face/coface poset view, not a literal trie layout. The exact `gudhi.SimplexTree` payload remains the source of truth.
- Strengthened `validate_interactive_audit_artifacts.py` so generated bundles must include the density-cloud page and the corrected SimplexTree face-coface wording. The latest `TropicalGT-I/outputs/multi_sample_browser/latest` bundle passed validation for all three sample rows.
- Browser QA at `http://127.0.0.1:8990/sample_000/got_nll_density_cloud_pca_3d.html` confirmed a rendered Plotly figure with visible local raw-NLL color scale, actual GoT state anchors, trajectory edges, and the explicit "not a model state" density-cloud hover contract.

## 2026-06-13 08:12:59 UTC - Strict Bifiltration And Density-Cloud Artifact Contract

- Kept the live full-dataset training run alive; no process restart or checkpoint mutation in this repair pass.
- Tightened the browser artifact validator so every nonempty sample must include `trajectory_persistence/two_parameter_bifiltration.html` and `trajectory_level_radius_bifiltration.json`.
- Validator now rejects missing/ambiguous 2-parameter persistence metadata: coefficient ring must be `F2[x_level,x_radius]`, parameter names must be `trajectory_level` and `radius`, radii must be finite nonnegative sorted min-to-max, grid axes must match levels/radii, rank-invariant samples must carry 2D grades, and chain-presentation diagnostics must not claim a real free resolution without a CAS certificate.
- Added the sample-browser entry `2-parameter F2[x_level,x_radius] bifiltration` and kept persistence landscapes separate from NLL/energy landscapes because the term is overloaded but both views are mathematically distinct and useful.
- Added a 3D PCA NLL density-cloud artifact generated from actual GoT graph-state PCA anchors and measured raw NLL values. The Gaussian cloud is explicitly visualization-only local mass around actual embeddings; cloud points are not treated as model states.
- Browser QA passed on the current served bundle at `http://127.0.0.1:8990`: sample index includes the bifiltration and density-cloud pages; direct bifiltration page renders the `F2[x_level,x_radius]` lattice; direct density-cloud page renders actual-state anchors plus local NLL cloud. Remaining visual issue: the bifiltration page is mathematically backed but still visually dense and needs a cleaner Miller-Sturmfels-style research layout.

## 2026-06-13 08:35:47 UTC - Persistence-Landscape Retrieval Regression Test

- Added a targeted analogical-memory regression test proving that, when embedding/signature/quality terms are tied, AnalogicalMemoryBank.retrieve ranks memories by GUDHI persistence-landscape vector similarity.
- The tested comparison uses the cached gudhi.representations.Landscape.vector payload shape and asserts L2 similarity/overlap metadata are exposed in retrieval rows.
- Updated the memory roundtrip test to reflect the current compact-memory policy: compact trajectory/probability complexes may be stored for analogical maps, but payload size remains bounded and summaries must match the model-derived trajectory object.

