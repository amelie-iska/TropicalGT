# 2026-06-11 GoT Visualization Audit

## Implemented

- Split the previous combined graph-of-thought audit into distinct dark-mode Plotly artifacts:
  - `got_embedding_map_3d.html`: model `graph_state` PCA only, with no tree-layout or synthetic coordinates.
  - `got_trajectory_pca_3d.html`: trajectory graph plus exact NLL surface.
  - `got_full_trajectory_complex.html`: full trajectory filtered simplicial complex with a filtration slider.
  - `reasoning_step_complex_maps/*.html`: one 3D PCoA/MDS radius-filtered simplicial complex map per sampled reasoning state.
- Added PCA diagnostics to payloads:
  - coordinate source,
  - explained variance,
  - pairwise distance correlation,
  - normalized stress.
- Added stochastic branch sampling for inference-time GoT audits:
  - temperature-controlled action sampling,
  - uniform exploration mixture,
  - deterministic per-node seeds for reproducible stochastic audits.
- Increased audit depth/width for training:
  - depth `8`,
  - width `10`,
  - branch factor `4`,
  - sampling temperature `2.5`,
  - exploration `0.45`.
- Changed checkpoint cadence to every `250` steps so each periodic audit has a reloadable model state.

## Step 250 Evidence

Fresh run: W&B `r5125kw6`, commit `c2ead9e`.

Main step-250 GoT audit:

- Candidates: `75`
- Edges: `74`
- Levels: `0..8`
- Branching nodes: `22`
- Max branch factor observed: `4`
- Unique paths: `75`
- Stochastic actions: `true`
- Temperature: `2.5`
- Exploration: `0.45`
- PCA source: `model graph_state embeddings`
- PCA pairwise-distance correlation: `0.9897913606560541`
- PCA normalized stress: `0.13932622249010618`
- PCA explained variance in first 3 components: `0.8936010608834728`
- NLL surface: `exact_delaunay_nll_mesh`
- NLL surface touches points: `true`
- NLL max point residual: `0.0`
- Reasoning-step complex maps: `75`

The two additional examples also have `75` candidates, `74` edges, depth `8`, stochastic sampling enabled, and exact NLL surfaces with zero point residual.

## Finding

The previous visual regularity was partly from the old renderer, but the corrected audit reveals a real model-side issue: early sampled graph states are still tightly clustered and NLL is nearly flat across branches.

Measured on the main step-250 audit:

- Mean pairwise embedding distance: `0.21800338093147495`
- Max pairwise embedding distance: `1.5862111676173911`
- Mean per-coordinate embedding standard deviation: `0.006168803698374065`
- NLL min/mean/max/std: `1.5630971193313599 / 1.5643782822291057 / 1.5644731521606445 / 0.00016039736308897623`

This is not a plotting artifact: the PCA metadata and high distance correlation indicate the projection is faithful to the current `graph_state` geometry. The next modeling improvement should target graph-state collapse directly, for example with a gated action/path embedding in the graph state, an auxiliary branch-diversity objective, or a stronger GFlowNet reward term that distinguishes reasoning-path topology and verifier/certificate changes.

## 2026-06-11 Iteration: NLL Landscape And Persistence Vectorizations

Implemented after the follow-up browser audit:

- Replaced the flat-looking raw NLL sheet with a two-layer audited NLL/fitness view:
  - Retired in the 2026-06-12 repair pass: the retired broad NLL/fitness surrogate is no longer rendered by default; the current view uses exact sampled anchors with the surface-contact projection contract so plotted trajectory points sit on the displayed NLL landscape.
  - `Exact GoT NLL anchor mesh`: exact Delaunay/triangular mesh through the sampled reasoning states with zero residual at anchors.
- Added `nll_progress` payload diagnostics:
  - per-edge raw NLL deltas,
  - improving-edge fraction,
  - terminal mean/best improvement from root,
  - by-level NLL summaries.
- Step-250 main audit now reports `0.0%` improving GoT edges and best terminal improvement about `-9.96e-4`, so the model/sampler is not yet producing downhill reasoning trajectories despite the improved visualization.
- Added GUDHI-backed persistence representation vectors to topology reports:
  - `Landscape` as the primary vector-space topology feature,
  - `BettiCurve` as the primary interpretable rank trace,
  - `PersistenceImage`, `Silhouette`, `PersistenceLengths`, `TopologicalVector`, and `Entropy` as auxiliary train/eval/retrieval features.
- Added `trajectory_persistence/persistence_representations.html`, a dark-mode interactive page for vectorized persistence growth over reasoning levels.
- Added backfilling in the renderer: older saved `trajectory_topological_algebra` payloads are upgraded from their persisted intervals when `persistence_representations` is absent.
- Sanitized non-finite vectorizer outputs before JSON/W&B logging.

Current vectorized persistence finding on the main step-250 audit:

- GUDHI vectorizations are available from the saved finite intervals.
- Landscape norm rises sharply around level 2 and then decays/stabilizes.
- Persistence length mass saturates by level 4.
- Entropy grows through the trajectory while the topological-vector norm remains small.
- The Betti-curve heatmap is nontrivial and now far more interpretable than the earlier decorative 3D spike view.

## Exact Versus Proxy/Surrogate Audit

- Exact:
  - filtered simplicial object construction for displayed finite complexes,
  - GUDHI `SimplexTree` persistence intervals when the backend is available,
  - F2 boundary ranks, homology ranks, and bounded Hochster/Taylor reports as labeled,
  - exact graph BPB byte accounting from masks/endpoint widths/graph JSON bytes,
  - exact NLL values at sampled reasoning states.
- Fast vectorized computations:
  - GUDHI landscapes, Betti curves, persistence images, silhouettes, persistence lengths, entropy, and topological vectors are computed from finite persistence intervals and are suitable for train/eval metrics, memory retrieval features, and optional inference artifacts.
- Explicit surrogates:
  - projected NLL/fitness landscape is a visualization surrogate over PCA space, not a dense model re-evaluation; metadata records provenance and anchor residual.
  - `multiparameter_free_resolution_proxy` remains a multigraded free-chain/free-resolution proxy, not a minimal free resolution. The label is intentionally retained until a CAS/backend computes minimal resolutions.
  - GraphCG condition numbers named `*_condition_proxy` are spectral diagnostics, not proof of well-conditioned causal directions.

Immediate modeling recommendation:

- Add or increase a GFlowNet reward component for negative raw-NLL edge deltas and terminal improvement, with diversity kept as a constraint rather than the only objective.
- Track `nll_progress.improving_edge_fraction`, terminal improvement, landscape norm, persistence length mass, and entropy under the priority W&B namespaces and ablate whether they predict BPB improvement.

## Current Artifact Locations

- Current repaired browser audit root: `TropicalGT-I/outputs/multi_sample_browser/latest`
- Local screenshots: `<local-screenshot-root>/multi_sample_browser_latest`
- Latest active checkpoint: `TropicalGT-I/checkpoints/tropicalgt_i_train_full_dataset_active.latest.pt`

## 2026-06-11 Historical Iteration: Actual Landscapes And Local NLL Sheet

Historical note, superseded by the 2026-06-12 repair pass. It is retained only to explain the earlier failure mode where trajectory vertices appeared outside the NLL surface.

- The 3D GoT NLL page now has audited NLL layers:
  - Retired in the 2026-06-12 repair pass: the smooth_projected_nll_fitness_landscape entry is recorded as unavailable instead of rendered as a default plot.
  - Superseded: the retired local NLL sheet and extra rendered anchors are no longer part of the default repaired view.
  - Current behavior: sampled model GoT states render against a displayed NLL surface-contact mesh; every plotted trajectory point and edge endpoint uses the displayed surface z-value, with raw centered/scaled NLL retained separately in JSON.
- Added `trajectory_persistence/persistence_landscapes.html`, a dark-mode page plotting actual GUDHI `Landscape` functions `lambda_k(t)` by trajectory-growth level, not only L2 norms.
- The landscape heatmap now uses the first finite homology dimension available in the audit instead of assuming finite H0 intervals. On the refreshed step-250 audit this selects H1, which is why the heatmap labels show `H1`.
- Refreshed all three step-250 audit rows from their saved `inference_audit.json` payloads.
- Validator result:
  - `<legacy-step-250-root>/interactive_audit_validation_landscapes.md`
  - status: `PASS`
  - rows checked: `3`
  - main row: `75` candidates, `74` edges, depth `8`, PCA distance correlation `0.989791`, NLL residual `0.0`.
- Local mirror:
  - `<legacy-local-browser-artifact-root>/legacy_step_250/got_audit/got_trajectory_pca_3d.html`
  - `<legacy-local-browser-artifact-root>/legacy_step_250/got_audit/trajectory_persistence/persistence_landscapes.html`
- Fresh screenshots:
  - `<legacy-local-browser-artifact-root>/legacy_step_250/screenshots_landscape_pass_v2/nll_landscape_local_sheet.png`
  - `<legacy-local-browser-artifact-root>/legacy_step_250/screenshots_landscape_pass_v2/actual_persistence_landscapes.png`

## 2026-06-11 Historical Iteration: Anchored Landscape, GraphCG, And Support Readability

Implemented after the follow-up critique that actual landscapes must be visible and that trajectory vertices appeared outside the NLL surface.

- Superseded: the retired broad NLL/fitness grid and extra rendered anchors are no longer accepted as the default repaired view. The current payload contract reports `trajectory_point_surface_residual_max: 0.0`, `surface_projected_z_by_record_id`, and per-node `touches_nll_surface: true`.
- The exact layer is now labeled `Actual sampled GoT NLL landscape (exact mesh)` and carries `actual_landscape_layer=true`, `actual_landscape_scope`, and provenance metadata. It is intentionally not described as a dense latent-space model evaluation.
- GraphCG visualization was redesigned from a hard-to-read label wall into:
  - a top-active-direction heatmap for readability,
  - a full-rank activity spectrum over all directions,
  - candidate activity and effective-direction counts,
  - a signed-bias versus absolute-activity scatter.
  The main refreshed payload reports matrix shape `[75, 1760]`, full-rank direction count `1760`, and active nonzero rank `1760`.
- Tropical support visualization now treats the step-250 result as true active-support collapse rather than a rendering artifact. The payload records query-to-support flow edges and a margin summary; the page shows the collapsed support strip, per-token margin profile, margin histogram, and collapse metrics table.
- Refreshed all three step-250 audit rows again from saved `inference_audit.json`.
- Stricter validator result:
  - `<legacy-step-250-root>/interactive_audit_validation_landscape_graphcg_support.md`
  - status: `PASS`
  - rows checked: `3`
  - main row: `75` candidates, `74` edges, depth `8`, PCA distance correlation `0.989791`, NLL residual `0.0`.
- Local screenshots:
  - `<legacy-local-browser-artifact-root>/legacy_step_250/screenshots_landscape_graphcg_support_pass/nll_actual_smoothed_landscape_webgl.png`
  - `<legacy-local-browser-artifact-root>/legacy_step_250/screenshots_landscape_graphcg_support_pass/graphcg_full_rank_audit.png`
  - `<legacy-local-browser-artifact-root>/legacy_step_250/screenshots_landscape_graphcg_support_pass/tropical_support_collapse_diagnostic.png`

Remaining limitation:

- The default repaired renderer no longer displays broad post-hoc NLL/fitness landscapes or extra rendered anchors. A genuinely dense local landscape still requires a separately defined perturbation family and additional model forward passes at those perturbations.

## Provenance Audit Status

Added `TropicalGT-I/src/tropicalgt/provenance.py` and `TropicalGT-I/scripts/audit_metric_provenance.py`.

Current report:

- `TropicalGT-I/outputs/metric_provenance_audit.json`
- `TropicalGT-I/outputs/metric_provenance_audit.md`
- registered entries: `24`
- risk-word findings in the active default scan: `106`
- directly covered findings: `106`
- uncovered generic findings: `0`
- gate: `audit_metric_provenance.py --fail-on-uncovered` passes.

The default scan covers active source, scripts, READMEs, and this current visualization-audit note. Historical planning logs are not part of the default gate because they intentionally preserve older iteration notes; scan them explicitly with `--scan planning` when reviewing archival status language.

## 2026-06-11 Iteration: Analogical Simplicial-Map Honesty

Implemented after the critique that the analogical-memory figure looked too regular and did not make the filtered-complex map readable.

- The analogical retrieval artifact now builds domain/codomain filtered complexes from the actual query and memory trajectory embeddings used by the audit payload.
- The displayed correspondence is checked simplex-by-simplex on the visible filtered skeleton:
  - preserved vertex correspondences are always shown,
  - preserved 1-simplex/2-simplex correspondences are counted separately,
  - failed edge/face correspondences are recorded and labeled as failures rather than silently rendered as a simplicial map.
- The 3D plot now separates:
  - domain complex vertices,
  - codomain complex vertices,
  - preserved simplex-map edges,
  - vertex-only correspondences where the displayed edge/face condition fails,
  - an explicit diagnostic table with domain/codomain simplex counts, preserved counts, failed counts, vectorized persistence distance, and warning text.
- Hover text was rewritten around the model record id, reasoning path, model output/preview text, NLL/reward, filtration, and simplex-map status.

Remaining limitation:

- The map is a nearest-neighbor/persistence-feature retrieval diagnostic over sampled trajectory complexes. It is not yet a learned functor between derived categories, and it is explicitly labeled as such. A true derived-equivalence certificate still requires a backend for minimal multigraded resolutions and chain-map verification.

## 2026-06-11 Iteration: Optional Meet-In-The-Middle Decoding

Implemented the requested meet-in-the-middle toggle from `references/2303.07295v1.pdf` as a graph-aware TropicalGT-I adaptation.

- Added `TropicalGT-I/src/tropicalgt/decoding.py` with:
  - `MeetInMiddleConfig`,
  - `meet_in_middle_config`,
  - `encode_record_bytes_reverse`,
  - `meet_in_middle_batch`.
- Added an off-by-default `meet_in_middle` config block to every active TropicalGT-I config.
- The implemented mode is `shared_weight_reverse_pass`:
  - graph-conditioned byte records are scored left-to-right and right-to-left by the same TropicalGT-I checkpoint,
  - causal DAG graphs keep their topological autoregressive order before reversal,
  - non-causal/cyclic graphs keep the deterministic seeded random autoregressive order before reversal,
  - the join token is evaluated at the configured split point.
- Logged/evaluated metrics include:
  - `mim_reverse_nll`,
  - `mim_bidirectional_nll`,
  - `mim_agreement_loss`,
  - `mim_join_token_match_rate`,
  - `mim_true_meet_logprob_mean`,
  - `mim_candidate_count`,
  - `mim_loss`,
  - `mim_agreement_weight`,
  - `mim_reverse_nll_weight`.
- Training only adds the MIM objective when `agreement_weight` or `reverse_nll_weight` is nonzero. This is deliberate because the current 1760-dimensional training run already consumes about 20 GiB VRAM on the RTX 4090.
- Evaluation and inference expose CLI overrides:
  - `eval_tropicalgt_i.py --meet-in-middle` / `--no-meet-in-middle`,
  - `infer_tropicalgt_i.py --meet-in-middle` / `--no-meet-in-middle`.
- W&B priorities now reserve `09_meet_in_middle` for MIM diagnostics, shifting system/optimizer namespaces to `10_system` and `11_optimization`.

Verification:

- Focused tests passed:
  - `pytest TropicalGT-I/tests/test_meet_in_middle_decoding.py TropicalGT-I/tests/test_losses_and_model.py TropicalGT-I/tests/test_training_metrics.py -q`
  - result: `13 passed`.
- Provenance coverage passed after registering the MIM metrics:
  - `audit_metric_provenance.py --fail-on-uncovered`
  - result: all active findings covered.

Current limitation:

- This is not yet a separately trained bidirectional pair of LTR/RTL language models. It is an honest shared-weight reverse-pass implementation compatible with the current artifact-size and VRAM constraints. A separate reverse checkpoint can be added later through `reverse_model_path` once the export budget is re-audited.

## 2026-06-11 Iteration: Sample-First Browser Index

Implemented after the critique that the browser view should be sample-based rather than artifact-category-based.

- Added `TropicalGT-I/scripts/build_sample_browser_index.py`.
- The builder scans an audit root for every sampled row/input directory with `inference_scaling_tree.json`.
- It emits `codex_browser_index.html` where each sampled row/input is the top-level unit.
- Each sample card includes:
  - record id prefix and directory,
  - candidate/state count,
  - edge count,
  - growth-level count,
  - NLL min/std,
  - PCA distance-correlation,
  - input/target/model-output preview,
  - NLL-progress summary,
  - analogical-map warning/failure counts.
- Each sample card links to that sample's:
  - GoT NLL landscape,
  - embedding map,
  - full trajectory complex,
  - reasoning-step complexes,
  - persistence barcode,
  - Betti/free-resolution growth,
  - vectorized persistence representations,
  - actual persistence landscapes,
  - analogical simplicial map,
  - analogical top-k index,
  - GraphCG direction audit,
  - tropical support audit.
- `validate_interactive_audit_artifacts.py` now optionally validates `codex_browser_index.html` for sample cards, per-sample artifact buttons, and broken relative links.
- Current local browser mirror:
  - `<legacy-local-browser-artifact-root>/legacy_step_250/got_audit/codex_browser_index.html`
- Current remote generated dashboard:
  - `<legacy-step-250-audit-root>/codex_browser_index.html`

Browser check:

- Codex browser loaded `http://127.0.0.1:8765/codex_browser_index.html`.
- The page reported `3` sample cards and `39` per-sample artifact buttons.
- Clicking `Sample 01 / GoT NLL landscape` correctly loaded `example_01/got_trajectory_pca_3d.html` and marked `Sample 01` active.

## 2026-06-12 Repair Pass: Real Filtrations and Browser Validation

- Removed the default retired broad NLL/fitness surrogate from the GoT NLL
  landscape. The page now renders sampled model-evaluated GoT NLL anchors and a
  surface-contact projection contract so every plotted trajectory marker and edge endpoint touches the displayed NLL landscape; missing NLLs
  are emitted as explicit unavailable diagnostics.
- Split trajectory topology into Euclidean radius and Jensen-Shannon
  probability complexes, each with a GUDHI simplex-tree inclusion-poset page.
- Added per-reasoning-step simplex-tree pages next to every reasoning-step
  filtered complex map.
- Tightened analogical memory retrieval into model-probability
  Jensen-Shannon vertex assignments, with filtration distortion,
  edge/2-simplex preservation, GUDHI provenance, and per-rank `pair_page`
  links. Vertex-only correspondences are intentionally de-emphasized.
- Extended the sample-first browser validator to check per-sample artifact
  buttons, `data-samples` coverage, broken relative targets, analogical map
  links, and legitimate unavailable states instead of accepting fabricated
  plots.

## 2026-06-12 Browser Review Note

Served the repaired multi-sample browser bundle through a local tunnel at `http://127.0.0.1:8977/browser_index.html`. Browser inspection confirmed the sample-first index, NLL page, embedding map, full radius complex, full radius simplex tree, Jensen-Shannon probability complex, probability simplex tree, reasoning-step simplex tree, trajectory persistence landscapes, GraphCG full-rank direction audit, and tropical support page all load as interactive Plotly pages with the expected controls/markers. The saved step-8250 analogical page was generated before the probability-map repair and remains an older artifact; the repaired `write_analogical_memory_visualization` path is covered by tests and will be used by the restarted training process for subsequent periodic audit bundles.

Training was restarted with `TropicalGT-I/configs/train_full_dataset_active.json`; the active run records its PID in `TropicalGT-I/outputs/train_full_dataset_active/latest_training.pid` and logs to `TropicalGT-I/outputs/train_full_dataset_active/full_dataset_active_training.log`.
