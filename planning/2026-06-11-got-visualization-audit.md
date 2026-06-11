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
  - `Smooth projected NLL/fitness landscape`: a dark-mode surrogate over actual model `graph_state` PCA coordinates, using inverse-distance centered NLL plus a local embedding-support energy that is zero at observed states.
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

- Remote audit root: `TropicalGT-I/outputs/train/periodic/step_00000250/got_audit`
- Local screenshots: `/Users/amelieschreiber/Documents/LaTeX-projects/TropicalGT-audit-screenshots/step_00000250`
- Latest checkpoint: `TropicalGT-I/checkpoints/tropicalgt_i_train.latest.pt`

## 2026-06-11 Iteration: Actual Landscapes And Local NLL Sheet

Implemented after the follow-up critique that actual landscapes were missing and the NLL energy surface appeared not to contain all trajectory vertices.

- The 3D GoT NLL page now has three distinct layers:
  - `Smooth projected NLL/fitness landscape`: a broad visual energy/fitness surrogate over sampled model states plus rendered reasoning microsteps.
  - `Local interpolating NLL sheet`: a sample-supported inverse-distance interpolant through model-state anchors and rendered microstep anchors. Metadata records `max_point_residual`, duplicate-coordinate collapse diagnostics, support radius, and masked fraction.
  - `Exact GoT NLL anchor mesh`: exact mesh through sampled model GoT states only.
- The key distinction is now explicit: the exact mesh is scoped to sampled model states, while the local sheet and broad landscape also include rendered microstep vertices so those vertices are not visually floating outside the NLL field.
- Added `trajectory_persistence/persistence_landscapes.html`, a dark-mode page plotting actual GUDHI `Landscape` functions `lambda_k(t)` by trajectory-growth level, not only L2 norms.
- The landscape heatmap now uses the first finite homology dimension available in the audit instead of assuming finite H0 intervals. On the refreshed step-250 audit this selects H1, which is why the heatmap labels show `H1`.
- Refreshed all three step-250 audit rows from their saved `inference_audit.json` payloads.
- Validator result:
  - `TropicalGT-I/outputs/train/periodic/step_00000250/interactive_audit_validation_landscapes.md`
  - status: `PASS`
  - rows checked: `3`
  - main row: `75` candidates, `74` edges, depth `8`, PCA distance correlation `0.989791`, NLL residual `0.0`.
- Local mirror:
  - `/Users/amelieschreiber/Documents/LaTeX-projects/TropicalGT-audit-browser/step_00000250/got_audit/got_trajectory_pca_3d.html`
  - `/Users/amelieschreiber/Documents/LaTeX-projects/TropicalGT-audit-browser/step_00000250/got_audit/trajectory_persistence/persistence_landscapes.html`
- Fresh screenshots:
  - `/Users/amelieschreiber/Documents/LaTeX-projects/TropicalGT-audit-browser/step_00000250/screenshots_landscape_pass_v2/nll_landscape_local_sheet.png`
  - `/Users/amelieschreiber/Documents/LaTeX-projects/TropicalGT-audit-browser/step_00000250/screenshots_landscape_pass_v2/actual_persistence_landscapes.png`

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
