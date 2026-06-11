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

## Current Artifact Locations

- Remote audit root: `TropicalGT-I/outputs/train/periodic/step_00000250/got_audit`
- Local screenshots: `/Users/amelieschreiber/Documents/LaTeX-projects/TropicalGT-audit-screenshots/step_00000250`
- Latest checkpoint: `TropicalGT-I/checkpoints/tropicalgt_i_train.latest.pt`
