# TropicalGT-I Sequential Repair Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every misleading, proxy, synthetic, or visually ambiguous TropicalGT-I algebra/topology/training artifact with a real computation, a certified unavailable state, or a fully implemented CAS-backed object, while keeping active BPB-oriented training and browser QA alive.

**Architecture:** Work sequentially. Each item produces one tested, browser-reviewed repair before moving to the next. The implementation boundary is `TropicalGT-I/src/tropicalgt/` plus CLI scripts in `TropicalGT-I/scripts/`; generated audit bundles remain ignored artifacts, while plans/docs/README and source changes are committed on the non-main branch.

**Tech Stack:** Python, PyTorch, Plotly, GUDHI, optional multipers/Sage/Macaulay2/Singular bridges, W&B, Tailscale SSH, pytest, HTML browser audit bundles.

---

## Immediate Rule Set

- Do all repository work on `iska` through `tailscale ssh iska@iska` in `/home/iska/Documents/amelie/bio/TropicalGT`.
- Stay on the non-main branch `tropicalgt-i-real-cas-no-proxy-20260614` or a newer non-main branch created from it.
- Work one item at a time. No shotgun edits.
- When a proxy is encountered, replace it with a real implementation immediately if possible; if a CAS dependency is unavailable, render a precise unavailable state and implement the dependency bridge as the next item.
- Browser QA is required for every visualization repair at `http://127.0.0.1:8991/browser_index.html` or the latest active tunnel.
- Keep the active BPB training run alive unless explicitly restarting under the 5K gate policy.
- Do not commit data, secrets, checkpoints, W&B runs, or generated audit outputs.

## Current Objective: 2-Parameter Module Figure Repair

**Problem:** The 2-parameter persistence plot visually collapses H0/H1 module layers and makes a constant H0 sheet dominate. The staircase panel can look empty or misleading when the minimal antichain is small. The term "affine toric exponent chart" may imply a full tropical-variety toric embedding that has not yet been implemented.

**Files:**
- Modify: `TropicalGT-I/src/tropicalgt/visualization.py`
- Modify: `TropicalGT-I/src/tropicalgt/algebra.py`
- Modify: `TropicalGT-I/tests/test_interactive_artifact_validator.py`
- Update: `planning/tropicalgt_i_cas_free_resolution_implementation_plan_2026-06-14.md`

- [x] Patch the 3D module plot so H0/H1 layers use tiny visual offsets while the z-axis remains the actual fiber rank.
- [x] Make H1/nonconstant layers visually primary when H0 is constant.
- [x] Render a proper 2D exponent lattice/staircase panel with horizontal `x_radius`, vertical `x_level`, actual lattice fibers, shifted chain-generator bidegrees, minimal antichain markers, and coordinate one dimensional cone(s); adjacent LCM syzygies remain certificate-gated rather than fabricated.
- [x] Replace broad "affine toric exponent chart" wording with scoped "coordinate exponent semigroup chart `Spec F2[x_level,x_radius]`" wording.
- [x] Regenerate the latest browser audit and visually inspect the repaired figure.
- [ ] Run focused tests and push the repair.

## Full Remaining Implementation List

### 1. Bifiltration Guarantees

- [ ] Guarantee every nonempty reasoning trajectory emits grades over `F2[x_level,x_radius]`.
- [ ] Remove missing-bifiltration paths except empty/invalid trajectory errors.
- [ ] Add tests proving radius grades and reasoning-level grades are present for every nonempty trajectory.
- [ ] Ensure all reasoning-step complexes begin as disjoint embedding vertices and grow by radius.

### 2. Actual 2-Parameter Persistence

- [ ] Compute grid fibers `K_(level,radius)` directly from GUDHI simplex trees and stored bifiltration grades.
- [ ] Compute structure maps between adjacent lattice fibers along `x_level` and `x_radius`.
- [ ] Render rank-invariant samples and Betti surfaces as lattice modules over `F2[x_level,x_radius]`.
- [ ] Add downloadable JSON with fiber bases, structure maps, ranks, grades, and provenance.
- [ ] Add a clearer 3D lattice plot with low separated module layers when multiple modules overlap, preserving fiber rank as z.

### 3. Review and Implement `references/2210.11433v1.pdf`

- [ ] Extract and read `./references/2210.11433v1.pdf` in full.
- [ ] Summarize methods relevant to multiparameter persistence modules, fitting ideals, Buchsbaum-Eisenbud multipliers, minors, resolutions, and module invariants.
- [ ] Write `planning/tropicalgt_i_2210_11433_cas_methodology_review.md` with exact transferable constructions.
- [ ] Implement applicable CAS computations from the paper for TropicalGT-I modules.
- [ ] Add tests comparing computed minors/Fitting ideals/BE diagnostics on small known bivariate modules.
- [ ] Render those objects in research-figure style with Macaulay2-like tables and diagrams.

### 4. CAS Integration: Only Real Resolutions

- [ ] Detect Macaulay2, SageMath, Singular, and optional `amelie-iska/BEMultipliers` availability on `iska`.
- [ ] Implement a CAS bridge module with strict provenance and no fabricated algebra.
- [ ] Compute minimal multigraded free resolutions over `F2[x_level,x_radius]` when CAS is available.
- [ ] Compute Betti tables, differential matrices, multidegree shifts, Fitting ideals, minors, Buchsbaum-Eisenbud diagnostics, and exactness/minimality certificates.
- [ ] Render unavailable only when the real CAS computation cannot run, and make the missing dependency/action explicit.
- [ ] Add tests with known monomial ideals: singleton generator, two-generator staircase, three-generator staircase, and a nontrivial adjacent-LCM syzygy case.

### 5. Derived and Analogical Maps

- [ ] Base analogical retrieval on model-predicted probability vectors and Jensen-Shannon distance, not arbitrary embedding-only assignment.
- [ ] Compute vertex assignments between query and memory filtered complexes using probability-vector optimal assignment.
- [ ] Validate simplicial map preservation on vertices, edges, faces, simplex-tree inclusions, and bifiltration grades.
- [ ] Compare free resolutions and derived objects only from certified CAS output.
- [ ] Remove contradictory outputs such as high derived similarity with zero resolution similarity unless a mathematically explicit reason is displayed.
- [ ] Retrieve many top-k analogies when memory is active; render insufficient-memory states early in training.

### 6. Simplex Trees and Simplicial Complexes

- [ ] Render actual GUDHI simplex-tree/trie or face-coface poset structure, not disconnected columns.
- [ ] Include every reasoning step's filtered complex and simplex tree.
- [ ] Add dotted directed edges for causal structure, decoding order, or both when they align.
- [ ] Use solid edges/faces filled in by radius slider for simplicial objects.
- [ ] Fix slider direction everywhere to min-to-max radius.
- [ ] Remove duplicate/extraneous panels where two panels show the same reasoning step.

### 7. NLL / Fitness / Density Landscapes

- [ ] Replace flat triangular NLL surfaces with a real 3D PCA NLL density cloud around actual trajectory/token embeddings.
- [ ] Generate local Gaussian support vectors around actual embeddings only for density estimation; do not render support vectors as model states.
- [ ] Color density by local NLL/fitness and keep actual model states/trajectory vertices visible and anchored.
- [ ] Retain a 2D surface when it genuinely interpolates around the trajectory neighborhood.
- [ ] Show edge-wise NLL improvement and terminal improvement statistics.

### 8. Tropical Support and Wall Crossing

- [ ] Make support heatmaps interpretable with grouped token labels, top-support summaries, margin profiles, collapse diagnostics, and wall-hit context.
- [ ] Audit wall-hit rate definition and explain when low wall-crossing is mathematically expected versus a metric issue.
- [ ] Fix/rename negative `tropical_margin_loss` so sign and objective direction are clear.
- [ ] Investigate rising `certificate_loss` and separate real certificate loss from diagnostic penalties.

### 9. GraphCG Full-Rank Visuals and Metrics

- [ ] Ensure GraphCG directions are full rank relative to embedding dimension.
- [ ] Improve direction heatmaps, spectra, candidate activity, and signed bias plots with readable layouts.
- [ ] Log GraphCG metrics in priority order for W&B and browser audit.
- [ ] Add tests that direction-rank configuration matches embedding dimension.

### 10. Decoding and Dataset Graph Causality

- [ ] Keep meet-in-the-middle decoding behind a config toggle.
- [ ] For causal DAGs, decode using forward plus reverse causal directions.
- [ ] For cyclic/noncausal graphs, use ROAR/random-order autoregressive decoding.
- [ ] Annotate dataset graphs that should have causal structure; preserve noncausal/cyclic graphs correctly.
- [ ] Add dotted decoding/causal edges to relevant visualizations.

### 11. Tropical/Toric Embedding Research and Paper Updates

- [ ] Research Macaulay2 `Tropical` package, Sage tropical polynomial/variety APIs, Maclagan tropical schemes in toric varieties, and `references/1710.10651v2.pdf`.
- [ ] Implement toric embeddings only where the embedding is mathematically correct and tool-backed.
- [ ] Use "one dimensional cone(s)" terminology in the paper/plans where rays/cone generators are meant.
- [ ] Add scheme/sheaf-theoretic and vector-bundle material to `TropicalGT-I/assets/tropicalgt_neurips_research_paper.tex` as implementations mature.
- [ ] Distinguish scoped monomial-ideal exponent-chart certificates from full tropical-variety embeddings into toric varieties.

### 12. Training, BPB, and W&B

- [ ] Keep the current BPB run alive until the 5K gate unless explicitly restarted.
- [ ] Optimize for BPB with the advanced losses/metrics that are actually implemented.
- [ ] Clean old local/online W&B runs and generated artifacts when they become irrelevant.
- [ ] Generate periodic visual audits every 250 steps.
- [ ] Restart only after the requested 5K gate or explicit user request.

### 13. Docs, README, Tests, Push

- [ ] Update `README.md` with current training, eval, inference, visualization, CAS, and browser audit commands.
- [ ] Update planning docs after each completed repair.
- [ ] Run focused pytest before each push.
- [ ] Keep browser open and visible during QA.
- [ ] Commit and push source/planning/docs changes to the non-main branch.


## 2026-06-14 Live Addendum: Current Implementation List

Current sequential order after browser review:

1. Finish the 2-parameter module figure repair: visually separated H0/H1 module layers, actual fiber rank on z, staircase panel as an exponent-lattice ideal diagram, and scoped coordinate exponent semigroup terminology.
2. Regenerate the browser bundle and visually check `two_parameter_bifiltration.html` in `127.0.0.1:8991`.
3. Review `references/2210.11433v1.pdf` in full and write `planning/tropicalgt_i_2210_11433_cas_methodology_review.md` before adding any further CAS claims.
4. Implement the paper-derived CAS computations item-by-item: Fitting ideals, minors, Buchsbaum-Eisenbud multiplier inputs/diagnostics, exact minimal resolutions where the CAS certificate exists, and explicit unavailable states otherwise.
5. Repair analogical maps so derived/resolution similarity is computed only from certified algebraic objects and probability/Jensen-Shannon filtered-complex assignments.
6. Repair NLL density and landscape plots: 3D PCA Gaussian NLL density around actual embeddings, optional 2D surfaces only when anchored and locally supported.
7. Repair simplex-tree visualizations and all radius/decoding sliders; add dotted decoding/causal edges and solid radius-gated simplicial edges/faces.
8. Reconcile training state: the current process table shows browser servers but no visible `train_tropicalgt_i.py` process, so the next training step must explicitly start or resume a BPB run and record its PID/log/W&B id before further claims of live training.
9. Update README and push the non-main branch after tests and browser QA.

Toric wording correction: the current bivariate monomial staircase is a scoped coordinate exponent semigroup chart `Spec F2[x_level,x_radius]`. It is not yet a full Maclagan-style tropical variety embedded into a toric variety. Full toric-variety embeddings remain a separate implementation item requiring certified fan/one dimensional cone data and compatible Sage/Macaulay2/polymake-style output.

## 2026-06-14 Current Repair Inventory Link

The live item-by-item repair inventory is maintained in `planning/tropicalgt_i_current_repair_inventory_2026-06-14.md`. The current linear order is: finish the 2-parameter bifiltration visualization, review `references/2210.11433v1.pdf`, implement certified CAS computations, then repair derived analogical maps, simplex trees, NLL density, tropical support, GraphCG, decoding, docs, tests, and push.


### Current Objective Update - Pass 4

- [x] Corrected the module-staircase orientation to match the Miller-Sturmfels monomial-ideal convention: `x_radius` runs horizontally and `x_level` runs vertically.
- [x] Added regression checks for adjacent `F2` structure maps and generated HTML language.
- [x] Browser-opened the regenerated `sample_000/trajectory_persistence/two_parameter_bifiltration.html` page on port `8991` and confirmed the corrected visible orientation strings.
- [x] Pushed this repair on the non-main branch.


### Current Objective Update - CAS BE Diagnostics Pass

- [x] Inspected the current CAS adapter and 2210.11433 methodology notes after the two-parameter bifiltration tests passed.
- [x] Added explicit tagged Buchsbaum-Eisenbud diagnostic blocks to Macaulay2, Singular, and Sage script templates.
- [x] Parsed BE diagnostics into certified CAS artifacts and surfaced them in the two-parameter CAS certificate display.
- [x] Added regression coverage proving Fitting ideals, minors, multigraded shifts, and BE diagnostic metadata are parsed from a tagged Macaulay2 certificate without implying BEMultipliers output.
- [x] Focused verification passed: algebraic persistence `15 passed`; visualization plus artifact validator `32 passed`.
- [x] Pushed this CAS diagnostic repair on the non-main branch.


### Current Objective Update - Bounded BEMultipliers Pass

- [x] Confirmed Macaulay2, Singular, Sage Python, and local BEMultipliers availability on the remote machine.
- [x] Added bounded Macaulay2 BEMultipliers execution for small certified resolutions, guarded by presentation-size/order limits.
- [x] Parsed and displayed actual `aMultiplier(1)` output shape/matrix text when the backend emits it.
- [x] Bumped the CAS adapter cache version to avoid stale certified artifacts without BE fields.
- [x] Focused verification passed: algebraic persistence `15 passed`; visualization plus artifact validator `32 passed`.
- [x] Pushed this bounded BEMultipliers repair on the non-main branch.


### Current Objective Update - Certified CAS Analogical Comparison Pass

- [x] Tightened analogical derived/free-resolution comparison so two certified CAS resolutions are not treated as safe derived-category evidence merely because both exist.
- [x] Added artifact-level comparison for ring, input hash, multigraded Betti shifts, differential summaries, Fitting ideals, minors, and Buchsbaum-Eisenbud multiplier output/status.
- [x] Updated analogical realization certificates to require `safe_for_derived_category_claims` from the CAS artifact comparison before claiming CAS-certified derived geometric realization.
- [x] Added regression coverage for matching certified CAS artifacts and mismatched certified CAS artifacts with otherwise compatible finite invariants.
- [x] Focused verification passed: visualization plus artifact validator `33 passed`; algebraic persistence `15 passed`; metrics and memory `8 passed`.
- [ ] Push this certified CAS analogical-comparison repair on the non-main branch after final diff/status review.
