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
- [x] Run focused tests and push the repair.

## Full Remaining Implementation List

### 1. Bifiltration Guarantees

- [x] Guarantee every nonempty reasoning trajectory emits grades over `F2[x_level,x_radius]`.
- [x] Remove missing-bifiltration paths except empty/invalid trajectory errors.
- [x] Add tests proving radius grades and reasoning-level grades are present for every nonempty trajectory.
- [x] Ensure all reasoning-step complexes begin as disjoint embedding vertices and grow by radius.

### 2. Actual 2-Parameter Persistence

- [x] Compute grid fibers `K_(level,radius)` directly from GUDHI simplex trees and stored bifiltration grades.
- [x] Compute structure maps between adjacent lattice fibers along `x_level` and `x_radius`.
- [x] Render rank-invariant samples and Betti surfaces as lattice modules over `F2[x_level,x_radius]`.
- [x] Add downloadable JSON with fiber bases, structure maps, ranks, grades, and provenance.
- [x] Add a clearer 3D lattice plot with low separated module layers when multiple modules overlap, preserving fiber rank as z.

### 3. Review and Implement `references/2210.11433v1.pdf`

- [x] Extract and read `./references/2210.11433v1.pdf` in full.
- [x] Summarize methods relevant to multiparameter persistence modules, fitting ideals, Buchsbaum-Eisenbud multipliers, minors, resolutions, and module invariants.
- [x] Write `planning/tropicalgt_i_2210_11433_cas_methodology_review.md` with exact transferable constructions.
- [x] Implement applicable CAS computations from the paper for TropicalGT-I modules.
- [x] Add tests comparing computed minors/Fitting ideals/BE diagnostics on small known bivariate modules.
- [x] Render those objects in research-figure style with Macaulay2-like tables and diagrams.

### 4. CAS Integration: Only Real Resolutions

- [x] Detect Macaulay2, SageMath, Singular, and optional `amelie-iska/BEMultipliers` availability on `iska`.
- [x] Implement a CAS bridge module with strict provenance and no fabricated algebra.
- [x] Compute minimal multigraded free resolutions over `F2[x_level,x_radius]` when CAS is available.
- [x] Compute Betti tables, differential matrices, multidegree shifts, Fitting ideals, minors, Buchsbaum-Eisenbud diagnostics, and exactness/minimality certificates.
- [x] Render unavailable only when the real CAS computation cannot run, and make the missing dependency/action explicit.
- [x] Add tests with known monomial ideals: singleton generator, two-generator staircase, three-generator staircase, and a nontrivial adjacent-LCM syzygy case.

### 5. Derived and Analogical Maps

- [x] Base analogical retrieval on model-predicted probability vectors and Jensen-Shannon distance, not arbitrary embedding-only assignment.
- [x] Compute vertex assignments between query and memory filtered complexes using probability-vector optimal assignment.
- [x] Validate simplicial map preservation on vertices, edges, faces, simplex-tree inclusions, and bifiltration grades.
- [x] Compare free resolutions and derived objects only from certified CAS output.
- [x] Remove contradictory outputs such as high derived similarity with zero resolution similarity unless a mathematically explicit reason is displayed.
- [x] Retrieve many top-k analogies when memory is active; render insufficient-memory states early in training.

### 6. Simplex Trees and Simplicial Complexes

- [x] Render actual GUDHI simplex-tree/trie or face-coface poset structure, not disconnected columns.
- [x] Include every reasoning step's filtered complex and simplex tree.
- [x] Add dotted directed edges for causal structure, decoding order, or both when they align.
- [x] Use solid edges/faces filled in by radius slider for simplicial objects.
- [x] Fix slider direction everywhere to min-to-max radius.
- [x] Remove duplicate/extraneous panels where two panels show the same reasoning step.

### 7. NLL / Fitness / Density Landscapes

- [x] Replace flat triangular NLL surfaces with a real 3D PCA NLL density cloud around actual trajectory/token embeddings.
- [x] Generate local Gaussian support vectors around actual embeddings only for density estimation; do not render support vectors as model states.
- [x] Color density by local NLL/fitness and keep actual model states/trajectory vertices visible and anchored.
- [x] Retain a 2D surface when it genuinely interpolates around the trajectory neighborhood.
- [x] Show edge-wise NLL improvement and terminal improvement statistics.

### 8. Tropical Support and Wall Crossing

- [x] Make support heatmaps interpretable with grouped token labels, top-support summaries, margin profiles, collapse diagnostics, and wall-hit context.
- [x] Audit wall-hit rate definition and explain when low wall-crossing is mathematically expected versus a metric issue.
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
- [x] Pushed this certified CAS analogical-comparison repair on the non-main branch as commit `1a21a74`.


### Current Objective Update - Certified CAS Retrieval Scoring Pass

- [x] Extended `AnalogicalMemoryBank.retrieve` with a certificate-gated `certified_cas_evidence` score component.
- [x] Retrieval CAS scoring now contributes only for exact artifact-level matches between query and memory certified real-free-resolution evidence: ring, input hash, stable artifact hash, multigraded Betti shifts, differential summaries, Fitting ideals, minors, and Buchsbaum-Eisenbud multiplier output/status.
- [x] Mismatched or unavailable CAS evidence is still serialized for audit with explicit reasons and mismatch components, but its score contribution is `0.0` and it does not assert derived-category equivalence.
- [x] Added memory regression coverage for matching certified CAS evidence, mismatched certified CAS evidence, and unavailable query evidence.
- [x] Focused verification passed: metrics and memory `9 passed`; visualization plus artifact validator `33 passed`; algebraic persistence `15 passed`.
- [x] Pushed this certified CAS retrieval-scoring repair on the non-main branch.


### Current Objective Update - SimplexTree Face/Coface Contract Pass

- [x] Confirmed the graph-of-thought SimplexTree page renders actual face-to-coface Hasse covers plus optional sorted-label trie-prefix links instead of disconnected simplex columns.
- [x] Updated the Plotly title contract to state that actual cover edges are primary and trie links are legend-only.
- [x] Added regression checks requiring `actual face-to-coface covers`, `optional sorted-label trie prefix links`, `not disconnected simplex columns`, and the empty-simplex root on the full trajectory SimplexTree page.
- [x] Preserved the Jensen-Shannon probability SimplexTree unavailable state when model probability vectors are absent.
- [x] Focused verification passed: visualization plus artifact validator `33 passed`.
- [x] Pushed this simplex-tree contract repair on the non-main branch.


### Current Objective Update - 2210 Methodology And Known Monomial Tests Pass

- [x] Refreshed the 2210.11433 CAS methodology review with the current backend state: Macaulay2 and Singular detected, Sage not on the active shell PATH, and BEMultipliers cloned as a diagnostic layer only.
- [x] Added exact known-ideal regression coverage for singleton, two-generator, and three-generator bivariate monomial staircase resolutions, including adjacent-LCM first syzygies and one dimensional cone language.
- [x] Marked the 2210 review/implementation checklist complete for the paper-derived Fitting/minor/BE/rank-invariant objects already wired into the certified CAS and visualization path. Remaining CAS hardening stays in section 4.
- [x] Focused verification passed: known monomial-ideal test `1 passed`; full algebraic persistence `21 passed`; full simplicial visualization `33 passed`; artifact validator `10 passed`.


### Current Objective Update - CAS Backend Probe Contract Pass

- [x] Added regression coverage for the real CAS backend probe contract so Macaulay2, Singular, and Sage availability is compared against the adapter's deterministic executable detector, including fixed remote paths such as `/usr/bin/M2` and `/usr/bin/Singular`.
- [x] Confirmed the BEMultipliers probe remains a diagnostic-layer report only: it is not treated as a resolution backend and its policy explicitly says never to substitute multiplier output for a free-resolution certificate.
- [x] Marked the section-4 backend-detection item complete while keeping broader CAS bridge, multigraded-resolution, certificate-content, and unavailable-rendering hardening items open for their own tested passes.
- [x] Focused verification passed: probe test `1 passed`; full algebraic persistence `22 passed`; full simplicial visualization `33 passed`; artifact validator `10 passed`.


### Current Objective Update - Strict CAS Bridge Provenance Pass

- [x] Fixed the failed-certificate path so unavailable CAS reports preserve the already-canonical module schema, original `input_sha256`, generator count, boundary count, and command templates instead of recanonicalizing an empty raw-module shell.
- [x] Added a regression proving a backend run with missing/false exactness certification returns `certificate_failed`, attaches no CAS artifacts, leaves `certificate_attached=false`, and keeps the original module provenance intact.
- [x] This completes the strict bridge/provenance checklist item: certified paths may expose CAS artifacts, while failed or unavailable paths expose only reasoned unavailable state, backend attempts, probes, templates, and module provenance.
- [x] Focused verification passed: failed-certificate provenance test `1 passed`; full algebraic persistence `23 passed`; full simplicial visualization `33 passed`; artifact validator `10 passed`.


### Current Objective Update - Explicit Macaulay2 Multigraded Resolution Pass

- [x] Updated the Macaulay2 bridge to construct explicit shifted free modules `F0` and `F1` from the canonical presentation's stored row/column generator multidegrees before computing `res coker M`.
- [x] Added a Macaulay2 homogeneity gate: if the presentation matrix is not homogeneous for the stored bifiltration degrees, the backend emits an uncertified response instead of producing a multigraded free-resolution claim.
- [x] Updated the live CAS smoke fixture to a homogeneous `F2[x_level,x_radius]` persistence presentation and asserted the certified multigraded Betti rows carry the expected bidegrees `(0,1)`, `(1,0)`, and `(1,1)`.
- [x] Added a no-claim regression proving a nonhomogeneous stored grading returns `certificate_failed` with no CAS artifacts.
- [x] Focused verification passed: real CAS smoke `1 passed`; nonhomogeneous rejection `1 passed`; full algebraic persistence `24 passed`; full simplicial visualization `33 passed`; artifact validator `10 passed`.


### Current Objective Update - Structured CAS Certificate Content Pass

- [x] Added `cas_artifacts.certificate_summary` to certified CAS outputs with backend, certificate type, attached/exact/minimal flags, homogeneity status, module hash, grading-safe render flags, backend-attempt count, and no-proxy policy.
- [x] Kept Betti rows, free modules, differential matrices, multidegree shifts, Fitting ideals, determinantal minors, Buchsbaum-Eisenbud rank diagnostics, and BEMultipliers diagnostics as structured CAS artifacts rather than raw-text-only evidence.
- [x] Passed `certificate_summary` through the two-parameter visualization adapter and certificate table so browser/audit payloads can disclose exactness/minimality evidence directly.
- [x] Focused verification passed: CAS parser/smoke `2 passed`; certified visualization table `1 passed`; full algebraic persistence `24 passed`; full simplicial visualization `33 passed`; artifact validator `10 passed`.


### Current Objective Update - Explicit CAS Unavailable Rendering Pass

- [x] Added structured `unavailable_diagnostic` and `unavailable_dependency_action` fields to uncertified CAS reports, with status-specific actions for missing backends, disabled execution, complexity guards, timeouts, invalid gradings, unsupported rings, and failed certificates.
- [x] The unavailable diagnostic states available backend names, backend attempt statuses, whether BEMultipliers is a resolution backend, safe unavailable rendering, and the no-proxy policy forbidding chain/rank/Fitting/minor/BEMultipliers substitution for a free resolution.
- [x] Added a deterministic no-backend regression proving `backend_not_installed` reports an install/activation action and remains renderable only as an unavailable diagnostic.
- [x] Focused verification passed: no-backend unavailable diagnostic `1 passed`; full algebraic persistence `24 passed`; full simplicial visualization `33 passed`; artifact validator `10 passed`.


### Current Objective Update - Derived Analogical Maps Completion Audit

- [x] Confirmed `AnalogicalMemoryBank.retrieve` uses retrieval-side `probability_simplicial_map_diagnostics`, model-probability Jensen-Shannon distances, and positive probability-map score only when the assignment extends to a filtration-preserving simplex-tree map.
- [x] Confirmed vertex assignment is solved over probability vectors, with edge and 2-simplex preservation counts, simplex-tree preservation rate, chain-map diagnostics, persistence-module morphism diagnostics, and transported persistence-landscape diagnostics serialized into retrieval rows.
- [x] Confirmed certified CAS evidence contributes to retrieval score only for exact certified real-free-resolution artifact matches; mismatched or unavailable CAS evidence is serialized with reasons and contributes `0.0`.
- [x] Confirmed the browser analogical-memory pages render top-k probability correspondences and explicit unavailable states for missing retrieval certificates, missing query probabilities, and no retrieved memories.
- [x] Verification passed: metrics/memory `11 passed`; full simplicial visualization `33 passed`; artifact validator `10 passed`.


### Current Objective Update - Rank-Invariant Table And 2-Parameter Persistence Pass

- [x] Added a secondary rank-invariant table to the two-parameter bifiltration page, backed only by computed `rank_invariant_samples` over `F2[x_level,x_radius]`.
- [x] Extended the visual payload with `rank_invariant_samples_table` and `rank_invariant_sample_count` so downstream validators and reviewers can audit this secondary view separately from the primary Miller-Sturmfels staircase.
- [x] Confirmed the full actual 2-parameter persistence section is covered by existing/new tests: GUDHI simplex-tree fiber provenance, adjacent structure maps, fiber bases/ranks/grades, downloadable JSON payload, rank-invariant samples, and the secondary 3D rank lattice with H0/H1 offsets.
- [x] Focused verification passed: bifiltration page test `1 passed`; full algebraic persistence `20 passed`; full simplicial visualization `33 passed`; artifact validator `10 passed`.


### Current Objective Update - Missing-Bifiltration Unavailable-State Pass

- [x] Materialized explicit unavailable `F2[x_level,x_radius]` bifiltration reports for legacy nonempty scaling payloads that have candidates but lack `trajectory_growth`.
- [x] Added regression coverage proving the raw sidecar is written, keeps the ring/parameter contract, and reports the exact nonempty-scaling missing-growth reason without creating fibers, ranks, maps, or synthetic edges.
- [x] Focused verification passed: new visualization writer test `1 passed`; full simplicial visualization `33 passed`; full algebraic persistence `20 passed`; artifact validator `10 passed`.


### Current Objective Update - Bifiltration Vertex-Only Start-State Pass

- [x] Updated the inference-scaling bifiltration policy so probability Jensen-Shannon radius complexes remain selected for nonempty vertex-only start states when every vertex has a real model probability vector.
- [x] Added regression coverage for a depth-0 trajectory proving the probability complex is a real single-vertex radius filtration, the `F2[x_level,x_radius]` bifiltration is available, and the selected object key is `probability_filtered_simplicial_object`.
- [x] Focused verification passed: new algebraic persistence test `1 passed`; full algebraic persistence `20 passed`; full simplicial visualization `32 passed`; artifact validator `10 passed`.


### Current Objective Update - Certified CAS Diagnostic Table Pass

- [x] Split the two-parameter bifiltration certified algebra disclosure into separate secondary tables for Betti-style rows, certified free modules, differentials, structured Fitting/minor diagnostics, Buchsbaum-Eisenbud rank/multiplier diagnostics, and CAS certificate summaries.
- [x] Required explicit structured `ideal_diagnostics` before rendering Fitting or determinantal-minor rows; legacy raw `fitting_ideals` and `minors` keys no longer substitute for that certificate.
- [x] Required explicit BE/certified-resolution evidence before rendering Buchsbaum-Eisenbud rows; missing evidence renders unavailable rather than defaulting to false exactness/minimality.
- [x] Focused verification passed: new CAS table helper test `1 passed`; bifiltration artifact test `1 passed`; full simplicial visualization `32 passed`; algebraic persistence `19 passed`; artifact validator `10 passed`.
- [x] Pushed this certified CAS diagnostic table repair on the non-main branch as commit `e70dbc8`.


### Current Objective Update - Duplicate Reasoning-Step Panel Removal Pass

- [x] Removed the duplicate static preview block from pages that already render the interactive selected filtered-complex panel, so the same reasoning-step complex is not shown in two adjacent side-panel surfaces.
- [x] Preserved the static same-payload preview path only for pages that explicitly disable the interactive selected-complex panel, keeping that path distinct rather than duplicative.
- [x] Updated regression coverage for validation PCA/NLL pages, GoT trajectory pages, and the static-preview-only WebGL path.
- [x] Section 6 is complete: actual GUDHI SimplexTree/face-coface views, per-step complex/tree pages, dotted causal/decoding overlays, solid radius edges, filled radius-gated 2-simplices, min-to-max sliders, and duplicate-panel removal are all covered.
- [x] Verification passed: focused duplicate-panel tests `3 passed`; full simplicial visualization `34 passed`; artifact validator `10 passed`; compile checks passed for modified source/tests.


### Current Objective Update - SimplexTree No-Fallback And Radius Slider Pass

- [x] Hardened the GUDHI-unavailable path so SimplexTree pages render an explicit `unavailable_gudhi_simplex_tree` state and do not display raw JSON simplex rows as a trie, face-coface poset, or SimplexTree substitute.
- [x] Added no-proxy/no-fallback certificate fields to the visualization and core simplicial serializers when a real GUDHI SimplexTree cannot be built.
- [x] Renamed the provenance registry entry from a JSON fallback to an explicit GUDHI unavailable state.
- [x] Confirmed every observed GoT reasoning step is linked to its own radius-filtered complex page and its own simplex-tree page through the step manifest.
- [x] Confirmed dotted causal/decoding overlays are attached from actual decoding-order reports and GoT parent-child order, while solid radius edges and filled radius-gated 2-simplices remain the simplicial objects controlled by the min-to-max radius slider.
- [x] Verification passed: full simplicial visualization `34 passed`; artifact validator `10 passed`; reasoning trajectory growth test `1 passed`; compile checks passed for modified source/tests.


### Current Objective Update - NLL Density Contract Pass

- [x] Confirmed the graph-of-thought audit writes `got_nll_density_cloud_pca_3d.html` and `got_nll_density_cloud_payload.json` from actual model graph-state PCA anchors and measured raw NLL values.
- [x] Added regression coverage proving Gaussian support samples are marked as visualization-only density samples, not model states.
- [x] Regression now requires exact model anchors, kernel-weighted local NLL metadata, density-volume provenance, and edge-wise NLL deltas.
- [x] Focused verification passed: visualization plus artifact validator `33 passed`.
- [x] Pushed this NLL density contract repair on the non-main branch.

### Current Objective Update - Tropical Support Grouped Labels Pass

- [x] Added grouped token summaries to the tropical support payload using only model graph-token trace fields: token kind plus explicit node type, edge type, active-support kind, or label when present.
- [x] Added a top-support summary with selected-query count, capture rate, mean selected margin, support group, and model support-probability mean when that probability was logged by the model.
- [x] Exposed grouped token labels, top-support group, collapse diagnostics, margin profile, and strict/near wall-hit context in both the observed-support matrix view and high-collapse diagnostic view.
- [x] Kept support heatmaps as observed assignment matrices only: yellow cells are selected model supports, confidence remains in model probability summaries and selected-margin profiles, and no proxy scores or fallback support objects are introduced.
- [x] Verification passed: focused tropical support tests `2 passed`; full simplicial visualization `34 passed`; artifact validator `10 passed`; compile checks passed for modified source/tests.

### Current Objective Update - Wall-Hit Metric Scope Pass

- [x] Reclassified tropical wall-hit reporting as a margin-threshold wall audit, not a certified normal-fan wall-crossing count.
- [x] Added machine-readable wall-audit scope, strict/near-wall definitions, low-strict interpretation status, interpretation text, and `metric_issue` status to `tropical_support_payload.json`.
- [x] Fixed threshold selection so an explicit `0.0` strict wall threshold is preserved instead of being replaced by the default threshold.
- [x] Updated validator requirements so current interactive audit artifacts must include the no-proxy metric scope and low-strict interpretation whenever wall-margin audit data is present.
- [x] Verification passed: focused tropical support wall tests `3 passed`; full simplicial visualization `35 passed`; artifact validator `10 passed`; compile checks passed for modified source/tests/validator.
