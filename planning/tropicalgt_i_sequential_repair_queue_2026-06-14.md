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
- [x] Fix/rename negative `tropical_margin_loss` so sign and objective direction are clear.
- [x] Investigate rising `certificate_loss` and separate real certificate loss from diagnostic penalties.

### 9. GraphCG Full-Rank Visuals and Metrics

- [x] Ensure GraphCG directions are full rank relative to embedding dimension.
- [x] Improve direction heatmaps, spectra, candidate activity, and signed bias plots with readable layouts.
- [x] Log GraphCG metrics in priority order for W&B and browser audit.
- [x] Add tests that direction-rank configuration matches embedding dimension.

### 10. Decoding and Dataset Graph Causality

- [x] Keep meet-in-the-middle decoding behind a config toggle.
- [x] For causal DAGs, decode using forward plus reverse causal directions.
- [x] For cyclic/noncausal graphs, use ROAR/random-order autoregressive decoding.
- [x] Annotate dataset graphs that should have causal structure; preserve noncausal/cyclic graphs correctly.
- [x] Add dotted decoding/causal edges to relevant visualizations.

### 11. Tropical/Toric Embedding Research and Paper Updates

- [x] Research Macaulay2 `Tropical` package, Sage tropical polynomial/variety APIs, Maclagan tropical schemes in toric varieties, and `references/1710.10651v2.pdf`.
- [x] Implement toric embeddings only where the embedding is mathematically correct and tool-backed.
- [x] Use "one dimensional cone(s)" terminology in the paper/plans where rays/cone generators are meant.
- [x] Add scheme/sheaf-theoretic and vector-bundle material to `TropicalGT-I/assets/tropicalgt_neurips_research_paper.tex` as implementations mature.
- [x] Distinguish scoped monomial-ideal exponent-chart certificates from full tropical-variety embeddings into toric varieties.

### 12. Training, BPB, and W&B

- [x] Keep the current BPB run alive until the 5K gate unless explicitly restarted.
- [ ] Optimize for BPB with the advanced losses/metrics that are actually implemented.
- [x] Audit old local W&B runs and generated artifacts; only cache directories were deleted automatically, while large old run outputs/checkpoints were preserved as provenance until explicit cleanup approval.
- [x] Generate periodic visual audits every 250 steps.
- [x] Restart only after the requested 5K gate or explicit user request; the 5K review currently blocks restart because the checkpoint evidence is unavailable.

### 13. Docs, README, Tests, Push

- [x] Update `README.md` with current training, eval, inference, visualization, CAS, and browser audit commands.
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

### Current Objective Update - Tropical Margin Loss Sign Pass

- [x] Changed ambiguous `tropical_margin_loss` telemetry to the nonnegative margin shortfall loss, matching the ordinary expectation that a metric named loss is nonnegative.
- [x] Kept the reward-maximizing training regularizer explicit as `tropical_margin_signed_loss` and `tropical_margin_signed_objective`, with `tropical_margin_reward = -tropical_margin_signed_objective`.
- [x] Added separate weighted telemetry for `loss_tropical_margin_signed_weighted` and `loss_tropical_margin_shortfall_weighted`; `loss_margin_weighted` remains as the legacy signed-objective alias for continuity.
- [x] Updated W&B priority grouping and tests so dashboards can distinguish the signed objective direction from the nonnegative shortfall diagnostic.
- [x] Verification passed: model/loss tests `9 passed`; training-metrics tests `12 passed`; focused model and W&B checks passed; compile checks passed for modified source/tests.

### Current Objective Update - Certificate Loss Decomposition Pass

- [x] Kept `certificate_loss` as the real negative-log allowed-support objective, with `certificate_objective_loss` carrying the same value explicitly.
- [x] Added `certificate_diagnostic_penalty` and `certificate_loss_reconstruction_error` so any future diagnostic penalty cannot be silently mixed into the certificate objective. Current diagnostic penalty is zero by construction.
- [x] Added certificate valid-token count and allowed-target count summaries to explain rising certificate loss through real target-set structure rather than hidden penalties.
- [x] Added weighted telemetry for `loss_certificate_objective_weighted` and `loss_certificate_diagnostic_penalty_weighted`; the legacy `loss_certificate_weighted` remains the objective-weight alias.
- [x] Section 8 is complete: tropical support heatmaps, wall-hit scope, margin-loss sign, and certificate-loss decomposition are all implemented and tested.
- [x] Verification passed: model/loss tests `9 passed`; training-metrics tests `12 passed`; focused certificate/W&B checks passed; compile checks passed for modified source/tests.

### Current Objective Update - GraphCG Embedding-Rank Direction Bank Pass

- [x] Model-level GraphCG now uses an effective direction bank with exactly `dim` directions, clamping undersized or oversized `graphcg_num_directions` requests to the embedding dimension while preserving the requested count in telemetry.
- [x] Increased the exact Stiefel/QR projection cap so the live `1760 x 1760` direction bank is projected through the full-rank effective basis rather than normalized raw rows.
- [x] Added GraphCG telemetry for requested directions, effective directions, embedding-span rank target, embedding-span full-rank flag, and whether the bank was clamped to the embedding dimension.
- [x] Updated W&B priority grouping and regression tests so the embedding-rank invariant is visible in local history and dashboard metrics.
- [x] Verification passed: focused GraphCG rank tests `4 passed`; training priority checks `2 passed`; full model/loss tests `10 passed`; training-metrics tests `12 passed`; compile checks passed for modified source/tests.

### Current Objective Update - GraphCG Readability Contract Pass

- [x] Added a machine-readable readability contract to the GraphCG trajectory payload proving the page contains four coordinated panels: all-direction heatmap, full-rank activity spectrum, candidate activity by observed GoT state, and direction signed-bias scatter.
- [x] The payload now records panel names/count, per-panel availability flags, and `directions_sampled_for_heatmap=false`; exact direction ids remain available in hover text and JSON.
- [x] Strengthened visualization regression coverage for all four visible panel titles plus candidate effective-direction and signed-mean arrays.
- [x] Verification passed: focused GraphCG visualization test `1 passed`; full simplicial visualization `35 passed`; artifact validator `10 passed`; compile checks passed for modified visualization source/test.

### Current Objective Update - GraphCG Metric Priority Pass

- [x] Reordered W&B GraphCG priority metrics so embedding-span full-rank status, direction-bank clamp status, requested/effective direction counts, and rank target are emitted before lower-level rank and singular-value diagnostics.
- [x] Added a browser metric priority tuple used by `training_metrics.html`, covering the same real GraphCG rank facts plus active-direction and singular-value diagnostics without proxy values.
- [x] Added regression coverage proving the W&B payload order and generated browser Plotly traces expose the GraphCG rank-audit metrics in the intended priority order.
- [x] Section 9 is complete: embedding-rank direction-bank enforcement, readable GraphCG trajectory panels, priority GraphCG metrics, and direction-rank configuration tests are all implemented.
- [x] b60 latest checked training state reached step `1542` with train loss/NLL `1.225/1.202`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- [x] Verification passed: compile checks for modified run/visualization/training-metrics files; focused GraphCG metric-order tests `2 passed`; full training-metrics tests `13 passed`; full simplicial visualization tests `35 passed`.

### Current Objective Update - Meet-In-The-Middle Toggle Pass

- [x] Hardened `meet_in_middle_config` so boolean config toggles are explicit: `true` enables zero-weight shared-parser defaults, `false` disables, and non-dict/non-bool values remain disabled.
- [x] Disabled meet-in-the-middle paths now emit explicit no-proxy reasons (`disabled_by_config`) and do not call the reverse-pass model when the toggle is off.
- [x] `evaluate_model`, `eval_tropicalgt_i.py`, and `infer_tropicalgt_i.py` now normalize meet-in-the-middle settings through the shared parser instead of silently treating boolean launch artifacts as empty configs.
- [x] b60 latest checked training state reached step `1732` with train loss/NLL `1.239/1.215`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- [x] Verification passed: compile checks for modified decoder/run/scripts/tests; `pytest TropicalGT-I/tests/test_meet_in_middle_decoding.py -q` (`10 passed`); `pytest TropicalGT-I/tests/test_training_metrics.py -q` (`13 passed`).

### Current Objective Update - Causal Forward/Reverse Decoding Pass

- [x] Confirmed causal DAG records emit both `decoding_order_kind=causal_dag` and `decoding_reverse_order_kind=reverse_causal_dag`, with reverse node order derived by reversing the topological order.
- [x] Added full-path meet-in-the-middle regression coverage for an explicit three-node causal DAG, proving the MIM report uses the forward causal context and reverse causal context under graph autoregressive decoding.
- [x] b60 latest checked training state reached step `1750` with train loss/NLL `1.151/1.128`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- [x] Verification passed: compile check for `test_meet_in_middle_decoding.py`; `pytest TropicalGT-I/tests/test_meet_in_middle_decoding.py -q` (`11 passed`).

### Current Objective Update - ROAR Random-Order Decoding Pass

- [x] Confirmed cyclic/noncausal graph records emit `decoding_order_kind=random_autoregressive` and `decoding_reverse_order_kind=reverse_random_autoregressive`, rather than pretending the graph is a causal DAG.
- [x] Added full-path meet-in-the-middle regression coverage for a cyclic graph, proving the report uses ROAR/random-order forward and reverse contexts under graph autoregressive decoding.
- [x] b60 latest checked training state remained at the latest parsed step `1750` with train loss/NLL `1.151/1.128`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- [x] Verification passed: compile check for `test_meet_in_middle_decoding.py`; `pytest TropicalGT-I/tests/test_meet_in_middle_decoding.py -q` (`12 passed`).

### Current Objective Update - Dataset Causal Annotation Preservation Pass

- [x] Confirmed dataset graph normalization annotates causal edge types (`depends_on`, `supports_answer`, sequence edges, and related temporal/control-flow relations) while preserving explicit noncausal or undirected edges as noncausal.
- [x] Added mixed-edge regression coverage proving a graph with causal reasoning edges plus an explicit noncausal similarity edge keeps the noncausal edge, avoids causal-DAG decoding, and uses ROAR/random-order forward and reverse contexts.
- [x] b60 latest checked training state remained at the latest parsed step `1750` with train loss/NLL `1.151/1.128`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- [x] Verification passed: compile check for `test_data_loader.py`; `pytest TropicalGT-I/tests/test_data_loader.py -q` (`13 passed`, `2 warnings`); `pytest TropicalGT-I/tests/test_meet_in_middle_decoding.py -q` (`12 passed`).

### Current Objective Update - Dotted Decoding/Causal Visualization Overlay Pass

- [x] Confirmed visualization payloads keep causal graph edges, forward decoding edges, and reverse decoding edges as dotted directed overlays, not as radius-filtration simplices.
- [x] Added cyclic-graph visualization regression coverage proving ROAR forward/reverse decoding edges remain dotted in the overlay payload and browser plot payload.
- [x] Section 10 is complete: meet-in-the-middle toggle semantics, causal forward/reverse decoding, ROAR/random-order decoding, dataset causal/noncausal preservation, and dotted visualization overlays are covered.
- [x] b60 latest checked training state reached step `1769` with train loss/NLL `1.174/1.150`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- [x] Verification passed: compile check for `test_simplicial_visualization.py`; focused decoding overlay tests `2 passed`; `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`36 passed`).

### Current Objective Update - Tropical/Toric Research Contract Pass

- [x] Reviewed the local `references/1710.10651v2.pdf` text and current official Macaulay2/Sage tropical APIs for Section 11 scope.
- [x] Recorded the strict certificate contract: Macaulay2 `Tropical` `tropicalVariety` is the fan/cycle certificate path; `isTropicalBasis` and `tropicalPrevariety` are side diagnostics; Sage tropical polynomial/variety APIs may support exact polynomial and hypersurface checks but do not replace this fan certificate.
- [x] Added machine-readable no-proxy contract fields to tropical fan diagnostics so support tokens, chain presentations, rank samples, embedding-only assignments, or visualization rows cannot be rendered as tropical fan certificates.
- [x] Added regression coverage for certified and unavailable tropical fan reports, including Sage-scope and no-proxy wording in the browser payload.
- [x] Verified the generated Plotly tropical fan diagnostic smoke artifact contains the certificate source, Sage scope, no-proxy policy, and one dimensional cone language without committing generated output.
- [x] Verification completed: py-compile for touched Python/test files, focused tropical-fan tests (`3 passed` + `2 passed`), `pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` (`24 passed`), `pytest TropicalGT-I/tests/test_simplicial_visualization.py -q` (`36 passed`), `pytest TropicalGT-I/tests/test_interactive_artifact_validator.py -q` (`10 passed`), generated HTML smoke, and `git diff --check`.
- [x] b60 latest checked training state reached step `1947` with train loss/NLL `1.129/1.105`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- [x] Next Section 11 item completed: finite exponent-matrix toric sidecars are now certified only by Macaulay2 `Quasidegrees` `toricIdeal(A,R)`, while neural chart-bundle toric metrics remain explicitly uncertified activation diagnostics.

### Current Objective Update - Tool-Backed Toric Embedding Sidecar Pass

- [x] Added `tropicalgt.cas_toric` as a real-only finite toric embedding sidecar: integer exponent matrices are canonicalized, hashed, cached, and sent to Macaulay2 `Quasidegrees` `toricIdeal(A,R)` with tagged certificate output.
- [x] Certified reports expose the finite monomial-map kernel ideal, generator text/count, codimension/dimension when Macaulay2 emits them, command template, backend probe, and a no-proxy certificate contract.
- [x] Unavailable reports remain unavailable with exact reason, backend attempts, and `safe_to_render_as_toric_embedding=false`; no Sage, chart-bundle, GraphCG, support-token, embedding, or visualization fallback is substituted.
- [x] Chart-bundle toric telemetry now carries `toric_embedding_certificate.status=uncertified_activation_chart`, making `toric_normal_fan_loss`, toric active rows, and GraphCG-cell agreement diagnostics only unless a real CAS sidecar is attached.
- [x] Live smoke certified the small exponent matrix `[[1,1,1],[0,1,2]]` as `ideal(z_1^2-z_0*z_2)` and explicitly kept `safe_to_use_as_normal_fan_certificate=false`.
- [x] Verification completed: py-compile for touched source/tests, focused toric embedding tests (`3 passed`), focused chart-bundle tests (`3 passed`), `pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` (`27 passed`), `pytest TropicalGT-I/tests/test_losses_and_model.py -q` (`10 passed`).
- [x] b60 latest checked training state remained at step `2000` with train loss/NLL `1.139/1.116`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- [x] Next Section 11 item completed: visible paper/planning prose now uses "one dimensional cone(s)" where fan rays/cone generators are meant; literal CAS method names such as `rays` remain unchanged.

### Current Objective Update - One Dimensional Cone Terminology Pass

- [x] Updated visible prose in `TropicalGT-I/assets/tropicalgt_neurips_research_paper.tex` from hyphenated one-dimensional-cone wording to the requested fan-theoretic "one dimensional cone(s)" terminology.
- [x] Updated stale planning bullets in `planning/tropicalgt_i_visual_math_repair_plan_2026-06-12.md` from `matroid/ray` and `ray-filtration` phrasing to one dimensional cone terminology.
- [x] Preserved literal CAS method/output names such as Macaulay2 `rays`, `rays(T)`, and `maxCones`; these are backend API names, not prose substitutes for the fan-theoretic language.
- [x] Verification completed: terminology audit found no visible hyphenated one-dimensional prose and no stale `ray-only`/`ray-filtration` planning phrases; lightweight TeX environment-balance check passed.
- [x] Remote `pdflatex` compile was attempted into `/tmp/tropicalgt_paper_compile`; it is blocked by missing TeX dependency `mathtools.sty`, so no PDF artifact was produced or staged.
- [x] b60 latest checked training state reached step `2005` with train loss/NLL `1.191/1.167`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- [x] Next Section 11 item completed: paper now records the certified-sidecar boundary for finite toric ideals and scheme/sheaf/vector-bundle diagnostics, with unavailable states required until exact fan, grading, module-presentation, backend-attempt, and certificate-hash data exist.

### Current Objective Update - Scheme/Sheaf Paper Scope Pass

- [x] Updated `TropicalGT-I/assets/tropicalgt_neurips_research_paper.tex` to tie finite toric sidecars to the real Macaulay2 `Quasidegrees` `toricIdeal(A,R)` certificate boundary.
- [x] Documented the current scheme/sheaf and vector-bundle implementation status: Cox-module, local-cohomology, and vector-bundle diagnostics remain unavailable unless a real CAS report supplies fan, grading, module presentation, backend attempt, and certificate hash.
- [x] Added explicit no-proxy paper language forbidding activation rows, chain ranks, support tokens, or embedding similarities from replacing missing scheme/sheaf certificates.
- [x] Normal-fan cell labels are now paper-gated behind certified fan/tropical sidecars; otherwise browser artifacts must retain `uncertified_activation_chart` status.
- [x] Verification completed for this paper-only pass: paper scope audit passed, unescaped Markdown backtick audit passed, requested no-proxy phrases are present, terminology audit passed, and TeX environment-balance audit passed.
- [x] Remote `pdflatex` compile was attempted into `/tmp/tropicalgt_paper_compile`; it remains blocked by missing TeX dependency `mathtools.sty`, so no PDF artifact was produced or staged.
- [x] b60 latest checked training state reached step `2157` with train loss/NLL `1.153/1.129`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- [x] Next Section 11 item completed: scoped exponent-chart diagnostics, finite monomial-map toric-ideal sidecars, and full tropical-variety embeddings into toric varieties are now separated in code contracts, paper prose, tests, and planning docs.

### Current Objective Update - Scoped Exponent-Chart Certificate Boundary Pass

- [x] Added machine-readable toric sidecar fields distinguishing finite monomial-map toric-ideal certificates from tropical-variety embeddings into toric varieties and global toric-variety models.
- [x] Chart-bundle metadata now marks activation charts as `uncertified_activation_chart_not_tropical_variety_embedding` and exposes false safety flags for tropical-variety/global toric-variety rendering.
- [x] Added `toric_activation_cell_margin_loss` and `loss_toric_activation_cell_margin_weighted` as correctly scoped aliases; legacy `toric_normal_fan_loss` keys remain compatibility aliases, not certificates.
- [x] Reworded the paper's trainable neural object as a scoped max-linear exponent-chart sidecar; normal-fan cell comparisons are allowed only with certified fan/tropical sidecars.
- [x] Updated planning docs to reserve toric-variety embedding language for certified finite sidecars and to label neural/runtime quantities as scoped exponent-chart diagnostics.
- [x] Verification completed: py-compile for touched source/tests, scoped exponent-chart paper audit, terminology/no-proxy grep, `git diff --check`, `pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` (`27 passed`), `pytest TropicalGT-I/tests/test_losses_and_model.py -q` (`10 passed`), and `pytest TropicalGT-I/tests/test_training_metrics.py -q` (`13 passed`).
- [x] Remote `pdflatex` compile was attempted into `/tmp/tropicalgt_paper_compile`; it remains blocked by missing TeX dependency `mathtools.sty`, so no PDF artifact was produced or staged.
- [x] b60 latest checked training state reached step `2250` with train loss/NLL `1.149/1.126`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- [ ] Next Section 12 item: keep the current BPB run alive until the 5K gate unless explicitly restarted.

### Current Objective Update - Section 12 Live 5K Gate Status

- [x] Confirmed the current b60 fresh step-0 trainer is alive as PID `378962` on the remote Tailscale machine and remains the only GPU compute process observed by `nvidia-smi` during this audit.
- [x] Confirmed watcher PID `379304` is alive and monitoring the trainer log until target step `5000`, with required step-5000 paths `periodic/step_00005000/validation_report.json` and `periodic/step_00005000/got_audit/inference_audit.html` before it records completion.
- [x] Confirmed the existing Codex heartbeat automation `check-tropicalgt-b59-5k-gate` is active on `FREQ=MINUTELY;INTERVAL=150`, i.e. every 2.5 hours, and targets this thread for post-5K follow-up.
- [x] Latest checked training state during this Section 12 audit reached step `2252/5000` with train loss/NLL `1.126/1.103`; W&B run id remains `ld5u55p5` for `amelie-iska-math/TropicalGT-I`.
- [ ] Section 12 5K gate remains open until the trainer produces the required step-5000 validation and audit artifacts; do not restart from step 0 before those artifacts exist unless the user explicitly changes the policy.

### Current Objective Update - Post-5K Subagent Review Contract Pass

- [x] Tightened `TropicalGT-I/scripts/parameter_golf_codex_review_loop.py` so the generated 5K review prompt instructs the main Codex coordinator to spawn or assign a fresh Codex subagent when subagent tools are available.
- [x] The review prompt now explicitly requires inspection of metrics, advanced sidecars, validation reports, W&B summaries, and topological, geometric, algebraic, GraphCG, GFlowNet, memory, and decoding visualizations from the artifact inventory.
- [x] The prompt now requires no proxies or fallbacks for unavailable CAS/topology/geometry/algebra/memory/visualization evidence and calls for a step-0 restart with adjusted hyperparameters/configs if BPB remains above target under the beginning restart policy.
- [x] `prepare_5k_review_bundle.py` now writes machine-readable `review_requirements` covering subagent review, advanced sidecar/visualization review, step-0 restart, and no-proxy evidence handling.
- [x] Verification completed: py-compile for the review-loop and bundle scripts/tests plus `pytest TropicalGT-I/tests/test_parameter_golf_review_loop.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py -q` (`5 passed`).
- [x] Latest checked training state reached step `2324/5000` with train loss/NLL `1.134/1.111`; trainer PID `378962` and watcher PID `379304` remain alive under the step-5000 gate.
- [ ] Next Section 12 item remains gated on step-5000 artifacts: run post-5K analysis/visualizations, route the review to a Codex subagent, then restart from step 0 with evidence-backed hyperparameter/config updates if BPB is still above target.

### Current Objective Update - Post-5K Report Schema Compatibility Pass

- [x] Added exact schema-aware metric lookup for post-5K review decisions so `eval.bpb` is read from real training reports, periodic validation manifests, or raw validation reports when those files use their documented field names (`eval.bpb`, `metrics.eval_bpb`, `bpb`, or `bpb_exact`).
- [x] Added the same schema handling for graph BPB and active-training-contract compression metrics, so the Codex review prompt and bundle cannot silently omit the observed BPB when step-5000 artifacts come from periodic validation.
- [x] Updated `prepare_5k_review_bundle.py` to prefer `periodic/step_00005000/periodic_validation_artifacts.json` and then `validation_report.json` when `train_report.json` is absent, while preserving explicit `--report` paths.
- [x] Added focused tests for periodic artifact manifests, top-level validation reports, active-contract metrics, and default step-5000 report discovery.
- [x] Verified with `python -m py_compile` on the touched scripts/tests and `python -m pytest TropicalGT-I/tests/test_parameter_golf_review_loop.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py -q` (`8 passed`).
- [x] Latest live pulse during this pass: trainer PID `378962` and watcher PID `379304` are alive; progress reached step `2434/5000` with train loss/NLL `1.142/1.119`; step-5000 validation/audit artifacts are not present yet.
- [ ] Section 12 5K gate remains open: keep the run alive, wait for required step-5000 artifacts, then run analyses/visualizations and route the evidence review through a Codex subagent before any step-0 restart.

### Current Objective Update - Section 12 Strict No-Fallback Data Config Pass

- [x] Audited the active b60 launch config and confirmed the live command is still capped by `--max-steps 5000`, with `validation_every_steps=250`, `visualization_every_steps=250`, `checkpoint_every=250`, online W&B run `ld5u55p5`, TokenGT graph-token config, hybrid HF plus OpenAI Parameter-Golf data, GFlowNet/GraphCG/tropical weights, causal forward+reverse DAG decoding, ROAR random-order decoding, and a 5K restart-review policy.
- [x] Found legacy dataset path-search fields (`fallback_roots`, `tokenizer_fallback_paths`) in full-dataset configs even though token-id fallback was disabled; implemented strict path handling so those fields are rejected by default unless `allow_config_path_fallbacks=true` is explicitly set.
- [x] Removed those fallback path fields from committed full-dataset training configs and set `allow_config_path_fallbacks=false` for strict future launches.
- [x] Preserved the old path-search behavior only in an explicit opt-in test fixture, and added regression tests proving path and tokenizer fallback fields are rejected by default.
- [x] Verified the live step-2500 periodic audit completed after its settle window: `validation_report.json`, `periodic_validation_artifacts.json`, and `got_audit/inference_audit.html` all exist under `periodic/step_00002500`; training resumed at step `2503/5000`.
- [x] Verified with `python -m py_compile TropicalGT-I/src/tropicalgt/data.py TropicalGT-I/tests/test_data_loader.py` and `PYTHONPATH=TropicalGT-I/src python -m pytest TropicalGT-I/tests/test_data_loader.py -q` (`15 passed`, only external SWIG deprecation warnings).
- [ ] Section 12 5K gate remains open: keep the current run alive until step 5000, then run analyses/visualizations and subagent evidence review before any step-0 restart.

### Current Objective Update - README And Browser Reattach Pass

- [x] Reattached the live Codex/browser review endpoint without copying artifacts: remote `python -m http.server` now serves the completed b60 step-2500 `got_audit` directory on remote `127.0.0.1:8991`, and the local `127.0.0.1:8991` endpoint is forwarded to it through a clean SSH tunnel.
- [x] Verified `http://127.0.0.1:8991/` returns the real `TropicalGT-I Inference Audit` HTML for the latest completed periodic audit; the generated `index.html` symlink, server PID, and server log remain under ignored `outputs/` and are not staged.
- [x] Updated `README.md` to reflect the live b60 step-0 5K-gate run, BPB target `< 1.12`, W&B run id `ld5u55p5`, strict no-config-path-fallback data policy, post-5K review bundle behavior, CAS unavailable-state policy, and current browser-serving/tunnel commands.
- [x] Removed stale README references to b44 as the current run, target BPB `1.18`, compatibility path fallbacks, and commutative-algebra proxy/fallback language.
- [ ] Section 12 5K gate remains open: keep trainer PID `378962` and watcher PID `379304` alive until step 5000 artifacts exist, then run analyses/visualizations and route evidence review through a Codex subagent before any step-0 restart.

### Current Objective Update - Executable Post-5K Review Bundle Pass

- [x] Extended `prepare_5k_review_bundle.py` with explicit `--run-eval-visualizations` and `--run-interactive-audit-validators` flags so the post-5K handoff can run the generated analysis/visualization and validator commands only when requested.
- [x] Added per-command stdout/stderr log capture under the generated review bundle `command_logs/` directory, return-code/timed-out summaries in `command_results`, and kept the default behavior path-only with no command execution.
- [x] Updated `README.md` with the post-5K bundle command that runs eval visualizations and interactive audit validators after step-5000 artifacts and checkpoint paths exist; generated logs remain under ignored outputs and must not be staged.
- [x] Added focused tests proving default bundles do not execute commands and `_run_shell_command` records stdout, stderr, return code, and timeout status.
- [ ] Section 12 5K gate remains open: wait for step-5000 artifacts before running the executable review bundle and assigning the Codex subagent evidence review.

### Current Objective Update - Post-5K Legacy Backfill Command Pass

- [x] Added exact `interactive_audit_backfill_commands` to the active-training artifact inventory whenever a real latest `got_audit` directory exists.
- [x] Extended `prepare_5k_review_bundle.py` with `--run-legacy-audit-backfill`, executed before strict interactive-audit validators and logged under the generated review bundle `command_logs/` directory.
- [x] Kept the default review bundle path-only; command execution remains explicit and generated backfill reports/logs must not be staged.
- [x] Documented that legacy audit backfill may only write explicit unavailable diagnostics or rerender visual contracts from existing raw payloads; no CAS, tropical-fan, persistence-module, or visualization proxy is allowed.
- [ ] Section 12 5K gate remains open: wait for step-5000 artifacts before running eval visualizations, legacy backfill, strict validators, and Codex subagent evidence review.

### Current Objective Update - Evidence-Bound Restart Schema Pass

- [x] Added a machine-readable `tropicalgt.restart_decision.v1` schema to the 5K review loop and post-5K review bundle.
- [x] Required every proposed post-5K hyperparameter/config change to include `dot_path`, old/new values, reason, evidence paths, expected BPB effect, and risk.
- [x] Added explicit allowed actions for target-met continuation, target-missed step-0 restart with evidence-backed config patch, and blocked missing-evidence/no-restart states.
- [x] Kept restart planning no-proxy: unavailable CAS/topology/geometry/algebra/memory/visual evidence remains unavailable with exact reasons and cannot justify a config change.
- [ ] Section 12 5K gate remains open: do not populate or execute the restart schema until step-5000 metrics, sidecars, visual audits, validators, and Codex subagent review exist.

### Current Objective Update - Post-5K Execution Readiness Gate Pass

- [x] Added an `execution_readiness` report to the post-5K review bundle with boundary step, observed report step, checkpoint path, latest audit path, and exact missing-evidence issues.
- [x] Kept path-only bundles available before the gate, but blocked all `--run-*` command execution until the boundary report is at/after step 5000, the checkpoint exists, and the latest audit directory exists when backfill/validators are requested.
- [x] Added regression coverage proving execution flags fail closed with `missing_checkpoint` instead of running eval/visualization/backfill/validator commands against incomplete evidence.
- [ ] Section 12 5K gate remains open: executable review commands stay blocked until the real step-5000 report, checkpoint, and audit artifacts exist.

### Current Objective Update - Step-3000 Browser Refresh Pass

- [x] Verified b60 step-3000 `validation_report.json`, `periodic_validation_artifacts.json`, and `got_audit/inference_audit.html` all exist as real generated artifacts.
- [x] Reattached remote `127.0.0.1:8991` and the local Codex tunnel to the step-3000 `got_audit` directory without copying generated artifacts; remote server PID at verification time: `593331`.
- [x] Recorded real step-3000 validation metrics: BPB/exact BPB `1.5293022093208184`, graph-BPB `20.20904657226864`, graph-conditioned BPB without side cost `1.3445352900384904`, NLL `1.0600315146148205`, and invalid graph rate `0.0`.
- [x] Confirmed step-5000 validation and audit artifacts are still absent, so post-5K analyses, visualization generation, Codex subagent review, and any step-0 restart remain blocked on real 5K evidence.
- [ ] Continue Section 12 linearly: preserve trainer PID `378962` and watcher PID `379304`, monitor disk under the configured generated-audit retention policy, and wait for step-5000 evidence before analysis, subagent review, or restart.

### Current Objective Update - Step-3250 Browser Refresh And Retention Pass

- [x] Verified b60 step-3250 `validation_report.json`, `periodic_validation_artifacts.json`, and `got_audit/inference_audit.html` all exist as real generated artifacts.
- [x] Reattached remote `127.0.0.1:8991` and the local Codex tunnel to the step-3250 `got_audit` directory without copying generated artifacts; remote server PID at verification time: `602885`.
- [x] Recorded real step-3250 validation metrics: BPB/exact BPB `1.5044848279550496`, graph-BPB `20.187227572172226`, graph-conditioned BPB without side cost `1.3227162899420746`, NLL `1.042829416692257`, and invalid graph rate `0.0`.
- [x] Pruned only generated `step_00002750/got_audit` under the active periodic retention policy while step-3250 audit generation was consuming disk; milestone audits, step 3000, step 3250, compact validation reports, and non-audit artifacts remain, with a manifest record in `periodic/got_audit_retention_manifest.jsonl`.
- [x] Confirmed training resumed after the step-3250 audit at step `3257/5000`; step-5000 validation and audit artifacts remain absent, so post-5K analyses, Codex subagent review, and any step-0 restart remain blocked on real 5K evidence.
- [ ] Continue Section 12 linearly: preserve trainer PID `378962` and watcher PID `379304`, keep the browser attached to the latest complete real audit, and wait for step-5000 evidence before analysis, subagent review, or restart.

### Current Objective Update - Step-3500 Browser Refresh And Retention Pass

- [x] Verified b60 step-3500 `validation_report.json`, `periodic_validation_artifacts.json`, and `got_audit/inference_audit.html` all exist as real generated artifacts.
- [x] Reattached remote `127.0.0.1:8991` and the local Codex tunnel to the step-3500 `got_audit` directory without copying generated artifacts; remote server PID at verification time: `612391`.
- [x] Recorded real step-3500 validation metrics: BPB/exact BPB `1.4940912929263006`, graph-BPB `20.178089761117462`, graph-conditioned BPB without side cost `1.3135784788873124`, NLL `1.0356251671910286`, and invalid graph rate `0.0`.
- [x] Pruned only generated `step_00003000/got_audit` under the active periodic retention policy while step-3500 audit generation was consuming disk; milestone audits, step 3250, step 3500, compact validation reports, and non-audit artifacts remain, with a manifest record in `periodic/got_audit_retention_manifest.jsonl`.
- [x] Confirmed training resumed after the step-3500 audit at step `3508/5000`; step-5000 validation and audit artifacts remain absent, so post-5K analyses, Codex subagent review, and any step-0 restart remain blocked on real 5K evidence.
- [ ] Continue Section 12 linearly: preserve trainer PID `378962` and watcher PID `379304`, keep the browser attached to the latest complete real audit, and wait for step-5000 evidence before analysis, subagent review, or restart.

### Current Objective Update - Step-3750 Browser Refresh And Retention Pass

- [x] Verified b60 step-3750 `validation_report.json`, `periodic_validation_artifacts.json`, and `got_audit/inference_audit.html` all exist as real generated artifacts.
- [x] Reattached remote `127.0.0.1:8991` and the local Codex tunnel to the step-3750 `got_audit` directory without copying generated artifacts; remote server PID at verification time: `621530`.
- [x] Recorded real step-3750 validation metrics: BPB/exact BPB `1.4739591865478667`, graph-BPB `20.16038997142015`, graph-conditioned BPB without side cost `1.2958786891900003`, NLL `1.0216706544160843`, and invalid graph rate `0.0`.
- [x] Pruned only generated `step_00003250/got_audit` under the active periodic retention policy while step-3750 audit generation was consuming disk; milestone audits, step 3500, step 3750, compact validation reports, and non-audit artifacts remain, with a manifest record in `periodic/got_audit_retention_manifest.jsonl`.
- [x] Confirmed training resumed after the step-3750 audit at step `3752/5000`; step-5000 validation and audit artifacts remain absent, so post-5K analyses, Codex subagent review, and any step-0 restart remain blocked on real 5K evidence.
- [ ] Continue Section 12 linearly: preserve trainer PID `378962` and watcher PID `379304`, keep the browser attached to the latest complete real audit, and wait for step-5000 evidence before analysis, subagent review, or restart.

### Current Objective Update - Step-4000 Browser Refresh And Retention Pass

- [x] Verified b60 step-4000 `validation_report.json`, `periodic_validation_artifacts.json`, and `got_audit/inference_audit.html` all exist as real generated artifacts.
- [x] Reattached remote `127.0.0.1:8991` and the local Codex tunnel to the step-4000 `got_audit` directory without copying generated artifacts; remote server PID at verification time: `630984`.
- [x] Recorded real step-4000 validation metrics: BPB/exact BPB `1.4740326876332792`, graph-BPB `20.160454592266568`, graph-conditioned BPB without side cost `1.2959433100364168`, NLL `1.021721601486206`, and invalid graph rate `0.0`.
- [x] Pruned only generated `step_00003500/got_audit` under the active periodic retention policy while step-4000 audit generation was consuming disk; milestone audits, step 3750, step 4000, compact validation reports, and non-audit artifacts remain, with a manifest record in `periodic/got_audit_retention_manifest.jsonl`.
- [x] Confirmed training resumed after the step-4000 audit at step `4006/5000`; step-5000 validation and audit artifacts remain absent, so post-5K analyses, Codex subagent review, and any step-0 restart remain blocked on real 5K evidence.
- [ ] Continue Section 12 linearly: preserve trainer PID `378962` and watcher PID `379304`, keep the browser attached to the latest complete real audit, and wait for step-5000 evidence before analysis, subagent review, or restart.

### Current Objective Update - Step-4250 Browser Refresh And Retention Pass

- [x] Verified b60 step-4250 `validation_report.json`, `periodic_validation_artifacts.json`, and `got_audit/inference_audit.html` all exist as real generated artifacts.
- [x] Reattached remote `127.0.0.1:8991` and the local Codex tunnel to the step-4250 `got_audit` directory without copying generated artifacts; remote server PID at verification time: `640368`.
- [x] Recorded real step-4250 validation metrics: BPB/exact BPB `1.4582135514242673`, graph-BPB `20.14654668919724`, graph-conditioned BPB without side cost `1.2820354069670894`, NLL `1.0107566118240356`, and invalid graph rate `0.0`.
- [x] Pruned only generated `step_00003750/got_audit` under the active periodic retention policy while step-4250 audit generation was consuming disk; milestone audits, step 4000, step 4250, compact validation reports, and non-audit artifacts remain, with a manifest record in `periodic/got_audit_retention_manifest.jsonl`.
- [x] Confirmed training resumed after the step-4250 audit at step `4256/5000`; step-5000 validation and audit artifacts remain absent, so post-5K analyses, Codex subagent review, and any step-0 restart remain blocked on real 5K evidence.
- [ ] Continue Section 12 linearly: preserve trainer PID `378962` and watcher PID `379304`, keep the browser attached to the latest complete real audit, and wait for step-5000 evidence before analysis, subagent review, or restart.

### Current Objective Update - Step-4500 Browser Refresh And Retention Pass

- [x] Verified b60 step-4500 `validation_report.json`, `periodic_validation_artifacts.json`, and `got_audit/inference_audit.html` all exist as real generated artifacts.
- [x] Reattached remote `127.0.0.1:8991` and the local Codex tunnel to the step-4500 `got_audit` directory without copying generated artifacts; remote server PID at verification time: `650940`.
- [x] Recorded real step-4500 validation metrics: BPB/exact BPB `1.4334303489231286`, graph-BPB `20.12475773855043`, graph-conditioned BPB without side cost `1.260246456320278`, NLL `0.9935782048851252`, and invalid graph rate `0.0`.
- [x] Pruned only generated `step_00004000/got_audit` under the active periodic retention policy while step-4500 audit generation was consuming disk; milestone audits, step 4250, step 4500, compact validation reports, and non-audit artifacts remain, with a manifest record in `periodic/got_audit_retention_manifest.jsonl`.
- [x] Confirmed training resumed after the step-4500 audit at step `4503/5000`; step-5000 validation and audit artifacts remain absent, so post-5K analyses, Codex subagent review, and any step-0 restart remain blocked on real 5K evidence.
- [ ] Continue Section 12 linearly: preserve trainer PID `378962` and watcher PID `379304`, keep the browser attached to the latest complete real audit, and wait for step-5000 evidence before analysis, subagent review, or restart.

### Current Objective Update - Step-4750 Browser Refresh And Retention Pass

- [x] Verified b60 step-4750 `validation_report.json`, `periodic_validation_artifacts.json`, and `got_audit/inference_audit.html` all exist as real generated artifacts.
- [x] Reattached remote `127.0.0.1:8991` and the local Codex tunnel to the step-4750 `got_audit` directory without copying generated artifacts; remote server PID at verification time: `660533`.
- [x] Recorded real step-4750 validation metrics: BPB/exact BPB `1.4367942113359327`, graph-BPB `20.127715186527215`, graph-conditioned BPB without side cost `1.263203904297063`, NLL `0.9959098566323519`, and invalid graph rate `0.0`.
- [x] Pruned only generated `step_00004250/got_audit` under the active periodic retention policy while step-4750 audit generation was consuming disk; milestone audits, step 4500, step 4750, compact validation reports, and non-audit artifacts remain, with a manifest record in `periodic/got_audit_retention_manifest.jsonl`.
- [x] Confirmed training resumed after the step-4750 audit at step `4759/5000`; step-5000 validation and audit artifacts remain absent, so post-5K analyses, Codex subagent review, and any step-0 restart remain blocked on real 5K evidence.
- [ ] Continue Section 12 linearly: preserve trainer PID `378962` and watcher PID `379304`, keep the browser attached to the latest complete real audit, and wait for step-5000 evidence before analysis, subagent review, or restart.

### Current Objective Update - Step-5000 Gate And Empty-Checkpoint Readiness Pass

- [x] Verified b60 step-5000 `validation_report.json`, `periodic_validation_artifacts.json`, and `got_audit/inference_audit.html` all exist as real generated artifacts.
- [x] Verified the trainer PID `378962` and watcher PID `379304` are stopped after the step-5000 gate, and the stop record `TropicalGT-I/outputs/training_stop_records/b60_fresh_step0_casrows_step5000_gate.json` reports no missing required step-5000 artifact paths.
- [x] Reattached remote `127.0.0.1:8991` and the local Codex tunnel to the step-5000 `got_audit` directory without copying generated artifacts; `curl http://127.0.0.1:8991/` returned the real `TropicalGT-I Inference Audit` HTML.
- [x] Recorded real step-5000 validation metrics: BPB/exact BPB `1.4304583543547733`, graph-BPB `20.122144813809587`, graph-conditioned BPB without side cost `1.2576335315794374`, NLL `0.9915181752294302`, and invalid graph rate `0.0`; target BPB `< 1.12` remains unmet.
- [x] Pruned only generated `step_00004500/got_audit` under the active periodic retention policy while step-5000 audit generation was consuming disk; milestone audits, step 4750, step 5000, compact validation reports, and non-audit artifacts remain.
- [x] Hardened post-5K readiness so zero-byte checkpoints are explicit unavailable evidence, not load attempts or proxies. `parameter_golf_codex_review_loop.py` returns `checkpoint_file_is_empty`, and `prepare_5k_review_bundle.py --run-*` fails closed with `empty_checkpoint:TropicalGT-I/checkpoints/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate.latest.pt`.
- [x] Generated the path-only 5K review bundle at `TropicalGT-I/outputs/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate/post_5k_review_bundle/review_bundle_step_00005000.json`; it records `execution_readiness.ready=false`, `issues=[empty_checkpoint:TropicalGT-I/checkpoints/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate.latest.pt]`, `command_results=[]`, BPB `1.4304583543547733`, graph-BPB `20.122144813809587`, and `restart_policy=beginning`.
- [x] Verification passed: py-compile for the touched bundle/review-loop scripts and tests; `pytest TropicalGT-I/tests/test_prepare_5k_review_bundle.py TropicalGT-I/tests/test_parameter_golf_review_loop.py -q` (`14 passed`); executable bundle attempt failed closed before running eval/backfill/validator commands with `empty_checkpoint`.
- [x] Routed the path-only real evidence bundle to Codex subagent Lagrange for metric/sidecar/visualization review; the empty checkpoint was treated as a missing-evidence blocker for executable post-5K analysis and any restart claim that would rely on checkpoint-backed sidecars.

### Current Objective Update - Codex 5K Evidence Review Result

- [x] Codex subagent Lagrange reviewed the real b60 step-5000 evidence bundle and wrote `planning/tropicalgt_i_b60_step5000_codex_evidence_review_2026-06-16.md`.
- [x] Recorded the restart-schema decision `blocked_missing_evidence_no_restart`: step-5000 BPB `1.4304583543547733` missed target `< 1.12`, but the required checkpoint `TropicalGT-I/checkpoints/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate.latest.pt` is a zero-byte unavailable checkpoint and blocks checkpoint-dependent executable analysis/visualization evidence.
- [x] Confirmed no hyperparameter/config changes were proposed, no training restart was launched, and no generated artifacts/checkpoints/W&B files were staged.
- [x] Available review evidence includes the path-only bundle `TropicalGT-I/outputs/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate/post_5k_review_bundle/review_bundle_step_00005000.json`, validation report, periodic artifact summary, step-5000 audit HTML/sidecars, topology/tropical/memory/GraphCG/tropical-support summaries where they exist, and explicit unavailable states for missing checkpoint/CAS/memory/W&B summary evidence.
- [x] Implemented source-side checkpoint-write hardening so future 5K gates cannot leave a zero-byte `.latest.pt` as the only checkpoint evidence; this was scoped to source/tests/docs and did not mutate generated checkpoints.

### Current Objective Update - Atomic Checkpoint Write Hardening Pass

- [x] Replaced direct `torch.save(path)` checkpoint writes with temp-file saves, nonempty/loadable verification, fsync, atomic `os.replace`, parent-directory fsync, and post-replace verification.
- [x] Added `checkpoint_verify_load` with default verified loading; setting it false only skips load verification, not the nonempty atomic-write guard.
- [x] Added regression coverage proving normal periodic `.latest.pt` checkpoints are nonempty/loadable and failed empty temp writes do not replace an existing checkpoint or leave temp files behind.
- [x] Verification passed: py-compile for `run.py`, `test_training_resume.py`, and `test_training_metrics.py`; `pytest TropicalGT-I/tests/test_training_resume.py TropicalGT-I/tests/test_training_metrics.py -q` (`16 passed`); `git diff --check` clean.
- [x] Audited cleanup candidates for stale W&B/generated artifacts without deleting required b60 5K evidence, checkpoints, review docs, or source-controlled files.

### Current Objective Update - Generated Artifact Cleanup Audit Pass

- [x] Audited remote disk pressure after b60 5K review: root filesystem remained at about `32G` available; `TropicalGT-I/outputs` was about `91G`, `TropicalGT-I/checkpoints` about `1.9G`, and local `wandb` about `317M`.
- [x] Removed only clearly safe cache directories: project `.pytest_cache` plus Python `__pycache__` directories under `TropicalGT-I/scripts`, `TropicalGT-I/tests`, `TropicalGT-I/src/tropicalgt`, and `external/parameter-golf`.
- [x] Preserved all b60 step-5000 evidence, review bundles, stop records, launch configs, and the zero-byte checkpoint path; no generated evidence was repaired, replaced, or fabricated.
- [x] Preserved old b55-b59 output/checkpoint/W&B provenance rather than deleting it silently. Largest cleanup candidates remain generated output dirs: b57 about `6.2G`, `multi_sample_browser` about `3.1G`, b59 fresh about `2.1G`, b58 about `1.1G`, plus older checkpoint files around `0.48G` each.
- [x] Verification: `git status --short` remained clean after cache deletion; no source-controlled files, datasets, checkpoints, W&B directories, or generated audit bundles were staged.
- [x] Left BPB restart blocked until real checkpoint-dependent evidence exists or the user explicitly changes the no-proxy evidence requirement; continued with the next source-side implementation/repair item that does not require a restart.

### Current Objective Update - BEMultipliers Diagnostic Contract Pass

- [x] Confirmed live CAS probes see Macaulay2 `/usr/bin/M2`, Singular, Sage, and local `external/BEMultipliers/BuchsbaumEisenbudMultipliers.m2` as available/loadable diagnostic backends.
- [x] Hardened Macaulay2 BEMultipliers script output with explicit contract fields: `bemultipliers_is_resolution_backend=false`, `requires_certified_macaulay2_chain_complex=true`, and `safe_to_substitute_for_resolution=false`.
- [x] Extended parsed Buchsbaum-Eisenbud diagnostics and certificate summaries with `safe_to_render_multiplier_output`, `is_resolution_backend=false`, `safe_to_substitute_for_resolution=false`, and a no-proxy diagnostic contract.
- [x] Added regression coverage proving computed `aMultiplier(1,C,ComputeRanks=>true)` output is renderable only as explicit post-resolution CAS diagnostic output and never as a free-resolution substitute.
- [x] Verification passed: py-compile for `cas_free_resolution.py` and `test_algebraic_persistence.py`; `pytest TropicalGT-I/tests/test_algebraic_persistence.py -q` (`28 passed`); `git diff --check` clean.
- [x] Inspected and patched remaining CAS/visualization surfaces so the new BEMultipliers diagnostic-contract fields render in browser tables without implying a resolution certificate.

### Current Objective Update - BEMultipliers Browser Contract Rendering Pass

- [x] Extended normalized CAS display payloads with `safe_to_render_multiplier_output`, `is_resolution_backend`, `requires_certified_macaulay2_chain_complex`, `safe_to_substitute_for_resolution`, and the structured diagnostic contract.
- [x] Added BEMultipliers contract rows to the Buchsbaum-Eisenbud diagnostic table and certificate table so browser users see that multiplier output is post-resolution diagnostic-only.
- [x] Included the BEMultipliers contract flags in certified-CAS resolution signatures used by derived/analogical comparison, preventing a multiplier hash from acting like resolution evidence.
- [x] Added visualization regression coverage for the browser table rows and normalized display payload fields.
- [x] Verification passed: py-compile for visualization/CAS files and tests; `pytest TropicalGT-I/tests/test_simplicial_visualization.py TropicalGT-I/tests/test_algebraic_persistence.py -q` (`64 passed`); `git diff --check` clean.
- [x] Next source-side repair item completed: derived/analogical memory comparison text now explains unavailable and diagnostic-only CAS components clearly when similarity components mismatch.

### Current Objective Update - Derived CAS No-Proxy Explanation Pass

- [x] Added explicit component-level explanations to certified real free-resolution comparisons for ring, CAS input hash, artifact hash, multigraded Betti data, differentials, Fitting ideals, minors, and Buchsbaum-Eisenbud/BEMultipliers sidecars.
- [x] Marked Buchsbaum-Eisenbud/BEMultipliers comparison data as diagnostic-only in the analogical/derived comparison payload, so it cannot be treated as a resolution backend or substituted for Betti, differential, Fitting, or minor agreement.
- [x] Added unavailable-state metadata for missing query or memory CAS evidence: unavailable reasons, no-proxy/no-fallback policy, and an explanation that missing certified resolutions are not estimated or replaced.
- [x] Updated derived/free-resolution interpretation text so mismatched CAS artifacts and diagnostic-only BEMultipliers sidecars are explained directly in browser-visible payloads.
- [x] Focused verification passed: visualization tests `36 passed`.

### Current Objective Update - Advanced Auxiliary Promotion Gate Pass

- [x] Added chart-bundle/toric ablation variants to `run_bpb_ablation_grid.py`, including telemetry-only zero-weight and nonzero `0p1x` variants.
- [x] Persisted `ablation_variant` and `ablation_overrides` into `train_report.json` so analysis reports can cite exact coefficient changes.
- [x] Added an `advanced_auxiliary_promotion_gate` to BPB ablation reports. It promotes no nonzero chart-bundle/toric coefficient unless matched-seed runs have the same final step and both held-out eval BPB and eval graph-BPB improve against baseline.
- [x] Kept no-proxy/no-fallback semantics: missing deltas, unmatched seeds/steps, telemetry-only rows, or unrelated variants are blocked or unavailable, not interpreted as wins.
- [x] Focused verification passed: ablation report/grid plus training resume tests `6 passed`.
