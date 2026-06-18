# TropicalGT-I Herschel 5K Review Protocol - 2026-06-16

Herschel (`019ed124-c3b0-7b00-81a9-1a39c76e5583`) is the dedicated 5K training-iteration reviewer and restart agent. Herschel's only job is to monitor each always-on 5K-step TropicalGT-I training iteration, review the complete evidence at the 5K gate, generate a complete report for Amelie, then write and launch an evidence-backed step-0 restart config when the no-proxy evidence gate allows it.

## Evidence Scope

At every 5K gate, Herschel must inspect all implemented and utilized evidence:

- Primary BPB evidence: BPB, text BPB, graph-conditioned BPB/no-side-cost when present, graph-BPB, NLL, loss, invalid graph rate, graph JSON fallback records, train/validation trends, W&B run metadata, logs, checkpoint integrity, stop records, and config/commit provenance.
- Advanced training evidence: all implemented losses, objectives, regularizers, loss weights, and scalar metrics from tropical/certificate losses, GraphCG, GFlowNet, memory, PH/topology, CAS/algebra, vector-bundle/chart-bundle, toric/sheaf, meet-in-the-middle, ROAR, causal DAG decoding, and any newly introduced metrics since the prior 5K iteration.
- Advanced sidecars and contracts: NLL density `visual_layer_contract`, tropical support `tropical_support_render_contract`, toric sidecar `toric_embedding_sidecar.html/json`, legacy toric sidecar backfill, analogical probability-map claim contracts, simplex-tree poset contracts, radius-slider contracts, CAS certificate summaries, tropical fan diagnostics, certified/free-resolution/Fitting/minor/BE artifacts, transported persistence-landscape memory diagnostics, chart-bundle/vector-bundle/toric/sheaf metrics, GraphCG full-rank diagnostics, and all periodic audit JSON/HTML artifacts.
- Visual and interactive artifacts: NLL trajectory/density, radius/probability complexes, SimplexTree posets, two-parameter bifiltration/staircase, CAS algebra tables, analogical top-k/maps, tropical support, GraphCG, tropical fan, toric sidecar, persistence barcode/landscape/representation, and every new artifact introduced since the previous gate.

Unavailable evidence must be recorded with the exact reason. Missing/proxy evidence cannot justify a hyperparameter or config change.

## Required Report

For each 5K iteration, Herschel must write a complete report under `planning/` or a run-specific review folder, with screenshot/plot artifacts in a sibling artifact directory. The report must include:

1. Run identity: run name, PID/monitor record, W&B id/URL, commit/config, checkpoint integrity, log path, validation step, output and artifact roots.
2. Primary metric tables and trends.
3. Advanced training readout covering all implemented losses, sidecars, contracts, and unavailable evidence.
4. Strict artifact validation after legacy backfill when applicable.
5. Screenshot and visual review of required interactive artifacts.
6. Charts, plots, or diagrams summarizing metric trends, loss interactions, artifact pass/fail status, and the restart decision path.
7. Evidence-backed hyperparameter/config recommendations, with one explicit evidence citation per change.
8. Restart execution record: written config, run name, PID, monitor PID, W&B id, log path, GPU/readiness check, first steps, first metrics, or a precise no-proxy block reason.

## Restart Rule

If checkpoint integrity, metrics, sidecars, visual audits, and validators satisfy the no-proxy evidence gate, Herschel may write the next step-0 config and launch training without waiting for the main agent to review the full report. If evidence is missing, empty, stale, proxy-derived, or contradictory, Herschel must block restart, write the reason, and leave the current always-on training policy to the main agent/user.

## Advanced Methodology Requirement

Each restart decision must consider all implemented paradigms: TokenGT-style graph tokenization, OAI Parameter-Golf graph data, available HF/reasoning data, tropical ring attention, long-context/multi-row training, GraphCG full-rank directions, GFlowNet graph-of-thought training, memory retrieval and analogical reports, BPB/graph-BPB logging, periodic visual audits, meet-in-the-middle, causal DAG forward/reverse decoding, ROAR/random-order decoding, CAS/free-resolution/Fitting/minor/BE evidence, PH/persistence/landscape features, chart-bundle/vector-bundle/toric/sheaf metrics, tropical fan/toric sidecars, and every no-proxy artifact contract implemented since the prior gate.

## GPU Launch Safety Update - 2026-06-18

Herschel and any campaign/review-loop automation must not launch, resume, or restart GPU training unless an explicit GPU launch clearance is present. The accepted clearance forms are `--allow-gpu-launch`, `TROPICALGT_ALLOW_GPU_LAUNCH=true`, or a declared `--gpu-memory-budget-mb` / `TROPICALGT_GPU_MEMORY_BUDGET_MB` paired with a human-readable clearance note. CPU-only report generation, checkpoint evidence inspection, prompt writing, and unavailable-evidence recording may proceed without clearance. Any blocked launch must write a `tropicalgt.gpu_launch_safety_contract.v1` JSON record with exact blockers, and no blocked GPU launch may be treated as a failed training result.

The current implementation writes this contract from both `parameter_golf_codex_review_loop.py` and `run_advanced_bpb_campaign.py`; this preserves the user's separate GPU work while keeping Herschel's 5K review/report path available.

## 2026-06-18 CPU-Only Herschel 5K Report Generation

Herschel's 5K evidence report path is now implemented as a deterministic CPU-only artifact generator over the existing post-5K review bundle. `prepare_5k_review_bundle.py` writes the normal no-proxy bundle, calls `write_herschel_5k_report.py`, then records `herschel_report_markdown`, `herschel_report_json`, and `herschel_report_summary` back into the bundle. The report includes run identity, BPB/graph-BPB, checkpoint availability and restart safety, execution readiness, advanced BPB gate failures, grouped advanced sidecar counts, restart blockers, and a Mermaid restart-decision diagram. It does not train, evaluate, browse, validate, load checkpoints, inspect GPUs, or synthesize missing evidence.

Blocked evidence remains blocked. Empty or unavailable checkpoints, missing post-5K command results, failed advanced BPB gates, and failed validators are copied into the report as exact blockers; no checkpoint-backed or sidecar-backed restart can be justified by this report unless the underlying bundle evidence already permits it. This keeps Herschel useful while the user may be running another GPU job.

2026-06-18 follow-up: Herschel report summaries now also read strict interactive-validator `--json-output` files from recorded validator command results and expose `validator_gap_evidence` under `artifact_evidence`. This carries the validator `evidence_gap_inventory` into Herschel's JSON/markdown report with combined category counts and per-source availability, while preserving failed validator status. Missing validator JSON is recorded as unavailable and cannot justify a restart, artifact pass, or hyperparameter change.

2026-06-18 visual-report follow-up: `write_herschel_5k_report.py` now also writes an optional HTML visual report, and `prepare_5k_review_bundle.py` records it as `herschel_report_html`. The HTML report renders metric cards, a restart-decision flow, sidecar-group bar charts, strict-validator gap bar charts, validator-source tables, blocker/warning panels, and a filterable sidecar list from the existing summary only. It is an evidence visualization layer, not a command runner or substitute for failed validators.

2026-06-18 validator-action follow-up: Herschel now preserves strict-validator `examples` and `required_action` fields as ranked category rows plus top example rows in JSON, Markdown, and HTML. This makes failed validators actionable without relaxing the gate: concrete missing artifacts and repair actions are displayed, but failed validator status still blocks restart/artifact-pass claims.

Validation:

```text
CUDA_VISIBLE_DEVICES="" python3 -m py_compile TropicalGT-I/scripts/write_herschel_5k_report.py TropicalGT-I/scripts/prepare_5k_review_bundle.py TropicalGT-I/tests/test_herschel_5k_report.py
# passed
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/bin/conda run -n tokengt python -m pytest -q tests/test_herschel_5k_report.py tests/test_prepare_5k_review_bundle.py tests/test_launch_safety.py tests/test_parameter_golf_review_loop.py
# 34 passed in 1.17s
```

## 2026-06-18 BPB-First Promotion Gates And Matched 5K Ablations

BPB-first ablation promotion is now stricter and evidence-backed:

- `build_bpb_ablation_report` now treats nonzero core advanced coefficients (`gflownet_weight`, `graphcg_weight`, `margin_weight`, `entropy_weight`, `certificate_weight`, `sequence_tropical_weight`) and chart-bundle/toric coefficients as promotion candidates.
- Promotion requires matched held-out `eval_bpb` and `eval_graph_bpb` improvement plus real held-out certificate and tropical-wall guardrail evidence. Missing or regressing `eval_certificate_*` or `eval_*wall*`/margin metrics blocks promotion; no train-history proxy or fallback can justify a coefficient increase.
- `evaluate_model` now aggregates actual scalar validation outputs for certificate and tropical-wall guardrails and leaves them absent when unavailable. Training reports persist those as `eval_*` metrics.
- `run_bpb_ablation_grid.py` now writes `tropicalgt.bpb_ablation_match_contract.v1` into the manifest and every generated config, with 5K boundary metadata, requested max steps, seed, data identity, graph-BPB side weight, and a base-config fingerprint. Training reports copy this contract forward so later analysis rejects mismatched runs.
- The analyzer rejects candidates whose match contract differs from baseline even if BPB improves and guardrails pass.

Validation:

```text
CUDA_VISIBLE_DEVICES="" python3 -m py_compile TropicalGT-I/src/tropicalgt/ablation.py TropicalGT-I/src/tropicalgt/run.py TropicalGT-I/scripts/run_bpb_ablation_grid.py TropicalGT-I/tests/test_bpb_ablation.py TropicalGT-I/tests/test_bpb_ablation_grid.py TropicalGT-I/tests/test_training_metrics.py
# passed
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/bin/conda run -n tokengt python -m pytest -q tests/test_bpb_ablation.py tests/test_bpb_ablation_grid.py tests/test_training_metrics.py
# 19 passed in 5.20s
```

## 2026-06-18 Advanced BPB Readiness Contract Strict Gates

Advanced BPB config readiness now has explicit optional strict gates for the next high-rigor training iterations:

- `require_graphcg_active_full_rank=true` adds `advanced_bpb_graphcg_active_directions_full_rank`, requiring active GraphCG directions to cover the embedding dimension rather than only requiring a positive active subset.
- `require_nontrivial_bundle_toric_losses=true` adds gates for chart-bundle auxiliary enablement and nonzero bundle/toric loss weights across transport, cocycle, flat-rank, normal-fan, GraphCG-cell agreement, chart-BPB consistency, and atom-stability coefficients.
- `advanced_bpb_max_visual_audit_interval` makes the visual-audit cadence gate configurable while preserving the default 250-step policy.

Validation:

```text
CUDA_VISIBLE_DEVICES="" python3 -m py_compile TropicalGT-I/src/tropicalgt/readiness_contracts.py TropicalGT-I/tests/test_readiness_audit.py
# passed
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/bin/conda run -n tokengt python -m pytest -q tests/test_readiness_audit.py tests/test_prepare_5k_review_bundle.py tests/test_parameter_golf_review_loop.py tests/test_training_resume.py
# 45 passed in 2.96s
```

2026-06-18 sidecar-inventory follow-up: `prepare_5k_review_bundle.py` now augments the active training contract before writing the contract, Codex prompt, bundle JSON/Markdown, and Herschel report. The augmentation scans only the recorded `latest_got_audit_dir` and only records real existing Herschel-required sidecars, currently `analogical_simplicial_maps.json`, under `advanced_sidecars_tail` plus `herschel_required_sidecars_present`. Missing sidecars remain missing/unavailable; the bundle does not copy, synthesize, or substitute evidence.

2026-06-18 sidecar-inventory validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/scripts/prepare_5k_review_bundle.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py TropicalGT-I/scripts/write_herschel_5k_report.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_prepare_5k_review_bundle.py TropicalGT-I/tests/test_herschel_5k_report.py
# 11 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_prepare_5k_review_bundle.py TropicalGT-I/tests/test_herschel_5k_report.py TropicalGT-I/tests/test_parameter_golf_review_loop.py TropicalGT-I/tests/test_metric_provenance.py
# 36 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/audit_metric_provenance.py --fail-on-uncovered
# findings=331 covered=331 uncovered=0
```

2026-06-18 tropical-support evidence follow-up: Herschel now treats `tropical_support_payload.json` as a required recorded GoT audit sidecar when the latest `got_audit` directory contains it. `write_herschel_5k_report.py` exposes `tropicalgt.herschel_tropical_support_evidence.v1` under `artifact_evidence`, but marks it available only when the sidecar carries the real `tropicalgt.tropical_support_render.v1` and `tropicalgt.tropical_support_readability.v1` contracts, `model_tropical_support_probabilities` provenance, observed support assignments, wall-margin rates, and no-proxy/no-fallback flags. Markdown and HTML reports now include `Tropical Support Evidence` plus a probability-source chart. The reported strict/near wall rates are explicitly model tropical-margin threshold audits, not certified normal-fan wall-crossing counts; missing or unsafe payloads remain unavailable and cannot justify restart or hyperparameter promotion.

2026-06-18 tropical-support evidence validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_herschel_5k_report.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py
# 11 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_herschel_5k_report.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py TropicalGT-I/tests/test_parameter_golf_review_loop.py TropicalGT-I/tests/test_metric_provenance.py
# 36 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/audit_metric_provenance.py --fail-on-uncovered
# findings=348 covered=348 uncovered=0
```

2026-06-18 GraphCG evidence follow-up: Herschel now treats `graphcg_direction_cosines_payload.json` as a required recorded GoT audit sidecar when it exists. `write_herschel_5k_report.py` exposes `tropicalgt.herschel_graphcg_direction_evidence.v1`, but marks it available only when the payload carries `tropicalgt.graphcg_direction_evidence.v1`, `tropicalgt.graphcg_direction_readability.v1`, all model-derived direction rows, unsampled all-direction panels, exact direction-id preservation, a projection-basis certificate with all candidate cosine rows, and no-proxy/no-fallback flags. Markdown and HTML reports now include `GraphCG Direction Evidence` plus a projection-basis chart. This evidence describes steering-basis activity and full-rank direction rendering; it is not a semantic identifiability proof or toric fan certificate.

2026-06-18 GraphCG evidence validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_herschel_5k_report.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py TropicalGT-I/tests/test_parameter_golf_review_loop.py TropicalGT-I/tests/test_metric_provenance.py
# 36 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/audit_metric_provenance.py --fail-on-uncovered
# findings=354 covered=354 uncovered=0
```

2026-06-18 NLL density evidence follow-up: Herschel now treats `got_nll_density_cloud_payload.json` as a required recorded GoT audit sidecar when it exists. `write_herschel_5k_report.py` exposes `tropicalgt.herschel_nll_density_evidence.v1`, but marks it available only when the payload carries `tropicalgt.nll_density_render.v1`, visible actual model GoT anchors, legend-only non-model Gaussian support samples, hidden support samples, PC3 z-axis policy, positive kernel bandwidth, NLL range, density-volume provenance, and no-proxy/no-fallback flags. Markdown and HTML reports now include `NLL Density Evidence` plus a visible-density-layer chart. Gaussian support samples are visualization support only, not model states, training data, or BPB evidence.

2026-06-18 NLL density evidence validation:

```text
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_herschel_5k_report.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py TropicalGT-I/tests/test_parameter_golf_review_loop.py TropicalGT-I/tests/test_metric_provenance.py
# 36 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/audit_metric_provenance.py --fail-on-uncovered
# findings=360 covered=360 uncovered=0
```

2026-06-18 chart/vector-bundle evidence follow-up: Herschel now treats `chart_bundle_transport_sidecar.json` as a required recorded GoT audit sidecar when it exists. `write_herschel_5k_report.py` exposes `tropicalgt.herschel_chart_bundle_transport_evidence.v1`, sourced only from recorded chart/vector-bundle transport payloads with exported chart metadata, monomial transport contracts, bundle matroid flat-incidence contracts, vector-bundle paper sidecars, completeness contracts, safety flags, and no-proxy/no-fallback markers. Available telemetry can be reported as partial or paper-ready according to the recorded completeness tier, but it is never promoted to a vector-bundle theorem certificate, toric embedding, tropical variety, global toric variety, normal-fan certificate, or restart justification by itself.

2026-06-18 chart/vector-bundle evidence validation:

```text
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/scripts/write_herschel_5k_report.py TropicalGT-I/scripts/prepare_5k_review_bundle.py TropicalGT-I/src/tropicalgt/provenance.py TropicalGT-I/tests/test_herschel_5k_report.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_herschel_5k_report.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py TropicalGT-I/tests/test_parameter_golf_review_loop.py TropicalGT-I/tests/test_metric_provenance.py
# 36 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/audit_metric_provenance.py --fail-on-uncovered
# findings=372 covered=372 uncovered=0
```

2026-06-18 toric/tropical CAS evidence follow-up: Herschel now treats `toric_embedding_sidecar.json` and `tropical_fan_diagnostics.json` as required recorded GoT audit sidecars when they exist. `write_herschel_5k_report.py` exposes `tropicalgt.herschel_toric_tropical_cas_evidence.v1`, sourced only from recorded finite toric-ideal and Macaulay2 Tropical fan diagnostic sidecars. Certified finite toric-ideal sidecars, certified tropical fan diagnostics, unavailable CAS states, backend statuses, missing no-proxy contracts, ray counts, and forbidden global/tropical/normal-fan claim flags are reported explicitly. This evidence is never promoted to a global toric-variety embedding, tropical-variety embedding, normal-fan certificate, vector-bundle theorem certificate, or BPB restart justification by itself.

2026-06-18 toric/tropical CAS evidence validation:

```text
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/scripts/write_herschel_5k_report.py TropicalGT-I/scripts/prepare_5k_review_bundle.py TropicalGT-I/src/tropicalgt/provenance.py TropicalGT-I/tests/test_herschel_5k_report.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_herschel_5k_report.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py TropicalGT-I/tests/test_parameter_golf_review_loop.py TropicalGT-I/tests/test_metric_provenance.py
# 36 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/audit_metric_provenance.py --fail-on-uncovered
# findings=382 covered=382 uncovered=0
```

2026-06-18 persistence-landscape PH evidence follow-up: Herschel now treats `trajectory_persistence/persistence_landscapes.json` as a required recorded GoT audit sidecar when it exists. `write_herschel_5k_report.py` exposes `tropicalgt.herschel_persistence_landscape_evidence.v1`, sourced only from recorded persistence-landscape visual contracts. Available evidence requires actual GUDHI `lambda_k(t)` rows, backend provenance, curve traces, finite interval evidence, no-proxy flags, and an explicit `not_nll_fitness_landscape`/not-norm-only contract. No-finite-interval cases can be reported as verified unavailable, but they are not converted into zero landscapes, norm-only summaries, NLL/fitness landscapes, or restart evidence.

2026-06-18 persistence-landscape PH evidence validation:

```text
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/scripts/write_herschel_5k_report.py TropicalGT-I/scripts/prepare_5k_review_bundle.py TropicalGT-I/tests/test_herschel_5k_report.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_herschel_5k_report.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py TropicalGT-I/tests/test_parameter_golf_review_loop.py TropicalGT-I/tests/test_metric_provenance.py
# 36 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/audit_metric_provenance.py --fail-on-uncovered
# findings=388 covered=388 uncovered=0
```

2026-06-18 bivariate module/free-resolution guard evidence follow-up: Herschel now treats `trajectory_level_radius_bifiltration.json` as a required recorded GoT audit sidecar when it exists. `write_herschel_5k_report.py` exposes `tropicalgt.herschel_bivariate_module_evidence.v1`, sourced only from the raw `F2[x_level,x_radius]` level/radius bifiltration payload. The report separates available module-grid evidence (fibers, structure maps, rank samples, chain generators) from nested `tropicalgt.real_free_resolution.v1` certification. Finite multigraded chain-presentation diagnostics remain useful but are not free resolutions; a free-resolution claim is available only when the nested CAS guard certifies exactness and multigraded resolution safety. Safe unavailable CAS guards are reported explicitly.

2026-06-18 bivariate module/free-resolution guard evidence validation:

```text
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/scripts/write_herschel_5k_report.py TropicalGT-I/scripts/prepare_5k_review_bundle.py TropicalGT-I/src/tropicalgt/provenance.py TropicalGT-I/tests/test_herschel_5k_report.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_herschel_5k_report.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py TropicalGT-I/tests/test_parameter_golf_review_loop.py TropicalGT-I/tests/test_metric_provenance.py
# 36 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/audit_metric_provenance.py --fail-on-uncovered
# findings=393 covered=393 uncovered=0
```

## 2026-06-18 Analogical Memory And Simplex-Tree Evidence Follow-Up

Herschel now treats `analogical_memory_retrieval.json` and `analogical_simplex_tree_analogy.json` as Herschel-required recorded GoT audit sidecars alongside `analogical_simplicial_maps.json` when those files exist in the latest `got_audit` directory. `write_herschel_5k_report.py` exposes `tropicalgt.herschel_analogical_memory_evidence.v1`, sourced only from recorded analogical retrieval, top-k probability-vector simplicial-map, and simplex-tree analogy sidecars.

Availability is deliberately conservative. Top-k analogical evidence requires `tropicalgt.analogical_topk.v1`, model-probability-vector retrieval, Jensen-Shannon assignment on model probability vectors, rejection of embedding-only assignment, `trajectory_probability_filtered_simplicial_object` query and codomain complexes, and no-proxy/no-fallback flags. Simplex-tree analogy evidence requires `tropicalgt.analogical_simplex_tree_analogy.v1`, source rows from `probability_simplicial_map.simplex_tree_map.rows`, no-proxy flags, and chain-map or persistence-module-morphism claims guarded by certified filtered simplicial maps. Empty or unqualified memory banks are reported as verified insufficient-memory states, not proxy analogies or restart evidence.

Validation:

```text
CUDA_VISIBLE_DEVICES="" /home/iska/miniconda3/envs/tokengt/bin/python -m py_compile TropicalGT-I/scripts/write_herschel_5k_report.py TropicalGT-I/scripts/prepare_5k_review_bundle.py TropicalGT-I/src/tropicalgt/provenance.py TropicalGT-I/tests/test_herschel_5k_report.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py
# passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_herschel_5k_report.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py
# 11 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q TropicalGT-I/tests/test_herschel_5k_report.py TropicalGT-I/tests/test_prepare_5k_review_bundle.py TropicalGT-I/tests/test_parameter_golf_review_loop.py TropicalGT-I/tests/test_metric_provenance.py
# 36 passed
CUDA_VISIBLE_DEVICES="" PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/audit_metric_provenance.py --fail-on-uncovered
# findings=412 covered=412 uncovered=0
```

## 2026-06-18 Simplicial Complex And Simplex-Tree Evidence Follow-Up

Herschel now treats the recorded full-trajectory complex payload, full-trajectory radius-slider contracts, full-trajectory SimplexTree poset contracts, and `reasoning_step_complex_maps/manifest.json` as Herschel-required audit sidecars when they exist in the latest `got_audit` directory. `write_herschel_5k_report.py` exposes `tropicalgt.herschel_simplicial_complex_evidence.v1`, sourced only from those recorded contracts.

Availability requires `tropicalgt.trajectory_complex_overlay_contract.v1`, `tropicalgt.radius_filtration_slider_contract.v1`, `tropicalgt.simplex_tree_poset.v1`, and `tropicalgt.reasoning_step_complex_maps.v1` evidence with actual-data/no-proxy flags. Radius sliders must start as disjoint vertices, grow min-to-max, keep solid edges/faces radius-gated, and remain monotone. SimplexTree posets must be GUDHI-backed, face-to-coface primary, not disconnected columns, and safe to render. Per-step manifests must confirm source contracts, sliders, and simplex-tree posets for every model-evaluated reasoning step. Missing or unsafe contracts remain unavailable and are not replaced by global trajectory plots or static proxy complexes.

Validation: focused Herschel/bundle tests passed (`11 passed`), broader Herschel/bundle/review-loop/provenance tests passed (`36 passed`), and metric provenance audit passed (`findings=423 covered=423 uncovered=0`).
