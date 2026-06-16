# TropicalGT-I b60 Step-5000 Codex Evidence Review

Date: 2026-06-16
Remote host/worktree: `iska-tailscale:/home/iska/Documents/amelie/bio/TropicalGT`
Branch/commit checked: `tropicalgt-i-real-cas-no-proxy-20260614` at `e1c9c1c`
Scope: real b60 step-5000 evidence only. No proxies, no fallbacks, no checkpoint-metric substitution, no fabricated analyses.

## Decision

`tropicalgt.restart_decision.v1` spirit decision:

```yaml
allowed_action: blocked_missing_evidence_no_restart
primary_metric: eval.bpb
primary_target: 1.12
observed_eval_bpb: 1.4304583543547733
target_result: target_missed
training_launched: false
proposed_config_changes: []
blocker: empty_checkpoint:TropicalGT-I/checkpoints/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate.latest.pt
```

The 5K BPB target was missed, but I do not recommend or launch a restart. The required checkpoint-dependent executable review evidence is unavailable because the referenced step-5000 checkpoint is a regular empty file of 0 bytes. The path-only review bundle records `execution_readiness.ready: false`, `command_results: []`, and the exact issue `empty_checkpoint:TropicalGT-I/checkpoints/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate.latest.pt`. Under the no-proxy policy, this blocks hyperparameter/config changes because any proposed change would require real command results or sidecar evidence for that change.

## Primary Evidence Paths

All relative paths are rooted at `/home/iska/Documents/amelie/bio/TropicalGT`.

- Config: `TropicalGT-I/outputs/launch_configs/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate.json`
- Output dir: `TropicalGT-I/outputs/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate`
- Stop record: `TropicalGT-I/outputs/training_stop_records/b60_fresh_step0_casrows_step5000_gate.json`
- Step-5000 validation: `TropicalGT-I/outputs/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate/periodic/step_00005000/validation_report.json`
- Step-5000 artifact summary: `TropicalGT-I/outputs/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate/periodic/step_00005000/periodic_validation_artifacts.json`
- Step-5000 audit dashboard: `TropicalGT-I/outputs/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate/periodic/step_00005000/got_audit/inference_audit.html`
- Path-only review bundle: `TropicalGT-I/outputs/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate/post_5k_review_bundle/review_bundle_step_00005000.json`
- Codex prompt: `TropicalGT-I/outputs/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate/post_5k_review_bundle/codex_review_step_00005000.md`
- Checkpoint: `TropicalGT-I/checkpoints/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate.latest.pt`, state `regular empty file | 0 bytes | 2026-06-16 11:10:26 +0000`

## Observed 5K Metrics

Source: `periodic/step_00005000/validation_report.json` and mirrored in `periodic_validation_artifacts.json`.

| Metric | Value | Status |
|---|---:|---|
| `eval.bpb` / `bpb_exact` | 1.4304583543547733 | Missed target `< 1.12` |
| `eval.text_bpb` | 1.4304583543547733 | Same as BPB |
| `eval.graph_bpb` | 20.122144813809587 | Secondary graph metric high |
| `eval.graph_conditioned_bpb_no_side_cost` | 1.2576335315794374 | Still above primary target |
| `eval.graph_sideinfo_bpb` | 22.88734311997977 | Side-cost dominated |
| `eval.nll` | 0.9915181752294302 | Observed validation NLL |
| `eval.nll_bits` | 93746.51871099442 | Observed validation bits |
| `eval.ppl` | 2.6953233408858814 | Observed perplexity |
| `eval.invalid_graph_rate` | 0.0 | No invalid graphs observed |
| `eval.graph_json_fallback_records` | 0 | No graph JSON fallback records |
| `eval.tokens` | 65536 | Two validation batches |
| `eval.graph_tokens` | 4588 | Node tokens 2315, edge tokens 2209 |
| `eval.parameter_golf_source_rate` | 0.34375 | Parameter-golf source fraction |
| `eval.mim_enabled` | 0.0 | Meet-in-middle metric disabled in validation |

Target result: the run did not meet the b60 target BPB of `< 1.12`.

## Artifact Inventory Summary

Available real artifacts:

- Launch config exists, 10429 bytes: `TropicalGT-I/outputs/launch_configs/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate.json`
- Stop record exists, 1019 bytes, with `target_step: 5000`, `latest_step: 5000`, `latest_loss_nll: ["0.992", "0.972"]`, and `action: target_reached_terminate`: `TropicalGT-I/outputs/training_stop_records/b60_fresh_step0_casrows_step5000_gate.json`
- Validation report exists, 57046036 bytes: `TropicalGT-I/outputs/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate/periodic/step_00005000/validation_report.json`
- Periodic validation artifact summary exists, 13994 bytes: `TropicalGT-I/outputs/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate/periodic/step_00005000/periodic_validation_artifacts.json`
- Audit dashboard exists, 7800 bytes: `TropicalGT-I/outputs/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate/periodic/step_00005000/got_audit/inference_audit.html`
- Review bundle exists, 42176 bytes, path-only with no command results: `TropicalGT-I/outputs/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate/post_5k_review_bundle/review_bundle_step_00005000.json`
- Codex prompt exists, 58635 bytes: `TropicalGT-I/outputs/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate/post_5k_review_bundle/codex_review_step_00005000.md`
- Local training log exists, 1087668 bytes: `TropicalGT-I/outputs/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate/logs/train_nohup_20260616T053650Z.log`
- Local W&B run metadata exists at `wandb/run-20260616_053659-ld5u55p5`, with run id `ld5u55p5` and run name `tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate` from the training log. No JSON summary/history file was found under that run directory; only `run-ld5u55p5.wandb` and metadata/log files were present.

Unavailable or blocked artifacts:

- Checkpoint is unavailable as a model state: `TropicalGT-I/checkpoints/tropicalgt_i_pg_bpb_step0_full24b_b60_20260616T053631Z_fresh_bpb112_casrows_5k_gate.latest.pt` is 0 bytes.
- Checkpoint-dependent executable post-5K evaluation/visualization is blocked. Evidence: review bundle `execution_readiness.ready: false`, `issues: ["empty_checkpoint:..."]`, and `command_results: []` in `post_5k_review_bundle/review_bundle_step_00005000.json`.
- `post_5k_review_bundle/eval_visualizations` is not available because the checkpoint-dependent `eval_tropicalgt_i.py --render-visualizations` command was not runnable under the empty-checkpoint readiness gate.
- W&B summary/history JSON is unavailable locally. Evidence: `wandb/run-20260616_053659-ld5u55p5` contains `files/wandb-metadata.json`, logs, and `run-ld5u55p5.wandb`, but no summary/history JSON path.

## Topology, Geometry, Algebra, CAS, Memory, GraphCG, GFlowNet, Tropical, Decoding

Topology and topological algebra evidence available:

- `got_audit/inference_topology.json` exists, 4397 bytes. It records an F2 chain complex with chain-group rank `{0: 96}`, Betti `{0: 96}`, Euler characteristic 96, Gudhi persistence `available: true`, and no persistence intervals. Graph metrics are `nodes: 124`, `edges: 7626`, `weak_components: 1`, `strong_components: 124`, `is_dag: true`, `dag_longest_path_length: 123`, `undirected_cycle_rank: 7503`, `density: 0.5`.
- `periodic/step_00005000/reasoning/reasoning_topological_algebra.json` exists, 53975290 bytes. The sampled record has the same topology summary under `topological_algebra`, including Gudhi availability and the derived-equivalence warning.
- `got_audit/trajectory_persistence/two_parameter_bifiltration.json` exists, 2216 bytes, with `actual_data_only: true`, `no_proxy_resolution_claim: true`, coefficient ring `F2[x_level,x_radius]`, axes `x_radius` and `x_level`, grid `x_level_grades: [0,1,2,3]`, `x_radius_grades: [0..9]`, 40 fiber rows, `h0_min = h0_max = 1.0`, `h1_min = 0.0`, `h1_max = 40.0`, and 151 total chain generators.
- Persistence and trajectory visualizations exist: `got_audit/persistence_barcode.html`, `got_audit/persistence_module_betti.html`, `got_audit/persistence_representations.html`, `got_audit/persistence_landscapes.html`, and the corresponding `got_audit/trajectory_persistence/*.html` files.
- Large topology payloads exist and were inventoried but not reinterpreted beyond their sidecar summaries because they are bulky generated evidence: `got_audit/trajectory_growth_topology.json` at 2581776010 bytes and `got_audit/inference_scaling_tree.json` at 3906518000 bytes.

Geometry evidence available:

- `got_audit/got_nll_density_cloud_payload.json` exists, 20814 bytes. It marks `available: true`, `actual_model_anchor_count: 12`, `support_sample_count: 1320`, `sample_points_are_model_states: false`, and explicitly says Gaussian cloud points are not model states while large labeled markers are actual model-evaluated GoT states.
- `got_audit/got_nll_density_cloud_pca_3d.html`, `got_audit/got_trajectory_pca_3d.html`, `got_audit/got_embedding_map_3d.html`, `reasoning/reasoning_trajectory_3d.html`, and `reasoning/reasoning_trajectory_pca_nll.html` exist as geometric visualizations.
- The NLL density payload reports root mean NLL 1.7013278007507324, terminal mean NLL 1.70064035483769, best terminal improvement from root 0.0011063814163208008, and improving edge fraction 0.5454545454545454. These are GoT audit geometry metrics, not checkpoint-recovered metrics.

Algebra and CAS evidence:

- `got_audit/inference_algebra.json` exists but is `{}` and contains no algebraic evidence.
- `got_audit/tropical_fan_diagnostics.json` exists, 1006 bytes, and marks fan/CAS evidence unavailable: `available: false`, `status: unavailable_no_model_derived_tropical_ideal`, backend `Macaulay2`, `certificate_attached: false`, `tropical_cycle_certified: false`, and `safe_to_render_as_tropical_fan: false`.
- The exact unavailable reason is: no explicit model-derived tropical ideal spec was exported. Expected one of `model_derived_tropical_ideal`, `tropical_ideal`, `tropical_fan_ideal`, or `tropical_variety_ideal` with variables and generators.
- `got_audit/tropical_fan_diagnostics.html` exists, but it renders the unavailable diagnostic state rather than a certified fan.

Memory evidence:

- `memory_bank/trajectory_memories.jsonl` exists but is 0 bytes.
- `got_audit/analogical_memory_retrieval.json` exists, 1667 bytes. It reports `bank_size: 0`, `records_added: 0`, one candidate seen, zero eligible, one rejected, reason `nll_improvement_below_threshold`, and `retrieved: []`.
- `got_audit/analogical_simplicial_maps.json` exists and says `available: false`, `reason: no_non_self_model_memory`, `maps: []`.
- Memory HTML views exist (`analogical_memory_retrieval.html`, `analogical_memory_topk_index.html`, `analogical_memory_map_02.html`) but the underlying JSON says no retrievals and no non-self memory maps.

GraphCG evidence:

- `got_audit/graphcg_direction_cosines_payload.json` exists, 204030 bytes. It reports `available: true`, matrix shape `[12, 1760]`, `full_rank_direction_count: 1760`, `display_count: 1760`, basis source `effective_full_rank_qr`, candidate count 12, all candidates have all direction cosines, `mean_abs_offdiag_cosine_max: 0.002083882689476013`, and `max_abs_offdiag_cosine_max: 0.30993935465812683`.
- GraphCG visualizations exist at `periodic/step_00005000/graphcg/graphcg_direction_gram.html`, `graphcg_direction_pca.html`, `graphcg_direction_singular_values.html`, and `got_audit/graphcg_direction_cosines.html`.

GFlowNet evidence:

- Config evidence exists for a nonzero GFlowNet weight in `TropicalGT-I/outputs/launch_configs/...5k_gate.json`, but no GFlowNet metric keys were found in `periodic_validation_artifacts.json` or `validation_report.json` for step 5000. Under the no-proxy policy, this is unavailable as performance evidence and cannot justify a config change.

Tropical support evidence:

- `got_audit/tropical_support_payload.json` exists, 351107 bytes. It reports real model support probabilities and an assignment matrix, not fabricated scores. The metrics include `token_count: 124`, `unique_support_count: 3`, `effective_supports: 2.0842928761090294`, support entropy 1.0595580133278242 bits, top support collapse rate 0.5, strict wall-hit rate 0.0, near-wall hit rate 0.008064516129032258, and support probability source `model_tropical_support_probabilities`.
- `got_audit/tropical_support_heatmap.html` exists. Tropical fan diagnostics are unavailable as CAS/fan evidence for the exact reason listed above.

Decoding evidence:

- Real validation metrics show graph autoregressive decoding was enabled: `graph_autoregressive_decoding_enabled: 1.0`.
- Causal DAG autoregressive rate was 1.0 and random graph autoregressive rate was 0.0.
- Invalid graph rate was 0.0 and graph JSON fallback records were 0.
- Config says `graph_decoding_policy.causal_dag: forward_plus_reverse_causal_directions`, `graph_decoding_policy.cyclic_or_noncausal: ROAR_random_order_autoregressive`, and `meet_in_middle_toggle: true`, but the validation metric `eval_mim_enabled` is 0.0.

## Config Change Review

No hyperparameter or config changes are proposed.

| Dot path | Old value | New value | Evidence paths | Expected BPB effect | Risk |
|---|---:|---:|---|---|---|
| none | n/a | n/a | n/a | n/a | n/a |

Reason: the target was missed, but checkpoint-dependent required evidence is blocked by the empty checkpoint. The available sidecars are sufficient to state observed metrics and available/unavailable evidence, but not sufficient to make an evidence-backed restart patch under the no-proxy rule.

## Final Action

No new training run was started. The correct action for this evidence state is `blocked_missing_evidence_no_restart` until a non-empty step-5000 checkpoint or other required real checkpoint-dependent executable evidence is available.
