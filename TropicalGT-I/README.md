# TropicalGT-I

[![Paper: main PDF](https://img.shields.io/badge/arXiv--style-main%20PDF-b31b1b?style=for-the-badge&logo=readthedocs&logoColor=white)](https://github.com/amelie-iska/TropicalGT/raw/tropicalgt-i-implementation/references/main.pdf)
[![Paper: TropicalGT-I PDF](https://img.shields.io/badge/arXiv--style-TropicalGT--I%20PDF-b31b1b?style=for-the-badge&logo=readthedocs&logoColor=white)](https://github.com/amelie-iska/TropicalGT/raw/tropicalgt-i-implementation/TropicalGT-I/assets/tropicalgt_neurips_research_paper.pdf)
[![Hugging Face dataset](https://img.shields.io/badge/Hugging%20Face-dataset-ffcc4d?style=for-the-badge&logo=huggingface&logoColor=111111)](https://huggingface.co/datasets/AmeliSchreiber/TropicalGT)
[![Hugging Face checkpoints](https://img.shields.io/badge/Hugging%20Face-checkpoints-ffcc4d?style=for-the-badge&logo=huggingface&logoColor=111111)](https://huggingface.co/AmeliSchreiber/TropicalGT)
[![GitHub codebase](https://img.shields.io/badge/GitHub-codebase-24292f?style=for-the-badge&logo=github&logoColor=white)](https://github.com/amelie-iska/TropicalGT)

TropicalGT-I is the active v1 implementation for tropical-geometry-guided reasoning in transformer embedding space. It combines TokenGT-style graph tokenization, tropical ring attention, GFlowNet graph-of-thought trajectories, GraphCG latent steering, and finite filtered simplicial diagnostics.

The main optimization metric remains byte-level BPB for the OpenAI Parameter-Golf style baseline. TropicalGT-I also reports graph-aware BPB variants for TokenGT conditioning:

- `bpb`: exact text bits-per-byte over non-padding UTF-8 target bytes.
- `graph_bpb`: NLL bits plus explicit graph side-information bits divided by target bytes plus graph structural bytes.
- `graph_sideinfo_bpb`: side-information-charged BPB on text bytes.
- `graph_conditioned_bpb_no_side_cost`: optimistic graph-conditioned BPB when graph structure is treated as already present in context.

The graph structural byte budget is derived from the actual TokenGT graph tuple: masks determine live graph tokens, node counts determine endpoint-id width, and edge endpoint ids are charged when supplied.

All training records are graph structured. The moved TropicalGT reasoning shards carry graph JSON and also receive deterministic sequential text path graphs. The OpenAI Parameter-Golf FineWeb stream is loaded from `external/oai-parameter-golf/data/datasets/fineweb10B_sp1024` and decoded with `external/oai-parameter-golf/data/tokenizers/fineweb_1024_bpe.model`; `train_full_dataset_active.json` keeps the older `external/parameter-golf` checkout as a compatibility fallback. Every sampled token window becomes a causal sequential DAG before TokenGT tokenization. Graphs with causal DAG structure are decoded autoregressively in topological order. Cyclic or explicitly non-causal graphs use deterministic seeded random autoregressive order.

The active full-dataset run uses `configs/train_full_dataset_active.json`. It requires both the moved Hugging Face reasoning shards and the full OpenAI Parameter-Golf SP1024 cache, and its startup data-budget gate requires more than `10,000,000,000` configured training token slots. The active shape is `seq_len: 1024`, `batch_size: 4`, `checkpoint_every: 1000`, `validation_every_steps: 500`, `visualization_every_steps: 5000`, and `max_steps: 2500000`, which schedules `10,240,000,000` sequence-token slots. Its budget report currently covers `117` HF train parquet shards plus `195` OAI train shards for `24,198,796,288` available train token slots.

The older `configs/train.json` remains a cap-sized review configuration unless its budget audit is explicitly refreshed.

W&B metrics are logged in priority-ordered namespaces so the dashboard opens around the important quantities first:

- `00_primary`: BPB, graph-BPB, loss, NLL, VRAM.
- `01_losses`: objective decomposition and weighted regularizers.
- `02_bpb`: byte, graph, and side-information accounting.
- `03_tropical`: support, margin, wall, sequence-ring, and certificate metrics.
- `04_gflownet`: trajectory-balance and reward diagnostics.
- `05_graphcg`: full-rank GraphCG diagnostics and direction spectra.
- `06_graph_data`: graph-token counts, causal/random graph AR rates, and OAI source rate.
- `07_algebra_topology`, `08_memory`, `09_meet_in_middle`, `10_system`, and `11_optimization`: topology/algebra, analogical memory, optional bidirectional decoding, throughput/VRAM, and optimizer/sampler metrics.

The online project has been cleaned so only the latest useful smoke run remains (`pzpw99m7`). Local `wandb/` directories are disposable smoke artifacts and can be removed after syncing useful runs.

All tropical margins, GraphCG factors, GFlowNet rewards, persistence summaries, and analogical memory metrics should be ablated by whether they improve held-out `bpb` and `graph_bpb`. GraphCG training now includes an explicit full-rank singular-value barrier and logs effective rank, numerical rank, singular min/max, and condition proxies so latent steering directions cannot collapse unnoticed.

Run `scripts/analyze_bpb_ablations.py` on one or more `train_report.json` files to generate JSON/Markdown/HTML screens ranking which logged metrics correlate with `bpb`, `graph_bpb`, `eval_bpb`, and `eval_graph_bpb`. The report also emits best-by-target summaries with the lowest run, baseline delta, runner-up, and runner-up margin for each BPB target. Use those rankings to choose matched ablations; do not treat correlations as causal wins. Run `scripts/run_bpb_ablation_grid.py` to generate same-seed variants such as `no_graphcg`, `no_gflownet`, `no_certificate`, `no_tropical_regularizers`, and `no_auxiliary`, optionally train them in sequence, and immediately analyze their BPB deltas.

Use `configs/gpu_ablation.json` for bounded data-backed RTX 4090 ablation ladders between smoke tests and the active full-dataset `configs/train_full_dataset_active.json` run. It keeps the moved parquet dataset, TokenGT graphification, topology audits, graph-BPB accounting, and dark Plotly correlation report, but caps train/validation records and step count so candidate auxiliary losses can be screened before longer training. The built-in grid variants include zero-ablation controls and nonzero half/quarter-weight settings such as `aux_0p5x`, `aux_0p25x`, `gflownet_0p25x`, `graphcg_0p25x`, and `tropical_0p25x`.

`meet_in_middle.enabled` optionally activates the graph-aware meet-in-the-middle adaptation. It scores the same graph-autoregressive byte stream left-to-right and right-to-left, using topological order for causal DAGs and deterministic random order for non-causal graphs. It is off by default because the reverse pass adds memory and compute; use `eval_tropicalgt_i.py --meet-in-middle` or `infer_tropicalgt_i.py --meet-in-middle` for inspection before adding nonzero training weights.

## Papers and Assets

- [Main TropicalGT paper](../references/main.pdf)
- [TropicalGT-I NeurIPS research paper](./assets/tropicalgt_neurips_research_paper.pdf)

## Runtime

From the repository root, use the `tokengt` environment with `PYTHONPATH=TropicalGT-I/src` for training, evaluation, validation, inference, and visualization scripts. The top-level README contains the full command inventory for CPU smoke tests, GPU smoke runs, data-backed training, checkpoint resume, and readiness audits.

The readiness audit should be treated as the pre-training acceptance check for this phase: it gates checkpoint reload, data-backed TokenGT conversion, sequential text graphification, finite eval, `bpb`, `graph_bpb`, topology-audit execution, ablation-tool availability, and generated visualization artifacts before a longer run is considered ready.


Before launching the full run, populate the full SP1024 OAI cache and audit the token budget:

```bash
cd external/oai-parameter-golf
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 195
cd ../..
PYTHONPATH=TropicalGT-I/src \
python \
TropicalGT-I/scripts/audit_training_data_budget.py \
--config TropicalGT-I/configs/train_full_dataset_active.json \
--output TropicalGT-I/outputs/train_full_dataset_active/data_budget.json
```

The audit must show both `tropicalgt_hf_reasoning` and `openai_parameter_golf`, at least `10,000,000,000` available train token slots, and at least `10,000,000,000` configured training token slots. Training startup runs the same guard before allocating the model, so a partial OAI cache fails fast instead of silently training on the wrong corpus.

For the active full-dataset `configs/train_full_dataset_active.json` path, run the readiness audit with `--train-dry-run --require-cuda --check-wandb-key` before launching the long job. That dry run samples the moved parquet train split, builds the configured model, performs one optimizer step on CUDA, and gates finite train loss, BPB, graph-BPB, and gradient norm.

For 5K-step BPB review, run:

```bash
PYTHONPATH=TropicalGT-I/src \
python \
TropicalGT-I/scripts/parameter_golf_codex_review_loop.py \
--config TropicalGT-I/configs/train.json \
--python python \
--review-every-steps 5000 \
--target-bpb 1.18 \
--max-total-steps 20000 \
--restart-policy beginning
```

Each boundary writes `active_training_contract_step_*.json/.md` with the active metrics, hyperparameters, losses, objectives, regularizers, dataset rates, VRAM/throughput, and restart decision. If `eval.bpb` is absent or above `1.18`, the generated Codex prompt requests a single-agent review of metrics, losses, hyperparameters, BPB behavior, and restart strategy.

The active full-dataset config validates every `500` steps and renders heavier browser/topology artifacts every `5000` steps. Each periodic validation round writes a step-local bundle under `TropicalGT-I/outputs/train_full_dataset_active/periodic/step_XXXXXXXX/`:

- `validation_report.json`: validation NLL, BPB, graph-BPB, graph source rates, and optional detailed record diagnostics.
- `reasoning/reasoning_trajectory_3d.html`: dark-mode PCA of graph-of-thought/graph-state embeddings with hover-rendered filtered simplicial complexes.
- `reasoning/reasoning_trajectory_pca_nll.html`: PCA/NLL view with exact model-evaluated NLL anchors and a surface-contact projection contract that keeps every plotted reasoning point on the displayed NLL energy surface.
- `reasoning/reasoning_topological_algebra.json`: per-node persistence and algebra diagnostics when the audit level is enabled.
- `got_audit/`: graph-of-thought scaling tree, GoT trajectory PCA, GraphCG direction cosines, tropical support heatmap, persistence barcodes, commutative algebra/free-resolution JSON, derived signatures, trajectory-growth topology, analogical memory retrieval, and an `inference_audit.html` dashboard.
- `graphcg/`: full-rank GraphCG Gram, PCA, and singular-value plots.
- `metrics/training_metrics.html`: live dark-mode metric traces up to the current step.
- `periodic_validation_artifacts.json`: manifest of every artifact path for that step.

The rolling manifest is `TropicalGT-I/outputs/train_full_dataset_active/periodic/manifest.jsonl`. W&B logs the scalar validation metrics at the same step and uploads the first configured HTML artifacts under `periodic_eval/step_XXXXXXXX/...`.

The visualization renderer is intentionally audit-oriented rather than decorative. Graph-of-thought trajectories are rendered from model `graph_state` embeddings and graph-of-thought candidate records. In the NLL landscape, every rendered trajectory marker and every edge endpoint uses the displayed surface z-value (`plot.z_surface`); the raw centered/scaled NLL remains in the payload for audit, but markers are projected onto the single displayed NLL mesh so they visibly touch it. Microstep interpolation markers and global surrogate NLL surfaces are disabled; unavailable model outputs write explicit `available: false` diagnostics instead of substitute geometry. Filtered simplicial objects are rendered as dark SVG/Plotly side-panel views with a scalar filtration-radius slider and multiparameter persistence data retained in the JSON payload. Long causal text-path complexes wrap into lanes for legibility.

The full GoT audit bundle now separates Euclidean embedding-radius and model-probability filtrations. The primary trajectory complex is `got_full_trajectory_complex.html`; its simplex-tree inclusion poset is `got_full_trajectory_simplex_tree_3d.html`. The Jensen-Shannon probability complex is `got_full_trajectory_complex_jensen_shannon.html`; its simplex tree is `got_full_trajectory_simplex_tree_3d_jensen_shannon.html`. Each sampled reasoning state also receives both `reasoning_step_###.html` and `reasoning_step_###_simplex_tree.html`. All radius sliders start at the disjoint 0-simplex cloud and grow min-to-max by true filtration threshold.

For browser review, generate the sample-first dashboard for any periodic audit directory:

```bash
PYTHONPATH=TropicalGT-I/src python \
TropicalGT-I/scripts/build_sample_browser_index.py \
TropicalGT-I/outputs/train_full_dataset_active/periodic/step_XXXXXXXX/got_audit \
--output TropicalGT-I/outputs/train_full_dataset_active/periodic/step_XXXXXXXX/got_audit/codex_browser_index.html
```

That index makes each sampled row/input the top-level unit, with links below it for that sample's GoT trajectory, NLL landscape, embedding map, full radius complex, full simplex tree, Jensen-Shannon probability complex, probability simplex tree, every reasoning-step complex and simplex tree, persistence/vectorized topology, per-rank analogical memory maps, GraphCG directions, tropical-support audit, raw sample index, and generated inference dashboard. The validator checks those per-sample buttons and relative targets.

Validate the generated interactive audit bundle with:

```bash
PYTHONPATH=TropicalGT-I/src python \
TropicalGT-I/scripts/validate_interactive_audit_artifacts.py \
--audit-root TropicalGT-I/outputs/train_full_dataset_active/periodic/step_XXXXXXXX/got_audit \
--min-rows 3 --min-candidates 8 --min-depth 2 \
--json-output TropicalGT-I/outputs/train_full_dataset_active/periodic/step_XXXXXXXX/interactive_artifact_validation.json \
--markdown-output TropicalGT-I/outputs/train_full_dataset_active/periodic/step_XXXXXXXX/interactive_artifact_validation.md
```

The trajectory persistence bundle now includes `trajectory_persistence/persistence_representations.html` and `trajectory_persistence/persistence_landscapes.html`. The representations page uses GUDHI vector methods to plot fast persistence landscapes, Betti curves, persistence images, silhouettes, persistence lengths, topological vectors, and entropy summaries. The landscapes page plots the actual GUDHI landscape functions `lambda_k(t)` over trajectory-growth levels, plus a heatmap for the first finite homology dimension available in the audit. These vectorized features are also summarized under the algebra/topology metric namespace for train/eval/W&B use. They are cached NumPy/scikit-learn features in this implementation, suitable for rewards, retrieval keys, diagnostics, and ablations; they are not PyTorch autograd losses unless replaced by a differentiable surrogate.

Run `TropicalGT-I/scripts/audit_metric_provenance.py --fail-on-uncovered` to refresh and gate `TropicalGT-I/outputs/metric_provenance_audit.{json,md}`. The registry labels exact BPB, graph-conditioned BPB, GraphCG condition diagnostics, surface-contact NLL diagnostics, unavailable visualization diagnostics, persistence vectorizations, algebraic free-resolution proxies, graph JSON fallbacks, and stripped-export estimates. The default scan covers active source, scripts, READMEs, and the current visualization-audit note; historical planning logs can be scanned explicitly when needed.

When reviewing early runs, treat tropical active-support collapse as a first-priority diagnostic. A heatmap where most graph tokens select the graph/root token is not a rendering failure; it means the model is using a very low-entropy tropical support pattern. Compare that against BPB, graph-BPB, graph-sideinfo BPB, and graph-conditioned BPB before increasing auxiliary weights.

Full inference audits can optionally emit the same family of graph-of-thought PCA trajectories, tropical support heatmaps, persistence barcodes, multiparameter algebra/free-resolution JSON, derived signatures, GraphCG direction plots, and analogical memory retrieval artifacts. The trajectory HTML is dark-mode; hovering over a reasoning node renders the node's filtered simplicial object in a cursor-following hover card and in the persistent side panel, and the 3D NLL view uses exact sampled NLL anchors plus a surface-contact projection contract so trajectory points and edge endpoints touch the rendered NLL landscape. Analogical-memory retrieval is rendered as per-rank 3D simplicial-map pages between query and memory `trajectory_probability_filtered_simplicial_object` complexes. The companion `analogical_simplicial_maps.json` reports `model_probability_jensen_shannon_assignment` vertex maps, Jensen-Shannon and assignment-cost summaries, filtration distortion, edge/2-simplex preservation, GUDHI SimplexTree provenance, persistent-homology similarity, free-resolution similarity, and derived-signature similarity. Legitimate unavailable states are explicit, such as `missing_model_probability_query_complex`, `missing_model_probability_codomain_complex`, or `no_non_self_model_memory`. Use `--audit-all` for the complete optional bundle:

```bash
PYTHONPATH=TropicalGT-I/src \
python \
TropicalGT-I/scripts/infer_tropicalgt_i.py \
--config TropicalGT-I/configs/train_full_dataset_active.json \
--checkpoint TropicalGT-I/checkpoints/tropicalgt_i_train_full_dataset_active.latest.pt \
--prompt "Question: add 2 and 3 Answer:" \
--audit-all \
--scale-depth 2 \
--scale-width 3 \
--scale-branch-factor 2 \
--memory-save \
--audit-output-dir TropicalGT-I/outputs/train_full_dataset_active/inference_full_audit \
--output TropicalGT-I/outputs/train_full_dataset_active/inference_full_audit.json
```

Standalone evaluation can also render the validation visualizations:

```bash
PYTHONPATH=TropicalGT-I/src \
python \
TropicalGT-I/scripts/eval_tropicalgt_i.py \
--config TropicalGT-I/configs/train_full_dataset_active.json \
--checkpoint TropicalGT-I/checkpoints/tropicalgt_i_train_full_dataset_active.latest.pt \
--split validation \
--audit-level full \
--audit-ph-backend gudhi \
--render-visualizations \
--visualization-output-dir TropicalGT-I/outputs/train_full_dataset_active
```
