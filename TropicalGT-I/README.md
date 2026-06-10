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

All training records are graph structured. The moved TropicalGT reasoning shards carry graph JSON and also receive deterministic sequential text path graphs. The OpenAI Parameter-Golf FineWeb stream is loaded from `external/parameter-golf/data/datasets/fineweb10B_sp1024` and decoded with `external/parameter-golf/data/tokenizers/fineweb_1024_bpe.model`; every sampled token window becomes a causal sequential DAG before TokenGT tokenization. Graphs with causal DAG structure are decoded autoregressively in topological order. Cyclic or explicitly non-causal graphs use deterministic seeded random autoregressive order.

The current full `configs/train.json` model is cap-sized for Parameter Golf: width/hidden size `1760`, memory width `220`, about `38.6M` raw parameters, and an estimated stripped int8+zlib competition export of `15,633,708` bytes. The CUDA training shape is `seq_len: 1024`, `batch_size: 80`, and `checkpoint_every: 5000`; the latest hybrid readiness dry run allocated `21,484 MB` on the RTX 4090 and passed all gates. Tropical ring attention is used in two places without duplicating parameters: exact blockwise graph-token attention (`graph_tropical_block_size: 32`) and pooled long-context sequence attention (`sequence_tropical_max_tokens: 32`, `sequence_tropical_block_size: 16`, residual weight `0.125`).

W&B metrics are logged in priority-ordered namespaces so the dashboard opens around the important quantities first:

- `00_primary`: BPB, graph-BPB, loss, NLL, VRAM.
- `01_losses`: objective decomposition and weighted regularizers.
- `02_bpb`: byte, graph, and side-information accounting.
- `03_tropical`: support, margin, wall, sequence-ring, and certificate metrics.
- `04_gflownet`: trajectory-balance and reward diagnostics.
- `05_graphcg`: full-rank GraphCG diagnostics and direction spectra.
- `06_graph_data`: graph-token counts, causal/random graph AR rates, and OAI source rate.
- `07_algebra_topology`, `08_memory`, `09_system`, and `10_optimization`: topology/algebra, analogical memory, throughput/VRAM, and optimizer/sampler metrics.

The online project has been cleaned so only the latest useful smoke run remains (`pzpw99m7`). Local `wandb/` directories are disposable smoke artifacts and can be removed after syncing useful runs.

All tropical margins, GraphCG factors, GFlowNet rewards, persistence summaries, and analogical memory metrics should be ablated by whether they improve held-out `bpb` and `graph_bpb`. GraphCG training now includes an explicit full-rank singular-value barrier and logs effective rank, numerical rank, singular min/max, and condition proxies so latent steering directions cannot collapse unnoticed.

Run `scripts/analyze_bpb_ablations.py` on one or more `train_report.json` files to generate JSON/Markdown/HTML screens ranking which logged metrics correlate with `bpb`, `graph_bpb`, `eval_bpb`, and `eval_graph_bpb`. The report also emits best-by-target summaries with the lowest run, baseline delta, runner-up, and runner-up margin for each BPB target. Use those rankings to choose matched ablations; do not treat correlations as causal wins. Run `scripts/run_bpb_ablation_grid.py` to generate same-seed variants such as `no_graphcg`, `no_gflownet`, `no_certificate`, `no_tropical_regularizers`, and `no_auxiliary`, optionally train them in sequence, and immediately analyze their BPB deltas.

Use `configs/gpu_ablation.json` for bounded data-backed RTX 4090 ablation ladders between smoke tests and the full `configs/train.json` run. It keeps the moved parquet dataset, TokenGT graphification, topology audits, graph-BPB accounting, and dark Plotly correlation report, but caps train/validation records and step count so candidate auxiliary losses can be screened before longer training. The built-in grid variants include zero-ablation controls and nonzero half/quarter-weight settings such as `aux_0p5x`, `aux_0p25x`, `gflownet_0p25x`, `graphcg_0p25x`, and `tropical_0p25x`.

## Papers and Assets

- [Main TropicalGT paper](../references/main.pdf)
- [TropicalGT-I NeurIPS research paper](./assets/tropicalgt_neurips_research_paper.pdf)

## Runtime

From the repository root, use the `tokengt` environment with `PYTHONPATH=TropicalGT-I/src` for training, evaluation, validation, inference, and visualization scripts. The top-level README contains the full command inventory for CPU smoke tests, GPU smoke runs, data-backed training, checkpoint resume, and readiness audits.

The readiness audit should be treated as the pre-training acceptance check for this phase: it gates checkpoint reload, data-backed TokenGT conversion, sequential text graphification, finite eval, `bpb`, `graph_bpb`, topology-audit execution, ablation-tool availability, and generated visualization artifacts before a longer run is considered ready.

For the full `configs/train.json` path, run the readiness audit with `--train-dry-run --require-cuda --check-wandb-key` before launching the long job. That dry run samples the moved parquet train split, builds the configured model, performs one optimizer step on CUDA, and gates finite train loss, BPB, graph-BPB, and gradient norm.

For 5K-step BPB review, run:

```bash
PYTHONPATH=TropicalGT-I/src \
/home/iska/miniconda3/envs/tokengt/bin/python \
TropicalGT-I/scripts/parameter_golf_codex_review_loop.py \
--config TropicalGT-I/configs/train.json \
--python /home/iska/miniconda3/envs/tokengt/bin/python \
--review-every-steps 5000 \
--target-bpb 1.18 \
--max-total-steps 20000 \
--restart-policy beginning
```

Each boundary writes `active_training_contract_step_*.json/.md` with the active metrics, hyperparameters, losses, objectives, regularizers, dataset rates, VRAM/throughput, and restart decision. If `eval.bpb` is absent or above `1.18`, the generated Codex prompt requests a single-agent review of metrics, losses, hyperparameters, BPB behavior, and restart strategy.

The full training config validates and renders artifacts every `250` steps. Each periodic validation round writes a step-local bundle under `TropicalGT-I/outputs/train/periodic/step_XXXXXXXX/`:

- `validation_report.json`: validation NLL, BPB, graph-BPB, graph source rates, and optional detailed record diagnostics.
- `reasoning/reasoning_trajectory_3d.html`: dark-mode PCA of graph-of-thought/graph-state embeddings with hover-rendered filtered simplicial complexes.
- `reasoning/reasoning_trajectory_pca_nll.html`: 2D PCA with NLL as height and a smoothed heatmapped NLL surface.
- `reasoning/reasoning_topological_algebra.json`: per-node persistence and algebra diagnostics when the audit level is enabled.
- `got_audit/`: graph-of-thought scaling tree, GoT trajectory PCA, GraphCG direction cosines, tropical support heatmap, persistence barcodes, commutative algebra/free-resolution JSON, derived signatures, trajectory-growth topology, analogical memory retrieval, and an `inference_audit.html` dashboard.
- `graphcg/`: full-rank GraphCG Gram, PCA, and singular-value plots.
- `metrics/training_metrics.html`: live dark-mode metric traces up to the current step.
- `periodic_validation_artifacts.json`: manifest of every artifact path for that step.

The rolling manifest is `TropicalGT-I/outputs/train/periodic/manifest.jsonl`. W&B logs the scalar validation metrics at the same step and uploads the first configured HTML artifacts under `periodic_eval/step_XXXXXXXX/...`.

The visualization renderer is intentionally audit-oriented rather than decorative. Graph-of-thought trajectories are rendered as branching graphs in PCA embedding space with NLL as height. Sparse trajectories use exact triangular or polyline interpolants through the sampled states; denser trajectories use a hull-masked smooth RBF surface whose JSON metadata records the point residual and whether the surface touches the sampled NLL values. Filtered simplicial objects are rendered as dark SVGs in the side panel and hover card, with a scalar filtration-radius slider and multiparameter persistence data retained in the JSON payload. Long causal text-path complexes wrap into lanes for legibility.

When reviewing early runs, treat tropical active-support collapse as a first-priority diagnostic. A heatmap where most graph tokens select the graph/root token is not a rendering failure; it means the model is using a very low-entropy tropical support pattern. Compare that against BPB, graph-BPB, graph-sideinfo BPB, and graph-conditioned BPB before increasing auxiliary weights.

Full inference audits can optionally emit the same family of graph-of-thought PCA trajectories, tropical support heatmaps, persistence barcodes, multiparameter algebra/free-resolution JSON, derived signatures, GraphCG direction plots, and analogical memory retrieval artifacts. The trajectory HTML is dark-mode; hovering over a reasoning node renders the node's filtered simplicial object as an SVG in a cursor-following hover card and in the persistent side panel, and the 3D NLL view includes an exact or smoothed NLL heatmapped surface under the reasoning graph. Analogical-memory retrieval is rendered as a 3D simplicial-map view between the query trajectory filtered complex and retrieved memory complexes; the companion `analogical_simplicial_maps.json` reports vertex maps, edge/2-simplex preservation, persistent-homology similarity, free-resolution similarity, and derived-signature similarity. Use `--audit-all` for the complete optional bundle:

```bash
PYTHONPATH=TropicalGT-I/src \
/home/iska/miniconda3/envs/tokengt/bin/python \
TropicalGT-I/scripts/infer_tropicalgt_i.py \
--config TropicalGT-I/configs/train.json \
--checkpoint TropicalGT-I/checkpoints/tropicalgt_i_train.pt \
--prompt "Question: add 2 and 3 Answer:" \
--audit-all \
--scale-depth 2 \
--scale-width 3 \
--scale-branch-factor 2 \
--memory-save \
--audit-output-dir TropicalGT-I/outputs/train/inference_full_audit \
--output TropicalGT-I/outputs/train/inference_full_audit.json
```

Standalone evaluation can also render the validation visualizations:

```bash
PYTHONPATH=TropicalGT-I/src \
/home/iska/miniconda3/envs/tokengt/bin/python \
TropicalGT-I/scripts/eval_tropicalgt_i.py \
--config TropicalGT-I/configs/train.json \
--checkpoint TropicalGT-I/checkpoints/tropicalgt_i_train.pt \
--split validation \
--audit-level full \
--audit-ph-backend gudhi \
--render-visualizations \
--visualization-output-dir TropicalGT-I/outputs/train
```
