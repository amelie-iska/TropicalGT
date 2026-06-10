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

All tropical margins, GraphCG factors, GFlowNet rewards, persistence summaries, and analogical memory metrics should be ablated by whether they improve held-out `bpb` and `graph_bpb`. GraphCG training now includes an explicit full-rank singular-value barrier and logs effective rank, numerical rank, singular min/max, and condition proxies so latent steering directions cannot collapse unnoticed.

Run `scripts/analyze_bpb_ablations.py` on one or more `train_report.json` files to generate JSON/Markdown/HTML screens ranking which logged metrics correlate with `bpb`, `graph_bpb`, `eval_bpb`, and `eval_graph_bpb`. Use those rankings to choose matched ablations; do not treat correlations as causal wins. Run `scripts/run_bpb_ablation_grid.py` to generate same-seed variants such as `no_graphcg`, `no_gflownet`, `no_certificate`, `no_tropical_regularizers`, and `no_auxiliary`, optionally train them in sequence, and immediately analyze their BPB deltas.

Use `configs/gpu_ablation.json` for bounded data-backed RTX 4090 ablation ladders between smoke tests and the full `configs/train.json` run. It keeps the moved parquet dataset, TokenGT graphification, topology audits, graph-BPB accounting, and dark Plotly correlation report, but caps train/validation records and step count so candidate auxiliary losses can be screened before longer training. The built-in grid variants include zero-ablation controls and nonzero half/quarter-weight settings such as `aux_0p5x`, `aux_0p25x`, `gflownet_0p25x`, `graphcg_0p25x`, and `tropical_0p25x`.

## Papers and Assets

- [Main TropicalGT paper](../references/main.pdf)
- [TropicalGT-I NeurIPS research paper](./assets/tropicalgt_neurips_research_paper.pdf)

## Runtime

From the repository root, use the `tokengt` environment with `PYTHONPATH=TropicalGT-I/src` for training, evaluation, validation, inference, and visualization scripts. The top-level README contains the full command inventory for CPU smoke tests, GPU smoke runs, data-backed training, checkpoint resume, and readiness audits.

The readiness audit should be treated as the pre-training acceptance check for this phase: it gates checkpoint reload, data-backed TokenGT conversion, sequential text graphification, finite eval, `bpb`, `graph_bpb`, topology-audit execution, ablation-tool availability, and generated visualization artifacts before a longer run is considered ready.

For the full `configs/train.json` path, run the readiness audit with `--train-dry-run --require-cuda --check-wandb-key` before launching the long job. That dry run samples the moved parquet train split, builds the configured model, performs one optimizer step on CUDA, and gates finite train loss, BPB, graph-BPB, and gradient norm.

Full inference audits can optionally emit graph-of-thought PCA trajectories, tropical support heatmaps, persistence barcodes, multiparameter algebra JSON, GraphCG direction plots, and analogical memory retrieval artifacts. The trajectory HTML is dark-mode; hovering over a reasoning node renders the node's filtered simplicial object as an SVG panel, and the 3D NLL view includes a smoothed NLL heatmapped surface under the reasoning graph:

```bash
PYTHONPATH=TropicalGT-I/src \
/home/iska/miniconda3/envs/tokengt/bin/python \
TropicalGT-I/scripts/infer_tropicalgt_i.py \
--config TropicalGT-I/configs/gpu_smoke.json \
--checkpoint TropicalGT-I/checkpoints/tropicalgt_i_gpu_smoke.pt \
--prompt "Question: add 2 and 3 Answer:" \
--scale-depth 2 \
--scale-width 3 \
--scale-branch-factor 2 \
--audit-level full \
--audit-ph-backend gudhi \
--audit-render-html \
--memory-bank TropicalGT-I/outputs/gpu_smoke/analogical_memory/reasoning_memory.jsonl \
--memory-save \
--memory-retrieve-top-k 3 \
--audit-output-dir TropicalGT-I/outputs/gpu_smoke/inference_full_audit \
--output TropicalGT-I/outputs/gpu_smoke/inference_full_audit.json
```
