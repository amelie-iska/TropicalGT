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

All tropical margins, GraphCG factors, GFlowNet rewards, persistence summaries, and analogical memory metrics should be ablated by whether they improve held-out `bpb` and `graph_bpb`.

## Papers and Assets

- [Main TropicalGT paper](../references/main.pdf)
- [TropicalGT-I NeurIPS research paper](./assets/tropicalgt_neurips_research_paper.pdf)

## Runtime

From the repository root, use the `tokengt` environment with `PYTHONPATH=TropicalGT-I/src` for training, evaluation, validation, inference, and visualization scripts. The top-level README contains the full command inventory for CPU smoke tests, GPU smoke runs, data-backed training, checkpoint resume, and readiness audits.

Full inference audits can optionally emit graph-of-thought PCA trajectories, tropical support heatmaps, persistence barcodes, multiparameter algebra JSON, GraphCG direction plots, and analogical memory retrieval artifacts:

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
