# TropicalGT

TropicalGT develops reasoning agents that use tropical geometry in transformer embedding space: TokenGT-style graph tokenization, tropical ring attention, graph-of-thought trajectories, GFlowNet training, GraphCG latent steering, and auditable reasoning visualizations.

## Remote workspace

All implementation work for this repo is under:

```bash
tailscale ssh iska@iska
cd /home/iska/Documents/amelie/bio/TropicalGT
```

Use the `tokengt` environment directly in noninteractive shells:

```bash
/home/iska/miniconda3/envs/tokengt/bin/python --version
```

## Repository layout

- `TropicalGT-I/` is the active v1 implementation.
- `TropicalGT-I/src/tropicalgt/` contains the runnable package.
- `TropicalGT-I/scripts/` contains training, eval, inference, validation, and visualization CLIs.
- `TropicalGT-I/configs/smoke.json` is a CPU fixture smoke config.
- `TropicalGT-I/configs/gpu_smoke.json` is the RTX 4090 data-backed smoke config.
- `TropicalGT-I/assets/tropicalgt_neurips_research_paper.tex` is the paper source.
- `planning/` contains reference synthesis and implementation notes.
- `external/` contains separate fork checkouts and is intentionally gitignored by this repo.

## Data

The ToricGT dataset has been moved, not copied, into:

```bash
TropicalGT-I/data/toricgt
```

The default training shard root is:

```bash
TropicalGT-I/data/toricgt/curated_hf_shards
```

Data is gitignored. Do not commit datasets, checkpoints, W&B runs, or `keys.txt`.

## Install/runtime notes

The `tokengt` env already provides PyTorch, pandas, pyarrow, datasets, tqdm, sklearn, transformers, and W&B. Plotly is required for interactive trajectory HTML and has been installed into the env.

## Run tests

```bash
cd /home/iska/Documents/amelie/bio/TropicalGT
/home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests -q
```

## CPU smoke

```bash
PYTHONPATH=TropicalGT-I/src \
/home/iska/miniconda3/envs/tokengt/bin/python \
TropicalGT-I/scripts/train_tropicalgt_i.py \
--config TropicalGT-I/configs/smoke.json
```

This writes:

- `TropicalGT-I/checkpoints/tropicalgt_i_cpu_smoke.pt`
- `TropicalGT-I/checkpoints/tropicalgt_i_cpu_smoke.latest.pt`
- `TropicalGT-I/outputs/smoke/train_report.json`
- `TropicalGT-I/outputs/smoke/reasoning_trajectory_3d.html`
- `TropicalGT-I/outputs/smoke/reasoning_trajectory_pca_nll.html`
- `TropicalGT-I/outputs/smoke/reasoning_trajectory_payloads.json`
- `TropicalGT-I/outputs/smoke/training_metrics.html`

## GPU smoke with W&B

```bash
PYTHONPATH=TropicalGT-I/src \
/home/iska/miniconda3/envs/tokengt/bin/python \
TropicalGT-I/scripts/train_tropicalgt_i.py \
--config TropicalGT-I/configs/gpu_smoke.json
```

The script reads the W&B API key from `keys.txt` when W&B is enabled. It logs NLL, BPB proxy, GFlowNet trajectory-balance loss, GraphCG losses, tropical support entropy, tropical margins, graph token counts, GPU memory, and generated HTML artifacts.

The reasoning payload JSON stores the hover text, PCA/NLL point coordinates, and the full finite filtered simplicial object for each visualized record. In v1 these objects include 0-simplices for graph vertices, 1-simplices for graph edges, directed length-2 path 2-simplices, filtration thresholds, and per-record summaries.

Training checkpoints contain model state, optimizer state, current step, metrics, history, config, and RNG state. Resume a run by pointing `train_tropicalgt_i.py` at a final or `.latest.pt` checkpoint:

```bash
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/train_tropicalgt_i.py --config TropicalGT-I/configs/gpu_smoke.json --resume-from TropicalGT-I/checkpoints/tropicalgt_i_gpu_smoke.latest.pt --max-steps 8
```

## Eval, inference, validation, visualization

```bash
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/validate_tropicalgt_i.py --config TropicalGT-I/configs/smoke.json
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/eval_tropicalgt_i.py --config TropicalGT-I/configs/smoke.json --checkpoint TropicalGT-I/checkpoints/tropicalgt_i_cpu_smoke.pt --details-limit 4
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/infer_tropicalgt_i.py --config TropicalGT-I/configs/smoke.json --checkpoint TropicalGT-I/checkpoints/tropicalgt_i_cpu_smoke.pt --prompt "Question: add 2 and 3 Answer:" --scale-depth 2 --scale-width 3 --scale-branch-factor 2 --output TropicalGT-I/outputs/smoke/inference_audit.json
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/render_reasoning_visualizations.py --config TropicalGT-I/configs/smoke.json --checkpoint TropicalGT-I/checkpoints/tropicalgt_i_cpu_smoke.pt
```

Validation reports graph JSON fallback rates, graph-token count statistics, node/edge statistics, and sample graph-token descriptors. Evaluation reports aggregate NLL/BPB plus optional per-record NLL, graph-token traces, tropical support histograms, and filtered simplicial objects. Inference emits the generated byte-level argmax text together with graph-token trace, tropical margins/supports, GFlowNet action probabilities, GraphCG direction diagnostics, and the filtered simplicial object for the prompt graph.

Inference-time scaling is enabled by `inference_scaling` config defaults or explicit CLI flags. The bounded controller expands a prompt graph by GFlowNet-preferred actions (`expand`, `merge`, `refine`, `stop`, `retrieve`, `verify`, `compress`, `reject`), scores candidates with prompt NLL, tropical margin, action probability, and graph-token budget terms, then reports the best graph-of-thought candidate with its action path and filtered object.

## External forks

- `external/TokenGT`: fork of `jw9730/tokengt` under `amelie-iska/TokenGT`.
- `external/parameter-golf`: OpenAI Parameter Golf fork; branch `tropicalgt-i-tokenization` contains `tropicalgt_tokengt_adapter.py`.
- `external/GraphCG`: GraphCG fork used for methodology reference.
- `external/Tropical-Attention`: Tropical Attention fork used for kernel/reference comparison.

These are separate git repositories and should be pushed separately from TropicalGT.

### Parameter-Golf TokenGT adapter

The Parameter-Golf baseline keeps its default language-model path unchanged. To enable graph conditioning in experiments that pass TokenGT-style graph tensors into `GPT.forward`, set:

```bash
TROPICALGT_GRAPH_ADAPTER=1
TROPICALGT_GRAPH_FEATURE_DIM=48
```

The optional `graph_tokens` argument is a tuple `(token_features, token_type_ids, graph_mask)`. The adapter pools graph node/edge tokens into a model-width context vector and adds it to each text-token embedding before the baseline transformer stack.
