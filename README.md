# TropicalGT

TropicalGT develops reasoning agents that use tropical geometry in transformer embedding space: TokenGT-style graph tokenization, tropical ring attention, graph-of-thought trajectories, GFlowNet training, GraphCG latent steering, and auditable reasoning visualizations.

[![Paper: TropicalGT main](https://img.shields.io/badge/arXiv--style-main%20paper-b31b1b?style=for-the-badge&logo=readthedocs&logoColor=white)](https://github.com/amelie-iska/TropicalGT/raw/tropicalgt-i-implementation/references/main.pdf)
[![Paper: TropicalGT-I](https://img.shields.io/badge/arXiv--style-TropicalGT--I-b31b1b?style=for-the-badge&logo=readthedocs&logoColor=white)](https://github.com/amelie-iska/TropicalGT/raw/tropicalgt-i-implementation/TropicalGT-I/assets/tropicalgt_neurips_research_paper.pdf)
[![Paper: TropicalGT-II](https://img.shields.io/badge/arXiv--style-TropicalGT--II-b31b1b?style=for-the-badge&logo=readthedocs&logoColor=white)](https://github.com/amelie-iska/TropicalGT/raw/tropicalgt-i-implementation/TropicalGT-II/assets/tropicalgt_ii_dynamical_tropical_geometry.pdf)
[![Paper: TropicalGT-III](https://img.shields.io/badge/arXiv--style-TropicalGT--III-b31b1b?style=for-the-badge&logo=readthedocs&logoColor=white)](https://github.com/amelie-iska/TropicalGT/raw/tropicalgt-i-implementation/TropicalGT-III/assets/tropicalgt_iii_context_protocols_memory_retrieval.pdf)
[![Paper: TropicalGT-IV](https://img.shields.io/badge/arXiv--style-TropicalGT--IV-b31b1b?style=for-the-badge&logo=readthedocs&logoColor=white)](https://github.com/amelie-iska/TropicalGT/raw/tropicalgt-i-implementation/TropicalGT-IV/assets/tropicalgt_iv_oracle_trajectory_classifiers.pdf)
[![Hugging Face dataset](https://img.shields.io/badge/Hugging%20Face-dataset-ffcc4d?style=for-the-badge&logo=huggingface&logoColor=111111)](https://huggingface.co/datasets/AmeliSchreiber/TropicalGT)
[![Hugging Face checkpoints](https://img.shields.io/badge/Hugging%20Face-checkpoints-ffcc4d?style=for-the-badge&logo=huggingface&logoColor=111111)](https://huggingface.co/AmeliSchreiber/TropicalGT)
[![GitHub codebase](https://img.shields.io/badge/GitHub-codebase-24292f?style=for-the-badge&logo=github&logoColor=white)](https://github.com/amelie-iska/TropicalGT)

![Dark 3D tropical hypersurface with Newton polytope](./assets/tropical_curve_newton_polytope_dark.png)

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
- `TropicalGT-I/scripts/` contains training, eval, inference, validation, visualization, and readiness-audit CLIs.
- `TropicalGT-I/configs/smoke.json` is a CPU fixture smoke config.
- `TropicalGT-I/configs/gpu_smoke.json` is the RTX 4090 data-backed smoke config.
- `TropicalGT-I/configs/gpu_ablation.json` is a bounded data-backed RTX 4090 config for matched BPB/graph-BPB ablation grids.
- `TropicalGT-I/configs/train.json` is the first full data-backed training config.
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

The full `TropicalGT-I/configs/train.json` path is hybrid by default. It mixes the moved TropicalGT reasoning shards with OpenAI Parameter-Golf FineWeb windows that are decoded as text and then represented as TokenGT-style sequential DAG records. A minimal SP1024 Parameter-Golf cache can be populated with:

```bash
cd external/parameter-golf
/home/iska/miniconda3/envs/tokengt/bin/python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
cd ../..
```

This creates `external/parameter-golf/data/datasets/fineweb10B_sp1024` and `external/parameter-golf/data/tokenizers/fineweb_1024_bpe.model`. Those data files are gitignored. Every OAI sample is graph structured before batching: each token window becomes a causal DAG of sequence chunks, while non-causal/cyclic graphs elsewhere use deterministic random autoregressive node order.

Data-backed configs set `require_data: true`, so missing or unreadable required parquet shards fail loudly instead of silently training on fixture examples. Optional hybrid OAI shards are reported when absent. The parquet loader builds a row-group metadata index over train/validation/test shards and reads records through a bounded row-group cache controlled by `cache_shards`; it does not concatenate the full moved dataset into memory. The full `train.json` config enables `chunk_shuffle`, which randomizes parquet row-group order while preserving row-group-local reads when the dataset is parquet-only; the hybrid sampler uses deterministic weighted indexed sampling over already graph-structured sources.

Generate a shard manifest and tokenization preflight report before a long run:

```bash
PYTHONPATH=TropicalGT-I/src \
/home/iska/miniconda3/envs/tokengt/bin/python \
TropicalGT-I/scripts/validate_tropicalgt_i.py \
--config TropicalGT-I/configs/train.json \
--split train \
--limit 64 \
--output TropicalGT-I/outputs/train/validate_train.json
```

## Install/runtime notes

The `tokengt` env already provides PyTorch, pandas, pyarrow, datasets, tqdm, sklearn, transformers, W&B, Gudhi, Ripser, Persim, NetworkX, SciPy, SymPy, and Plotly. The optional `multipers` package can be installed later for richer multiparameter signed-measure backends; until then TropicalGT-I uses its in-repo bounded exact multiparameter persistence and commutative-algebra fallback.

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

The script reads the W&B API key from `keys.txt` when W&B is enabled. It logs NLL, exact text BPB, graph-BPB, graph side-information BPB, optimistic graph-conditioned BPB, GFlowNet trajectory-balance loss, GraphCG losses, full-rank singular-spectrum diagnostics, direction geometry, finite graph-certificate loss/agreement, tropical support entropy, tropical margins, margin-threshold wall-hit rate, graph token counts, graph structural byte counts, explicit graph JSON byte counts, graph JSON fallback/sequentialization rates, algebraic persistence summaries, analogical-memory query norms, examples/sec, tokens/sec, GPU memory, and generated HTML artifacts.

W&B scalar metrics are organized by prefixed priority groups rather than logged as one flat pile:

- `00_primary`: eval/train BPB, graph-BPB, loss, NLL, GPU memory.
- `01_losses`: total loss, NLL, weighted/unweighted auxiliary losses and regularizer ratio.
- `02_bpb`: text BPB, graph-BPB, side-information BPB, byte counts, graph structural bytes.
- `03_tropical`: graph/sequence tropical supports, margins, wall hits, certificates.
- `04_gflownet`: trajectory-balance, reward, diversity, and flow diagnostics.
- `05_graphcg`: GraphCG loss, full-rank barrier, singular values, effective/numerical rank.
- `06_graph_data`: graph-token counts, node/edge ratios, causal/random graph AR rates, OAI source rate.
- `07_algebra_topology`: persistence, Betti, free-resolution, derived-signature summaries.
- `08_memory`: analogical memory query and bank metrics.
- `09_system`: VRAM, step time, examples/sec, tokens/sec.
- `10_optimization`: learning rate, gradient norm, sampler settings.

Local W&B run directories are disposable and should be cleaned after smoke iterations:

```bash
rm -rf wandb
```

The online `amelie-iska-math/TropicalGT-I` project was cleaned to keep only the latest useful smoke run, `pzpw99m7`; future long runs should use distinctive names or tags so review-worthy runs are not mixed with throwaway smoke jobs.

The primary Parameter-Golf-style metric is text BPB:

```text
bpb = NLL_bits / predicted_target_bytes
```

For TokenGT-style graph conditioning the reports also include:

```text
graph_bpb = (NLL_bits + 8 * graph_bpb_side_weight * explicit_graph_json_bytes)
            / (predicted_target_bytes + graph_token_structural_bytes)
graph_sideinfo_bpb = (NLL_bits + side_info_bits) / predicted_target_bytes
graph_conditioned_bpb_no_side_cost = NLL_bits / (predicted_target_bytes + graph_token_structural_bytes)
```

`graph_token_structural_bytes` is computed from the actual TokenGT tuple: live-token masks, node counts, and edge endpoint ids determine the structural byte budget.

Sequential text path graphs are deterministic from the byte stream and are excluded from explicit side-information byte accounting. All auxiliary metrics should be ablated by whether they improve held-out `bpb` and `graph_bpb`; a prettier trajectory or cleaner algebraic invariant is not a win by itself.

The certificate metrics are finite graph-structure checks: edge tokens are rewarded when their active tropical support lies on the edge itself or one of its endpoint vertex tokens. These certificates are useful for auditing whether tropical supports follow the TokenGT graph skeleton, but they are not semantic correctness proofs and must be interpreted beside task loss, verifier scores, and held-out BPB.

The reasoning HTML artifacts are dark-mode interactive Plotly views. Hovering over a reasoning node now opens a cursor-following dark hover card with the node's filtered simplicial object rendered as SVG, while also updating the persistent side panel. The SVG shows vertices, edges, 2-simplices, filtration-colored structure, and filtration-layer ticks rather than only textual metadata; the PCA/NLL views also include a smoothed NLL heatmap surface below or through the trajectory. The payload JSON stores hover text, PCA/NLL point coordinates, NLL-surface metadata, and the full finite filtered simplicial object for each visualized record. In v1 these objects include 0-simplices for graph vertices, 1-simplices for graph edges, directed length-2 path 2-simplices, filtration thresholds, and per-record summaries.

Training checkpoints contain model state, optimizer state, current step, metrics, history, config, and RNG state. Resume a run by pointing `train_tropicalgt_i.py` at a final or `.latest.pt` checkpoint:

```bash
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/train_tropicalgt_i.py --config TropicalGT-I/configs/gpu_smoke.json --resume-from TropicalGT-I/checkpoints/tropicalgt_i_gpu_smoke.latest.pt --max-steps 8
```

## Data-backed training launch

The current full training config is sized for the OpenAI Parameter-Golf 16MB artifact cap while using the RTX 4090. Its core settings are:

- model width/hidden size `1760`, memory width `220`, about `38.6M` parameters before int8+zlib export.
- estimated stripped competition export `15,633,708` bytes, leaving about `366,292` bytes under the `16,000,000` byte cap.
- `seq_len: 1024`, `batch_size: 80`, `checkpoint_every: 5000`, `max_steps: 20000`.
- exact blockwise tropical ring attention over graph tokens with `graph_tropical_block_size: 32`.
- pooled long-context sequence tropical ring attention with `sequence_tropical_max_tokens: 32`, `sequence_tropical_block_size: 16`, and residual weight `0.125`.
- graph-aware autoregressive decoding: causal topological order for DAGs, deterministic random order for non-causal/cyclic graphs.
- hybrid data weights `0.7` TropicalGT reasoning shards and `0.3` OpenAI Parameter-Golf SP1024 windows when the local OAI cache is present.

Before launching the first full run, execute a CUDA dry-run readiness audit against `train.json`. This does not write a training checkpoint; it samples moved parquet data, builds the configured model, runs one optimizer step, checks the W&B key can be found, and gates finite train loss/BPB/graph-BPB:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=TropicalGT-I/src \
/home/iska/miniconda3/envs/tokengt/bin/python \
TropicalGT-I/scripts/audit_tropicalgt_i_readiness.py \
--config TropicalGT-I/configs/train.json \
--split train \
--sample-limit 80 \
--details-limit 0 \
--trace-limit 8 \
--scale-depth 0 \
--audit-level topology \
--audit-ph-backend gudhi \
--audit-max-simplices 128 \
--check-ablation-tools \
--check-wandb-key \
--train-dry-run \
--require-cuda \
--output TropicalGT-I/outputs/train/readiness_train_dry_run.json
```

The latest hybrid cap-sized dry run used `21,484 MB` CUDA allocation, mixed in Parameter-Golf graph records at about `31%` of the sampled batch, and passed all readiness gates.

After the dry-run preflight report is clean, launch the first full TropicalGT-I run with:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=TropicalGT-I/src \
/home/iska/miniconda3/envs/tokengt/bin/python \
TropicalGT-I/scripts/train_tropicalgt_i.py \
--config TropicalGT-I/configs/train.json
```

Resume it with:

```bash
PYTHONPATH=TropicalGT-I/src \
/home/iska/miniconda3/envs/tokengt/bin/python \
TropicalGT-I/scripts/train_tropicalgt_i.py \
--config TropicalGT-I/configs/train.json \
--resume-from TropicalGT-I/checkpoints/tropicalgt_i_train.latest.pt
```

The training report includes a parquet manifest for the train and validation splits in addition to losses, graph-token counts, certificate metrics, throughput metrics, W&B metadata, visualization paths, and checkpoint paths.

For the 5K Parameter-Golf review cadence, run the single-agent Codex review loop:

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

At every 5K boundary it writes a Codex prompt plus `active_training_contract_step_*.json/.md` containing active hyperparameters, losses, objective weights, BPB/graph-BPB metrics, tropical metrics, GFlowNet metrics, GraphCG full-rank metrics, algebra/topology metrics, memory metrics, data-source rates, throughput, VRAM, and visualization paths. If `eval.bpb` is missing or above `1.18`, the boundary is marked for review and restart according to the chosen policy.

## Eval, inference, validation, visualization

```bash
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/validate_tropicalgt_i.py --config TropicalGT-I/configs/smoke.json
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/eval_tropicalgt_i.py --config TropicalGT-I/configs/smoke.json --checkpoint TropicalGT-I/checkpoints/tropicalgt_i_cpu_smoke.pt --details-limit 4
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/infer_tropicalgt_i.py --config TropicalGT-I/configs/smoke.json --checkpoint TropicalGT-I/checkpoints/tropicalgt_i_cpu_smoke.pt --prompt "Question: add 2 and 3 Answer:" --scale-depth 2 --scale-width 3 --scale-branch-factor 2 --output TropicalGT-I/outputs/smoke/inference_audit.json
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/render_reasoning_visualizations.py --config TropicalGT-I/configs/smoke.json --checkpoint TropicalGT-I/checkpoints/tropicalgt_i_cpu_smoke.pt
```

Validation reports graph JSON fallback rates, graph-token count statistics, node/edge statistics, and sample graph-token descriptors. Evaluation reports aggregate NLL/BPB/graph-BPB plus optional per-record NLL, graph-token traces, tropical support histograms, filtered simplicial objects, and, when `--audit-level topology|algebra|full` is supplied, topological algebra diagnostics. Inference emits the generated byte-level argmax text together with BPB accounting, graph-token trace, tropical margins/supports, GFlowNet action probabilities, GraphCG direction diagnostics, and the filtered simplicial object for the prompt graph.

Inference-time scaling is enabled by `inference_scaling` config defaults or explicit CLI flags. The bounded controller expands a prompt graph by GFlowNet-preferred actions (`expand`, `merge`, `refine`, `stop`, `retrieve`, `verify`, `compress`, `reject`), scores candidates with prompt NLL, tropical margin, action probability, and graph-token budget terms, then reports the best graph-of-thought candidate with its action path and filtered object.

For a full algebraic/topological inference audit with analogical memory retrieval:

```bash
PYTHONPATH=TropicalGT-I/src \
/home/iska/miniconda3/envs/tokengt/bin/python \
TropicalGT-I/scripts/infer_tropicalgt_i.py \
--config TropicalGT-I/configs/gpu_smoke.json \
--checkpoint TropicalGT-I/checkpoints/tropicalgt_i_gpu_smoke.pt \
--prompt "Question: prove that a two-step reasoning chain can be represented as a graph. Answer:" \
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

This writes a dark-mode graph-of-thought PCA trajectory whose nodes are reasoning candidates and whose edges are parent-child expansions. Hovering over a node renders the filtered simplicial complex attached to that reasoning step directly in a cursor-following hover card and in the side panel. The 3D trajectory includes three labeled NLL layers: a smooth projected NLL/fitness landscape over actual model `graph_state` PCA coordinates plus rendered microstep anchors with grid cells pinned to measured anchor values, a local sample-supported interpolating NLL sheet that passes through those anchors, and an actual sampled NLL landscape mesh through sampled reasoning states, with residual/provenance metadata in JSON. The persistence bundle includes barcodes, Betti/free-resolution growth, `persistence_representations.html` for GUDHI vector summaries, and `persistence_landscapes.html` for the actual GUDHI landscape functions `lambda_k(t)` by trajectory-growth level. The JSON payload stores the full complex, multiparameter persistence report, vectorized persistence summaries, commutative-algebra proxies, derived-equivalence signature, GraphCG direction diagnostics, NLL-surface metadata, NLL-progress diagnostics, and analogical memory retrieval details.

Metric and visualization provenance can be audited with:

```bash
PYTHONPATH=TropicalGT-I/src /home/iska/miniconda3/envs/tokengt/bin/python TropicalGT-I/scripts/audit_metric_provenance.py
```

The report registers exact metrics, fast vectorized topology features, sample interpolants, visual surrogates, spectral diagnostics, and known fallbacks. The default active-code/docs scan now passes with `--fail-on-uncovered`; historical planning logs can still be audited explicitly with `--scan planning` when reviewing older status notes.

## BPB ablation analysis

Use the ablation analyzer after smoke runs or matched-seed experiment ladders to screen which auxiliary metrics correlate with the primary compression targets:

```bash
PYTHONPATH=TropicalGT-I/src \
/home/iska/miniconda3/envs/tokengt/bin/python \
TropicalGT-I/scripts/analyze_bpb_ablations.py \
TropicalGT-I/outputs/gpu_smoke/train_report.json \
--output-dir TropicalGT-I/outputs/gpu_smoke/bpb_ablation
```

For multiple matched runs, pass every `train_report.json` and set `--baseline` to the baseline report. The tool writes JSON, Markdown, and Plotly HTML rankings for correlations against `bpb`, `graph_bpb`, `eval_bpb`, and `eval_graph_bpb`, plus a best-by-target table with the lowest run, baseline delta, runner-up, and runner-up margin for each compression target. Treat the output as a screen for the next ablation, not as causal evidence; promote an auxiliary only when the matched run improves held-out `bpb` or `graph_bpb`.

To generate a matched ablation grid and optionally train the variants in sequence:

```bash
PYTHONPATH=TropicalGT-I/src \
/home/iska/miniconda3/envs/tokengt/bin/python \
TropicalGT-I/scripts/run_bpb_ablation_grid.py \
--config TropicalGT-I/configs/gpu_ablation.json \
--output-dir TropicalGT-I/outputs/gpu_ablation/grid_iter19 \
--variants baseline,aux_0p5x,aux_0p25x,gflownet_0p25x,graphcg_0p25x,tropical_0p25x,no_auxiliary \
--max-steps 8 \
--seed 1729 \
--run
```

By default the grid runner disables W&B media/network logging for the variants and keeps a shared seed across all runs. Add `--wandb` only when online logging for every ablation variant is desired. Add `--fixture --device cpu` for a quick local sanity check that writes configs and reports without touching the moved parquet dataset. The `gpu_ablation.json` config is intentionally bounded (`train_limit`, `val_limit`, and `max_steps`) so it can test whether an auxiliary is directionally helpful before a longer `train.json` run; it should not be treated as final leaderboard evidence. The built-in variant ladder includes zero-ablation variants plus nonzero half/quarter-weight and isolated low-weight GFlowNet, GraphCG, and tropical-regularizer settings.

For a checkpoint-backed smoke proof bundle, write a readiness report that checks the environment, packages, data manifest, sample TokenGT conversion, sequential text graphification, paper artifacts, checkpoint reload, finite eval, BPB/graph-BPB accounting, bounded inference scaling, optional topological audits, ablation-tool availability, and visualization generation:

```bash
PYTHONPATH=TropicalGT-I/src \
/home/iska/miniconda3/envs/tokengt/bin/python \
TropicalGT-I/scripts/audit_tropicalgt_i_readiness.py \
--config TropicalGT-I/configs/gpu_smoke.json \
--checkpoint TropicalGT-I/checkpoints/tropicalgt_i_gpu_smoke.pt \
--split validation \
--sample-limit 16 \
--details-limit 2 \
--scale-depth 1 \
--scale-width 2 \
--scale-branch-factor 2 \
--audit-level topology \
--audit-ph-backend gudhi \
--audit-max-simplices 128 \
--check-ablation-tools \
--render-visualizations \
--output TropicalGT-I/outputs/gpu_smoke/readiness_audit.json
```

The command also writes `readiness_audit.md` next to the JSON report and exits nonzero if any required gate fails. In the latest GPU-smoke readiness bundle, both `eval_reports_bpb` and `eval_reports_graph_bpb` are hard gates; in the train-config dry-run bundle, `train_dry_run_reports_bpb` and `train_dry_run_reports_graph_bpb` are hard gates. Auxiliary tropical, topological, GraphCG, GFlowNet, and memory metrics remain candidates to ablate against those compression targets.

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

The optional `graph_tokens` argument accepts either `(token_features, token_type_ids, graph_mask)` or `(token_features, token_type_ids, endpoint_ids, graph_mask)`. The four-tensor form is the TokenGT incidence path: endpoint ids attach edge tokens to vertex-token ids, with `-1` used as endpoint padding. The adapter pools graph node/edge tokens into a model-width context vector and adds it to each text-token embedding before the baseline transformer stack. `tropicalgt_tokengt_adapter.py` also exposes `graph_bpb_metrics(...)`, which reports ordinary text BPB, graph-BPB, graph side-information BPB, optimistic graph-conditioned BPB, graph structural bytes, explicit graph JSON bytes, and side-information bits.

To build an OpenAI Parameter-Golf stripped package without the full TropicalGT-I research stack, run from `external/parameter-golf`:

```bash
/home/iska/miniconda3/envs/tokengt/bin/python scripts/export_tropicalgt_parameter_golf.py \
  --model-artifact final_model.int8.ptz \
  --output-dir parameter_golf_export
```

The export zip contains only `train_gpt.py`, `tropicalgt_tokengt_adapter.py`, `final_model.int8.ptz`, and `manifest.json`. The manifest checks the local 16,000,000-byte competition cap against code bytes plus compressed artifact bytes and records the BPB/graph-BPB metric contract. Datasets, checkpoints other than the final compressed artifact, W&B runs, topological audit JSON, and visualization HTML are intentionally excluded from the stripped package.
