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

## Repository layout

- `TropicalGT-I/` is the active v1 implementation.
- `TropicalGT-I/src/tropicalgt/` contains the runnable package.
- `TropicalGT-I/scripts/` contains training, eval, inference, validation, visualization, and readiness-audit CLIs.
- `TropicalGT-I/configs/smoke.json` is a CPU fixture smoke config.
- `TropicalGT-I/configs/gpu_smoke.json` is the RTX 4090 data-backed smoke config.
- `TropicalGT-I/configs/gpu_ablation.json` is a bounded data-backed RTX 4090 config for matched BPB/graph-BPB ablation grids.
- `TropicalGT-I/configs/train_full_dataset_pg_bpb_step0_full24b_b44.json` is the current fresh step-0 full-dataset Parameter-Golf BPB run config; `TropicalGT-I/configs/train_full_dataset_active.json` is an older full-dataset path, and `TropicalGT-I/configs/train.json` is a legacy cap-sized review config.
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

The active full-dataset `TropicalGT-I/configs/train_full_dataset_active.json` path is hybrid by default. It mixes the moved TropicalGT reasoning shards with OpenAI Parameter-Golf FineWeb windows that are decoded as text and then represented as TokenGT-style sequential DAG records. The primary local OAI path is:

```bash
external/oai-parameter-golf
```

On this workstation that path can be a symlink to the existing ignored checkout at `external/parameter-golf`; `train_full_dataset_active.json` uses `external/oai-parameter-golf` first and keeps `external/parameter-golf` as a compatibility fallback. Populate the full SP1024 cache before a full run:

```bash
cd external/oai-parameter-golf
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 195
cd ../..
```

This creates `external/oai-parameter-golf/data/datasets/fineweb10B_sp1024` and `external/oai-parameter-golf/data/tokenizers/fineweb_1024_bpe.model`. The manifest reports `19,473,201,340` SP1024 train tokens across `195` train shards, plus the moved Hugging Face reasoning parquet shards. Those data files are gitignored. Every OAI sample is graph structured before batching: each token window becomes a causal DAG of sequence chunks, while non-causal/cyclic graphs elsewhere use deterministic random autoregressive node order.

Data-backed configs set `require_data: true`, and the OAI source is required. Missing or unreadable required parquet/OAI shards fail loudly instead of silently training on fixture examples or a partial hybrid. The current `train_full_dataset_pg_bpb_step0_full24b_b44.json` run requires both `tropicalgt_hf_reasoning` and `openai_parameter_golf` and uses the full audited train token-slot budget, not a 10B floor. The parquet loader builds a row-group metadata index over train/validation/test shards and reads records through a bounded row-group cache controlled by `cache_shards`; it does not concatenate the full moved dataset into memory. The hybrid sampler uses deterministic weighted indexed sampling over already graph-structured sources.

Audit the data budget before a long run:

```bash
PYTHONPATH=TropicalGT-I/src \
python \
TropicalGT-I/scripts/audit_training_data_budget.py \
--config TropicalGT-I/configs/train_full_dataset_active.json \
--output TropicalGT-I/outputs/train_full_dataset_active/data_budget.json
```

Generate a shard manifest and tokenization preflight report before a long run:

```bash
PYTHONPATH=TropicalGT-I/src \
python \
TropicalGT-I/scripts/validate_tropicalgt_i.py \
--config TropicalGT-I/configs/train_full_dataset_active.json \
--split train \
--limit 64 \
--output TropicalGT-I/outputs/train_full_dataset_active/validate_train.json
```

## Install/runtime notes

The `tokengt` env already provides PyTorch, pandas, pyarrow, datasets, tqdm, sklearn, transformers, W&B, Gudhi, Ripser, Persim, NetworkX, SciPy, SymPy, and Plotly. The optional `multipers` package can be installed later for richer multiparameter signed-measure backends; until then TropicalGT-I uses its in-repo bounded exact multiparameter persistence and commutative-algebra fallback.

## Run tests

```bash
python -m pytest TropicalGT-I/tests -q
```

## CPU smoke

```bash
PYTHONPATH=TropicalGT-I/src \
python \
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
python \
TropicalGT-I/scripts/train_tropicalgt_i.py \
--config TropicalGT-I/configs/gpu_smoke.json
```

The script reads the W&B API key from `keys.txt` when W&B is enabled. It logs NLL, exact text BPB, graph-BPB, graph side-information BPB, optimistic graph-conditioned BPB, GFlowNet trajectory-balance loss, GraphCG losses, full-rank singular-spectrum diagnostics, direction geometry, finite graph-certificate loss/agreement, tropical support entropy, tropical margins, margin-threshold wall-hit rate, graph token counts, graph structural byte counts, explicit graph JSON byte counts, graph JSON fallback/sequentialization rates, algebraic persistence summaries, analogical-memory query norms, examples/sec, tokens/sec, GPU memory, losses, objectives, and regularizers. Interactive HTML artifacts are not logged to W&B unless explicitly opted in with `wandb.log_interactive_artifacts: true` and a positive `wandb.html_artifact_limit`.

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
- `09_meet_in_middle`: optional MIM reverse-pass diagnostics, bidirectional NLL, join-token agreement, and MIM objective weights.
- `10_system`: VRAM, step time, examples/sec, tokens/sec.
- `11_optimization`: learning rate, gradient norm, sampler settings.

### Meet-in-the-middle decoding toggle

`meet_in_middle` is an off-by-default config block in every TropicalGT-I config:

```json
"meet_in_middle": {
  "enabled": false,
  "mode": "shared_weight_reverse_pass",
  "split_ratio": 0.5,
  "agreement_weight": 0.0,
  "reverse_nll_weight": 0.0,
  "max_records": 0,
  "reverse_model_path": ""
}
```

When enabled, TropicalGT-I runs a graph-aware adaptation of Meet-in-the-Middle decoding: the same graph-autoregressive byte stream is scored left-to-right and right-to-left, using causal topological order for DAGs and seeded random order for non-causal graphs. The current implementation uses a shared TropicalGT-I checkpoint for the reverse pass unless `reverse_model_path` is supplied later. It reports `mim_reverse_nll`, `mim_bidirectional_nll`, `mim_agreement_loss`, `mim_join_token_match_rate`, and per-record join diagnostics. Nonzero `agreement_weight` or `reverse_nll_weight` adds the MIM objective during training, so reduce batch/context first on the 1760-dim train config if GPU headroom is tight. For one-off checks, use `eval_tropicalgt_i.py --meet-in-middle` or `infer_tropicalgt_i.py --meet-in-middle`.

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

The reasoning HTML artifacts are dark-mode interactive Plotly views. Hovering over a reasoning node opens a cursor-following dark hover card with the node's filtered simplicial object rendered as SVG/Plotly data, while also updating the persistent side panel. The PCA/NLL views use model-evaluated NLL anchors projected onto the displayed NLL surface so every rendered reasoning-trajectory point lies on that surface; missing model outputs are rendered as explicit unavailable diagnostics. The payload JSON stores hover text, PCA/NLL point coordinates, NLL-surface metadata, Euclidean radius complexes, Jensen-Shannon probability complexes when model probability vectors exist, simplex-tree provenance, GraphCG projection-basis certificates, and the finite filtered simplicial object for each visualized record.

Training checkpoints contain model state, optimizer state, current step, metrics, history, config, and RNG state. Resume a run by pointing `train_tropicalgt_i.py` at a final or `.latest.pt` checkpoint:

```bash
PYTHONPATH=TropicalGT-I/src python TropicalGT-I/scripts/train_tropicalgt_i.py --config TropicalGT-I/configs/gpu_smoke.json --resume-from TropicalGT-I/checkpoints/tropicalgt_i_gpu_smoke.latest.pt --max-steps 8
```

## Data-backed training launch

### Current full-dataset BPB repair run

The current fresh step-0 Parameter-Golf BPB repair run uses:

```text
config: TropicalGT-I/configs/train_full_dataset_pg_bpb_step0_full24b_b44.json
run name: tropicalgt_i_pg_bpb_step0_full24b_b44
batch_size: 44
seq_len: 1024
max_steps: 537083
configured token slots: 24,198,811,648
audited available token slots: 24,198,796,288
dataset roots:
  - TropicalGT-I/data/toricgt/curated_hf_shards
  - external/oai-parameter-golf/data/datasets/fineweb10B_sp1024
```

This config is intended to use roughly 18-20GB VRAM on an RTX 4090 while
leaving headroom for implementation tests and browser/inference probes. Its
readiness audit must pass the required-source gates for both the moved
Hugging Face reasoning shards and the OpenAI Parameter-Golf SP1024 cache. Do
not describe a run as full-dataset unless it uses both data roots and the full
audited token-slot budget.

```bash
PYTHONPATH=TropicalGT-I/src \
python TropicalGT-I/scripts/train_tropicalgt_i.py \
--config TropicalGT-I/configs/train_full_dataset_pg_bpb_step0_full24b_b44.json
```

W&B should remain scalar-first by default: losses, BPB metrics, objectives,
regularizers, throughput, graph/tropical/GraphCG diagnostics, and memory/topology
metrics are logged online. Periodic and final interactive browser artifacts are
local optional outputs controlled by `periodic_interactive_artifacts_enabled`
and `final_interactive_artifacts_enabled`. HTML/media upload to W&B is a separate
explicit opt-in: set `wandb.log_interactive_artifacts: true` and a positive
`wandb.html_artifact_limit`.

The older full-dataset run path uses `TropicalGT-I/configs/train_full_dataset_active.json`, not the bounded smoke or review-loop configs. It is intentionally configured to keep training over more than ten billion sequence-token slots while requiring both real data sources:

- `seq_len: 1024`, `batch_size: 4`, `max_steps: 2500000`, for `10,240,000,000` configured training token slots.
- Required source `tropicalgt_hf_reasoning`: `117` train parquet shards, `4,633,582` examples, `4,744,787,968` token slots.
- Required source `openai_parameter_golf`: full SP1024 cache, `195` train shards, `19,473,201,340` raw tokens, `19,454,008,320` token slots.
- Total available train token slots reported by `outputs/train_full_dataset_active/data_budget.json`: `24,198,796,288`.
- Active output root: `TropicalGT-I/outputs/train_full_dataset_active`.
- Active validation/checkpoint cadence: validation every `500` steps, checkpoints every `1000` steps, heavier visualization/topology audit every `10000` steps.

The active run writes its PID to `TropicalGT-I/outputs/train_full_dataset_active/latest_training.pid` and logs to `TropicalGT-I/outputs/train_full_dataset_active/full_dataset_active_training.log`. W&B run names for this path use `tropicalgt_i_train_full_dataset_active`. Do not describe a run as full-dataset unless its data-budget audit passes the required-source and >10B slot gates.

Launch or resume the active full-dataset path with:

```bash
PYTHONPATH=TropicalGT-I/src python TropicalGT-I/scripts/train_tropicalgt_i.py --config TropicalGT-I/configs/train_full_dataset_active.json
```

For the legacy cap-sized `configs/train.json` path, treat it as a separate review configuration unless its data-budget audit is refreshed and its configured token-slot gate still passes.

Before launching or resuming the active full-dataset run, execute a CUDA dry-run readiness audit against `train_full_dataset_active.json`. This does not write a training checkpoint; it samples moved parquet data, builds the configured model, runs one optimizer step, checks the W&B key can be found, and gates finite train loss/BPB/graph-BPB:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=TropicalGT-I/src \
python \
TropicalGT-I/scripts/audit_tropicalgt_i_readiness.py \
--config TropicalGT-I/configs/train_full_dataset_active.json \
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
--output TropicalGT-I/outputs/train_full_dataset_active/readiness_train_dry_run.json
```

The latest hybrid cap-sized dry run used `21,484 MB` CUDA allocation, mixed in Parameter-Golf graph records at about `31%` of the sampled batch, and passed all readiness gates.

After the dry-run preflight report is clean, launch the active full-dataset TropicalGT-I run with:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=TropicalGT-I/src \
python \
TropicalGT-I/scripts/train_tropicalgt_i.py \
--config TropicalGT-I/configs/train_full_dataset_active.json
```

Resume it with:

```bash
PYTHONPATH=TropicalGT-I/src \
python \
TropicalGT-I/scripts/train_tropicalgt_i.py \
--config TropicalGT-I/configs/train_full_dataset_active.json \
--resume-from TropicalGT-I/checkpoints/tropicalgt_i_train_full_dataset_active.latest.pt
```

The training report includes a parquet manifest for the train and validation splits in addition to losses, graph-token counts, certificate metrics, throughput metrics, W&B metadata, visualization paths, and checkpoint paths.

For the 5K Parameter-Golf review cadence, run the single-agent Codex review loop:

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

At every 5K boundary it writes a Codex prompt plus `active_training_contract_step_*.json/.md` containing active hyperparameters, losses, objective weights, BPB/graph-BPB metrics, tropical metrics, GFlowNet metrics, GraphCG full-rank metrics, algebra/topology metrics, memory metrics, data-source rates, throughput, VRAM, and visualization paths. If `eval.bpb` is missing or above `1.18`, the boundary is marked for review and restart according to the chosen policy.

## Eval, inference, validation, visualization

```bash
PYTHONPATH=TropicalGT-I/src python TropicalGT-I/scripts/validate_tropicalgt_i.py --config TropicalGT-I/configs/smoke.json
PYTHONPATH=TropicalGT-I/src python TropicalGT-I/scripts/eval_tropicalgt_i.py --config TropicalGT-I/configs/smoke.json --checkpoint TropicalGT-I/checkpoints/tropicalgt_i_cpu_smoke.pt --details-limit 4
PYTHONPATH=TropicalGT-I/src python TropicalGT-I/scripts/infer_tropicalgt_i.py --config TropicalGT-I/configs/smoke.json --checkpoint TropicalGT-I/checkpoints/tropicalgt_i_cpu_smoke.pt --prompt "Question: add 2 and 3 Answer:" --scale-depth 2 --scale-width 3 --scale-branch-factor 2 --output TropicalGT-I/outputs/smoke/inference_audit.json
PYTHONPATH=TropicalGT-I/src python TropicalGT-I/scripts/render_reasoning_visualizations.py --config TropicalGT-I/configs/smoke.json --checkpoint TropicalGT-I/checkpoints/tropicalgt_i_cpu_smoke.pt
```

Validation reports graph JSON fallback rates, graph-token count statistics, node/edge statistics, and sample graph-token descriptors. Evaluation reports aggregate NLL/BPB/graph-BPB plus optional per-record NLL, graph-token traces, tropical support histograms, filtered simplicial objects, and, when `--audit-level topology|algebra|full` is supplied, topological algebra diagnostics. Inference emits the generated byte-level argmax text together with BPB accounting, graph-token trace, tropical margins/supports, GFlowNet action probabilities, GraphCG direction diagnostics, and the filtered simplicial object for the prompt graph.

Inference-time scaling is enabled by `inference_scaling` config defaults or explicit CLI flags. The bounded controller expands a prompt graph by GFlowNet-preferred actions (`expand`, `merge`, `refine`, `stop`, `retrieve`, `verify`, `compress`, `reject`), scores candidates with prompt NLL, tropical margin, action probability, and graph-token budget terms, then reports the best graph-of-thought candidate with its action path and filtered object.

For a full algebraic/topological inference audit with analogical memory retrieval:

```bash
PYTHONPATH=TropicalGT-I/src \
python \
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

This writes a dark-mode graph-of-thought PCA trajectory whose nodes are reasoning candidates and whose edges are parent-child expansions. Hovering over a node renders the filtered simplicial complex attached to that reasoning step directly in a cursor-following hover card and in the side panel. The 3D trajectory uses observed model-evaluated NLL anchors and a surface-contact projection contract: every plotted reasoning node and edge endpoint lies on the displayed NLL energy landscape with payload plot.z equal to plot.z_surface, while each node retains the raw centered/scaled NLL value for audit. Missing NLL, embedding, or probability outputs are reported as unavailable diagnostics instead of synthetic/proxy geometry. The persistence bundle includes barcodes, Betti/free-resolution growth, `persistence_representations.html` for GUDHI vector summaries, and `persistence_landscapes.html` for the actual GUDHI landscape functions `lambda_k(t)` by trajectory-growth level. The JSON payload stores the full Euclidean radius complex, the Jensen-Shannon probability complex, simplex-tree provenance, multiparameter persistence report, vectorized persistence summaries, commutative-algebra proxies, derived-equivalence signature, GraphCG direction diagnostics with projection-basis proof, NLL-surface metadata, NLL-progress diagnostics, and analogical memory retrieval details.

For Codex/browser review of a periodic audit, generate the sample-first dashboard so each inference row/input is the top-level unit. Each sample card links to the GoT NLL landscape, embedding map, full radius complex, full simplex tree, Jensen-Shannon probability complex, probability simplex tree, every reasoning-step complex/tree, persistence pages, per-rank analogical probability correspondences, GraphCG, tropical support, and metric/audit pages:

```bash
PYTHONPATH=TropicalGT-I/src python \
TropicalGT-I/scripts/build_sample_browser_index.py \
TropicalGT-I/outputs/train_full_dataset_active/periodic/step_XXXXXXXX/got_audit \
--output TropicalGT-I/outputs/train_full_dataset_active/periodic/step_XXXXXXXX/got_audit/codex_browser_index.html
```

For fresh sample-based browser review, run multiple independent inference audits into a clean output root and serve the generated `browser_index.html`:

```bash
PYTHONPATH=TropicalGT-I/src python \
TropicalGT-I/scripts/run_multi_inference_audits.py \
--config TropicalGT-I/configs/train_full_dataset_active.json \
--checkpoint TropicalGT-I/checkpoints/tropicalgt_i_train_full_dataset_active.latest.pt \
--samples 6 \
--scale-depth 3 \
--scale-width 4 \
--scale-branch-factor 3 \
--audit-ph-backend auto \
--memory-save \
--memory-retrieve-top-k 3 \
--output-root TropicalGT-I/outputs/multi_sample_browser/latest

python -m http.server 8977 \
--bind 127.0.0.1 \
--directory TropicalGT-I/outputs/multi_sample_browser/latest
```

Open `http://127.0.0.1:8977/browser_index.html`. When `--memory-save` or retrieval is enabled and no explicit `--memory-bank` is supplied, `run_multi_inference_audits.py` creates a clean bundle-local bank at `TropicalGT-I/outputs/multi_sample_browser/latest/browser_memory/reasoning_memory.jsonl`; pass `--memory-bank` only when intentionally reusing an existing bank. The first sample may have no non-self memory yet; later samples should retrieve earlier sample trajectories and render per-rank model-probability correspondences from `trajectory_probability_filtered_simplicial_object` complexes with `model_probability_jensen_shannon_assignment` provenance. These pages are not allowed to imply a passed simplicial map when the finite certificate fails: `analogical_simplicial_maps.json` records relative per-rank `pair_page` links, Jensen-Shannon assignment costs, edge/face preservation, filtration distortion, and the pass/fail certificate. If a browser cannot create a WebGL context, pages with selected simplicial objects promote a static SVG preview from the same payload instead of silently leaving only the Plotly WebGL error. Top-level non-growth `persistence_landscapes.html` pages redirect to the real `trajectory_persistence/persistence_landscapes.html` growth landscape plot when the bundle includes it. Add `--skip-existing` only when intentionally reusing prior sample directories.

For deeper/wider acceptance browser review from a stable checkpoint snapshot,
use the full preset. It enforces minimum depth `5`, width `8`, branch factor
`4`, trace limit `8192`, topology budget `8192`, and memory retrieval top-k
`8`. Browser audit commands pass `--require-complete-reasoning-steps`, so a
rendered trajectory fails closed if any candidate lacks model NLL, graph-state
embedding, action probabilities, complete graph-token trace, Jensen-Shannon
probability complex, Euclidean graph-token embedding complex, GraphCG
directions, or a directed GoT parent edge. Pass
`--no-scale-stochastic-actions` when the acceptance artifact must use
deterministic branch expansion from model action probabilities:

```bash
PYTHONPATH=TropicalGT-I/src:TropicalGT-I/scripts \
python TropicalGT-I/scripts/run_multi_inference_audits.py \
--config TropicalGT-I/configs/train_full_dataset_pg_bpb_step0_full24b_b44.json \
--checkpoint TropicalGT-I/checkpoints/tropicalgt_i_pg_bpb_step0_full24b_b44.latest.pt \
--output-root TropicalGT-I/outputs/multi_sample_browser/full_audit_deepwide \
--samples 1 \
--audit-preset full \
--no-scale-stochastic-actions \
--memory-bank TropicalGT-I/outputs/multi_sample_browser/latest/browser_memory/reasoning_memory.jsonl \
--memory-retrieve-top-k 8 \
--title "TropicalGT-I Full Deep/Wide Model-Derived Reasoning Audit"
```

`--audit-output-dir` alone writes JSON but should not be treated as a request
for browser HTML. Use `--interactive-artifacts` or `--audit-all` for local
interactive pages, and opt into W&B HTML uploads separately with
`wandb.log_interactive_artifacts` plus a positive `wandb.html_artifact_limit`.

Metric and visualization provenance can be audited with:

```bash
PYTHONPATH=TropicalGT-I/src python TropicalGT-I/scripts/audit_metric_provenance.py
```

The report registers exact metrics, fast vectorized topology features, surface-contact NLL diagnostics, unavailable diagnostics, spectral diagnostics, and historical guarded terms. The default active-code/docs scan now passes with `--fail-on-uncovered`; historical planning logs can still be audited explicitly with `--scan planning` when reviewing older status notes.

## BPB ablation analysis

Use the ablation analyzer after smoke runs or matched-seed experiment ladders to screen which auxiliary metrics correlate with the primary compression targets:

```bash
PYTHONPATH=TropicalGT-I/src \
python \
TropicalGT-I/scripts/analyze_bpb_ablations.py \
TropicalGT-I/outputs/gpu_smoke/train_report.json \
--output-dir TropicalGT-I/outputs/gpu_smoke/bpb_ablation
```

For multiple matched runs, pass every `train_report.json` and set `--baseline` to the baseline report. The tool writes JSON, Markdown, and Plotly HTML rankings for correlations against `bpb`, `graph_bpb`, `eval_bpb`, and `eval_graph_bpb`, plus a best-by-target table with the lowest run, baseline delta, runner-up, and runner-up margin for each compression target. Treat the output as a screen for the next ablation, not as causal evidence; promote an auxiliary only when the matched run improves held-out `bpb` or `graph_bpb`.

To generate a matched ablation grid and optionally train the variants in sequence:

```bash
PYTHONPATH=TropicalGT-I/src \
python \
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
python \
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
python scripts/export_tropicalgt_parameter_golf.py \
  --model-artifact final_model.int8.ptz \
  --output-dir parameter_golf_export
```

The export zip contains only `train_gpt.py`, `tropicalgt_tokengt_adapter.py`, `final_model.int8.ptz`, and `manifest.json`. The manifest checks the local 16,000,000-byte competition cap against code bytes plus compressed artifact bytes and records the BPB/graph-BPB metric contract. Datasets, checkpoints other than the final compressed artifact, W&B runs, topological audit JSON, and visualization HTML are intentionally excluded from the stripped package.
