# TropicalGT-I Implementation Status - 2026-06-10

## Completed
- Killed the previously running ToricGT training/watch/mirror processes and verified no GPU compute process remained.
- Created branch `tropicalgt-i-implementation` in the TropicalGT repo.
- Cloned `amelie-iska/TokenGT` into `external/TokenGT` and created branch `tropicalgt-i-tokenization`.
- Created branch `tropicalgt-i-tokenization` in `external/parameter-golf`.
- Moved `/home/iska/Documents/amelie/bio/ToricGT/data` to `TropicalGT-I/data/toricgt`.
- Added the TropicalGT-I package, configs, scripts, tests, README updates, paper implementation addendum, and reference synthesis.
- Added a minimal Parameter-Golf TokenGT adapter in the external fork.
- Iteration 2 adds explicit filtered simplicial objects to reasoning visualization payloads, per-step Plotly metric dashboards, and an opt-in Parameter-Golf graph adapter path controlled by `TROPICALGT_GRAPH_ADAPTER`.

## Verification evidence
- Unit tests: `/home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests -q` -> `6 passed`.
- CPU smoke: `train_tropicalgt_i.py --config TropicalGT-I/configs/smoke.json` completed and wrote checkpoint plus Plotly HTML.
- Dataset validation: `validate_tropicalgt_i.py --config TropicalGT-I/configs/gpu_smoke.json` read 12 moved-data records and produced graph-token batches.
- GPU smoke: `train_tropicalgt_i.py --config TropicalGT-I/configs/gpu_smoke.json` completed on CUDA.
- W&B run: https://wandb.ai/amelie-iska-math/TropicalGT-I/runs/52scyeiw
- GPU eval: checkpoint `TropicalGT-I/checkpoints/tropicalgt_i_gpu_smoke.pt` evaluated with validation NLL `5.374286810557048` and BPB proxy `7.753456929905617`.
- Paper: compiled locally from the remote TeX source to 45 pages and copied back to `TropicalGT-I/assets/tropicalgt_neurips_research_paper.pdf`.
- Iteration 2 unit tests: `/home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests -q` -> `8 passed`.
- Parameter-Golf adapter check: `py_compile train_gpt.py tropicalgt_tokengt_adapter.py` succeeded; tiny `GPT(..., use_graph_adapter=True)` forward/backward returned finite loss `3.4679651260375977` with `880` adapter parameters.
- CPU smoke after iteration 2 wrote `history`, `reasoning_trajectory_payloads.json`, and `training_metrics.html`; payload keys were `filtered_simplicial_objects`, `hover`, and `points`.
- GPU smoke after iteration 2 completed on CUDA with W&B run https://wandb.ai/amelie-iska-math/TropicalGT-I/runs/p762tx2d, validation NLL `5.408573945363362`, BPB proxy `7.8029228092569785`, and checkpoint `TropicalGT-I/checkpoints/tropicalgt_i_gpu_smoke.pt`.
- Separate GPU validation/eval/infer/render commands completed. The final GPU payload contained `6` filtered simplicial objects; the first summary was `15` vertices, `14` edges, `13` directed path \(2\)-simplices, and `28` thresholds.
- Process/GPU cleanup check after the smoke run found no lingering training or GPU compute process.

## Remaining research risks
- The model is a first functional iteration, not a competitive Parameter-Golf artifact.
- The GFlowNet reward is smoke-stage likelihood/margin based and should be replaced with verifier and task rewards in longer runs.
- GraphCG is currently a light latent-chart auxiliary and should be calibrated after longer training.
- TokenGT identifiers are deterministic finite-chart features for v1 smoke; high-scale runs should review the full upstream TokenGT implementation before architectural scaling.
