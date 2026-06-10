# TropicalGT-I Implementation Status - 2026-06-10

## Completed
- Killed the previously running ToricGT training/watch/mirror processes and verified no GPU compute process remained.
- Created branch `tropicalgt-i-implementation` in the TropicalGT repo.
- Cloned `amelie-iska/TokenGT` into `external/TokenGT` and created branch `tropicalgt-i-tokenization`.
- Created branch `tropicalgt-i-tokenization` in `external/parameter-golf`.
- Moved `/home/iska/Documents/amelie/bio/ToricGT/data` to `TropicalGT-I/data/toricgt`.
- Added the TropicalGT-I package, configs, scripts, tests, README updates, paper implementation addendum, and reference synthesis.
- Added a minimal Parameter-Golf TokenGT adapter in the external fork.

## Verification evidence
- Unit tests: `/home/iska/miniconda3/envs/tokengt/bin/python -m pytest TropicalGT-I/tests -q` -> `6 passed`.
- CPU smoke: `train_tropicalgt_i.py --config TropicalGT-I/configs/smoke.json` completed and wrote checkpoint plus Plotly HTML.
- Dataset validation: `validate_tropicalgt_i.py --config TropicalGT-I/configs/gpu_smoke.json` read 12 moved-data records and produced graph-token batches.
- GPU smoke: `train_tropicalgt_i.py --config TropicalGT-I/configs/gpu_smoke.json` completed on CUDA.
- W&B run: https://wandb.ai/amelie-iska-math/TropicalGT-I/runs/52scyeiw
- GPU eval: checkpoint `TropicalGT-I/checkpoints/tropicalgt_i_gpu_smoke.pt` evaluated with validation NLL `5.374286810557048` and BPB proxy `7.753456929905617`.
- Paper: compiled locally from the remote TeX source to 45 pages and copied back to `TropicalGT-I/assets/tropicalgt_neurips_research_paper.pdf`.

## Remaining research risks
- The model is a first functional iteration, not a competitive Parameter-Golf artifact.
- The GFlowNet reward is smoke-stage likelihood/margin based and should be replaced with verifier and task rewards in longer runs.
- GraphCG is currently a light latent-chart auxiliary and should be calibrated after longer training.
- TokenGT identifiers are deterministic finite-chart features for v1 smoke; high-scale runs should review the full upstream TokenGT implementation before architectural scaling.
