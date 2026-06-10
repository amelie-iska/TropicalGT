# TropicalGT-I

[![Paper: main PDF](https://img.shields.io/badge/arXiv--style-main%20PDF-b31b1b?style=for-the-badge&logo=readthedocs&logoColor=white)](https://github.com/amelie-iska/TropicalGT/raw/tropicalgt-i-implementation/references/main.pdf)
[![Paper: TropicalGT-I PDF](https://img.shields.io/badge/arXiv--style-TropicalGT--I%20PDF-b31b1b?style=for-the-badge&logo=readthedocs&logoColor=white)](https://github.com/amelie-iska/TropicalGT/raw/tropicalgt-i-implementation/TropicalGT-I/assets/tropicalgt_neurips_research_paper.pdf)
[![Hugging Face dataset](https://img.shields.io/badge/Hugging%20Face-dataset-ffcc4d?style=for-the-badge&logo=huggingface&logoColor=111111)](https://huggingface.co/datasets/AmeliSchreiber/TropicalGT)
[![Hugging Face checkpoints](https://img.shields.io/badge/Hugging%20Face-checkpoints-ffcc4d?style=for-the-badge&logo=huggingface&logoColor=111111)](https://huggingface.co/AmeliSchreiber/TropicalGT)
[![GitHub codebase](https://img.shields.io/badge/GitHub-codebase-24292f?style=for-the-badge&logo=github&logoColor=white)](https://github.com/amelie-iska/TropicalGT)

TropicalGT-I is the active v1 implementation for tropical-geometry-guided reasoning in transformer embedding space. It combines TokenGT-style graph tokenization, tropical ring attention, GFlowNet graph-of-thought trajectories, GraphCG latent steering, and finite filtered simplicial diagnostics.

## Papers and Assets

- [Main TropicalGT paper](../references/main.pdf)
- [TropicalGT-I NeurIPS research paper](./assets/tropicalgt_neurips_research_paper.pdf)

## Runtime

From the repository root, use the `tokengt` environment with `PYTHONPATH=TropicalGT-I/src` for training, evaluation, validation, inference, and visualization scripts. The top-level README contains the full command inventory for CPU smoke tests, GPU smoke runs, data-backed training, checkpoint resume, and readiness audits.
