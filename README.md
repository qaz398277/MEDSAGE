# MEDSAGE: Structured Medical Visual Reasoning

MEDSAGE formulates medical visual reasoning as a four-stage scaffold (Localization, Visual Analysis, Knowledge Matching, Conclusion) and trains medical VLMs with structured supervised fine-tuning plus self-check-guided RL. The goal is robust, faithful, and clinically aligned reasoning for Med-VQA.

## Key Contributions
- Structured reasoning path: fixed LVKC stages to reduce visual-text misalignment and shortcut reasoning.
- High-quality data: from 61k medical images we build SAGE-sft20K (full-stage reasoning traces) and SAGE-rl10K (balanced RL data); PubMedVision samples are pHash-deduped against eval sets to avoid leakage.
- Self-check RL: GRPO with format reward and self-check correction; rewards apply only when retry fixes errors, encouraging reliable self-verification.

## Method Overview
![MEDSAGE](/static/images/method.png "MEDSAGE")
- Trajectory construction: RoI perturbation and text-based localization with template-based knowledge matching to create multiple answer-consistent paths (Reasoning Path Augmentation).
- Reasoning-guided SFT: LVKC-tagged full-sequence supervision teaches structured reasoning and answers.
- RL refinement: Easy-R1 GRPO with format, answer-accuracy, and self-check rewards plus dynamic weight normalization to keep reward scale stable.

## Datasets
- Sources: DeepLesion 26,851; Roboflow 15,057; PubMedVision 19,097 (total 61,005).
- SAGE-sft20K: filtered four-stage trajectories for structured SFT.
- SAGE-rl10K: 10k balanced samples for RL training.

## Main Results
- Five benchmarks: RAD 70.4 / SLAKE 79.8 / PathVQA 66.7 / PMC 58.3 / MMMU-Med 65.8; average 69.8 (best among open-source models, on par with or slightly above ViTAR).
- GPT-score evaluations show LVKC improves localization, visual analysis, knowledge alignment, and reasoning quality.
- Ablations: from baseline SFT to RPA+RL, overall accuracy rises 39.3 → 71.3, highlighting gains from structured data and self-check RL.

## Training Setup
- Hardware: 4×A100, bfloat16, DeepSpeed.
- SFT: LLaMA-Factory, full-parameter (vision encoder frozen), 4 epochs, lr 1e-5, seq len 4096, batch 4, grad accum 16.
- RL (R-GRPO): Easy-R1, 8 epochs, lr 1e-5, seq len 2048, max gen 1024, batch 2, grad accum 8, 6 candidates per sample.

## Repo Map
- index.html and static/ for the project webpage.
- medical_rl/acl_latex.tex is the paper source with full method and experiments.

## Acknowledgments
Built on public medical imaging data and auxiliary annotations generated with multimodal LLMs; thanks to the open-source community.
