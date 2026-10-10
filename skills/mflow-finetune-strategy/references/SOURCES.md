# Sources — mflow-finetune-strategy

Everything in SKILL.md is paraphrased (often translated into Chinese) and adapted to MFlow's commands and
thresholds; no passages are copied. Numbers taken from a source are attributed below; numbers that are
MFlow's own (thresholds.yaml, tested in this repository) are not listed.

## Skill collections (read 2026-10-10 at the pinned commit)

| Source | What was used | License |
|---|---|---|
| [CatMaster](https://github.com/q734738781/catmaster/tree/06b856814f980ca35f97fb01d7b970b5b18fd230) | demos/reference_scripts/mace_training_example/vasp_to_mace_finetune.md, run_train.sh (mode choice, LoRA rank 4/2/8, replay settings, E0 frozen to json, rank-deficient E0) | Apache-2.0 |
| [CatMaster](https://github.com/q734738781/catmaster/tree/06b856814f980ca35f97fb01d7b970b5b18fd230) | skills/ml_worker/mace-finetuning-and-benchmark/SKILL.md (15–25 epoch cap, keep best-validation checkpoint) | Apache-2.0 |
| [AtomisticSkills](https://github.com/learningmatter-mit/atomisticskills/tree/62574443f2a772e23dd5951f3712b65133b06715/skills) | ml-mace-finetune (reuse foundation E0 for same functional; OMAT-small needs lr 1e-4; patience) — the skill is internally inconsistent, numbers used as heuristics | MIT |
| [AtomisticSkills](https://github.com/learningmatter-mit/atomisticskills/tree/62574443f2a772e23dd5951f3712b65133b06715/skills) | ml-mlip-automl (≈10 short trials, one change at a time) | MIT |

## Literature and documentation

- Tompa … Bernstein, Csányi, *Fine-tuning MLIP foundation models*, arXiv:2606.12704 (2026) — multihead replay keeps out-of-distribution robustness
- MACE documentation, fine-tuning section (mace-torch 0.3.16)

## Caveats

- AtomisticSkills' fine-tuning skill lists inconsistent defaults; its numbers are treated as rough heuristics.
- Committee-deviation thresholds are only comparable between tools that define the deviation the same way;
- Skill format follows agentskills.io/specification (name = directory, description ≤ 1024 characters).
