# Sources — mflow-dataset-curation

Everything in SKILL.md is paraphrased (often translated into Chinese) and adapted to MFlow's commands and
thresholds; no passages are copied. Numbers taken from a source are attributed below; numbers that are
MFlow's own (thresholds.yaml, tested in this repository) are not listed.

## Skill collections (read 2026-10-10 at the pinned commit)

| Source | What was used | License |
|---|---|---|
| [CatMaster](https://github.com/q734738781/catmaster/tree/06b856814f980ca35f97fb01d7b970b5b18fd230) | skills/ml_worker/mace-dataset-curation/SKILL.md and vasp_to_mace_finetune.md §2–5 (all ionic steps with SCF flag, count check, provenance, split by source run, 5 %/5 %) | Apache-2.0 |
| [AtomisticSkills](https://github.com/learningmatter-mit/atomisticskills/tree/62574443f2a772e23dd5951f3712b65133b06715/skills) | general-property-units, ml-mlip-benchmark (stress units and sign check) | MIT |
| [DP-GEN](https://github.com/deepmodeling/dpgen/tree/0b9acec837474d2a93fa010598472feb8f69f82a) | skills/dpgen-run/references/monitoring.md (reject incomplete/duplicate/unconverged labels) | LGPL-3.0 (ideas only, no text copied) |

## Caveats

- AtomisticSkills' fine-tuning skill lists inconsistent defaults; its numbers are treated as rough heuristics.
- Committee-deviation thresholds are only comparable between tools that define the deviation the same way;
- Skill format follows agentskills.io/specification (name = directory, description ≤ 1024 characters).
