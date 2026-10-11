# Sources — mflow-potential-acceptance

Everything in SKILL.md is paraphrased (often translated into Chinese) and adapted to MFlow's commands and
thresholds; no passages are copied. Numbers taken from a source are attributed below; numbers that are
MFlow's own (thresholds.yaml, tested in this repository) are not listed.

## Skill collections (read 2026-10-10 at the pinned commit)

| Source | What was used | License |
|---|---|---|
| [AtomisticSkills](https://github.com/learningmatter-mit/atomisticskills/tree/62574443f2a772e23dd5951f3712b65133b06715/skills) | ml-mlip-benchmark (accuracy bands, reconcile units first), ml-mace-finetune (benchmark the foundation model first) | MIT |
| [CatMaster](https://github.com/q734738781/catmaster/tree/06b856814f980ca35f97fb01d7b970b5b18fd230) | mace-finetuning-and-benchmark (same split for comparisons, stress only with reference) | Apache-2.0 |
| [DP-GEN](https://github.com/deepmodeling/dpgen/tree/0b9acec837474d2a93fa010598472feb8f69f82a) | monitoring.md (per-system before aggregate; low error + high deviation = coverage gap) | LGPL-3.0 (ideas only, no text copied) |

## Literature and documentation

- Morrow, Gardner, Deringer, *J. Chem. Phys.* 158, 121501 (2023) — validation beyond test errors
- Fu et al., arXiv:2210.07237 — force accuracy does not imply MD stability
- LAMBench — NVE drift tolerance 5e-4 eV/atom/ps
- MLIP Arena, mlip_arena/flows/diatomics.py @6239aea — dimer grid and metrics
- Wen et al., *Deep Potentials for Materials Science*, Mater. Futures 1, 022601 (2022), arXiv:2203.00393 — practical guide, Table I (trust windows, accurate-ratio rule, ~100 labels/iteration)

## Caveats

- AtomisticSkills' fine-tuning skill lists inconsistent defaults; its numbers are treated as rough heuristics.
- Committee-deviation thresholds are only comparable between tools that define the deviation the same way;
- Skill format follows agentskills.io/specification (name = directory, description ≤ 1024 characters).
