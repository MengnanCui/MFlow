# Sources — mflow-failure-triage

Everything in SKILL.md is paraphrased (often translated into Chinese) and adapted to MFlow's commands and
thresholds; no passages are copied. Numbers taken from a source are attributed below; numbers that are
MFlow's own (thresholds.yaml, tested in this repository) are not listed.

## Skill collections (read 2026-10-10 at the pinned commit)

| Source | What was used | License |
|---|---|---|
| [HeshamFS materials-simulation-skills](https://github.com/heshamfs/materials-simulation-skills/tree/fa1ce8de76b2ab7a856f81891525cfa227fadfe9/skills) | robustness/simulation-failure-triage (+ references/failure_patterns.md) (preserve evidence, first warning, one change per rung, when to stop, failure categories, 'runs' ≠ valid) | Apache-2.0 |
| [DP-GEN](https://github.com/deepmodeling/dpgen/tree/0b9acec837474d2a93fa010598472feb8f69f82a) | monitoring.md (training symptom table, never retune silently, never add failed labels) | LGPL-3.0 (ideas only, no text copied) |
| [AtomisticSkills](https://github.com/learningmatter-mit/atomisticskills/tree/62574443f2a772e23dd5951f3712b65133b06715/skills) | mat-md-monitors, ml-mace-finetune (explosions out of domain; stress unit divergence) | MIT |

## Caveats

- AtomisticSkills' fine-tuning skill lists inconsistent defaults; its numbers are treated as rough heuristics.
- Committee-deviation thresholds are only comparable between tools that define the deviation the same way;
- Skill format follows agentskills.io/specification (name = directory, description ≤ 1024 characters).
