# Sources — mflow-lab-notebook

Everything in SKILL.md is paraphrased (often translated into Chinese) and adapted to MFlow's commands and
thresholds; no passages are copied. Numbers taken from a source are attributed below; numbers that are
MFlow's own (thresholds.yaml, tested in this repository) are not listed.

## Skill collections (read 2026-10-10 at the pinned commit)

| Source | What was used | License |
|---|---|---|
| [EveryInc compound-engineering](https://github.com/EveryInc/compound-engineering-plugin/tree/67035e931c5cb26e80f198a7019a4502a9788273/skills/ce-compound) | skills/ce-compound/SKILL.md (the ce-durable-bar block: counterfactual test, verified and solved only, update instead of duplicate), assets/resolution-template.md (bug / knowledge tracks, retire_when) | MIT |
| [obra/superpowers](https://github.com/obra/superpowers/blob/bb92a77741419a4ab5f06e711a283343f1ada0c3/skills/verification-before-completion/SKILL.md) | verification-before-completion (re-run before claiming) | MIT |
| [K-Dense scientific skills](https://github.com/K-Dense-AI/claude-scientific-skills/blob/92ace75ac21efe19a620434e0ca4e356081fe807/skills/hypothesis-generation/SKILL.md) | hypothesis-generation (record what was seen, dated deviations) | MIT |
| [AtomisticSkills](https://github.com/learningmatter-mit/atomisticskills/tree/62574443f2a772e23dd5951f3712b65133b06715/skills) | general-atomisticskills-rules (every number with model, parameters, raw-output path) | MIT |

## Caveats

- AtomisticSkills' fine-tuning skill lists inconsistent defaults; its numbers are treated as rough heuristics.
- Committee-deviation thresholds are only comparable between tools that define the deviation the same way;
- Skill format follows agentskills.io/specification (name = directory, description ≤ 1024 characters).
