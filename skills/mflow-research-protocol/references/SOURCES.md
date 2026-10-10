# Sources — mflow-research-protocol

Everything in SKILL.md is paraphrased (often translated into Chinese) and adapted to MFlow's commands and
thresholds; no passages are copied. Numbers taken from a source are attributed below; numbers that are
MFlow's own (thresholds.yaml, tested in this repository) are not listed.

## Skill collections (read 2026-10-10 at the pinned commit)

| Source | What was used | License |
|---|---|---|
| [AtomisticSkills](https://github.com/learningmatter-mit/atomisticskills/tree/62574443f2a772e23dd5951f3712b65133b06715/skills) | skills/general-atomisticskills-rules (request types, research directory, plan before running, ask before installing, GPU memory estimate) | MIT |
| [K-Dense scientific skills](https://github.com/K-Dense-AI/claude-scientific-skills/blob/92ace75ac21efe19a620434e0ca4e356081fe807/skills/hypothesis-generation/SKILL.md) | hypothesis-generation (rival explanations, refutation tests, recording the plan before outcomes, scoped 'not found') | MIT |
| [obra/superpowers](https://github.com/obra/superpowers/blob/bb92a77741419a4ab5f06e711a283343f1ada0c3/skills/verification-before-completion/SKILL.md) | verification-before-completion (evidence before claims, re-run the proving command, verify delegated work) | MIT |

## Caveats

- AtomisticSkills' fine-tuning skill lists inconsistent defaults; its numbers are treated as rough heuristics.
- Committee-deviation thresholds are only comparable between tools that define the deviation the same way;
- Skill format follows agentskills.io/specification (name = directory, description ≤ 1024 characters).
