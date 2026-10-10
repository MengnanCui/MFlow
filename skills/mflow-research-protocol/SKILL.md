---
name: mflow-research-protocol
description: Ground rules for any MLIP research task done with MFlow — classify the request, set up a run directory, write a plan and get the user's approval before expensive steps, estimate GPU and DFT cost, cite the artifact behind every number, and never claim a result that was not verified. Use at the start of every new user request that involves training, active learning, MD or property calculations, and before reporting a conclusion.
---

# 研究流程守则

## 1. 先分类

| 类型 | 例子 | 走哪条路 |
|---|---|---|
| 评估现有模型 | "MPA 对我的数据准不准" | `mflow check` / `calc` → `mflow-report-reading` |
| 训练 / 微调 | "用这些 DFT 数据微调" | `mflow-finetune-strategy` → `mflow-train` |
| 补数据 | "MD 崩了 / 想让模型覆盖高温" | `mflow-active-learning-campaign` |
| 跑 MD / 算性质 | "算 1000 K 的扩散系数" | `mflow-md-setup` → `mflow-property-protocols` → `mflow-md` |
| 判断够不够用 | "这个势能用了吗" | `mflow-potential-acceptance` |
| 出错了 | 任何 fail | `mflow-failure-triage` |
| 直接问一个数 / 概念 | "MACE 的 r_max 是多少" | 直接回答，不需要计划和运行目录 |

拿不准属于哪类，就问用户要的**最终数字或结论**是什么。
每个计算任务一个带日期的运行目录（`2026-10-10_cu_diffusion/`），所有产物放在里面，不放进 skill 目录。

## 2. 计划，等批准

在跑任何超过几分钟 GPU 或需要 DFT 的步骤之前，写 `plan.md`（或在 NOTEBOOK 里写一条）并给用户看：

```
目标：…（要得到的数字 / 判断）
步骤：1. mflow … 2. …
成本：GPU ≈ … h（模型、原子数、步数、ns/day 估计）；DFT ≈ … 个单点
判据：什么结果算成功（引用 thresholds.yaml 的键）
风险：…
```

用户确认后再执行。用户改了方案，原话记进 NOTEBOOK。

**估算：** MD 的 GPU 时间 = 步数 × dt / (ns/day)；先跑 `-steps` 很小的一段读 `ns_per_day`。训练的显存约为推理的 3 倍；先小规模试跑再放大。主动学习一轮的 DFT 数 = `max_per_round`（默认 150）。

**先查已有的：** 已经有训练好的模型、已有的活动目录或 LESSONS 条目时，先用它们，不要重复训练。

**有疑问的现象，先列竞争解释再动手：** 例如"MD 崩了"可能是 dt、势函数外推、初始结构、导出错误——每个解释写出"如果它成立，哪个试验会给出不同结果"，优先做能区分它们的试验。计划在看到结果之前写下；之后的偏离注明日期和原因。

## 3. 执行时

- 只用 `mflow` 命令和它的选项；不手写 `mace_run_train` 配置或 LAMMPS 输入（除非用户明确要求）。
- 命令失败时不要悄悄换成手写的替代脚本；装新包、提交集群作业前先问用户。
- 每一步先 `-dry`（有的话），看过生成的输入再跑。
- 每个命令后读 JSON 报告，fail 就停（`mflow-report-reading`）。
- 记录：`mflow-lab-notebook`。

## 4. 汇报：证据先于结论

- **每个数字都要能追溯**：值 + 来源（`<dir>/<file>.json` 的字段名）+ 模型 + 关键参数。
- "完成了"之前确认：命令 exit 0、报告 status 不是 fail、产物文件存在、数字是从产物里读出来的而不是从记忆里。
- 区分**测过的**和**推断的**：没跑的验证明确说没跑。
- warn 要列出来，不能只报好消息。
- 不确定时说不确定，并说怎样能确定（哪个命令、多少成本）。"应该""大概"不是结论。
- 文献或数据检索只能说"在查过的范围内没找到"，不能说"不存在"。
- 子 agent 或别的工具报告"成功"时，自己看产物确认。

## 5. 不做的事

- 不在用户没同意时提交 DFT 或长时间 GPU 任务。
- 不绕过一致性检查、不删别人的运行目录、不覆盖原始数据。
- 不把训练误差当作 MD 可靠性的证据（见 `mflow-potential-acceptance`）。

出处与改写说明见 `references/SOURCES.md`。
