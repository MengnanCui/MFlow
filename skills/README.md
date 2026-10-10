# skills/ — MFlow 的 Agent Skills（给 pi 等通用 agent 用）

符合 [agentskills.io](https://agentskills.io/specification) 标准。pi 会从这些目录自动加载：
`~/.pi/agent/skills/`、`~/.agents/skills/`、项目里的 `.pi/skills/` 或 `.agents/skills/`。

```bash
# 任选一种：链接到全局，或放进项目
ln -s "$PWD/skills/"* ~/.agents/skills/
```

| skill | 用途 |
|---|---|
| `mflow-train` | 数据体检 → 训练 / 微调 → 训练报告 |
| `mflow-md` | 导出模型 → 检查 LAMMPS → MD → MD 报告 |
| `mflow-report-reading` | 怎么读三类 JSON 报告、每种 warn/fail 的下一步 |

共同原则：agent **只调用 `mflow` 命令、只读 JSON 报告**，不手写 mace_run_train / LAMMPS 输入，
不提交集群作业。所有判断标准写在代码里（`status` = pass / warn / fail），agent 负责的是"下一步做什么"。
