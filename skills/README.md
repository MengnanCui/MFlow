# skills/ — MFlow 的 Agent Skills（给 pi 等通用 agent 用）

符合 [agentskills.io](https://agentskills.io/specification) 标准。pi 会从这些目录自动加载：
`~/.pi/agent/skills/`、`~/.agents/skills/`、项目里的 `.pi/skills/` 或 `.agents/skills/`。

```bash
# 任选一种：链接到全局，或放进项目
ln -s "$PWD/skills/"* ~/.agents/skills/
```

**流程类**（怎么调用 mflow）：

| skill | 用途 |
|---|---|
| `mflow-train` | 数据体检 → 训练 / 微调 → 训练报告 |
| `mflow-md` | 导出模型 → 检查 LAMMPS → MD（含协议）→ MD 报告 |
| `mflow-report-reading` | 怎么读各类 JSON 报告、每种 warn/fail 的下一步 |

**经验类**（选什么、够不够、下一步）：

| skill | 用途 |
|---|---|
| `mflow-research-protocol` | 请求分类、运行目录、计划与批准、成本估算、证据先于结论 |
| `mflow-finetune-strategy` | 基础模型与 head、四种模式、E0、学习率与 epoch、委员会、遗忘的征兆 |
| `mflow-dataset-curation` | 标签与单位、应力符号、按来源切分、固定测试集、清洗与出处 |
| `mflow-potential-acceptance` | 四层验证、按用途设门槛、误差 × 委员会偏差怎么读 |
| `mflow-active-learning-campaign` | 探索器、信任窗口（DP 文献的经验值）、校准、导入拒收、何时停 |
| `mflow-md-setup` | 系综/恒温器/恒压器/协议、步长、阻尼、盒子、中止条件 |
| `mflow-property-protocols` | 熔点、扩散、RDF、VDOS、非晶、静态性质、报告模板 |
| `mflow-failure-triage` | 保留证据、一次改一个变量、分诊表、何时停 |
| `mflow-lab-notebook` | `NOTEBOOK.md`（只追加）与 `LESSONS.md`（过门槛才写），模板在 references/ |

经验类 skill 改写自公开的 skill 集合（AtomisticSkills、CatMaster、HeshamFS、superpowers、compound-engineering、K-Dense、ai2-kit、DP-GEN）和文献，
不照抄原文；每个 skill 的 `references/SOURCES.md` 列出用了哪个文件的什么内容、固定到哪个 commit、许可证。
阈值一律引用 `mflow/thresholds.yaml` 的键名，代码和经验用同一个数。`tests/test_skills.py` 检查格式和交叉引用。

共同原则：agent **只调用 `mflow` 命令、只读 JSON 报告**，不手写 mace_run_train / LAMMPS 输入，
不提交集群作业。所有判断标准写在代码里（`status` = pass / warn / fail），agent 负责的是"下一步做什么"。
