---
name: mflow-failure-triage
description: Systematic debugging when an MFlow run fails or gives suspicious results — MD that blows up or loses atoms, energy drift, LAMMPS consistency-check failures, training that diverges or plateaus, a committee that is over-confident, active-learning rounds that stop making progress. Use as soon as a report says fail or a number looks wrong, before changing parameters.
---

# 出问题时怎么查（经验）

## 原则

1. **先保留证据。** 不要删运行目录、不要用同名 `-outdir` 覆盖。新的尝试用新目录（`md_300K_dt05`）。
2. **每次只改一个变量。** 同时改 dt 和模型，就不知道是哪个起了作用。
3. **先读报告，再猜。** `*_report.json` 里第一个 fail 的 `name`/`message`；日志里**第一条** WARNING/ERROR，而不是最后崩溃的那一行。
   分清类别：显存不足 ≠ 超时，段错误 ≠ IO 失败，各有各的查法。
4. **不要用人为手段让它"稳定"。** 加强恒温器、限制位移、`-nocheck`、`-force` 掩盖的问题会污染后面所有结果。
5. **知道什么时候停。** 同一现象三次尝试都没定位，或结果只有靠人为稳定手段才成立，或势函数/输入无法核实 → 停下，把证据整理给用户。
6. **"能跑了"不等于"对了"。** 修好之后仍要过物理验证（`mflow-potential-acceptance`）。不要悄悄改参数重跑：写下证据、改了什么、预期效果。

## 分诊表

| 现象（报告里的 check） | 最可能的原因 | 下一步（一次一个） |
|---|---|---|
| `consistency: *` fail | 元素顺序 / head / 导出环境与 LAMMPS 内嵌 Python 版本不一致 / 单位 | 看 `deploy.json` 的 elements 和 head；`mflow lmp-check -deploy`；**绝不绕过** |
| `skiprun` fail | 输入语法、LAMMPS 缺包 | `skiprun.out` 第一条 ERROR；`mflow lmp-check` |
| `runaway halt` / `lost atoms` | dt 太大，或势函数外推 | 1) dt 减半重跑；2) 还崩 → `mflow validate dimer`；3) 用委员会给崩之前的帧打分（`mflow calc -model a,b,c`），高偏差 = 外推 → 主动学习补数据 |
| `closest pair` fail | 短程墙有洞 | `validate dimer`；补压缩构型 |
| `NVE energy drift` warn/fail | dt；fp32 导出；势函数不光滑 | dt/2 对比；fp64 导出对比；`validate dimer` 的 `conservation_deviation` |
| `temperature` / `halves` warn | 没平衡 | 延长，或加大 `-skip` 重出报告（`md-report`） |
| `volume drift` / `melt: volume` fail | 相变、沸腾、势函数在大体积下错误 | 看 V(t)；`validate eos` 是否单极小 |
| 训练 loss NaN / 发散 | 应力单位错（160 倍）、学习率 | `mflow check` 的应力检查；降 lr |
| 训练、验证误差都高 | 单位、标签键、E0、元素映射 | `check` 报告；`zero_shot.raw_offset` |
| 训练降、验证停 | 数据泄漏或重复、过拟合 | 按来源重新切分；`check` 的 duplicates |
| 训练好但测试差 | 测试集分布不同 | 按组误差；补数据 |
| MD 里 NaN | dt 或几何（重叠） | dt 减半；看起始结构最近距离 |
| 压强爆炸 | 原子重叠、密度不对、恒压耦合太强 | 先 minimize；`-pdamp` 默认 1000·dt |
| 校准 Spearman < 0.2 | 委员会成员太像 | `-bagging 0.8`；成员数 ≥ 4；从头训练成员 |
| `al`：每轮候选比例不降 | 探索太窄或太宽 | 看 `select` 的 accurate/failed 比例：几乎全 accurate → 探索加温度/步数/minhop；几乎全 failed → 模型没准备好，先多标注一轮 |
| `al import` 大量拒收 | DFT 设置问题、error fix | `rejects.jsonl` 的 reason 分布；SCF 不收敛 → 调 `run_label.py`（ALGO、NELM）；**失败的标注绝不进数据集** |

## 排错记录（写进 NOTEBOOK.md）

```
现象：<check 名 + message>，目录 <run_dir>
假设：A …  B …
试验：只改 <变量>，目录 <new_dir> → 结果 …
结论：原因是 …；证据：<文件/字段>
```

能推广的结论按 `mflow-lab-notebook` 的门槛写进 `LESSONS.md`。

出处与改写说明见 `references/SOURCES.md`。
