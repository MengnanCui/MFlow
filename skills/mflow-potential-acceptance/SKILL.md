---
name: mflow-potential-acceptance
description: Decide whether a trained MACE potential is good enough for the user's purpose, using MFlow's four validation layers — test-set errors, static properties (dimer, EOS, phonons, defects), dynamics gates (NVE drift, stability, RDF vs AIMD), and application-level DFT spot checks on production trajectories. Use when the user asks "is this model good enough", before production MD, and before reporting any property computed with a new model.
---

# 势函数够不够用（经验）

**低训练误差不等于可用。** 力误差小的模型照样可能在 MD 里崩溃或给出错误性质（Fu et al. 2023；Morrow et al. 2023）。按用途逐层验证，每层都要有产物。

## 四层

| 层 | 命令 | 通过 |
|---|---|---|
| 0 误差 | `mflow train-report` / `mflow calc -ref` | grade good 以上（10 meV/atom、50 meV/Å，`thresholds.yaml` 的 `train`）；**每个元素**（`forces_by_element`）和每个组都看；大力构型单独看 |
| 1 静态 | `mflow validate dimer / eos / phonon / defect` | dimer 无 fail（短程墙完整）；EOS、声子、缺陷与 DFT 参考差 < 10 %（`validate.ref_pass`） |
| 2 动力学 | `mflow md`（NVE + 目标条件）、`md-report -ref aimd.xyz` | NVE 漂移 < 5e-4 eV/atom/ps；无 halt / lost / closest pair；RDF 与 AIMD 差 < 0.1；平衡检验通过 |
| 3 应用 | 从**生产轨迹**抽 20–50 帧，DFT 复算（`mflow al` 的 export/import，或 TFlow 直接算） | 这些帧上的误差和测试集误差相当；委员会 σ 不高于训练时 |

第 3 层是最有说服力的一条：它检验的正是你要用的那些构型。

**粗略分级**（MFlow 报 RMSE；文献常报 MAE，MAE 一般比 RMSE 小 20–30 %）：

| | 很好 | 好 | 差 |
|---|---|---|---|
| 能量 | < 5 meV/atom | 5–20 | > 50 |
| 力 | < 20 meV/Å | 20–50 | > 100 |

**怎么读误差与委员会偏差的组合：**
- 测试误差低、委员会偏差高（在生产构型上）→ **覆盖不够**，去采样（主动学习）；
- 已覆盖的数据上误差就高 → 数据或训练的问题（标签、单位、E0、超参）；
- 长轨迹后期委员会偏差上升 → 轨迹走出了训练分布，后半段结果不可信。

先报告每个体系 / 条件的结果，再给总平均：好看的总数不能掩盖一个失败的体系。
先测基础模型，再在**同一留出集**上测微调模型；有参考应力时才报应力误差。

## 按用途设门槛

| 用途 | 重点 | 额外要求 |
|---|---|---|
| 晶格常数、弹性 | 1 层 EOS | V₀ < 1–2 %、B₀ < 10 % |
| 声子、热导 | 1 层声子 | 无虚频；色散与 DFT 对齐（最大频率 < 5 %） |
| 熔点 | 1 + 2 层，液体帧在训练集 | 固液都在训练/测试里；共存协议多个起点收敛 |
| 扩散、离子电导 | 2 + 3 层 | 生产温度的轨迹抽帧复算；NVE 漂移 |
| 表面、催化 | 1 层缺陷/表面 + 3 层 | 表面构型在训练集；吸附能单独验证 |
| 高温 / 冲击 | dimer 短程墙 | 压缩构型在训练集 |

## 报告

```
模型：<path>（基础模型、模式、数据集版本）
用途：…
第 0 层：E … meV/atom、F … meV/Å（测试集 N 帧）；最差元素 …
第 1 层：dimer …；EOS V₀ … vs … ；声子 …
第 2 层：NVE 漂移 …；<条件> 下 … ps 无失败；RDF vs AIMD …
第 3 层：生产轨迹 N 帧，E … F …
结论：可用于 … / 不可用于 …；还缺 …
```

没做的层要写"未验证"，不能省略。

## 验证清单

- [ ] 测试集固定且与训练按来源分开
- [ ] 每个元素的力误差都看过
- [ ] dimer 跑过
- [ ] 至少一条目标条件下的 MD 通过全部闸门
- [ ] 生产轨迹抽帧 DFT 复算（或明确说没做）

出处与改写说明见 `references/SOURCES.md`。
