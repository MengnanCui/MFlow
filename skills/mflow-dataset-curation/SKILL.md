---
name: mflow-dataset-curation
description: Prepare and maintain DFT training data for MACE with MFlow — label keys and units, stress sign, isolated-atom energies, deduplication and outliers, train/validation/test splits by source trajectory, a fixed test set, and documenting what the data covers. Use when the user brings new DFT data, merges datasets, imports active-learning labels, or asks why a model trained on "good" data behaves badly.
---

# 训练数据整理（经验）

`mflow check` 做自动检查；这里是检查不了的判断。

## 1. 标签

| 项 | 要求 | 常见错误 |
|---|---|---|
| 能量 | 总能量（eV），`info` 里；同一套 DFT 设置 | 每原子能量；混了不同 ENCUT / +U |
| 力 | eV/Å，`arrays` 里 | — |
| 应力 | eV/Å³，**ASE 符号**（压缩为负） | 直接用 OUTCAR 的 kBar → 大约 160 倍、符号反 |
| 孤立原子 | `config_type=IsolatedAtom`，同一 DFT 设置、自旋极化 | 缺失 → E0 只能估计 |
| 自由能 vs σ→0 能量 | 和力一致的是 free energy（TOTEN） | 混用两种 |

换算：eV/Å³ = −kBar / 1602.18。自检：压缩过的晶胞，应力对角元为负；`mflow check` 的应力检查通过。

**从 VASP 轨迹提取：** 取所有离子步，但每帧记下 SCF 步数和 NELM，之后再按"SCF 收敛"过滤，而不是提取时就丢；读出的帧数和 vasprun 里的计算步数对不上时，跳过整个文件，不要截断。每帧保留出处：源路径、离子步序号、SCF 步数、关键 INCAR（ENCUT、EDIFF、ISIF）。

## 2. 切分

- **按来源切，不按帧切。** 同一条 AIMD 或同一次弛豫的帧高度相关；随机按帧切会让测试误差偏乐观。用 `config_type` 或来源标签分组，整组进训练或测试。
- **测试集固定。** 活动开始时定好，之后每轮不变；换了测试集，轮次之间就不能比。
- 验证集（训练时选最佳 epoch）和测试集（最终报告）分开；大数据集常用验证 5 % + 测试 5 %，固定 seed。
- 比较不同模型时用同一个切分，不同切分上的指标不能比。
- 测试集要覆盖你关心的用途：要跑 1000 K 液体，测试集里就要有 1000 K 液体的帧。

## 3. 清洗

| 问题 | 怎么发现 | 处理 |
|---|---|---|
| 重复帧 | `check` 的 duplicates | 删除；主动学习每轮会用 FPS 相对已有数据去重 |
| 能量离群 | `check` 的 outliers（按成分拟合后 MAD） | 先查 DFT（SCF 不收敛？设置不同？），再决定删 |
| 力过大 / NaN | `check` | 删；查 DFT |
| 原子过近 | `check` 的 min distance | 删，或确认是有意的高压构型 |
| SCF 未收敛 | AL 导入时自动拒收；其他来源要自己查 OSZICAR | 删 |

删数据要记在 NOTEBOOK（删了哪些、为什么）；每次导出用新目录，跳过的文件记账。合并前每帧都要有有限的能量和力。

## 4. 覆盖与记录

每个数据集版本写清楚：
- 成分范围、相（晶体/液体/表面/缺陷）、温度和压力范围、帧数（按组）；
- DFT 设置（泛函、ENCUT、k 点、+U、自旋、ISMEAR/SIGMA）；
- 来源（AIMD、AL 轮次、弛豫轨迹）。

主动学习里数据集每轮从 `initial` + 各轮标注**重建**（`data/train_vNN.xyz`），不要手工追加。

## 5. 误区

| 误区 | 后果 |
|---|---|
| 只有近平衡结构 | MD 一偏离就外推；补高温、压缩、拉伸、缺陷 |
| 不同来源 DFT 设置混用 | 系统偏移，模型学到的是噪声 |
| 测试集跟着训练集一起变 | 进步是假的 |
| 只看总 RMSE | 少数元素、少数构型的误差被平均掉；看 `forces_by_element`、按组误差 |

出处与改写说明见 `references/SOURCES.md`。
