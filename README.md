# MFlow

**MACE 势函数的一条龙工具：训练 → 评估 → 推理 → LAMMPS 动力学 → 质量判定。**
当前版本 **v0.2.0a1**（Python 包 `mflow/`，命令 `mflow`）：v0.1 的训练/推理/MD 之上，加了委员会训练与不确定度校准、主动学习闭环（接 TFlow）、`validate` 静态性质、LAMMPS 协议库。设计与依据见 [v0.2 路线图](docs/roadmap-v0.2.md)。

两条设计原则：

1. **执行交给代码，结果可以复查。** 每个命令都写一份 JSON 报告：`status` 是 `pass / warn / fail` 之一，附带逐项检查和建议。判定标准写在代码里，不靠人或大模型临场判断。
2. **经验写进 skill。** "该选哪种微调""这个误差够不够""用什么系综"这类判断放在 [`skills/`](skills/)。pi 之类的通用 agent 读 skill、调用 `mflow`、读 JSON，就能走完全流程。

数据生成（DFT 计算）不在本项目范围内，由 TFlow 负责。MFlow 的起点是一个**已经带 DFT 标签的 xyz 文件**。

---

## 目录

- [安装](#安装)
- [五分钟走完全程](#五分钟走完全程)
- [一、训练](#一训练)：数据与标签 · 体检 · 选方案 · 训练与监控 · **精度怎么看** · 评估方案
- [二、推理](#二推理)：标签 · 精度 · 不确定度（委员会）· 弛豫
- [三、LAMMPS 动力学](#三lammps-动力学)：导出 · 环境检查 · **输入如何生成** · **系综怎么选** · 质量闸门 · 可参考的 skill
- [四、主动学习：`mflow al`](#四主动学习mflow-al)：探索 → 筛选 → TFlow 标注 → 收回 → 委员会再训练 → 何时停
- [五、给 agent 用：skills](#五给-agent-用skills)
- [六、输出文件与约定](#六输出文件与约定)
- [七、版本](#七版本)

---

## 安装

```bash
# 1) 先装与 CUDA 匹配的 torch，再在仓库根目录：
pip install -e ".[mace]"          # 得到命令 mflow（也可以 python -m mflow ...）
pip install -e ".[gpu,test]"      # 可选：cuequivariance-torch、nvidia-ml-py（GPU 监控）、pytest
pip install -e ".[validate]"      # 可选：phonopy + seekpath（mflow validate phonon）
pytest                            # 42 个测试：装了 mace-torch 会真训练，PATH 里有 lmp 会真跑 LAMMPS（LJ / EAM）
```

- **LAMMPS** 需要另外从源码编译，打开 KOKKOS(CUDA)、ML-IAP、ML-SNAP、PYTHON、`MLIAP_ENABLE_PYTHON`，见 [3.1](#31-导出与环境检查)。
- **`mace-torch` 固定为 0.3.16**：上游正在重写 v1.0，命令行参数会变。

## 五分钟走完全程

```bash
mflow check -in dft.xyz -model mpa -split 0.1 -outdir check                  # 1 体检 + 零样本 + 切测试集
mflow train -in check/dft_train.xyz -test check/dft_test.xyz -name cu -dry   # 2 先看 config.yaml
mflow train -in check/dft_train.xyz -test check/dft_test.xyz -name cu        # 3 训练 + 实时监控 + 报告
mflow calc  -in new_dft.xyz -model cu/cu.model -e0 none -group config_type   # 4 在新数据上评估
mflow deploy -model cu/cu.model -outdir deploy                               # 5 导出给 LAMMPS（在 GPU 上）
mflow lmp-check -deploy deploy                                               # 6 LAMMPS 能不能跑它
mflow md -in start.xyz -deploy deploy -ensemble npt -temp 600 -steps 200000 -outdir md600   # 7 MD + 闸门
```

每一步结束都会打印 `verdict PASS / WARN / FAIL`。**出现 FAIL 就先看建议，不要往下走。**

---

## 一、训练

### 1.1 数据与标签：输入要长什么样

输入是 **extxyz**，一个文件放多个结构。每个结构需要以下内容：

| 内容 | 放在哪 | 单位 | 说明 |
|---|---|---|---|
| 总能量 | `info["<前缀>energy"]` | eV | **总能量**，不是每原子能量 |
| 力 | `arrays["<前缀>forces"]` | eV/Å | 每个原子 3 个分量 |
| 应力（可选） | `info["<前缀>stress"]` | eV/Å³，ASE 符号约定 | 3×3 或 Voigt 6 分量；要么所有结构都有，要么都没有 |
| 构型类别（推荐） | `info["config_type"]` | — | `bulk` / `surface` / `liquid` 等；用于按组统计误差和分层切分 |
| 孤立原子（推荐） | `config_type="IsolatedAtom"` 的单原子结构 | eV | 用同一套 DFT 设置算出的每个元素的孤立原子能量，也就是 E0 |

**前缀会自动识别**，依次尝试：无前缀（`energy`）、`REF_`、`ref_`、`dft_`、`DFT_`、`pbe_`、`scan_`。也可以用 `-ref dft_` 指定。
训练前，MFlow 把识别到的标签**复制**成 MACE 的默认键 `REF_energy / REF_forces / REF_stress`，写到 `<run>/data/`，原文件不动。

**DFT 标签最容易出错的三件事：**

1. **应力的单位和符号。** VASP 的 OUTCAR 和 vasprun.xml 输出 kBar，而且符号与 ASE 相反。
   - 用 `ase.io.read` 读出来的值已经转好了（eV/Å³，ASE 符号）。
   - 自己解析 OUTCAR 时要乘 −1/1602.18；否则应力大约 160 倍，训练会发散。
   - 快速自检：压缩过的晶胞，应力对角元应当为负。
2. **DFT 设置与基础模型一致。** 微调基础模型时，你的 DFT 设置最好和它的训练数据一致；否则会出现每原子几百 meV 的系统偏移：

   | 基础模型 / head | 它的训练数据 | 你的 VASP 最好用 |
   |---|---|---|
   | MACE-MP-0 / MPA-0 | MPtrj / sAlex，PBE(+U) | pymatgen `MPStaticSet`（ISMEAR 改为 0） |
   | MACE-OMAT-0、MH-1 的 `omat_pbe` head | OMat24（PBE，PBE_54 赝势） | `OMat24StaticSet`（fairchem-data-omat） |
   | MATPES 系列、MH-1 的 `matpes_r2scan` head | MatPES（PBE 或 r2SCAN，不加 U） | `MatPESStaticSet` |

   - 用 `mflow check -model ...` 看**零样本原始偏移**。超过 0.4 eV/atom，基本就是 DFT 设置不一致。
   - 处理办法：给每个元素补 IsolatedAtom；或者训练时用 `E0s=estimated`，同时考虑多头微调。
3. **不同泛函不能混。** PBE 和 PBE+U、PBE 和 r2SCAN 的标签不能放进同一个 head。要么分开训练，要么用多头。

### 1.2 体检：`mflow check`

```bash
mflow check -in dft.xyz                               # 只做不需要模型的检查
mflow check -in dft.xyz -model mpa -split 0.1         # + 基础模型零样本 + 按 config_type 分层切出 10% 测试集
mflow check -in train.xyz -test test.xyz              # + 测试集标签检查、训练集与测试集是否重叠
```

| 检查 | 判据 | 不通过时的建议 |
|---|---|---|
| 标签完整 | 每帧都有能量和力 | 补齐或删掉 |
| 数值健康 | NaN/inf → fail；单个力分量 > 50 eV/Å → warn，> 200 → fail | 多半是 SCF 不收敛或原子重叠 |
| 最近原子对 | < 0.6 Å → fail | 结构坏了 |
| 重复帧 | 几何指纹相同 | 去重，否则会在训练/验证/测试集之间泄漏 |
| 能量离群 | 对"逐元素线性拟合"的残差超过 6 MAD（先剔除粗大离群，再重拟合） | 多半是 DFT 设置混用；但表面、团簇本来就离得远，先看 config_type |
| E0 | 每个元素是否都有 IsolatedAtom | 没有就用 `--E0s estimated` |
| 零样本（`-model`） | 是否覆盖所有元素、误差多少、原始偏移是否超过 0.4 eV/atom | 换基础模型，或检查 DFT 设置 |
| 规模与分组 | 总帧数 < 50 → 提示；某组 < 10 → 提示 | 补该类构型 |
| 测试集 | 有没有独立测试集；和训练集是否重叠 | 用 `-split` 或 `-test` |

- 报告：`<stem>_check.json`，其中 `flags` 列出每类坏帧的索引。
- 图：`<stem>_check.png`，三格分别是每原子能量分布、力分布、零样本 parity。

### 1.3 选训练方案

| 情况 | `-mode` | `-foundation` | 说明 |
|---|---|---|---|
| 只关心一个体系，几十到几千帧 | **`finetune`**（默认） | `mpa`（MIT 许可）/ `mh1` / `omat` | 收敛最快 |
| 还要保住在其他体系上的泛化能力 | `multihead` | `mh1` / `mpa` | 回放预训练数据（`-pt`，默认 `mp`，需要联网下载），能保住短程排斥、减少"遗忘" |
| 数据很少，想少改动 | `lora` | 同上 | 对 MACE，LoRA 比冻结层效果好 |
| 基础模型没有你的元素，或数据超过约 5k 帧 | `scratch` | — | 最后 25% 的 epoch 进入 stage two |

超参采用 MACE 官方微调指南的推荐值：

| mode | lr | EMA | clip | 能量 : 力 权重 |
|---|---|---|---|---|
| finetune | 1e-3 | 0.999 | 1.0 | 1 : 10，全程不变 |
| multihead | 1e-4 | 0.9999 | 1.0 | 1 : 10 |
| lora | 1e-2 | 0.99 | 10 | 1 : 10 |
| scratch | 1e-2 | 0.99 | 10 | 1 : 100，之后进入 stage two |

- 比"选哪种方案"影响更大的是：**基础模型的质量、E0 是否正确、超参是否合理**（Tompa et al. 2026, arXiv:2606.12704）。
- 指南没给推荐值的参数（epoch、batch、stress 权重、r_max）用保守默认值。可以用 `-set key=value` 覆盖；键名按 `mace_run_train --help` 校验，不合法直接 fail。

### 1.4 训练与实时监控：`mflow train`

```bash
mflow train -in train.xyz -test test.xyz -name cu -dry                  # 只写 <run>/config.yaml，先看
mflow train -in train.xyz -test test.xyz -name cu                       # 训练
mflow train -in train.xyz -name cu -mode multihead -foundation mh1      # 多头回放
mflow train -in train.xyz -name cu -set lr=5e-4 max_num_epochs=30       # 覆盖超参
mflow train -in train.xyz -name cu -resume                              # 中断后续跑
```

`train` 按以下顺序执行：

1. **门槛检查。** 规则与 `check` 相同；有 fail 就不训练（`-force` 可以跳过）。
2. **准备数据。** 复制进 run 目录，统一标签键名。
3. **选 E0。** 每个元素都有孤立原子时用孤立原子；否则微调用 `estimated`，从头训练用 `average`。
4. **生成配置。** 写 `config.yaml`，并校验每个键名。
5. **启动 `mace_run_train` 并实时跟踪** `results/*_train.txt`：
   - 进度条显示 epoch、验证集 E/F RMSE，以及 GPU/CPU/内存占用。
   - loss 出现 NaN 立即停止，判 fail。
   - 验证 loss 连续 5 次上升，告警。
   - 显存超过 90%，告警并建议降低 `-batch`。
6. **出报告。** 结束后自动执行 `train-report`。

run 目录可以完整复现：`config.yaml`、`data/`、`mace_train.out`，以及 MACE 自己写出的 `results/ logs/ checkpoints/ <name>.model`。

### 1.5 精度怎么看：`mflow train-report`

精度按四层给出，从"训练有没有出问题"一直到"在你关心的结构上准不准"：

| 层 | 看什么 | 在哪里 |
|---|---|---|
| ① 训练过程 | 学习曲线：train/valid 的 loss，valid 的 E/F RMSE 随 epoch 的变化；标出最佳 epoch | `<run>_learning_curve.png`；报告里的 `curves` |
| ② 训练是否健康 | 无 NaN；过拟合（最终验证 loss 超过最佳值 1.5 倍时告警）；训练前后验证误差改善的倍数 | `checks` |
| ③ 测试集误差 | 能量（meV/atom）、力（meV/Å）的 MAE / RMSE / 最大误差 / R²；**按 config_type 拆分**，从差到好排序；**最差的 5 个结构** | `<run>_test_summary.png`；`test.errors`、`test.by_group` |
| ④ 分级 | excellent：能量 < 5 meV/atom 且力 < 20 meV/Å；good：< 10 且 < 50；其余为 needs work | `test.grade` |

- 测试集误差与 `mflow calc` 用**完全相同**的定义（见 [2.2](#22-推理精度)），所以训练报告和日后的评估可以直接比。
- 测试集与训练集来自同一套 DFT，E0 已经在模型里，所以测试集误差**不做 E0 拟合**（`E0 none`）。

> ⚠️ **力误差低不等于 MD 稳定**（Fu et al., arXiv:2210.07237）：在他们的基准里，力误差最低的模型最先在 MD 中崩溃。
> 训练报告只是第一关，后面还要过第三部分的 MD 闸门。

### 1.6 精度评估的系列方案

验证类文献（Morrow, Gardner & Deringer, *J. Chem. Phys.* 158, 121501 (2023)；LAMBench；MLIP Arena）里的检查可以归成四层。MFlow 目前的覆盖情况：

| 层 | 内容 | MFlow |
|---|---|---|
| **0 误差** | 测试集 E/F 的 RMSE、MAE；按组、**按元素**拆分；\|ΔF\| 对 \|F\|（大力构型单独看）；相对力误差 | `train-report` / `calc -group` |
| **1 静态性质** | 双原子曲线（平滑、短程排斥）、E–V/EOS（V₀、B₀、B′）、声子（有无虚频）、空位/表面能 | `mflow validate dimer / eos / phonon / defect`（见 1.7） |
| **2 动力学** | NVE 能量漂移、平衡检验、与 AIMD 的 RDF 对比、扩散（含扩散区判断）、VDOS | `md` 协议 + `md-report` 闸门（见 3.3–3.4） |
| **3 应用相关** | 从**你自己的生产轨迹**抽帧送 DFT 复算（Morrow 等最推荐的一条）；直接算目标性质 | `mflow al`（见第四章），TFlow 做 DFT |

### 1.7 静态性质：`mflow validate`

```bash
mflow validate dimer  -model cu/cu.model -elements Cu,O                        # 每个元素对的双原子曲线
mflow validate eos    -model cu/cu.model -in cu_relaxed.xyz -ref "V0=11.8,B0=140"
mflow validate phonon -model cu/cu.model -in cu_relaxed.xyz -supercell 4       # 需要 pip install phonopy seekpath
mflow validate defect -model cu/cu.model -in cu_conv.xyz -surface 111,100      # 空位形成能、表面能
```

| 子命令 | 怎么做 | 判据（`thresholds.yaml` 的 `validate` 段） |
|---|---|---|
| `dimer` | MLIP Arena 的网格（0.9·r_cov → 3.1·r_vdW，0.01 Å），扣孤立原子能量 | 短程墙有"洞"或 E(r_min) ≤ E(∞) → **fail**（MD 会塌缩）；力变号次数 ≠ 1、多余极值、能量跳变、\|F + dE/dr\| > 0.05 eV/Å、尾部不归零 → warn |
| `eos` | ±4 % 线应变 11 个体积，每个体积内弛豫原子位置，Birch–Murnaghan 拟合 | 极小值在采样范围内；E(V) 凸；与 `-ref` 相对误差 < 10 % pass、< 25 % warn |
| `phonon` | 先在固定晶胞下对称弛豫（fmax 1e-3），phonopy 有限位移 0.01 Å，Γ 中心 q 网格 | 低于 −0.1 THz 的频率算虚频；出现即 warn，低于 −0.5 THz fail |
| `defect` | 晶胞扩到每个方向 ≥ 10 Å；去一个原子后弛豫；`ase.build.surface` 建板 | 单质给 E_f = E(N−1) − (N−1)/N·E(N)；化合物只给原始差值（形成能需要化学势） |

所有命令都会先检查输入是否已经弛豫（|F| < 0.05 eV/Å、|σ| < 0.5 GPa），没弛豫会 warn：先 `mflow relax` 或 `mflow md -protocol minimize`。
`-model emt` 用 ASE 的 EMT 代替 MACE，方便没有模型时先试命令。

---

## 二、推理

### 2.1 推理与输出标签：`mflow calc`

```bash
mflow calc -in data.xyz -model cu/cu.model                  # 预测
mflow calc -in data.xyz -model cu/cu.model -ref dft_ -e0 none   # 与同一套 DFT 对比
mflow calc -in data.xyz -model mpa -group config_type       # 基础模型（在线下载），误差按组拆分
mflow calc -in data.xyz -model a.model,b.model,c.model      # 委员会：均值 + 不确定度
mflow plot -in data_mace.xyz -ref dft_ -group formula       # 只重画，不重算
```

**写进输出 xyz 的标签**（`<p>` 为前缀，默认 `mpa0_`）。输入里已有的标签**全部保留**；前缀冲突时自动顺延为 `mpa0_c1_`，不会覆盖：

| key | 含义 |
|---|---|
| `<p>energy`、`<p>forces` | 预测的总能量（eV）和力（eV/Å） |
| `<p>dE` | 与参考值之差，meV/atom，带符号，已做 E0 对齐（有参考时才有） |
| `<p>dF` | 该结构的力 RMSE，meV/Å（有参考时才有） |
| `<p>Fdev`（逐原子）、`<p>Fdev_max`、`<p>Edev` | 委员会力偏差（DP-GEN 定义：√(各模型 \|F_m − F̄\|² 的平均)）、它在该结构内的最大值、能量标准差（meV/atom） |

| 参数 | 默认 | 说明 |
|------|------|------|
| `-model` | 本地 MPA-0 路径 | `.model` 文件路径；或基础模型名（`small` / `medium` / `large` / `medium-mpa-0` / `medium-omat-0`，别名 `mpa` / `omat`），会在线下载；用逗号分隔多个模型即为委员会 |
| `-batch` | `32` | batch 大小；批量前向失败时自动退回逐结构计算 |
| `-prefix` | `mpa0_` | 输出标签前缀 |
| `-ref` | 自动探测 | 参考标签前缀（`dft_` / `REF_` / 无前缀） |
| `-e0` | `fit` | 能量对齐方式：`fit` / `mean` / `none` / json 文件 |
| `-group` | 无 | 按已有标签分组：任意 info 键，或虚拟键 `formula` / `natoms`；数值键自动切成四分位 |
| `-index` | `:` | ASE 切片，例如 `:100` |
| `-device` `-dtype` | `auto` `float64` | |
| `-head` | 无 | 多 head 模型指定使用哪个 head |
| `-out` `-outdir` `-log` `-overwrite` `-noplot` | | 输出控制 |

### 2.2 推理精度

```
error  mpa0_ vs dft_
                          MAE       RMSE     max|Δ|        R²      N        unit
energy (E0 aligned)      7.30       8.70      18.69    0.9982     40    meV/atom
energy (raw)           116.03     147.65     354.00    0.8951     40    meV/atom
forces                  15.39      19.32      73.22    0.9975  1,302       meV/Å
E0 alignment (fit), eV per atom of element: O -0.2076 · Si +0.3542
```

- **E0 对齐。** 不同 DFT 设置之间（或 DFT 与基础模型之间），总能量的零点**每种元素差一个常数**：E_pred − E_ref = Σ_j n_ij·δ_j。
  - 默认 `-e0 fit` 用最小二乘从数据里解出 δ，扣掉后再算误差。
  - 看 **`energy (E0 aligned)`** 这一行；`raw` 那行只用来看偏移有多大。力不受 E0 影响。
  - ⚠️ δ 是从被评估的数据里拟合出来的，会把模型真实的系统偏差也吸收进去。用**自己训练的模型**评估**同一套 DFT** 的数据时，请用 `-e0 none`。
- **分组。** `-group config_type` 按组列出误差，从差到好排序，图上多一列排名条形图。输出 xyz 里的 `<p>dE` / `<p>dF` 可以作为下一轮的分组键（例如 `-group mpa0_dE`，即按误差大小分组）。
- **最差的结构。** 列出索引、所在组、化学式、ΔE 和力 RMSE，方便按图索骥。
- **图。** `<stem>_summary.png` 包含能量 parity、力 parity 和两个误差分布；点数超过 2 万时自动改用 hexbin。

### 2.3 没有 DFT 参考时：用委员会估计不确定度

新结构（例如从 MD 里抽出来的帧）往往没有 DFT 标签，这时只能看模型对它有多"拿不准"：

```bash
mflow calc -in md_frames.xyz -model s1.model,s2.model,s3.model -prefix com_
```

- **委员会怎么组。** `mflow train -committee 4` 一键训练 4 个成员：每个成员换 seed，微调时默认再做 **bagging**（每个成员随机用 80 % 的训练集）。原因：从同一个基础模型微调出来的成员起点相同，只换 seed 时彼此高度相关，σ 会严重偏小。输出 `committee.json` 和 `<name>_committee_report.json`（测试集上的误差 + 校准）。
- **主动学习看什么。** 看 `com_Fdev_max`，即每个结构里最大的原子力偏差。`metrics.json` 的 `committee` 段给出它的分布。DP-GEN 惯用的信任窗口是 0.05–0.15 eV/Å：
  - 低于下限：模型已经会了；
  - 高于上限：多半是不物理的结构；
  - 落在窗口中间：送去 DFT。
- **只用力偏差排序。** 能量标准差的指示性比力差。
- ⚠️ **不确定度不等于误差。** 委员会的不确定度通常校准得不好。在有参考标签的数据上 `calc` 会给出校准表（`<stem>_calibration.png`）：
  - **Spearman ρ**（σ 与 \|误差\| 的秩相关）：≥ 0.5 pass，≥ 0.2 warn，否则 σ 连排序都不可靠；
  - **miscalibration area**、RMS 校准误差、sharpness；
  - **RMS(误差)/RMS(σ)**：> 1 表示过度自信。主动学习筛选时会用这个比值把 σ 放大（variance scaling）。

### 2.4 结构弛豫：`mflow relax`

```bash
mflow relax -in data.xyz                           # 位置 + 晶胞，fmax 0.05
mflow relax -in data.xyz -cell none -fmax 0.02     # 固定晶胞，更严
mflow relax -in data.xyz -nproc 3 -dtype float32   # 3 个 worker 共享一张卡
mflow relax -in data.xyz -resume                   # 接着上次中断的地方继续
```

- **优化器。** 用 ASE 自带的（`fire` / `fire2` / `lbfgs` / `bfgs`）；`-cell full` 时套用 `FrechetCellFilter`。
- **输出标签。** `<p>steps`、`<p>converged`、`<p>energy0`、`<p>Erelax`、`<p>fmax`、`<p>dmax`、`<p>dvol`、`<p>walltime`。
  - 未收敛的结构也会写出，**请按 `<p>converged` 筛选**，不要按 `<p>fmax`。
- **并行。** 用 spawn 方式开多进程，每个 worker 只加载一次模型。实测 `-nproc 3` 与 `-nproc 1` 的能量差在 2×10⁻¹⁴ eV。
- **中断恢复。** 结果流式写入 `.part` 文件，进程被 kill 也不丢，`-resume` 接着跑。
- **资源监控。** 进度条右侧实时显示整卡/整机的 GPU、CPU、内存占用。
  - GPU 利用率低于 30% 显示为红色，说明该加 worker；显存超过 90% 说明该减 worker。
  - 设 `MFLOW_NO_MONITOR=1` 可以关闭监控。
- ⚠️ **单卡上跑不满。** ASE 这条路线无法占满一张 GPU；大批量的小结构以后会改用 torch-sim（v0.2）。

---

## 三、LAMMPS 动力学

MACE 通过 **ML-IAP unified + KOKKOS** 在 GPU 上运行。**积分、恒温、恒压、dump、restart 全部由 LAMMPS 自己的模块完成**。
MFlow 只负责生成输入、检查、启动和分析，进程内不加载 LAMMPS。

### 3.1 导出与环境检查

```bash
mflow deploy -model cu/cu.model -outdir deploy                  # 单 head 模型
mflow deploy -model cu/cu.model -head Default -outdir deploy    # 多 head 必须指定（多头微调后用自己的 head，不要用 pt_head）
mflow deploy -model cu/cu.model -dtype float64 -outdir deploy64 # 另导一份双精度，用于严格对比
mflow lmp-check -deploy deploy                                  # 换机器或换 LAMMPS 后做一次
```

**`deploy`**
- 调用 `mace_create_lammps_model --format mliap`。**要在跑 LAMMPS 的那种 GPU 上执行。**
- 写出 `deploy.json`，内容包括：源模型和导出文件的 sha256、head、**元素顺序**（即 LAMMPS 原子类型的顺序）、r_max、精度、版本号。
- 后续所有命令只认 `deploy.json`。

**`lmp-check`** 依次检查三件事：
1. `lmp -h` 里是否有 KOKKOS(CUDA)、ML-IAP、PYTHON。
2. **LAMMPS 内嵌的 Python** 能否 import mace、torch、cuequivariance、cupy，torch 能否看到 GPU。
3. 用导出的模型实际跑一次 2 原子的 `run 0`。

`lmp` 可执行文件的查找顺序：`-lmp` > 环境变量 `MFLOW_LMP` > PATH。

LAMMPS 编译选项（摘自 MACE 文档）：

```bash
cmake ../cmake -D BUILD_MPI=ON -D BUILD_SHARED_LIBS=ON -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_<你的GPU>=ON -D PKG_ML-IAP=ON -D PKG_ML-SNAP=ON -D PKG_PYTHON=ON -D MLIAP_ENABLE_PYTHON=ON
```

### 3.2 输入是怎么生成的

```bash
mflow md -in start.xyz -deploy deploy -ensemble nvt -temp 300 -steps 100000 -outdir md300 -dry   # 只写输入，先看
mflow md -in start.xyz -deploy deploy -ensemble nvt -temp 300 -steps 100000 -outdir md300        # 运行
mflow md -in start.xyz -deploy deploy -steps 200000 -outdir md300 -resume                         # 续跑；-steps 是总步数
```

**只有一个固定模板，所有选择都写在文件里：**

```
units metal / atom_style atomic / atom_modify map yes / newton on / boundary p p p
read_data       data.lmp                       # 由 ASE 写出；原子类型 = 结构里出现的元素，按原子序数排序
pair_style      mliap unified <模型>.pt 0
pair_coeff      * * Al Cu                      # 与 data 文件、dump_modify element 用同一份元素列表
neighbor 2.0 bin / neigh_modify every 1 delay 0 check yes
timestep        <dt/1000>                      # -dt 的单位是 fs
thermo_style    custom step time temp pe ke etotal press vol density pxx pyy pzz
thermo_modify   lost error
velocity        all create T seed dist gaussian mom yes rot yes
fix             md all nvt|npt|nve ...         # 见 3.3
fix             runaway all halt N v_tnow > 5×T error soft     # 温度失控就停
dump            traj all custom N traj.lammpstrj id type element xu yu zu vx vy vz fx fy fz
restart         N restart.a restart.b          # 两个文件轮流写
run             STEPS upto                     # 续跑时跑到总步数为止
```

**运行前自动做三件事：**

1. **参数检查。**
   - 步长：含 H 时 ≤ 1 fs，否则 ≤ 2 fs。
   - 阻尼：tdamp ≈ 100·dt，pdamp ≈ 1000·dt。
   - NPT 必须是全周期体系。
   - 每张卡 1000–4000 原子最合适。
   - 晶胞宽度小于 2·r_max 时提示扩胞。
2. **`lmp -sr`。** 把真实输入完整解析、初始化一遍，但不积分。语法错误、缺少的 style、原子类型数不对，都会在这一步暴露。
3. **一致性检查。** 对起始结构 `run 0`，要求 LAMMPS 算出的**能量、逐原子力、压强**与同一模型在 LAMMPS 外的计算一致。
   - 容差：fp64 导出时为 1e-5 eV/atom、1e-4 eV/Å；fp32 导出时为 1e-3、2e-2。
   - **元素顺序错位、单位错误、维里符号、导错 head、版本不一致**，都会在这里被拦下。
   - **一致性检查失败时，不要用 `-nocheck` 或 `-force` 绕过。**

运行中实时显示步数、温度、ns/day 和 GPU 占用；结束后自动执行 `md-report`。

### 3.3 系综怎么选

`-ensemble nvt|npt|nve`，`-thermostat nh|csvr|langevin`（Nosé–Hoover 链 / Bussi 随机速度重标 / Langevin），`-baro iso|aniso|tri`；
多阶段的常用流程用 `-protocol`（见下一小节的表）。

| 目的 | 推荐做法 | 原因 |
|---|---|---|
| 给定 T、P 下的平衡结构 / 密度 | `-protocol equilibrate -temp T -press P`；液体、非晶、立方晶体 `-baro iso`，正交非立方 `aniso`，三斜 `tri` | NPT 让体积平衡，再固定在后半段平均体积上做 NVT |
| 给定体积下的结构、RDF | 先 npt 平衡，再 `-ensemble nvt` | 体积固定，涨落更小 |
| **扩散系数、热导等输运性质** | `-protocol transport`：NPT → 平均体积 → NVT → **NVE 生产段** | 强耦合的局部恒温器（如 Langevin）会**压低扩散**；NH、CSVR 与 NVE 在统计上无差别（Basconi & Shirts, *JCTC* 2013） |
| 检查势函数是否守恒、步长是否合适 | `-ensemble nve`，看 NVE 漂移闸门 | 漂移 < 5×10⁻⁴ eV/atom/ps 为通过（LAMBench 标准） |
| 升温 / 退火 / 淬火 | `-protocol anneal` 或 `quench`，`-temp T1 -temp_end T2` | 报告给出 K/ps，> 10 K/ps warn；越慢越接近真实 |
| 熔点 | `-protocol melt`：固液两相共存，**NPH** 下温度自己落到 Tm；按层的固相键数判断两相是否都在 | 直接加热到熔化会有很大的过热滞后（完美晶体可过热到约 1.2 Tm） |
| 0 K 结构 / 晶格常数 | `mflow relax`，或 `-protocol minimize`（LAMMPS 的 `fix box/relax` + 反复 `minimize`，输出 `relaxed.xyz`） | box/relax 不是严格的极小化，要重复到晶胞不变（LAMMPS 文档） |
| 初始结构很差（原子重叠、能量很高） | 先 `-protocol minimize`，再 `-thermostat csvr` 或 `langevin` 平衡 | 稳健。不要用 Berendsen 做生产：系综不对，还会出现 flying ice cube |

**参数经验：**

| 参数 | 经验值 | 说明 |
|---|---|---|
| `-dt` | 1 fs；含 H 或温度高于约 2000 K 时 0.5 fs；不含 H 的金属、氧化物 ≤ 2 fs | 用 NVE 漂移闸门验证 |
| `-tdamp` | 默认 100·dt | 太小温度剧烈振荡，太大平衡很慢（LAMMPS `fix nh` 文档） |
| `-pdamp` | 默认 1000·dt | 同上 |
| 原子数 | 每张卡 1000–4000 | 太少 GPU 闲置；晶胞宽度不要小于 2·r_max |
| `-skip` | 默认 0.2 | 平衡段所占比例，这部分不参与闸门 |
| 是否已平衡 | 生产段的温度、体积、势能都不再有趋势 | `md-report` 自动比较前后两半（差 > 3 个分块误差 warn） |

**协议一览（`-protocol`）：**

| 协议 | 阶段 | 报告里多出的内容 |
|---|---|---|
| `equilibrate` | NPT 50 %（结束时设为后半段平均体积）→ NVT 50 % | 闸门只看最后一段 |
| `transport` | NPT 20 % → 平均体积 → NVT 10 % → NVE 70 % | NVE 漂移；D（总的和 x/y/z 分量）、MSD 双对数斜率（0.85–1.15 才算扩散区）；`-viscosity` 加 Yeh–Hummer 修正 |
| `anneal` / `quench` | 升温–保温–降温 / 保温–降温 | 变温速率 K/ps |
| `minimize` | box/relax + minimize 循环（最多 20 次） | 最大力、压强、体积变化、`relaxed.xyz` |
| `melt` | NPT 晶体 → 上半盒在 `-melt_temp` 熔化（下半冻结）→ 两半都设到 `-temp` → NPH | T_m 估计 ± 分块误差、液相比例、体积是否失控（沸腾） |

续跑（`-resume`）按阶段进行：已完成的阶段跳过，`-steps` 仍是总步数。

### 3.4 质量闸门：`mflow md-report`

```bash
mflow md-report -dir md300                    # md 结束时会自动执行
mflow md-report -dir md300 -skip 0.3 -stride 5
mflow md-report -dir md300 -ref aimd.xyz      # 与 AIMD 比较 RDF
```

| 闸门 | 判据 |
|---|---|
| LAMMPS ERROR、丢原子、fix halt 触发 | 出现即 fail |
| Dangerous builds | > 0 告警（步长或 skin 不合适） |
| NVE 能量漂移 | 生产段 E_total/atom 对时间的斜率：< 5×10⁻⁴ eV/atom/ps 为 pass，< 5×10⁻³ 为 warn |
| 温度（NVT/NPT） | 生产段均值与目标值相差 < 5% 为 pass，< 15% 为 warn；生产段线性趋势需 < 5% |
| 体积（NPT） | 生产段体积变化 < 2% 为 pass，< 5% 为 warn |
| 最近原子对 | 任何一帧 < 0.6 Å 即 fail（势函数在这里有"洞"，需要补数据） |
| RDF 对比参考（`-ref`） | 每个元素对 g(r) 的平均绝对差 < 0.1 为 pass |
| 平衡（前后两半） | T、PE（NPT 时加 V）前后两半之差 < 3 个分块误差 |
| 扩散区 | 可动元素的 MSD 双对数斜率在 0.85–1.15；在 NVT/NPT 下测扩散会提示改用 transport |

统计：均值的误差用分块平均（Flyvbjerg–Petersen），报告给出 `± SE` 和有效样本数 `n_eff`。
分析与作图（`md_summary.png`，2×3）：T、P、E、V 随时间的变化（灰色为平衡段，虚线为协议阶段分界）、按元素对分开的 RDF、VDOS（帧间隔 ≤ 25 fs 时，否则画 MSD）。VDOS 由质量加权的速度自相关函数算出，写进 `vdos.dat`。

**MD 崩了怎么办：** 先把 `-dt` 减半重跑。如果还崩，说明势函数在这个区域是外推，需要把这些构型补进训练集（见第四章的主动学习），并跑 `mflow validate dimer` 看短程墙。

### 3.5 有没有 skill 能辅助设定

有。[`skills/mflow-md`](skills/mflow-md/SKILL.md) 规定执行顺序、参数范围和常见问题；
[`mflow-md-setup`](skills/mflow-md-setup/SKILL.md)（系综、步长、盒子）、[`mflow-property-protocols`](skills/mflow-property-protocols/SKILL.md)（熔点、扩散、RDF、非晶）、[`mflow-failure-triage`](skills/mflow-failure-triage/SKILL.md)（崩溃排错）是经验类 skill。
它们参考了下面这些公开 skill，只借鉴写法和经验、不直接复制，出处记在各 skill 的 `references/SOURCES.md`：

| 来源 | 可借鉴的经验 |
|---|---|
| [AtomisticSkills](https://github.com/learningmatter-mit/AtomisticSkills) `mat-md-monitors` / `mat-melting-point` / `mat-diffusion-analysis` / `mat-amorphization`（MIT 许可） | 爆炸、过冲、体积失控时的停止规则；固液共存法测熔点的建模与判据；扩散的温度范围、盒子大小与 Arrhenius 拟合；熔融–淬火速率 |
| [HeshamFS/materials-simulation-skills](https://github.com/HeshamFS/materials-simulation-skills) `md-analysis-planner` / `simulation-failure-triage`（Apache-2.0） | 检验 MSD 是否处于线性区、Yeh–Hummer 有限尺寸修正、分块误差棒；"每次只改一个变量"的逐级排错 |
| [CatMaster](https://github.com/q734738781/CatMaster) `lammps-preparation` / `trajectory-analysis`（Apache-2.0） | 不根据成分猜势函数或类型映射；先确认轨迹健康，再解读结果 |

---

## 四、主动学习：`mflow al`

**MFlow 负责"提议、挑选、交出去、收回来、再训练、判断何时停"，DFT 由 [TFlow](https://github.com/MengnanCui/TFlow) 算。**

```bash
mflow al init   -name cu_al -system start.xyz -data dft.xyz -test test.xyz -foundation mpa -committee 4
mflow al run    -name cu_al                 # 第 0 轮：基础模型探索 → FPS 选帧 → 导出给 TFlow，然后停下
#   …… 在 TFlow 里算 round_00/export/candidates.extxyz（命令写在 round_00/export/README.txt）……
mflow al import -name cu_al -tflow /path/to/calc_dir
mflow al run    -name cu_al                 # 训练委员会 → 探索 → 委员会筛选 → 导出（下一轮），再停下
mflow al status -name cu_al                 # 台账 + 停止判据
```

每一步也能单独跑：`al explore / select / export / import / train`。`al run` 走到"需要 DFT"时一定停下（人工闸门）。

### 4.1 一轮做什么

| 步骤 | 做法 | 在哪改 |
|---|---|---|
| 探索 | `md`：T（和 P）网格上的短 MD，每轮温度乘 `widen`（默认 1.25）；引擎 `ase`（Langevin）或 `lammps`（走 `mflow md`，自动导出成员 0）。`perturb`：rattle + 随机应变（+ 空位）。`minhop`：ASE 极小值跳跃，**所有到过的局部极小**都收（不只被接受的） | `campaign.yaml` 的 `explore` |
| 筛选 | 第 0 轮没有委员会：只做多样性（MACE 描述子上的 FPS）。之后：委员会力偏差 `Fdev_max`，DP-GEN 式分类 accurate / candidate / failed；信任窗口 `fixed`（0.05–0.15 eV/Å）、`auto`（训练集偏差的 Q3 + 1.5·IQR，委员会没收敛时自动退回 fixed）、`relative`；σ 先按测试集上的 RMS(误差)/RMS(σ) 放大；候选再用 FPS（相对已有训练集）限量 `max_per_round` | `select` |
| 导出 | `round_NN/export/candidates.extxyz`（只有几何和 `mflow_id`）、`run_label.py`（给 `vaspctl submit --src-run` 的单点脚本）、`manifest.json`、`README.txt`（四条命令） | `run_label.py` 里的 INCAR：**整个活动保持一致**，并和基础模型的 DFT 设置对齐 |
| 收回 | 读 TFlow 的 `job_map.json`，按 `info.mflow_id` 对回去，解析 `static/OUTCAR`；**拒收**：状态不是 done、跑过 error fix（`error_job_*`、KPOINTS 被改）、SCF 步数达到 NELM（TFlow 对 static 不查这个）、结构被改过、OUTCAR 读不了；再过一遍 `check` 的规则（NaN、力过大、原子过近）。拒收原因写 `rejects.jsonl` | `-tflow <calc_dir>`，或 `-in labelled.xyz`（任何带 `mflow_id` 和参考标签的 extxyz） |
| 训练 | 数据集**每轮重建**（initial + 所有已标注轮次），测试集固定；委员会 `train -committee N`（bagging），记下 Spearman 和 σ 比值 | `train` |
| 判断 | `al status`：候选比例 < 5 %；**新帧误差**（这一轮 DFT 刚算、模型没见过的帧）能量 < 10 meV/atom、力 < 50 meV/Å。都满足才建议停，然后还要跑生产 MD 看闸门 | `stop`，或 `thresholds.yaml` 的 `al` 段 |

`ledger.jsonl` 只追加：每一步的设置、数量、模型、误差都记在里面；`state.json` 是当前轮次和阶段。

### 4.2 与 TFlow 的约定

```bash
vaspctl prepare <calc_dir> round_00/export/candidates.extxyz --id-key mflow_id --tag mflow_r00
vaspctl submit  <calc_dir> --src-run round_00/export/run_label.py --src-submit <submit.sh> --max-concurrent 20 --auto-resume --watch
vaspctl report  <calc_dir> --task-dirs static --no-queue
mflow al import -name cu_al -tflow <calc_dir>
```

- `REF_energy` 取 VASP 的 free energy（TOTEN，力是它的导数），σ→0 的能量另存为 `REF_energy_sigma0`；`REF_stress` 是 ASE 约定（eV/Å³，符号已翻转）。
- TFlow 的错误修复可能缩放体积或改 KPOINTS，修复后算出来的不是你要的那个结构或设置，MFlow 一律拒收而不是"将就"。

### 4.3 方案怎么选

依据见[路线图第 1 节](docs/roadmap-v0.2.md#1-训练方案全景来自-20242026-年的综述)（Kulichenko 2024 *Chem. Rev.*、Grasselli 2025、Tompa 2026 等）。经验写在 [`mflow-active-learning-campaign`](skills/mflow-active-learning-campaign/SKILL.md)：

- 体相、液体：`md` 网格 + `perturb`；团簇、表面：加 `minhop`。
- 委员会 ≥ 3，建议 4；微调时保持 bagging。
- `trust: fixed` 起步；委员会校准好（Spearman ≥ 0.5）后再试 `auto`。
- 一轮 50–150 个结构；连续两轮候选比例不降，说明探索不够（加温度、加步数、开 minhop），而不是模型够好了。

---

## 五、给 agent 用：skills

[`skills/`](skills/) 里的 skill 遵循 [agentskills.io](https://agentskills.io/specification) 标准，pi 会自动加载。
安装：`ln -s "$PWD/skills/"* ~/.agents/skills/`。

| skill | 用途 |
|---|---|
| [`mflow-train`](skills/mflow-train/SKILL.md) | 体检 → dry run → 训练 → 报告；怎么选模式；超参只能用 `-set` 改 |
| [`mflow-md`](skills/mflow-md/SKILL.md) | 导出 → 环境检查 → dry run → MD → 报告；参数范围；一致性检查失败绝不绕过 |
| [`mflow-report-reading`](skills/mflow-report-reading/SKILL.md) | 怎么读 JSON 报告；遇到 fail 必须停；不编造数字 |

**分工：** 代码控制执行（顺序、校验、闸门）；skill 提供经验（选什么、够不够、下一步做什么）。
v0.1 的 skill 以流程为主；v0.2 会补 9 个经验类 skill，见[路线图](docs/roadmap-v0.2.md#4-经验类-skills)。

---

## 六、输出文件与约定

| 命令 | 主要输出 |
|---|---|
| `check` | `<stem>_check.json` · `<stem>_check.png` · （加 `-split` 时）`<stem>_train.xyz` / `<stem>_test.xyz` |
| `train` / `train-report` | `<run>/config.yaml` · `<run>/<run>.model` · `<run>_train_report.json` · `_learning_curve.png` · `_test_summary.png` · `_test_mace.xyz` |
| `calc` / `plot` | `<stem>_mace.xyz` · `<stem>_metrics.json` · `<stem>_summary.png` |
| `relax` | `<stem>_relax.xyz` · `<stem>_relax_metrics.json` · `<stem>_relax.png` |
| `deploy` | `deploy/deploy.json` · `*.model-mliap_lammps.pt` · `deploy_report.json` |
| `lmp-check` | `lmp_check/lmp_env.json` |
| `md` / `md-report` | `<dir>/in.lammps` · `data.lmp` · `log.lammps` · `traj.lammpstrj` · `md.json` · `md_report.json` · `md_summary.png`；协议另有 `vdos.dat` · `phase_profile.dat`（melt）· `relaxed.xyz`（minimize） |
| `train -committee N` | `<dir>/m0..m{N-1}/` · `committee.json` · `<name>_committee_report.json` · `committee_calibration.png` |
| `validate` | `validate/<stem>_<kind>.json` · `.png` |
| `al` | `<campaign>/campaign.yaml` · `state.json` · `ledger.jsonl` · `status.json` · `data/train_vNN.xyz` · `round_NN/{explore,select,export,train}/` |
| 所有命令 | `py.log`：终端输出的无色副本 |

**约定：**
- 参数一律用多字母（`-in`、`-model`……），避免冲突。
- 终端彩色、日志无色，内容相同。
- 不覆盖已有标签。
- 长任务都能 `-resume`。

**其他文档：**
- MFlow 替你做的所有决定和对应的风险点，记录在 [`docs/conventions.html`](docs/conventions.html)：E0 对齐、单位、训练默认值、LAMMPS 的单位 / 元素顺序 / 维里约定、一致性检查容差。
- 针对特定体系的小工具放在 [`tools/`](tools/README.md)，例如 `izo_formation.py`（In–Zn–O 形成能）。

---

## 七、版本

| 版本 | 内容 |
|---|---|
| v0.1 | 包结构；`check` / `train` / `train-report`；`calc`（含委员会）/ `relax` / `plot`；`deploy` / `lmp-check` / `md` / `md-report`；3 个流程类 skill |
| **v0.2.0a1（当前）** | a：`train -committee`（bagging）、不确定度校准、力误差按元素、`thresholds.yaml`；b：`mflow al` 主动学习闭环 + TFlow 对接；c：极小值跳跃探索；d：`validate`（dimer/eos/phonon/defect）、LAMMPS 协议库（恒温器/恒压器、equilibrate/transport/anneal/quench/minimize/melt）、`md-report` 统计（分块误差、平衡检验、扩散区、VDOS、Yeh–Hummer）；e：9 个经验类 skill。42 个测试。**尚未在 GPU + 真实 MACE/LAMMPS 上做端到端验证**（测试用 EMT / LJ / EAM 代替） |
| 以后 | 随机结构搜索探索、多头委员会（MACE PR #800 合并后）、不确定度偏置 MD、PLUMED 增强采样模板、集群提交，见路线图 |

v0.1 之前是单文件脚本 `mflow.py`（内部版本 0.2–0.6，只有 calc/relax/plot），现已并入当前的包，计算结果逐位一致。
