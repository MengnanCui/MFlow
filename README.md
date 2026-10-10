# MFlow

**MACE 势函数的一条龙工具：训练 → 评估 → 推理 → LAMMPS 动力学 → 质量判定。**
当前版本 **v0.1**（Python 包 `mflow/`，命令 `mflow`）；下一版的规划见 [v0.2 路线图](docs/roadmap-v0.2.md)。

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
- [四、给 agent 用：skills](#四给-agent-用skills)
- [五、输出文件与约定](#五输出文件与约定)
- [六、版本](#六版本)

---

## 安装

```bash
# 1) 先装与 CUDA 匹配的 torch，再在仓库根目录：
pip install -e ".[mace]"          # 得到命令 mflow（也可以 python -m mflow ...）
pip install -e ".[gpu,test]"      # 可选：cuequivariance-torch、nvidia-ml-py（GPU 监控）、pytest
pytest                            # 22 个测试：装了 mace-torch 会真训练 2 个 epoch，PATH 里有 lmp 会真跑 LAMMPS
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

| 层 | 内容 | v0.1 | v0.2 计划 |
|---|---|---|---|
| **0 误差** | 测试集 E/F 的 RMSE、MAE；按组拆分；最差结构 | ✅ `train-report` / `calc -group` | 力误差按元素拆分；\|ΔF\| 对 \|F\| 作图；相对力误差 |
| **1 静态性质** | 双原子曲线（平滑、短程排斥）、E–V/EOS（a₀、B₀）、弹性常数、声子（有无虚频）、空位/表面能 | ❌ | `mflow validate`：dimer / eos / phonon / defect |
| **2 动力学** | NVE 能量漂移、升温稳定性、与 AIMD 的 RDF 对比、扩散、VDOS | ✅ `md` + `md-report` 闸门（NVE 漂移、RDF 对比、MSD） | 升温稳定性阶梯；VDOS |
| **3 应用相关** | 从**你自己的生产轨迹**抽帧送 DFT 复算（Morrow 等最推荐的一条）；直接算目标性质 | 半自动：`calc -model a,b,c` 用委员会选帧 | `mflow al` 主动学习闭环，接 TFlow 做 DFT |

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

- **委员会怎么组。** 同样的数据、同样的架构，用 3–5 个不同的 seed 训练（v0.2 会提供 `train -committee N` 一键训练）。
- **主动学习看什么。** 看 `com_Fdev_max`，即每个结构里最大的原子力偏差。`metrics.json` 的 `committee` 段给出它的分布。DP-GEN 惯用的信任窗口是 0.05–0.15 eV/Å：
  - 低于下限：模型已经会了；
  - 高于上限：多半是不物理的结构；
  - 落在窗口中间：送去 DFT。
- **只用力偏差排序。** 能量标准差的指示性比力差。
- ⚠️ **不确定度不等于误差。** 委员会的不确定度通常校准得不好，只能用来**排序**，不能当误差棒。v0.2 会在有参考的数据上报告不确定度与真实误差的 Spearman 相关和 miscalibration area，用来判断这组委员会是否可信。

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

v0.1 的 `-ensemble` 提供三种：`nvt`（Nosé–Hoover 链）、`npt`（Nosé–Hoover，各向同性 `iso`）、`nve`。
下表是选择指南。标 **v0.2** 的选项需要等下一版（v0.1 里只能手写 LAMMPS 输入）。

| 目的 | 推荐做法 | 原因 |
|---|---|---|
| 给定 T、P 下的平衡结构 / 密度 | `-ensemble npt -temp T -press P`；液体、非晶、立方晶体用 iso，正交非立方用 aniso，三斜用 tri（aniso/tri 为 **v0.2**） | 让体积自己达到平衡 |
| 给定体积下的结构、RDF | 先 npt 平衡，再 `-ensemble nvt` | 体积固定，涨落更小 |
| **扩散系数、热导等输运性质** | npt/nvt 平衡之后，**生产段用 `nve`**，或用弱耦合的全局恒温器（NH/CSVR） | 强耦合的局部恒温器（如 Langevin）会**压低扩散**；NH、CSVR 与 NVE 在统计上无差别（Basconi & Shirts, *JCTC* 2013） |
| 检查势函数是否守恒、步长是否合适 | `-ensemble nve`，看 NVE 漂移闸门 | 漂移 < 5×10⁻⁴ eV/atom/ps 为通过（LAMBench 标准） |
| 升温 / 退火 / 淬火 | `-temp T1 -temp_end T2`（线性变温） | 速率 1–10 K/ps；越慢越接近真实 |
| 熔点 | 固液两相共存，在 **NPH** 下让温度自己落到 Tm（**v0.2** 协议） | 直接加热到熔化会有很大的过热滞后 |
| 0 K 结构 / 晶格常数 | `mflow relax`，或 LAMMPS 的 `fix box/relax` + `minimize`（**v0.2** 协议） | |
| 初始结构很差（原子重叠、能量很高） | 先用 CSVR（Bussi）或 Langevin 平衡（**v0.2** 的 `-thermostat`） | 稳健。不要用 Berendsen 做生产：系综不对，还会出现 flying ice cube |

**参数经验：**

| 参数 | 经验值 | 说明 |
|---|---|---|
| `-dt` | 1 fs；含 H 或温度高于约 2000 K 时 0.5 fs；不含 H 的金属、氧化物 ≤ 2 fs | 用 NVE 漂移闸门验证 |
| `-tdamp` | 默认 100·dt | 太小温度剧烈振荡，太大平衡很慢（LAMMPS `fix nh` 文档） |
| `-pdamp` | 默认 1000·dt | 同上 |
| 原子数 | 每张卡 1000–4000 | 太少 GPU 闲置；晶胞宽度不要小于 2·r_max |
| `-skip` | 默认 0.2 | 平衡段所占比例，这部分不参与闸门 |
| 是否已平衡 | 生产段的温度、体积、势能都不再有趋势 | 跑完不等于平衡了；拿前后两半对比 |

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

分析与作图（`md_summary.png`，2×3）：T、P、E、V 随时间的变化（灰色区域为平衡段）、按元素对分开的 RDF、MSD 与扩散系数。

**MD 崩了怎么办：** 先把 `-dt` 减半重跑。如果还崩，说明势函数在这个区域是外推，需要把这些构型补进训练集（见 v0.2 的主动学习）。

### 3.5 有没有 skill 能辅助设定

有。仓库自带 [`skills/mflow-md`](skills/mflow-md/SKILL.md)，规定了固定的执行顺序、参数范围和常见问题的处理。
v0.2 会参考下面这些公开 skill 补充"经验"类 skill，只借鉴写法和经验，不直接复制：

| 来源 | 可借鉴的经验 |
|---|---|
| [AtomisticSkills](https://github.com/learningmatter-mit/AtomisticSkills) `mat-md-monitors` / `mat-melting-point` / `mat-diffusion-analysis` / `mat-amorphization`（MIT 许可） | 爆炸、过冲、体积失控时的停止规则；固液共存法测熔点的建模与判据；扩散的温度范围、盒子大小与 Arrhenius 拟合；熔融–淬火速率 |
| [HeshamFS/materials-simulation-skills](https://github.com/HeshamFS/materials-simulation-skills) `md-analysis-planner` / `simulation-failure-triage`（Apache-2.0） | 检验 MSD 是否处于线性区、Yeh–Hummer 有限尺寸修正、分块误差棒；"每次只改一个变量"的逐级排错 |
| [CatMaster](https://github.com/q734738781/CatMaster) `lammps-preparation` / `trajectory-analysis`（Apache-2.0） | 不根据成分猜势函数或类型映射；先确认轨迹健康，再解读结果 |

---

## 四、给 agent 用：skills

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

## 五、输出文件与约定

| 命令 | 主要输出 |
|---|---|
| `check` | `<stem>_check.json` · `<stem>_check.png` · （加 `-split` 时）`<stem>_train.xyz` / `<stem>_test.xyz` |
| `train` / `train-report` | `<run>/config.yaml` · `<run>/<run>.model` · `<run>_train_report.json` · `_learning_curve.png` · `_test_summary.png` · `_test_mace.xyz` |
| `calc` / `plot` | `<stem>_mace.xyz` · `<stem>_metrics.json` · `<stem>_summary.png` |
| `relax` | `<stem>_relax.xyz` · `<stem>_relax_metrics.json` · `<stem>_relax.png` |
| `deploy` | `deploy/deploy.json` · `*.model-mliap_lammps.pt` · `deploy_report.json` |
| `lmp-check` | `lmp_check/lmp_env.json` |
| `md` / `md-report` | `<dir>/in.lammps` · `data.lmp` · `log.lammps` · `traj.lammpstrj` · `md.json` · `md_report.json` · `md_summary.png` |
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

## 六、版本

| 版本 | 内容 |
|---|---|
| **v0.1（当前）** | 包结构；`check` / `train` / `train-report`；`calc`（含委员会）/ `relax` / `plot`；`deploy` / `lmp-check` / `md` / `md-report`；3 个流程类 skill；22 个测试 |
| **v0.2（规划中）** | 精度评估第 1、2 层（`validate`）；不确定度校准；LAMMPS 协议库（多种恒温器/恒压器、熔点、输运、0 K）；**训练方案：委员会训练 + 主动学习闭环**（用 MD / 极小值跳跃 / 随机结构搜索探索 → 委员会或多样性筛选 → TFlow 标注 → 再训练 → 判断何时停止）；9 个经验类 skill。详见 [docs/roadmap-v0.2.md](docs/roadmap-v0.2.md) |

v0.1 之前是单文件脚本 `mflow.py`（内部版本 0.2–0.6，只有 calc/relax/plot），现已并入当前的包，计算结果逐位一致。
