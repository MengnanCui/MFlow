# MFlow

机器学习势工具箱（v1.0.0，Python 包 `mflow/`，命令 `mflow`）。一条线走完 MACE 的 **数据体检 → 训练/微调 → 评估 → 导出 → LAMMPS 动力学 → 质量闸门**，
每一步都输出机器可读的 `pass / warn / fail` 报告，可以直接交给 pi 之类的通用 agent 调用（见 [`skills/`](skills/)）。

| 阶段 | 命令 | 报告 |
|---|---|---|
| 数据体检 | `mflow check` | `<stem>_check.json` |
| 训练 / 微调 + 实时监控 | `mflow train` | `<run>/<run>_train_report.json` |
| 训练分析 | `mflow train-report` | 同上 |
| 推理 / 委员会 / 弛豫 | `mflow calc` · `mflow relax` · `mflow plot` | `<stem>_metrics.json` |
| 导出到 LAMMPS（ML-IAP） | `mflow deploy` | `deploy/deploy.json` |
| LAMMPS 环境检查 | `mflow lmp-check` | `lmp_check/lmp_env.json` |
| LAMMPS MD（KOKKOS GPU） | `mflow md` | `<dir>/md_report.json` |
| MD 分析与闸门 | `mflow md-report` | 同上 |

计算过程中的**约定、默认和风险点**写在 [`docs/conventions.html`](docs/conventions.html) —— 尤其是能量对齐那一节，看数字之前先看它。

```bash
# 先装和 CUDA 匹配的 torch，再在仓库根目录：
pip install -e ".[mace]"        # 装好后命令行就有 mflow；也可以 python -m mflow ...
pip install -e ".[gpu,test]"    # 可选：cuEquivariance、pynvml、pytest
pytest                          # 22 个测试；有 mace-torch 会真训练 2 个 epoch，有 lmp 会真跑 LAMMPS
```

## 从数据到 MD：完整流程

```bash
mflow check -in dft.xyz -model mpa -split 0.1 -outdir check        # 体检 + 零样本 + 分层切分
mflow train -in check/dft_train.xyz -test check/dft_test.xyz -name cu -dry   # 先看 config.yaml
mflow train -in check/dft_train.xyz -test check/dft_test.xyz -name cu        # 训练 + 监控 + 报告
mflow deploy -model cu/cu.model -outdir deploy                      # ML-IAP 导出（GPU 上）
mflow lmp-check -deploy deploy                                      # LAMMPS 能不能跑它
mflow md -in start.xyz -deploy deploy -temp 600 -steps 100000 -outdir md600   # MD + 闸门
```

## 用法

```bash
mflow calc -in data.xyz                        # 默认模型，batch 32
mflow calc -in data.xyz -model mpa -batch 64   # 换成在线下载的基础模型
mflow calc -in data.xyz -model ./my.model -prefix mace_
mflow calc -in data.xyz -ref dft_              # 和 dft_energy/dft_forces 比较
mflow calc -in data.xyz -group config_type     # 按已有标签把误差拆开
mflow plot -in data_mace.xyz -ref dft_         # 只重画，不重算

mflow relax -in data.xyz                       # 弛豫：位置+晶胞，fmax 0.05
mflow relax -in data.xyz -cell none -fmax 0.02 # 固定晶胞，更严
mflow relax -in data.xyz -nproc 3 -dtype float32   # 3 个 worker 共享一张卡
mflow relax -in data.xyz -resume               # 接着上次中断的继续
```

参数一律多字母，不用单字母，避免撞车：

| 参数 | 默认 | 说明 |
|------|------|------|
| `-in` | 必填 | 输入 xyz / extxyz |
| `-model` | `/home/cmn01/software/mace-main032026/models/mace-mpa-0-medium.model` | `.model` 路径，或基础模型名（`small`/`medium`/`large`/`medium-mpa-0`/`medium-omat-0`，别名 `mpa`/`omat`）在线下载 |
| `-batch` | `32` | batch size |
| `-prefix` | `mpa0_` | 输出标签前缀 → `mpa0_energy` / `mpa0_forces` |
| `-ref` | 自动探测 | 参考标签前缀（`dft_` / `REF_` / 无前缀 `energy`） |
| `-e0` | `fit` | 能量对齐：`fit` 逐元素最小二乘 ／ `mean` 单一常数 ／ `none` ／ json 文件 |
| `-group` | 无 | 按已有标签分类误差：任意 info 键，或虚拟键 `formula` / `natoms`；数值键自动切四分位 |
| `-out` `-outdir` `-log` | `<stem>_mace.xyz` `.` `py.log` | 输出位置 |
| `-index` | `:` | ASE 切片，如 `:100` |
| `-device` `-dtype` | `auto` `float64` | cuda/cpu/mps；float64/float32 |
| `-head` | 无 | 多 head 模型指定 head |
| `-overwrite` | 关 | 前缀已存在时强制覆盖（默认自动顺延成 `mpa0_c1_`） |
| `-noplot` | 关 | 跳过画图 |

`relax` 专有：

| 参数 | 默认 | 说明 |
|------|------|------|
| `-fmax` | `0.05` | 收敛判据，eV/Å（ASE 定义：原子受力模长的最大值） |
| `-steps` | `500` | 每个结构最大优化步数 |
| `-opt` | `fire` | ASE 优化器：`fire` / `fire2` / `lbfgs` / `bfgs` |
| `-cell` | `full` | `full` 用 `FrechetCellFilter` 放开晶胞；`none` 只优化位置。非周期结构自动退回位置优化 |
| `-nproc` | `auto` | worker 进程数，每个自带一份模型。auto：GPU → 1，CPU → 核数/4 |
| `-traj` | 无 | 把所有中间帧写进一个 xyz |
| `-resume` | 关 | 从 `<stem>_relax.part.xyz` 接着跑 |

## 输出

| 文件 | 内容 |
|------|------|
| `<stem>_mace.xyz` | 结构 + `mpa0_energy`（info）/ `mpa0_forces`（arrays），原标签保留 |
| `<stem>_summary.png` | 一张图四格：能量 parity、力 parity、两个误差分布（无参考时退化为两格分布图） |
| `<stem>_metrics.json` | 参数、预测统计、误差、版本信息 |
| `py.log` | 终端输出的无色副本，时间戳只打在小节标题上 |

终端和日志同一份内容：`rich` 负责上色、表格、进度条；日志文件去掉颜色。

```
12:00:07 MFlow 0.3.0 · calc ───────────────────────────────────────
     input  data.xyz                       index  :
     model  mace-mpa-0-medium.model         kind  local file
batch size  32                            device  cuda
...
error  mpa0_ vs dft_
                          MAE       RMSE     max|Δ|        R²      N        unit
energy (E0 aligned)      7.30       8.70      18.69    0.9982     40    meV/atom
energy (raw)           116.03     147.65     354.00    0.8951     40    meV/atom
forces                  15.39      19.32      73.22    0.9975  1,302       meV/Å
E0 alignment (fit), eV per atom of element: O -0.2076 · Si +0.3542
```

## 能量对齐（E0）

DFT 和 MACE 的总能量不在同一个零点上，差的是**每种元素一个常数**：

```
E_pred(i) - E_ref(i) = Σ_j n_ij · δ_j + ε_i
```

不需要提供单原子能量 —— 代码用 `np.linalg.lstsq` 从数据集本身解出 δ，扣掉之后再算误差，
这就是默认的 `-e0 fit`。报告里 **`energy (E0 aligned)` 是要看的那一行**，`energy (raw)` 只用来看偏移有多大。
力不受 E0 影响，无需对齐。

⚠️ δ 是从被评估的数据里拟合的，会吸收模型真实的系统性偏差 —— 详见 `docs/conventions.html`。

## 分类（-group）

用数据集**已有的标签**把误差拆开，按 RMSE 从大到小排，图上多出一列排名条形图：

```
by config_type  (worst first)
group       N     E MAE    E RMSE    F RMSE
cluster    20    105.85    124.66    115.19
surface    20     29.48     40.78     30.90
bulk       20     11.93     13.85      4.10

worst 5 structures by |ΔE|
#       group    formula    atoms         ΔE    F RMSE
5     cluster      O3Si8       11    -236.91     89.19
20    cluster       O2Si        6    -212.26    125.90
```

KEY 可以是任意 `info` 键（`config_type` / `step` / `temperature_K`…）、虚拟键 `formula` / `natoms`，
或**上一轮自己写进文件的** `<prefix>dE` / `<prefix>dF` —— 即按误差大小分类。数值键超过 8 个取值时自动切四分位。

每次跑完还会把 `<prefix>dE`（meV/atom，有符号）和 `<prefix>dF`（该结构力 RMSE，meV/Å）写进输出 xyz，
方便你自己排序、筛选，或下一轮直接 `-group mpa0_dE`。

## 结构弛豫（relax）

优化器**全部用 ASE 现成的**，MFlow 只负责喂结构、收结果、记账。

```
relax  mpa0_
                   unit        min     median        max       mean
steps                      21.0000    28.5000    35.0000    27.6667
Erelax          eV/atom    -0.3434    -0.0701    -0.0351    -0.1138
final fmax         eV/Å     0.0108     0.0337     0.0503     0.0330
displacement          Å     0.0523     0.1583     0.3439     0.1761
12/12 converged · 0 hit the step limit · 0 failed · 0:00:42 · 0.29 struct/s · 1 worker(s)
```

写进输出 xyz 的标签（`<p>` = 前缀）：

| key | 含义 |
|-----|------|
| **`<p>steps`** | **优化步数** |
| `<p>converged` | 是否收敛（未收敛的结构照样写出，用这个筛） |
| `<p>energy` `<p>forces` | 弛豫后的能量与力 |
| `<p>energy0` `<p>Erelax` | 弛豫前能量；(E_末−E_初)/N，eV/atom |
| `<p>fmax` `<p>dmax` `<p>dvol` | 最终最大受力；原子最大位移 Å；体积变化 % |
| `<p>walltime` `<p>index` | 该结构耗时；在输入文件里的原始位置 |

### 运行时资源监控

进度条右侧实时显示整卡/整机用量，**跑的时候就能判断该开几个 worker**：

```
relaxing ━━━━━━━━━━╺━━━━  45/120 structures  0:01:23  eta 0:02:10 │ GPU  87% 6.2/24GB · CPU 340% · RAM 12/64GB
```

配色即结论：**GPU 利用率 ≥70% 绿 / 30–70% 黄 / <30% 红**（红 = GPU 闲着，该加 worker）；
显存和内存 ≥90% 红（该减 worker）；CPU 超过核数 95% 变黄（线程超订）。
终端窄时从右往左自动省略，优先保住 GPU。

结束时终端和 `py.log` 各一行汇总，同样的数字进 `metrics.json` 的 `resources` 段：

```
resources · GPU util mean 64% peak 92% · GPU mem peak 6.2/24.0 GB · CPU mean 340% of 1600% · RAM peak 12.1/64.0 GB
```

采样后端按可用性降级，缺哪个就少显示哪个，**任何后端出错都只是关掉该项，不影响计算**：

| 指标 | 首选 | 退路 | 再退 |
|------|------|------|------|
| GPU | `pynvml`（`pip install nvidia-ml-py`） | `nvidia-smi`（GPU 节点必有） | 无 |
| CPU / 内存 | `psutil` | `/proc/stat`、`/proc/meminfo` | `os.getloadavg()` |

采样 1 Hz，实测开关监控的耗时差异在运行噪声内。要完全关掉：`MFLOW_NO_MONITOR=1`。

⚠️ 这些数字是**整卡/整机**口径（含同卡上别人的任务）—— 调 `-nproc` 要看的正是这个，但它不等于"MFlow 自己用了多少"。

### 并行与内存

ASE 优化器逐结构串行，并行只能来自进程级：`multiprocessing` 的 **spawn**（fork 继承 CUDA context 会崩），
每个 worker 只加载一次模型。进子进程的只有几何，父进程的 `Atoms` 从不进子进程，**标签因此不受并行影响**。
实测 `-nproc 3` 与 `-nproc 1` 能量差 2×10⁻¹⁴ eV，步数完全一致，输出仍按输入顺序。

- 每个 worker ≈ 一份模型 + 一个 CUDA context（约 0.5 GB）+ 单结构的图 → **worker 数是唯一的显存旋钮**。
- `-dtype float32` 减半，弛豫场景性价比最高。
- 结果流式落盘到 `.part`，内存与结构数无关，被 kill 也不丢；`-resume` 接着跑，完成后自动删除。
- 结束时报告实测峰值 RSS 和 GPU 峰值显存，`-nproc` 据此调。

⚠️ **单张 GPU 上 ASE 路线跑不满** —— 一个小结构的 forward 填不满 GPU。能做的只有 `-nproc 2~4` 共享一张卡 +
`float32`。真正的 GPU 批量弛豫需要把优化器本身向量化（torch-sim 那类），本工具不做。

⚠️ `-cell full` 时 ASE 用 filter 的力（含应力）判收敛，写出的 `<p>fmax` 是原始原子力，两者可能差一点点 ——
**按 `<p>converged` 筛，不要按 `<p>fmax < 0.05` 筛**。

## 标签保留

**输入 xyz 里已有的标签全部保留**：`config_type`、`step`、自定义 per-atom 数组、
挂在 calculator 上的 `energy`/`forces`/`stress`，都原样写出。MACE 拿到的是只含几何的副本，碰不到你的标签。
只新增 `<prefix>energy`、`<prefix>forces`、`<prefix>dE`、`<prefix>dF` 四个 key。

**同名前缀不会被覆盖，而是自动顺延**：若 `mpa0_energy` 已存在，本次写成 `mpa0_c1_*`，
再来一次是 `mpa0_c2_*`，旧值原封不动。要强制沿用原前缀加 `-overwrite`。

## 说明

- 批量前向为主；一旦失败（版本漂移、显存不足）自动退回逐结构的 ASE 计算器，不用管。
- 一次性读进内存，超大数据集用 `-index` 分块。
- 画图点数 > 2 万自动 hexbin，> 50 万随机降采样并在日志里说明。
- 数据处理约定参考 [ptbplus](https://gitlab.com/mncui/ptbplus)。

## tools/

体系专属的小工具放在 [`tools/`](tools/)，共用 `mflow` 包的读写、日志和绘图风格。
目前有 `izo_formation.py`（In–Zn–O 形成能 vs In 比例），用法见 [tools/README.md](tools/README.md)。

```bash
python tools/izo_formation.py -in izo_mace.xyz -eZnO zno_mace.xyz -eIn2O3 in2o3_mace.xyz
```

## 训练：check / train / train-report

**`mflow check`** —— 训练前的体检。每一项都是 pass / warn / fail，加一句建议：

| 检查 | 判据 |
|---|---|
| 标签 | 每帧都有能量和力；stress 要么全有要么全没有 |
| 数值 | NaN/inf → fail；单个力分量 > 50 eV/Å warn，> 200 fail |
| 几何 | 最近原子对 < 0.6 Å → fail；周期结构晶胞体积为 0 → fail |
| 重复 | 几何指纹相同的帧（会在 train/valid/test 之间泄漏） |
| 能量离群 | 逐元素线性拟合（剔除粗大离群后重拟合）的残差，> 6 MAD |
| E0 | 是否每个元素都有 IsolatedAtom；没有就用 `--E0s estimated` |
| 零样本（`-model`） | 基础模型是否覆盖所有元素、误差多大、原始能量偏移 > 0.4 eV/atom 提示 DFT 设置不一致 |
| 规模 / 分组 | 总帧数、每个 `config_type` 的帧数 |

`-split 0.1` 按分组分层切出测试集，IsolatedAtom 永远留在训练集。

**`mflow train`** —— 同样的体检作为门槛（fail 就不训练，`-force` 跳过）→ 数据复制到 `<run>/data/` 并把标签统一成
`REF_energy/REF_forces/REF_stress` → 按模式生成 `config.yaml` → 用 `mace_run_train --help` 校验每个键 → 运行并实时跟踪
`results/*_train.txt`（NaN 立即停、验证 loss 连升 5 次告警、显存 > 90% 告警）→ 自动出报告。

| `-mode` | lr | EMA | clip | 说明 |
|---|---|---|---|---|
| `finetune`（默认） | 1e-3 | 0.999 | 1.0 | 单一体系最好 |
| `multihead` | 1e-4 | 0.9999 | 1.0 | 回放预训练数据（`-pt`，默认 `mp`），保泛化 |
| `lora` | 1e-2 | 0.99 | 10 | 低秩适配 |
| `scratch` | 1e-2 | 0.99 | 10 | 从头训练，最后 25% 进入 stage two |

数值来自 MACE 官方微调指南；指南没给的（epoch、batch、stress 权重、r_max）用保守默认，通过
`-set key=value` 覆盖（键名不合法会直接 fail）。`-foundation` 可以是 `mpa` / `omat` / `mh1` / `matpes` 或一个 `.model`。

**`mflow train-report -dir <run>`** —— 学习曲线（标出最佳 epoch）、过拟合（最终 / 最佳验证 loss > 1.5）、
训练前后验证误差改善倍数、测试集走和 `mflow calc` **完全相同**的误差定义（按组拆分、最差 5 个结构），
并按 能量 < 5 / 10 meV/atom、力 < 20 / 50 meV/Å 分成 excellent / good / needs work。

**委员会推理**：`mflow calc -model a.model,b.model,c.model` 写出平均能量/力，以及
`<prefix>Fdev`（逐原子力偏差，DP-GEN 定义）、`<prefix>Fdev_max`、`<prefix>Edev` —— 主动学习选帧用。

## LAMMPS 动力学：deploy / lmp-check / md / md-report

MACE 通过 **ML-IAP unified + KOKKOS** 在 GPU 上跑，积分、恒温、恒压、dump、restart **全部由 LAMMPS 完成**。
需要的 LAMMPS 编译选项（MACE 文档）：`PKG_KOKKOS + Kokkos_ENABLE_CUDA`、`PKG_ML-IAP`、`PKG_ML-SNAP`、`PKG_PYTHON`、
`MLIAP_ENABLE_PYTHON`、`BUILD_SHARED_LIBS`；LAMMPS 内嵌的 Python 里要有 mace-torch、cuequivariance-torch、cupy。

1. **`mflow deploy -model x.model [-head H] [-dtype float32|float64]`** —— 调 `mace_create_lammps_model --format mliap`
   （要在跑 LAMMPS 的同型号 GPU 上做），写 `deploy.json`：源模型和导出文件的 sha256、head、**元素顺序**、r_max、精度、版本。
   之后所有命令只认 `deploy.json`。多 head 模型必须指定 `-head`。
2. **`mflow lmp-check [-lmp PATH] -deploy deploy`** —— 读 `lmp -h` 确认 KOKKOS(CUDA) / ML-IAP / PYTHON；在 LAMMPS 内嵌 Python
   里 import mace / torch / cuequivariance / cupy 并确认能看到 GPU；用导出模型实跑一次 2 原子的 `run 0`。
   lmp 的位置：`-lmp` > 环境变量 `MFLOW_LMP` > PATH。
3. **`mflow md -in start.xyz -deploy deploy -ensemble nvt|npt|nve -temp T -steps N`**
   - 输入来自一个固定模板（units metal，`velocity create`，`fix nvt/npt/nve`，`fix halt` 温度失控保护，unwrapped dump，
     两个 restart 文件轮换，`run N upto`），参数检查：步长、阻尼、每卡原子数、晶胞是否小于 2·r_max。
   - `lmp -sr` 先把真实输入完整解析、初始化一遍（不积分）。
   - **一致性检查**：对起始结构 `run 0`，LAMMPS 的能量、逐原子力、压强必须和同一模型在 LAMMPS 外的计算一致
     （fp64 导出：1e-5 eV/atom、1e-4 eV/Å；fp32：1e-3、2e-2）。元素顺序错位、单位、维里符号、导错 head 都会在这里被拦下。
   - 正式运行时实时显示步数、温度、ns/day 和 GPU 占用；`-resume` 从最新 restart 接着跑（`-steps` 是总步数）。
4. **`mflow md-report -dir md600`** —— 闸门 + 分析 + 一张 2×3 的图：

| 闸门 | 判据 |
|---|---|
| LAMMPS ERROR / 丢原子 / fix halt | 出现即 fail |
| Dangerous builds | > 0 warn |
| NVE 能量漂移 | 生产段线性拟合斜率 < 5×10⁻⁴ eV/atom/ps pass，< 5×10⁻³ warn（LAMBench 标准） |
| 温度（NVT/NPT） | 生产段均值与目标差 < 5 % pass，< 15 % warn；生产段线性趋势 < 5 % |
| 体积（NPT） | 生产段体积变化 < 2 % pass，< 5 % warn |
| 最近原子对 | 任何一帧 < 0.6 Å → fail |

分析：热力学量均值/涨落、分元素 RDF（`-ref aimd.xyz` 时给出与参考的 RDF 差）、多时间原点 MSD 与扩散系数。
`-skip` 是平衡段比例（默认 0.2），不参与闸门。

## Roadmap

主动学习闭环（委员会对 LAMMPS dump 重新打分 → 信任窗口 + FPS 选帧 → VASP 标注 → `check` → 再训练）·
VASP 标注（pymatgen 输入集 + custodian + dpdispatcher）· Optuna 调参 + 评分卡 · symmetrix 后端 · torch-sim 批量弛豫 · 声子/EOS 验证
