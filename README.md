# MFlow

单文件机器学习势工具箱（v0.6.0，`mflow.py`，约 1360 行）。功能：静态 MACE 计算、结构弛豫、按标签分类误差，运行时资源监控。

计算过程中的**约定、默认和风险点**写在 [`docs/conventions.html`](docs/conventions.html) —— 尤其是能量对齐那一节，看数字之前先看它。

```bash
pip install mace-torch ase numpy matplotlib rich
```

## 用法

```bash
python mflow.py calc -in data.xyz                        # 默认模型，batch 32
python mflow.py calc -in data.xyz -model mpa -batch 64   # 换成在线下载的基础模型
python mflow.py calc -in data.xyz -model ./my.model -prefix mace_
python mflow.py calc -in data.xyz -ref dft_              # 和 dft_energy/dft_forces 比较
python mflow.py calc -in data.xyz -group config_type     # 按已有标签把误差拆开
python mflow.py plot -in data_mace.xyz -ref dft_         # 只重画，不重算

python mflow.py relax -in data.xyz                       # 弛豫：位置+晶胞，fmax 0.05
python mflow.py relax -in data.xyz -cell none -fmax 0.02 # 固定晶胞，更严
python mflow.py relax -in data.xyz -nproc 3 -dtype float32   # 3 个 worker 共享一张卡
python mflow.py relax -in data.xyz -resume               # 接着上次中断的继续
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

体系专属的小工具放在 [`tools/`](tools/)，共用 `mflow.py` 的读写、日志和绘图风格。
目前有 `izo_formation.py`（In–Zn–O 形成能 vs In 比例），用法见 [tools/README.md](tools/README.md)。

```bash
python tools/izo_formation.py -in izo_mace.xyz -eZnO zno_mace.xyz -eIn2O3 in2o3_mace.xyz
```

## Roadmap

数据集工具（merge/split/delta/filter）· 训练与迁移学习封装 · 收敛曲线 · MD / 结构优化
