# MFlow

单文件机器学习势工具箱（v0.4.0，`mflow.py`，约 800 行）。当前功能：静态 MACE 计算能量和力，按标签分类误差。

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
| `-noplot` | 关 | 跳过画图 |

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

## 标签保留

**输入 xyz 里已有的标签全部保留**：`config_type`、`step`、自定义 per-atom 数组、
挂在 calculator 上的 `energy`/`forces`/`stress`，都原样写出。MACE 拿到的是只含几何的副本，碰不到你的标签。
只新增 `<prefix>energy`、`<prefix>forces`、`<prefix>dE`、`<prefix>dF` 四个 key。

⚠️ 用同一个 `-prefix` 重跑会覆盖上一轮的这四个 key；比较两个模型请用不同前缀。

## 说明

- 批量前向为主；一旦失败（版本漂移、显存不足）自动退回逐结构的 ASE 计算器，不用管。
- 一次性读进内存，超大数据集用 `-index` 分块。
- 画图点数 > 2 万自动 hexbin，> 50 万随机降采样并在日志里说明。
- 数据处理约定参考 [ptbplus](https://gitlab.com/mncui/ptbplus)。

## Roadmap

数据集工具（merge/split/delta/filter）· 训练与迁移学习封装 · 收敛曲线 · MD / 结构优化
