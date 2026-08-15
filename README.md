# MFlow

单文件机器学习势工具箱（v0.2.0，`mflow.py`，约 600 行）。当前功能：静态 MACE 计算能量和力。

```bash
pip install mace-torch ase numpy matplotlib rich
```

## 用法

```bash
python mflow.py calc -in data.xyz                        # 默认模型，batch 32
python mflow.py calc -in data.xyz -model mpa -batch 64   # 换成在线下载的基础模型
python mflow.py calc -in data.xyz -model ./my.model -prefix mace_
python mflow.py calc -in data.xyz -ref dft_              # 和 dft_energy/dft_forces 比较
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
12:00:07 MFlow 0.2.0 · calc ───────────────────────────────────────
     input  small.xyz                      index  :
     model  small                           kind  foundation
batch size  4                             device  cpu
...
error  mpa0_ vs dft_
                       MAE       RMSE     max|Δ|        R²      N        unit
energy               97.24     108.60     189.39    0.9861     12    meV/atom
energy (shifted)     41.72      52.32     107.65    0.9861     12    meV/atom
forces              971.37    1368.22    6783.39    0.0000    228       meV/Å
constant offset +95.2 meV/atom — the shifted row is the one that describes the shape
```

**energy vs energy (shifted)**：两种方法的原子参考能不同，总能量差一个常数；`shifted` 是两边各自减均值后的误差，描述"形状"是否一致，通常看这一行。

## 说明

- 批量前向为主；一旦失败（版本漂移、显存不足）自动退回逐结构的 ASE 计算器，不用管。
- 一次性读进内存，超大数据集用 `-index` 分块。
- 画图点数 > 2 万自动 hexbin，> 50 万随机降采样并在日志里说明。
- 数据处理约定参考 [ptbplus](https://gitlab.com/mncui/ptbplus)。

## Roadmap

数据集工具（merge/split/delta/filter）· 训练与迁移学习封装 · 收敛曲线 · MD / 结构优化
