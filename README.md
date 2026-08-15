# MFlow

**单文件**的机器学习势工具箱（v0.1.0）。所有代码都在 `mflow.py` 里，方便直接复制到集群上使用。

当前版本 = **静态 MACE 计算**：输入一个 xyz 数据集 → 用 MACE 算能量和力 → 输出带标签的 xyz + 日志 + 误差 + 可视化。

---

## 1. 安装

```bash
pip install mace-torch ase numpy matplotlib
```

只做画图/分析（`plot` 子命令）时不需要 `mace-torch` 和 `torch`。

## 2. 快速开始

```bash
# 默认：mace-mp-0 medium，batch size 32，输出标签 mpa0_energy / mpa0_forces
python mflow.py calc -i data.xyz

# 换模型 + 换 batch size
python mflow.py calc -i data.xyz -m medium-mpa-0 -b 64

# 用自己的模型文件（微调/迁移学习得到的 .model）
python mflow.py calc -i data.xyz -m ./MACE_model_swa.model

# 和文件里已有的 DFT 标签比较（dft_energy / dft_forces）
python mflow.py calc -i data.xyz --ref-prefix dft_

# 只重新分析/画图，不重新计算
python mflow.py plot -i data_mace.xyz --pred-prefix mpa0_ --ref-prefix dft_
```

## 3. 输出文件

| 文件 | 内容 |
|------|------|
| `<stem>_mace.xyz` | 全部结构 + `mpa0_energy`（info）/ `mpa0_forces`（arrays），原有标签保留 |
| `py.log` | 详细日志：参数、环境、每个 batch 的进度/耗时/ETA、RMSE、traceback |
| `<stem>_metrics.json` | 机器可读的汇总：数据集统计、预测统计、误差、运行环境 |
| `<stem>_energy_parity.png` | 能量 parity 图（eV/atom，各自减去均值），标注 RMSE / R² |
| `<stem>_forces_parity.png` | 力分量 parity 图（eV/Å），标注 RMSE / R² |
| `<stem>_error_hist.png` | 能量误差（meV/atom）与力误差（meV/Å）分布 |
| `<stem>_distribution.png` | 没有参考标签时：预测的 E/atom 与 max&#124;F&#124; 分布 |

日志分两路：**终端**只打印关键参数和结果，**日志文件**记录全部细节（含时间戳、函数名、DEBUG 信息）。

## 4. 主要参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `-i, --input` | 必填 | 输入 xyz / extxyz |
| `--index` | `:` | ASE 切片，例如 `:100` 只算前 100 个结构 |
| `-m, --model` | `medium` | 基础模型名（`small` / `medium` / `large` / `medium-mpa-0` / `medium-omat-0` …）**或**本地 `.model` 路径。别名：`mpa`→`medium-mpa-0`，`omat`→`medium-omat-0` |
| `-b, --batch-size` | `32` | batch size |
| `-p, --prefix` | `mpa0_` | 输出标签前缀 → `mpa0_energy` / `mpa0_forces` |
| `--ref-prefix` | 自动探测 | 参考标签前缀（`dft_`、`REF_`、或不带前缀的 `energy`/`forces`） |
| `--engine` | `auto` | `batch`（批量 torch 循环，快）/ `ase`（逐结构，稳）/ `auto`（先 batch，失败自动退回 ase） |
| `--device` | `auto` | `cuda` / `cpu` / `mps` |
| `--dtype` | `float64` | `float64` / `float32` |
| `--head` | 无 | 多 head 模型指定 head |
| `--outdir` | `.` | xyz / json / png 输出目录 |
| `--log` | `py.log` | 日志文件；`--log-append` 追加而非覆盖 |
| `--no-plot` | 关 | 跳过画图 |

完整帮助：`python mflow.py calc --help`

## 5. 误差怎么读

不同方法（DFT / MACE）的原子参考能不同，总能量会差一个常数偏移，因此同时输出：

- `RMSE_meV_per_atom` —— 直接的每原子误差（含偏移）
- `RMSE_shifted_meV_per_atom` —— **各自减去均值之后**的误差，这才是描述"形状"是否一致的量
- `mean_offset_meV_per_atom` —— 两者的常数偏移

力是逐分量比较（eV/Å），另外给出 R² 和最大绝对误差。

## 6. 说明

- 输入用 `ase.io.read(index=":")` 一次性读入内存；超大数据集建议先用 `--index` 分块跑。
- 画图点数 > 20,000 自动切换为 hexbin；> 500,000 会随机降采样（日志里会说明降了多少）。
- 绘图风格：publication style（tick 朝内、`axes.linewidth=2`、300 dpi）。

## 7. 参考

思路和数据处理约定参考 [ptbplus](https://gitlab.com/mncui/ptbplus)（`Mtools.py` / `multi_fidelity.py`）。

## 8. Roadmap

- [ ] 数据集工具：merge / split / delta / filter / 单位换算
- [ ] MACE 训练与迁移学习封装
- [ ] 训练收敛曲线可视化
- [ ] 分子动力学 / 结构优化
