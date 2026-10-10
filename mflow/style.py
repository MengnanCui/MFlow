# -*- coding: utf-8 -*-
"""Plot style (publication) and the colour/marker cycle shared by every figure."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- plot style (publication) -------------------------------------------------
COLORS = ["#2470a0", "#ca3e47", "#f29c2b", "#1f640a", "#2ca02c", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#005555"]
MARKERS, LINESTYLES = ["o", "v", "s", "*", "p", "P"], ["-", "--", "-.", ":"]
plt.rcParams.update({
    "font.size": 15, "axes.linewidth": 2, "axes.labelsize": 15, "axes.titlesize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 2, "ytick.major.width": 2, "xtick.minor.width": 1.5, "ytick.minor.width": 1.5,
    "xtick.major.size": 6, "ytick.major.size": 6, "xtick.minor.size": 3, "ytick.minor.size": 3,
    "lines.linewidth": 2, "lines.markersize": 13, "lines.markeredgewidth": 2,
    "legend.fontsize": 10, "legend.markerscale": 1.5, "legend.framealpha": 0.5,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

HEXBIN_ABOVE, PLOT_MAX_POINTS = 20_000, 500_000
