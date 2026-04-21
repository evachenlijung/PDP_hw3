#!/bin/bash

python3 << 'EOF'
import csv
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import numpy as np
from collections import defaultdict

warnings.filterwarnings("ignore")

# --- load CSVs ---
def load_csv(path):
    data = defaultdict(dict)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name    = row["testcase"]
            np_node = int(row["npernode"])
            try:
                scatter   = float(row["scatter_time"])   if row.get("scatter_time")   else 0.0
                render    = float(row["avg_render_time"]) if row.get("avg_render_time") else 0.0
                composite = float(row["composite_time"]) if row.get("composite_time") else 0.0
                total     = scatter + render + composite
                data[name][np_node] = {
                    "scatter":   scatter,
                    "render":    render,
                    "composite": composite,
                    "total":     total,
                }
            except ValueError:
                continue
    return data

old_csv = os.path.expanduser("~/hw3/eva/logs/q1_mpi/q1_mpi.csv")
new_csv = os.path.expanduser("~/hw3/eva/logs/q1_mpi_new/q1_mpi_new.csv")

old_data = load_csv(old_csv)
new_data = load_csv(new_csv)

testcase_order = [
    "imbalance_c100000",
    "medium_c200000",
    "large_c1000000",
    "large_c2000000",
    "large_c4000000",
]
NPERNODE_LIST = [1, 2, 4, 8, 12, 16]

names = [n for n in testcase_order
         if n in old_data and n in new_data
         and all(np_node in old_data[n] for np_node in NPERNODE_LIST)
         and all(np_node in new_data[n] for np_node in NPERNODE_LIST)]

tc_colors = {
    n: mcolors.to_hex(c)
    for n, c in zip(testcase_order,
                    plt.cm.coolwarm(np.linspace(0.05, 0.95, len(testcase_order))))
}
tc_colors["large_c1000000"] = "#4d4d4d"

x = np.array(NPERNODE_LIST, dtype=float)

# --- print comparison table ---
print(f"{'testcase':<25} {'npernode':>8} {'nprocs':>6} "
      f"{'old_total':>12} {'new_total':>12} {'diff':>10} {'improvement':>12}")
print("-" * 90)

for n in names:
    for np_node in NPERNODE_LIST:
        nprocs   = np_node * 4
        old_t    = old_data[n][np_node]["total"]
        new_t    = new_data[n][np_node]["total"]
        diff     = old_t - new_t
        improve  = (diff / old_t * 100) if old_t > 0 else 0
        marker   = "✓" if diff > 0 else ("=" if diff == 0 else "✗")
        print(f"{n:<25} {np_node:>8} {nprocs:>6} "
              f"{old_t:>12.6f} {new_t:>12.6f} {diff:>+10.6f} {improve:>11.2f}% {marker}")
    print()

# --- save comparison to CSV ---
compare_csv = os.path.join(os.path.expanduser("~/hw3/eva/logs"), "q1_mpi_compare.csv")
with open(compare_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["testcase", "npernode", "nprocs", "old_total", "new_total", "diff", "improvement_pct", "result"])
    for n in names:
        for np_node in NPERNODE_LIST:
            nprocs  = np_node * 4
            old_t   = old_data[n][np_node]["total"]
            new_t   = new_data[n][np_node]["total"]
            diff    = old_t - new_t
            improve = (diff / old_t * 100) if old_t > 0 else 0
            marker  = "win" if diff > 0 else ("tie" if diff == 0 else "loss")
            writer.writerow([n, np_node, nprocs,
                             f"{old_t:.6f}", f"{new_t:.6f}",
                             f"{diff:+.6f}", f"{improve:.2f}", marker])
print(f"Comparison CSV saved to {compare_csv}")

# --- figure: 3 subplots (scatter, render, composite) ---
fig, axes = plt.subplots(1, 3, figsize=(22, 8))
fig.suptitle("malloc vs MPI_Alloc_mem: Timing Comparison", fontsize=14)

time_keys   = ["scatter",               "render",              "composite"]
time_labels = ["Read+Scatter Time (s)", "Avg Render Time (s)", "Composite+Write Time (s)"]

for ax, tkey, tlabel in zip(axes, time_keys, time_labels):
    all_vals = []
    for n in names:
        for np_node in NPERNODE_LIST:
            v1 = old_data[n][np_node][tkey]
            v2 = new_data[n][np_node][tkey]
            if v1 > 0: all_vals.append(v1)
            if v2 > 0: all_vals.append(v2)

    max_val = max(all_vals) if all_vals else 1.0
    min_val = min(all_vals) if all_vals else 0.001

    for n in names:
        color = tc_colors[n]
        rgb   = mcolors.to_rgb(color)
        light = tuple(c * 0.5 + 0.5 for c in rgb)

        old_vals = [old_data[n][np_node][tkey] for np_node in NPERNODE_LIST]
        new_vals = [new_data[n][np_node][tkey] for np_node in NPERNODE_LIST]

        # old: dashed lighter color
        ax.plot(x, old_vals, marker="o", color=mcolors.to_hex(light),
                linewidth=1.5, markersize=5, linestyle="--",
                label=f"{n} (malloc)" if n == names[0] else "_nolegend_", zorder=3)
        # new: solid full color
        ax.plot(x, new_vals, marker="D", color=color,
                linewidth=2, markersize=7, linestyle="-",
                label=f"{n} (MPI_Alloc_mem)" if n == names[0] else "_nolegend_", zorder=4)

    ax.set_yscale('log')
    ax.set_ylim(min_val * 0.4, max_val * 50.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))

    log_max = np.log10(max_val * 50.0)
    log_min = np.log10(min_val * 0.4)

    for xi, np_node in enumerate(NPERNODE_LIST):
        col_points = []
        for n in names:
            old_t = old_data[n][np_node][tkey]
            new_t = new_data[n][np_node][tkey]
            diff  = old_t - new_t
            pct   = (diff / old_t * 100) if old_t > 0 else 0
            color = tc_colors[n]
            col_points.append((new_t, old_t, pct, color))

        col_points.sort(key=lambda p: p[0], reverse=True)

        label_log_top    = log_max - (log_max - log_min) * 0.02
        label_log_bottom = log_max - (log_max - log_min) * 0.40
        step = (label_log_top - label_log_bottom) / (len(col_points) - 1) if len(col_points) > 1 else 0

        x_offset = 0.4 if xi == 1 else 0.0

        for rank_idx, (new_t, old_t, pct, color) in enumerate(col_points):
            y_pos  = 10 ** (label_log_top - step * rank_idx)
            marker = "▼" if pct > 0 else ("▲" if pct < 0 else "=")
            ax.annotate(
                f"{new_t:.4f}s\n{pct:+.1f}% {marker}",
                xy=(x[xi], new_t),
                xytext=(x[xi] + x_offset, y_pos),
                ha='center', va='top', fontsize=7,
                color=color, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                arrowprops=dict(arrowstyle="-", color=color, lw=0.4, alpha=0.5),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{int(n)}\n(total {int(n)*4})" for n in NPERNODE_LIST],
        fontsize=9
    )
    ax.set_xlim(0, max(NPERNODE_LIST) * 1.1)
    ax.set_xlabel("Processes per Node\n(total = npernode x 4 nodes)", fontsize=10)
    ax.set_ylabel("Time (s)", fontsize=11)
    ax.set_title(tlabel, fontsize=12)
    ax.grid(axis='y', which='both', linestyle='--', alpha=0.4, zorder=0)

# legend: one entry per style, not per testcase
from matplotlib.lines import Line2D
legend_handles = []
for n in names:
    color = tc_colors[n]
    rgb   = mcolors.to_rgb(color)
    light = tuple(c * 0.5 + 0.5 for c in rgb)
    legend_handles.append(
        Line2D([0], [0], color=mcolors.to_hex(light), linewidth=1.5,
               linestyle="--", marker="o", markersize=5, label=f"{n} malloc"))
    legend_handles.append(
        Line2D([0], [0], color=color, linewidth=2,
               linestyle="-", marker="D", markersize=7, label=f"{n} MPI_Alloc_mem"))

fig.legend(handles=legend_handles, fontsize=8,
           loc="center left", bbox_to_anchor=(1.01, 0.5),
           framealpha=0.9, borderaxespad=0)

plt.tight_layout()
logs_dir = os.path.expanduser("~/hw3/eva/logs")
outpath  = os.path.join(logs_dir, "q1_mpi_compare.png")
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {outpath}")
EOF