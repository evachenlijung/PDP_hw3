python3 << 'EOF'
import os
import csv
import warnings
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import numpy as np
from collections import defaultdict

warnings.filterwarnings("ignore")

serial_csv = os.path.expanduser("~/hw3/eva/logs/q1_serial/q1_serial.csv")
q2_csv     = os.path.expanduser("~/hw3/eva/logs/q2/q2_strong_scaling.csv")
logs_dir   = os.path.expanduser("~/hw3/eva/logs/q4")

testcase_order = [
    "imbalance_c100000",
    "medium_c200000",
    "large_c1000000",
    "large_c2000000",
    "large_c4000000",
]
NPERNODE_LIST = [1, 2, 4, 8, 12, 16]

serial_time = {}
with open(serial_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        serial_time[row["testcase"]] = float(row["total_render_time"])

par_data = defaultdict(dict)
with open(q2_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        name    = row["testcase"]
        np_node = int(row["npernode"])
        try:
            par_data[name][np_node] = float(row["total_runtime"]) if row["total_runtime"] else 0.0
        except ValueError:
            continue

names = [n for n in testcase_order if n in par_data and n in serial_time and
         all(np_node in par_data[n] for np_node in NPERNODE_LIST)]

tc_colors = {
    n: mcolors.to_hex(c)
    for n, c in zip(testcase_order,
                    plt.cm.coolwarm(np.linspace(0.05, 0.95, len(testcase_order))))
}
tc_colors["large_c1000000"] = "#4d4d4d"

x = np.array(NPERNODE_LIST, dtype=float)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
fig.suptitle("Speedup and Efficiency vs Processes per Node (relative to serial baseline)", fontsize=14)

def plot_panel(ax, metric, ylabel, title, ideal_label, ideal_val_fn, val_fmt):
    all_vals = []
    for n in names:
        for np_node in NPERNODE_LIST:
            t_par  = par_data[n][np_node]
            t_ser  = serial_time[n]
            nprocs = np_node * 4
            if t_par > 0:
                all_vals.append(metric(t_ser, t_par, nprocs))

    max_val = max(all_vals) if all_vals else 1.0
    min_val = min(all_vals) if all_vals else 0.01

    for n in names:
        color = tc_colors[n]
        t_ser = serial_time[n]
        vals  = [metric(t_ser, par_data[n][np_node], np_node*4)
                 if par_data[n][np_node] > 0 else 0
                 for np_node in NPERNODE_LIST]
        ax.plot(x, vals, marker="o", color=color,
                linewidth=2, markersize=7, label=n, zorder=4)

    if ideal_val_fn is not None:
        ideal_vals = [ideal_val_fn(np_node * 4) for np_node in NPERNODE_LIST]
        ax.plot(x, ideal_vals, linestyle="--", color="gray",
                linewidth=1.5, alpha=0.6, label=ideal_label, zorder=3)

    ax.set_yscale('log')
    ax.set_ylim(min_val * 0.5, max_val * 50.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))

    log_max = np.log10(max_val * 50.0)
    log_min = np.log10(min_val * 0.5)

    for xi, np_node in enumerate(NPERNODE_LIST):
        col_points = []
        for n in names:
            t_par  = par_data[n][np_node]
            t_ser  = serial_time[n]
            nprocs = np_node * 4
            val    = metric(t_ser, t_par, nprocs) if t_par > 0 else 0
            color  = tc_colors[n]
            col_points.append((val, t_par, color))

        col_points.sort(key=lambda p: p[0], reverse=True)

        label_log_top    = log_max - (log_max - log_min) * 0.02
        label_log_bottom = log_max - (log_max - log_min) * 0.40
        step = (label_log_top - label_log_bottom) / (len(col_points) - 1) if len(col_points) > 1 else 0

        x_offset = 0.4 if xi == 1 else 0.0

        for rank_idx, (val, t_par, color) in enumerate(col_points):
            y_pos = 10 ** (label_log_top - step * rank_idx)
            ax.annotate(
                f"{t_par:.6f}s\n{val_fmt.format(val)}",
                xy=(x[xi], val),
                xytext=(x[xi] + x_offset, y_pos),
                ha='center', va='top', fontsize=8,
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
    ax.set_xlabel("Processes per Node\n(total processes = npernode x 4 nodes)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(axis='y', which='both', linestyle='--', alpha=0.4, zorder=0)

# speedup panel
plot_panel(
    ax1,
    metric       = lambda t_ser, t_par, nprocs: t_ser / t_par,
    ylabel       = "Speedup (vs serial baseline)",
    title        = "Speedup",
    ideal_label  = "Ideal Speedup (= nprocs, efficiency = 100%)",
    ideal_val_fn = lambda nprocs: nprocs,
    val_fmt      = "{:.2f}x",
)

# # serial baselines for speedup panel
# serial_csv_path = os.path.expanduser("~/hw3/eva/logs/q1_serial/q1_serial.csv")
# serial_time_local = {}
# with open(serial_csv_path) as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         serial_time_local[row["testcase"]] = float(row["total_render_time"])

# for n in names:
#     if n not in serial_time_local:
#         continue
#     color = tc_colors[n]
#     rgb   = mcolors.to_rgb(color)
#     light = tuple(c * 0.7 + 0.3 for c in rgb)
#     t_ser = serial_time_local[n]
#     ax1.axhline(
#         y=t_ser,
#         color=mcolors.to_hex(light),
#         linestyle="-",
#         linewidth=1.5,
#         alpha=0.95,
#         zorder=2,
#         label=f"{n} serial ({t_ser:.2f}s)"
#     )

# efficiency panel
plot_panel(
    ax2,
    metric       = lambda t_ser, t_par, nprocs: (t_ser / t_par) / nprocs,
    ylabel       = "Efficiency (Speedup / nprocs)",
    title        = "Efficiency",
    ideal_label  = "Ideal Efficiency (= 1.0)",
    ideal_val_fn = lambda nprocs: 1.0,
    val_fmt      = "{:.2%}",
)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, fontsize=9,
           loc="center left", bbox_to_anchor=(1.01, 0.5),
           framealpha=0.9, borderaxespad=0)

plt.tight_layout()
outpath = os.path.join(logs_dir, "q4_speedup_efficiency.png")
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Plot saved to {outpath}")
EOF