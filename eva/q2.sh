# #!/bin/bash

# mpicc renderer_mpi.c -o renderer_mpi -lm -march=native
# for i in 1 2 3; do scp renderer_mpi rdma$i:~/hw3/eva/; done

# mkdir -p ~/hw3/eva/output/q2
# mkdir -p ~/hw3/eva/logs/q2

# Q2_CSV=~/hw3/eva/logs/q2/q2_strong_scaling.csv
# echo "testcase,npernode,nprocs,total_runtime" > $Q2_CSV

# TESTCASES=(
#     "../testcases/imbalance_c100000.bin"
#     "../testcases/medium_c200000.bin"
#     "../testcases/large_c1000000.bin"
#     "../testcases/large_c2000000.bin"
#     "../testcases/large_c4000000.bin"
# )

# NPERNODE_LIST=(1 2 4 8 12 16)

# for bin in "${TESTCASES[@]}"; do
#     name=$(basename $bin .bin)

#     for npernode in "${NPERNODE_LIST[@]}"; do
#         nprocs=$((npernode * 4))
#         outpng=~/hw3/eva/output/q2/${name}_npernode${npernode}.png
#         logfile=~/hw3/eva/logs/q2/${name}_npernode${npernode}.log

#         echo "=== Running: $name npernode=$npernode (total $nprocs procs) ==="

#         UCX_TLS=rc,sm,self \
#         UCX_NET_DEVICES=rocep23s0:1 \
#         mpirun \
#           --hostfile hosts \
#           -npernode $npernode \
#           --mca pml ucx \
#           --mca btl ^tcp \
#           ./renderer_mpi $bin $outpng \
#           2>&1 | tee $logfile

#         total=$(grep "Total runtime" $logfile | awk '{print $(NF-1)}')

#         echo "$name,$npernode,$nprocs,$total" >> $Q2_CSV
#         echo "  total_runtime=$total"
#         echo ""
#     done
# done

# echo "=== Done. Results written to $Q2_CSV ==="

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

logs_dir = os.path.expanduser("~/hw3/eva/logs/q2")
csv_path = os.path.join(logs_dir, "q2_strong_scaling.csv")

testcase_order = [
    "imbalance_c100000",
    "medium_c200000",
    "large_c1000000",
    "large_c2000000",
    "large_c4000000",
]
NPERNODE_LIST = [1, 2, 4, 8, 12, 16]

data = defaultdict(dict)
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        name    = row["testcase"]
        np_node = int(row["npernode"])
        try:
            data[name][np_node] = float(row["total_runtime"]) if row["total_runtime"] else 0.0
        except ValueError:
            continue

names = [n for n in testcase_order if n in data and
         all(np_node in data[n] for np_node in NPERNODE_LIST)]

tc_colors = {
    n: mcolors.to_hex(c)
    for n, c in zip(testcase_order,
                    plt.cm.coolwarm(np.linspace(0.05, 0.95, len(testcase_order))))
}
tc_colors["large_c1000000"] = "#4d4d4d"

x = np.array(NPERNODE_LIST, dtype=float)

all_times = [data[n][np_node] for n in names for np_node in NPERNODE_LIST if data[n][np_node] > 0]
max_time  = max(all_times) if all_times else 1.0
min_time  = min(all_times) if all_times else 0.1

fig, ax = plt.subplots(figsize=(12, 7))
fig.suptitle("Strong Scaling — Total Runtime vs Processes per Node", fontsize=14)

for n in names:
    color = tc_colors[n]
    times = [data[n][np_node] for np_node in NPERNODE_LIST]
    ax.plot(x, times, marker="o", color=color,
            linewidth=2, markersize=7, label=n, zorder=4)

# log scale
ax.set_yscale('log')
ax.set_ylim(min_time * 0.4, max_time * 50.0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))

ax.set_xticks(x)
ax.set_xticklabels(
    [f"{int(n)}\n(total {int(n)*4})" for n in NPERNODE_LIST],
    fontsize=9
)
ax.set_xlim(0, max(NPERNODE_LIST) * 1.1)
ax.set_xlabel("Processes per Node\n(total processes = npernode × 4 nodes)", fontsize=10)
ax.set_ylabel("Total Runtime (s)", fontsize=11)
ax.grid(axis='y', which='both', linestyle='--', alpha=0.4, zorder=0)

# sorted labels in log space
log_max = np.log10(max_time * 50.0)
log_min = np.log10(min_time * 0.4)

for xi, np_node in enumerate(NPERNODE_LIST):
    col_points = []
    for n in names:
        t_val   = data[n][np_node]
        base_t  = data[n][1]
        speedup = base_t / t_val if t_val > 0 else 0
        color   = tc_colors[n]
        col_points.append((t_val, speedup, color))

    col_points.sort(key=lambda p: p[0], reverse=True)

    label_log_top    = log_max - (log_max - log_min) * 0.02
    label_log_bottom = log_max - (log_max - log_min) * 0.3
    step = (label_log_top - label_log_bottom) / (len(col_points) - 1) if len(col_points) > 1 else 0

    for rank_idx, (t_val, speedup, color) in enumerate(col_points):
        y_pos = 10 ** (label_log_top - step * rank_idx)
        if xi == 1:
            x_offset = 0.4
        else:
            x_offset = 0.0
        ax.annotate(
            f"{t_val:.6f}s\n{speedup:.2f}x",
            xy=(x[xi], t_val),
            xytext=(x[xi] + x_offset, y_pos),
            ha='center', va='top', fontsize=8.5,
            color=color, fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            arrowprops=dict(arrowstyle="-", color=color, lw=0.4, alpha=0.5),
        )

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, fontsize=9,
           loc="center left",
           bbox_to_anchor=(1.01, 0.5),
           framealpha=0.9,
           borderaxespad=0)

plt.tight_layout()
outpath = os.path.join(logs_dir, "q2_strong_scaling.png")
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Plot saved to {outpath}")
EOF