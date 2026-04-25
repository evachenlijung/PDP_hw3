#!/bin/bash

mpicc renderer_mpi_v4_reduce.c -o renderer_mpi_v4_reduce -lm -march=native
for i in 1 2 3; do scp renderer_mpi_v4_reduce rdma$i:~/hw3/eva/; done

mkdir -p ~/hw3/eva/output/mpi
mkdir -p ~/hw3/eva/logs/q1_mpi

Q1_MPI_CSV=~/hw3/eva/logs/q1_mpi/q1_mpi.csv

echo "testcase,npernode,nprocs,image_size,total_weight,target_per_proc,scatter_time,min_render_time,max_render_time,avg_render_time,composite_time,total_time" > $Q1_MPI_CSV

TESTCASES=(
    "../testcases/imbalance_c100000.bin"
    "../testcases/medium_c200000.bin"
    "../testcases/large_c1000000.bin"
    "../testcases/large_c2000000.bin"
    "../testcases/large_c4000000.bin"
)

NPERNODE_LIST=(1 2 4 8 12 16)

# NPERNODE_LIST=(1)

declare -A RAW_IMAGE_SIZE
declare -A RAW_WEIGHT
declare -A RAW_SCATTER
declare -A RAW_AVG_RENDER
declare -A RAW_COMPOSITE
declare -A RAW_TOTAL

for bin in "${TESTCASES[@]}"; do

    name=$(basename $bin .bin)

    for npernode in "${NPERNODE_LIST[@]}"; do
        nprocs=$((npernode * 4))
        outpng=~/hw3/eva/output/mpi/${name}_npernode${npernode}.png
        logfile=~/hw3/eva/logs/q1_mpi/${name}_npernode${npernode}.log

        UCX_TLS=rc,sm,self \
        UCX_NET_DEVICES=rocep23s0:1 \
        UCX_LOG_LEVEL=info \
        mpirun \
        --hostfile hosts \
        -npernode $npernode \
        --mca pml ucx \
        --mca btl ^tcp \
        ./renderer_mpi_v4_reduce $bin $outpng \
        2>&1 | tee $logfile

        image_size=$(grep "rank0: magic=CRDR" $logfile | head -1 | sed 's/.*image \([0-9]*\)x\([0-9]*\).*/\1 \2/' | awk '{print $1 * $2}')
        total_weight=$(grep "total weight:" $logfile | head -1 | awk '{print $NF}')
        target=$(grep "target per process:" $logfile | head -1 | awk '{print $NF}')
        scatter=$(grep "rank0: read+scatter time:" $logfile | awk '{print $(NF-1)}' | awk 'BEGIN{val="0"} $1+0 != 0 {val=$1} END{print val}')
        min_render=$(grep "per-rank render time" $logfile | awk -F'min=' '{print $2}' | awk -F' ' '{print $1}')
        max_render=$(grep "per-rank render time" $logfile | awk -F'max=' '{print $2}' | awk -F' ' '{print $1}')
        avg_render=$(grep "per-rank render time" $logfile | awk -F'avg=' '{print $2}')
        composite=$(grep "rank0: composite+write time:" $logfile | awk '{print $(NF-1)}')
        
        total=$(grep "Total runtime" $logfile | awk '{print $(NF-1)}')

        RAW_IMAGE_SIZE["$name"]=$image_size
        RAW_WEIGHT["$name"]=$total_weight
        RAW_SCATTER["${name}_${npernode}"]=$scatter
        RAW_AVG_RENDER["${name}_${npernode}"]=$avg_render
        RAW_COMPOSITE["${name}_${npernode}"]=$composite
        RAW_TOTAL["${name}_${npernode}"]=$total

        echo "image_size=$image_size total_weight=$total_weight target=$target scatter=$scatter min_render=$min_render max_render=$max_render avg_render=$avg_render composite=$composite total=$total"
        echo "$name,$npernode,$nprocs,$image_size,$total_weight,$target,$scatter,$min_render,$max_render,$avg_render,$composite,$total" >> $Q1_MPI_CSV    
    done
done

echo "=== Done. Results written to $Q1_MPI_CSV ==="

python3 << 'EOF'
import os
import re
import csv
import warnings
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import numpy as np
from collections import defaultdict

warnings.filterwarnings("ignore")

logs_dir = os.path.expanduser("~/hw3/eva/logs/q1_mpi")
csv_path = os.path.join(logs_dir, "q1_mpi.csv")

testcase_order = [
    "imbalance_c100000",
    "medium_c200000",
    "large_c1000000",
    "large_c2000000",
    "large_c4000000",
]
NPERNODE_LIST = [1, 2, 4, 8, 12, 16]

# --- 1. 從所有 Log 檔動態解析數據 ---
rows = []
for name in testcase_order:
    for npernode in NPERNODE_LIST:
        nprocs = npernode * 4
        logfile = os.path.join(logs_dir, f"{name}_npernode{npernode}.log")
        if not os.path.exists(logfile):
            print(f"⚠️ Warning: Missing {logfile}. Skipping this point.")
            continue
        
        with open(logfile) as f:
            content = f.read()

        def grep(pattern, text, default=""):
            m = re.search(pattern, text)
            return m.group(1) if m else default

        image_size_m = re.search(r"image (\d+)x(\d+)", content)
        image_size   = int(image_size_m.group(1)) * int(image_size_m.group(2)) if image_size_m else ""
        total_weight = grep(r"total weight: ([\d.]+)", content)
        target       = grep(r"target per process: ([\d.]+)", content)
        scatter      = grep(r"rank0: read\+scatter time: ([\d.]+)", content)
        min_render   = grep(r"min=([\d.]+)", content)
        max_render   = grep(r"max=([\d.]+)", content)
        avg_render   = grep(r"avg=([\d.]+)", content)
        composite    = grep(r"rank0: composite\+write time: ([\d.]+)", content)
        total        = grep(r"Total runtime.*: ([\d.]+)", content)

        rows.append({
            "testcase": name, "npernode": npernode, "nprocs": nprocs,
            "image_size": image_size, "total_weight": total_weight,
            "target_per_proc": target, "scatter_time": scatter,
            "min_render_time": min_render, "max_render_time": max_render,
            "avg_render_time": avg_render, "composite_time": composite,
            "total_time": total,
        })

# --- 2. 把乾淨無重複的資料寫入 CSV ---
fieldnames = ["testcase","npernode","nprocs","image_size","total_weight",
              "target_per_proc","scatter_time","min_render_time","max_render_time","avg_render_time","composite_time","total_time"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"CSV written to {csv_path}")

# --- 3. 讀取 CSV 準備畫圖 ---
data = defaultdict(dict)
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        name    = row["testcase"]
        np_node = int(row["npernode"])
        try:
            data[name][np_node] = {
                "image_size":   int(row["image_size"])        if row["image_size"]        else 0,
                "total_weight": float(row["total_weight"])    if row["total_weight"]       else 0.0,
                "scatter":      float(row["scatter_time"])    if row["scatter_time"]       else 0.0,
                "avg_render":   float(row["avg_render_time"]) if row["avg_render_time"]    else 0.0,
                "composite":    float(row["composite_time"])  if row["composite_time"]     else 0.0,
            }
        except ValueError:
            continue

names = [n for n in testcase_order if n in data and
         all(np_node in data[n] for np_node in NPERNODE_LIST)]

base_name = "imbalance_c100000"
base_img  = data[base_name][16]["image_size"] if 16 in data[base_name] else 0
base_wt   = data[base_name][16]["total_weight"] if 16 in data[base_name] else 0

tc_colors = {
    n: mcolors.to_hex(c)
    for n, c in zip(testcase_order, plt.cm.coolwarm(np.linspace(0.05, 0.95, len(testcase_order))))
}
tc_colors["large_c1000000"] = "#4d4d4d"

x = np.array(NPERNODE_LIST, dtype=float)

time_keys   = ["scatter",               "avg_render",          "composite"]
time_labels = ["Read+Scatter Time (s)", "Avg Render Time (s)", "Composite+Write Time (s)"]
time_fmt    = ["{:.6f}s",               "{:.6f}s",             "{:.6f}s"]
markers     = ["^", "D", "P"]

fig, axes = plt.subplots(1, 3, figsize=(24, 9))
fig.suptitle("MPI Renderer: Timing Breakdown vs npernode by Testcase", fontsize=14)

for col, (ax, tkey, tlabel, marker, fmt) in enumerate(zip(axes, time_keys, time_labels, markers, time_fmt)):

    all_times = [data[n][np_node][tkey] for n in names for np_node in NPERNODE_LIST if data[n][np_node][tkey] > 0]
    if not all_times:
        continue
    max_time = max(all_times)
    min_time = min(all_times)

    for n in names:
        color  = tc_colors[n]
        img_g  = data[n][16]["image_size"]   / base_img if base_img else 0
        wt_g   = data[n][16]["total_weight"] / base_wt  if base_wt  else 0
        label  = f"{n}\n(img {img_g:.2f}× wt {wt_g:.2f}×)"
        times  = [data[n][np_node][tkey] for np_node in NPERNODE_LIST]
        ax.plot(x, times, marker=marker, color=color, linewidth=2, markersize=7, label=label, zorder=4)

    ax.set_yscale('log')
    ax.set_ylim(min_time * 0.4, max_time * 4.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(n)) for n in NPERNODE_LIST], fontsize=9)
    ax.set_xlim(0, max(NPERNODE_LIST) * 1.1)
    ax.set_xlabel("Processes per Node", fontsize=10)
    ax.set_ylabel("Time (s)", fontsize=11)
    ax.set_title(tlabel, fontsize=12)
    ax.grid(axis='y', which='both', linestyle='--', alpha=0.4, zorder=0)

    log_max = np.log10(max_time * 4.0)
    log_min = np.log10(min_time * 0.4)

    for xi, np_node in enumerate(NPERNODE_LIST):
        col_points = []
        for n in names:
            t_val   = data[n][np_node][tkey]
            base_t  = data[n][1][tkey]
            speedup = base_t / t_val if t_val > 0 else 0
            color   = tc_colors[n]
            col_points.append((t_val, speedup, color))

        col_points.sort(key=lambda p: p[0], reverse=True)

        label_log_top    = log_max - (log_max - log_min) * 0.01
        label_log_bottom = log_max - (log_max - log_min) * 0.45
        step = (label_log_top - label_log_bottom) / (len(col_points) - 1) if len(col_points) > 1 else 0

        for rank_idx, (t_val, speedup, color) in enumerate(col_points):
            y_pos = 10 ** (label_log_top - step * rank_idx)
            if xi == 1:
                y_pos = 10 ** (np.log10(y_pos) - (log_max - log_min) * 0.06)
            ax.annotate(
                f"{fmt.format(t_val)}\n{speedup:.2f}x",
                xy=(x[xi], t_val),
                xytext=(x[xi], y_pos),
                ha='center', va='top', fontsize=11,
                color=color, fontweight='bold', zorder=10,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                arrowprops=dict(arrowstyle="-", color=color, lw=0.4, alpha=0.5, zorder=5),
            )

handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.9, borderaxespad=0)

plt.tight_layout()
outpath = os.path.join(logs_dir, "q1_mpi_plot.png")
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Plot saved to {outpath}")
EOF