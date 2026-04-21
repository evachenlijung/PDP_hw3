#!/bin/bash

TESTCASES=(
    "../testcases/imbalance_c100000.bin"
    "../testcases/medium_c200000.bin"
    "../testcases/large_c1000000.bin"
    "../testcases/large_c2000000.bin"
    "../testcases/large_c4000000.bin"
)

NPERNODE_LIST=(1 2 4 8 12 16)

mkdir -p ~/hw3/eva/output
mkdir -p ~/hw3/eva/logs

SUMMARY=~/hw3/eva/logs/summary.csv
echo "testcase,npernode,nprocs,scatter_time,min_render,max_render,avg_render,imbalance_ratio,composite_time,total_time" > $SUMMARY

for bin in "${TESTCASES[@]}"; do
    name=$(basename $bin .bin)

    for npernode in "${NPERNODE_LIST[@]}"; do
        nprocs=$((npernode * 4))
        logfile=~/hw3/eva/logs/${name}_npernode${npernode}.log

        echo "=== Running $name with npernode=$npernode (total $nprocs procs) ==="

        UCX_TLS=rc,sm,self \
        UCX_NET_DEVICES=rocep23s0:1 \
        mpirun \
          --hostfile hosts \
          -npernode $npernode \
          --mca pml ucx \
          --mca btl ^tcp \
          ./renderer_mpi $bin ~/hw3/eva/output/${name}_npernode${npernode}_output.png \
          2>&1 | tee $logfile

        scatter=$(grep "rank0: read+scatter time:" $logfile | tail -1 | awk '{print $NF}' | tr -d 's')
        min_r=$(grep "per-rank render time" $logfile | awk -F'min=' '{print $2}' | awk -F' ' '{print $1}')
        max_r=$(grep "per-rank render time" $logfile | awk -F'max=' '{print $2}' | awk -F' ' '{print $1}')
        avg_r=$(grep "per-rank render time" $logfile | awk -F'avg=' '{print $2}')
        composite=$(grep "rank0: composite+write time:" $logfile | awk '{print $NF}' | tr -d 's')

        imbalance=$(echo "$max_r $min_r $avg_r" | awk '{printf "%.6f", ($1 - $2) / $3}')
        total=$(echo "$scatter $max_r $composite" | awk '{printf "%.6f", $1 + $2 + $3}')

        echo "$name,$npernode,$nprocs,$scatter,$min_r,$max_r,$avg_r,$imbalance,$composite,$total" >> $SUMMARY

        echo "  scatter=$scatter max_render=$max_r avg_render=$avg_r imbalance=$imbalance total=$total"
        echo ""
    done
done

echo "=== Done. Summary written to $SUMMARY ==="

# --- Plot ---
python3 << 'EOF'
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from collections import defaultdict

summary_path = os.path.expanduser("~/hw3/eva/logs/summary.csv")

data = defaultdict(lambda: {"npernode": [], "avg_render": []})

with open(summary_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row["testcase"]
        data[name]["npernode"].append(int(row["npernode"]))
        data[name]["avg_render"].append(float(row["avg_render"]))

testcase_order = [
    "imbalance_c100000",
    "medium_c200000",
    "large_c1000000",
    "large_c2000000",
    "large_c4000000",
]

colors = [mcolors.to_hex(c) for c in plt.cm.coolwarm(np.linspace(0, 1, len(testcase_order)))]
colors[2] = "#969696"
# colors = [mcolors.to_hex(c) for c in plt.cm.plasma(np.linspace(0, 1, len(testcase_order)))[::-1]]

fig, ax = plt.subplots(figsize=(9, 6))

for i, name in enumerate(testcase_order):
    if name not in data:
        continue
    d = data[name]
    sorted_pairs = sorted(zip(d["npernode"], d["avg_render"]))
    x, y = zip(*sorted_pairs)
    ax.plot(x, y, marker="o", color=colors[i], linewidth=2, markersize=6, label=name)
    for xi, yi in sorted_pairs:
        ax.annotate(f"{yi:.6f}", xy=(xi, yi), textcoords="offset points",
                    xytext=(5, 3), fontsize=6, color=colors[i])

ax.set_xlabel("Processes per Node", fontsize=13)
ax.set_ylabel("Avg Render Time (s)", fontsize=13)
ax.set_title("Avg Render Time vs Processes per Node", fontsize=14)
ax.set_yscale("log")
ax.set_xticks([1, 2, 4, 8, 12, 16])
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.5)

outpath = os.path.expanduser("~/hw3/eva/logs/avg_render_plot.png")
plt.tight_layout()
plt.savefig(outpath, dpi=150)
print(f"Plot saved to {outpath}")
EOF