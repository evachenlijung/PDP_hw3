# #!/bin/bash

# # ==========================================
# # 1. 編譯與分發執行檔
# # ==========================================
# mpicc renderer_mpi_new.c -o renderer_mpi_new -lm -march=native
# for i in 1 2 3; do scp renderer_mpi_new rdma$i:~/hw3/eva/; done

# # 建立輸出與日誌目錄
# mkdir -p ~/hw3/eva/output/q3_new
# mkdir -p ~/hw3/eva/logs/q3_new

# Q3_CSV=~/hw3/eva/logs/q3_new/q3_new_weak_scaling.csv
# # 🌟 更新 CSV 標頭：加入 avg_render_time
# echo "testcase,npernode,nprocs,total_runtime,avg_render_time" > $Q3_CSV

# # 僅針對三個 large 測資進行 Weak Scaling 測試
# TESTCASES=(
#     "large_c1000000"
#     "large_c2000000"
#     "large_c4000000"
# )

# # ==========================================
# # 2. 執行 MPI 測試並收集數據
# # ==========================================
# for name in "${TESTCASES[@]}"; do
#     bin="../testcases/${name}.bin"

#     # 根據不同的測資大小，精準指派對應的 npernode 陣列
#     case $name in
#         "large_c1000000") NPERNODE_LIST=(1 2 3 4) ;;
#         "large_c2000000") NPERNODE_LIST=(2 4 6 8) ;;
#         "large_c4000000") NPERNODE_LIST=(4 8 12 16) ;;
#     esac

#     for npernode in "${NPERNODE_LIST[@]}"; do
#         nprocs=$((npernode * 4))
#         outpng=~/hw3/eva/output/q3_new/${name}_npernode${npernode}.png
#         logfile=~/hw3/eva/logs/q3_new/${name}_npernode${npernode}.log

#         echo "=== Running: $name npernode=$npernode (total $nprocs procs) ==="

#         UCX_TLS=rc,sm,self \
#         UCX_NET_DEVICES=rocep23s0:1 \
#         mpirun \
#           --hostfile hosts \
#           -npernode $npernode \
#           --mca pml ucx \
#           --mca btl ^tcp \
#           ./renderer_mpi_new $bin $outpng \
#           2>&1 | tee $logfile

#         # 抓取 Total Runtime 與 Local Render Time
#         total=$(grep "Total runtime" $logfile | awk '{print $(NF-1)}')
#         avg_render=$(grep "per-rank render time" $logfile | awk -F'avg=' '{print $2}')

#         # 寫入 CSV
#         echo "$name,$npernode,$nprocs,$total,$avg_render" >> $Q3_CSV
#         echo "  total=$total avg_render=$avg_render"
#         echo ""
#     done
# done

# echo "=== Done. Results written to $Q3_CSV ==="
# echo "=== Generating Weak Scaling Plots... ==="


python3 << 'EOF'
import os
import csv
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
import numpy as np
from collections import defaultdict

warnings.filterwarnings("ignore")

logs_dir = os.path.expanduser("~/hw3/eva/logs/q3_new")
csv_path = os.path.join(logs_dir, "q3_new_weak_scaling.csv")

testcase_order = [
    "large_c1000000",
    "large_c2000000",
    "large_c4000000"
]

x_labels = ["(1, 2, 4)", "(2, 4, 8)", "(3, 6, 12)", "(4, 8, 16)"]

npernode_mapping = {
    "large_c1000000": [1, 2, 3, 4],
    "large_c2000000": [2, 4, 6, 8],
    "large_c4000000": [4, 8, 12, 16]
}

data = defaultdict(dict)
try:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["testcase"]
            if name in testcase_order:
                np_node = int(row["npernode"])
                try:
                    data[name][np_node] = {
                        "total": float(row["total_runtime"]) if row.get("total_runtime") else 0.0,
                        "render": float(row["avg_render_time"]) if row.get("avg_render_time") else 0.0
                    }
                except ValueError:
                    continue
except FileNotFoundError:
    print(f"Warning: {csv_path} not found. Please run the bash script first.")

cmap = plt.cm.coolwarm
tc_colors = {
    testcase_order[0]: mcolors.to_hex(cmap(0.1)),
    testcase_order[1]: "#666666",                  
    testcase_order[2]: mcolors.to_hex(cmap(0.9))   
}

# 🌟 終極排版策略：三向爆破偏移
# 藍線(1M)往下推，紅線(4M)往上推，灰線(2M)往右推到兩個 X 軸刻度的中間空白處！
label_configs = {
    "large_c1000000": [ 
        (0, -45, 'top', 'center'),
        (0, -45, 'top', 'center'),
        (0, -45, 'top', 'center'),
        (0, -45, 'top', 'center')
    ],
    "large_c2000000": [ 
        (45, 0, 'center', 'left'),    # 點0：往右推到空白處
        (45, 0, 'center', 'left'),    # 點1：往右推到空白處
        (45, 0, 'center', 'left'),    # 點2：往右推到空白處
        (-45, 0, 'center', 'right')   # 點3：最後一點往左推，避免超出圖表右邊界！
    ],
    "large_c4000000": [ 
        (0, 45, 'bottom', 'center'),  
        (0, 45, 'bottom', 'center'),  
        (0, 45, 'bottom', 'center'),  
        (0, 45, 'bottom', 'center')   
    ]
}

fig, axes = plt.subplots(1, 2, figsize=(24, 11))
fig.suptitle("Weak Scaling Analysis (Total vs. Local Render) - Log Scale", fontsize=22, fontweight='bold', y=0.98)

metrics = [
    {"key": "total", "ax": axes[0], "title": "Total Runtime (Total Time)"},
    {"key": "render", "ax": axes[1], "title": "Local Render Time (Compute Only)"}
]

for metric in metrics:
    ax = metric["ax"]
    key = metric["key"]
    
    max_time = 0.0
    min_time = float('inf') 
    
    base_times = []
    for np_node in npernode_mapping["large_c1000000"]:
        base_times.append(data.get("large_c1000000", {}).get(np_node, {}).get(key, 0.0))

    for name in testcase_order:
        color = tc_colors[name]
        target_npernodes = npernode_mapping[name]
        
        y_values = []
        for np_node in target_npernodes:
            val = data.get(name, {}).get(np_node, {}).get(key, 0.0) 
            y_values.append(val)
            if val > max_time:
                max_time = val
            if 0 < val < min_time: 
                min_time = val

        x_positions = np.arange(len(x_labels))
        
        ax.plot(x_positions, y_values, marker="o", color=color,
                linewidth=2.5, markersize=8, label=name, zorder=4)

        for i, val in enumerate(y_values):
            if val > 0:
                x_off, y_off, v_align, h_align = label_configs[name][i]
                col_base_time = base_times[i]
                speedup = col_base_time / val if val > 0 and col_base_time > 0 else 0.0

                ax.annotate(
                    f"{val:.6f}s\n{speedup:.2f}x",
                    xy=(x_positions[i], val),
                    xytext=(x_off, y_off),
                    textcoords="offset points",
                    ha=h_align, va=v_align, fontsize=11,
                    color=color, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                    # 🌟 加上引導線：即使文字被推到 45 像素外，也能清楚指回原本的資料點
                    arrowprops=dict(arrowstyle="-", color=color, lw=1.5, alpha=0.5, shrinkA=0, shrinkB=5)
                )

    ax.set_yscale('log')
    if min_time != float('inf'):
        # 為了容納被推遠的標籤，稍微擴大 Y 軸下限的視覺緩衝 (從 0.6 改為 0.45)
        ax.set_ylim(min_time * 0.45, max_time * 2.5)

    # 稍微拓寬 X 軸左右兩側的空間，防止被往外推的標籤遭到裁切
    ax.set_xlim(-0.4, len(x_labels) - 0.6)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}'))

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=12)

    ax.set_xlabel("Npernode Configuration Tuples\n(1M, 2M, 4M)", fontsize=14, labelpad=10)
    ax.set_ylabel("Time (s) - Log Scale", fontsize=14)
    ax.set_title(metric["title"], fontsize=18, pad=15)
    
    ax.grid(axis='y', which='both', linestyle='--', alpha=0.5, zorder=0)

axes[1].legend(title="Testcases", fontsize=12, 
               loc="upper left", bbox_to_anchor=(1.02, 1), 
               framealpha=0.9, borderaxespad=0.)

plt.tight_layout()
outpath = os.path.join(logs_dir, "q3_new_weak_scaling.png")
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Plot saved to {outpath}")
EOF