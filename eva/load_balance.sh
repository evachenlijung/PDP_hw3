#!/bin/bash

# 確保輸出資料夾存在
mkdir -p ~/hw3/eva/logs/load_balance

echo "=== Parsing CSV and generating Load Balance Efficiency Plot ==="

python3 << 'EOF'
import os
import csv
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
import matplotlib.patheffects as pe
from collections import defaultdict

warnings.filterwarnings("ignore")

# 設定路徑 (自動解析波浪號 ~)
logs_dir = os.path.expanduser("~/hw3/eva/logs/q1_mpi")
csv_path = os.path.join(logs_dir, "q1_mpi.csv")

out_dir = os.path.expanduser("~/hw3/eva/logs/load_balance")
out_path = os.path.join(out_dir, "load_balance_efficiency.png")

testcase_order = [
    "imbalance_c100000",
    "medium_c200000",
    "large_c1000000",
    "large_c2000000",
    "large_c4000000",
]
NPERNODE_LIST = [1, 2, 4, 8, 12, 16]

# --- 1. 讀取 CSV 並計算 Efficiency ---
data = defaultdict(dict)
try:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["testcase"]
            if name in testcase_order:
                np_node = int(row["npernode"])
                try:
                    avg_t = float(row["avg_render_time"])
                    max_t = float(row["max_render_time"])
                    # 避免除以零的錯誤
                    if max_t > 0:
                        efficiency = (avg_t / max_t) * 100.0
                    else:
                        efficiency = 0.0
                    data[name][np_node] = efficiency
                except ValueError:
                    continue
except FileNotFoundError:
    print(f"❌ Error: {csv_path} not found! Please run the MPI test script first.")
    exit(1)

# --- 2. 設定冷到暖的漸層色彩 (Cool to Warm) ---
cmap = plt.cm.coolwarm
tc_colors = {
    n: mcolors.to_hex(c)
    for n, c in zip(testcase_order, cmap(np.linspace(0.05, 0.95, len(testcase_order))))
}
# 🌟 針對要求：將 1M 強制覆寫為更深的灰色 (Medium Dark Gray)
tc_colors["large_c1000000"] = "#888888"

# --- 3. 開始繪圖 ---
fig, ax = plt.subplots(figsize=(14, 8))
fig.suptitle("Load Balance Efficiency vs Npernode", fontsize=20, fontweight='bold', y=0.96)

x_positions = np.array(NPERNODE_LIST, dtype=float)
markers = ["o", "s", "^", "D", "P"]

# 畫線與點 (不再這裡加標籤)
for idx, name in enumerate(testcase_order):
    if name not in data or not data[name]:
        continue
        
    color = tc_colors[name]
    marker = markers[idx % len(markers)]
    y_values = [data[name].get(np_node, np.nan) for np_node in NPERNODE_LIST]
    
    ax.plot(x_positions, y_values, marker=marker, color=color, 
            linewidth=2.5, markersize=9, label=name, zorder=4)

# 🌟 終極完美排版：帶虛線引導的垂直排序標籤
# 稍微往下挖深 Y 軸即可，將 -30 到 0 的空間當作標籤區，大幅縮小與上方折線的距離
ax.set_ylim(-30, 115) 

# (已經將 ax.axhline 刪除，不再有灰色的分隔線)

for xi, np_node in zip(x_positions, NPERNODE_LIST):
    col_data = []
    
    # 收集這個節點所有的有效數據與顏色
    for name in testcase_order:
        if name in data and np_node in data[name]:
            val = data[name][np_node]
            if not np.isnan(val):
                col_data.append((val, tc_colors[name]))
                
    # 依照效能由低到高排序
    col_data.sort(key=lambda item: item[0])
    
    start_y_lowest = -20
    step_y = 9 
    
    # 🌟 完美排版微調：因為 X 軸 1 和 2 距離太近，我們把它們的標籤稍微推開
    label_x = xi
    if np_node == 2 or np_node == 4:
        label_x = xi + 1  # 滿足你的需求，把 2 往右推多一點！
        
    for idx, (val, color) in enumerate(col_data):
        fixed_y = start_y_lowest + (idx * step_y)
        
        # 利用 annotate 同時畫出文字與虛線引導線
        ax.annotate(
            f"{val:.1f}%",
            xy=(xi, val),              # 箭頭起點：保留在上方真實的 X 座標
            xytext=(label_x, fixed_y), # 箭頭終點與文字：使用微調推開後的 X 座標
            ha='center', va='center', fontsize=12,
            color=color, fontweight='bold', zorder=10,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            arrowprops=dict(arrowstyle="-", linestyle="--", color=color, alpha=0.5, lw=1.2, zorder=3)
        )
# --- 4. 圖表格式化 ---
# 🌟 強制設定 Y 軸的刻度只顯示 0 到 100
ax.set_yticks(np.arange(0, 120, 20))
ax.yaxis.set_major_formatter(mtick.PercentFormatter()) 

ax.set_xticks(x_positions)
ax.set_xticklabels([str(int(n)) for n in NPERNODE_LIST], fontsize=13)
ax.set_xlim(-0.5, max(NPERNODE_LIST) + 1.5)

ax.set_xlabel("Processes per Node", fontsize=15, labelpad=15)
ax.set_ylabel("Efficiency (Avg / Max)", fontsize=15)

# 網格線只畫大於等於 0 的部分
ax.grid(axis='y', which='major', linestyle='--', alpha=0.5, zorder=0)

ax.legend(title="Testcases (Less → More Data)", fontsize=12, title_fontsize=12,
          loc="center left", bbox_to_anchor=(1.02, 0.5), 
          framealpha=0.9, borderaxespad=0.)

plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"✅ Plot successfully saved to {out_path}")
EOF