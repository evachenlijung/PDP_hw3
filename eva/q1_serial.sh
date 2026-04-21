#!/bin/bash

# add total_weight and image_size print to renderer.c before compiling
gcc renderer.c -o renderer -lm -march=native

mkdir -p ~/hw3/eva/output/serial
mkdir -p ~/hw3/eva/logs/q1_serial

Q1_CSV=~/hw3/eva/logs/q1_serial/q1_serial.csv
echo "testcase,image_size,total_weight,total_render_time" > $Q1_CSV

TESTCASES=(
    "../testcases/imbalance_c100000.bin"
    "../testcases/medium_c200000.bin"
    "../testcases/large_c1000000.bin"
    "../testcases/large_c2000000.bin"
    "../testcases/large_c4000000.bin"
)

for bin in "${TESTCASES[@]}"; do
    name=$(basename $bin .bin)
    outpng=~/hw3/eva/output/serial/${name}.png

    echo "=== Running serial: $name ==="

    log=$(./renderer $bin $outpng 2>&1)
    echo "$log"

    image_size=$(echo "$log" | grep "^bbox:" | sed 's/.*image \([0-9]*\)x\([0-9]*\).*/\1 \2/' | awk '{print $1 * $2}')
    total_weight=$(echo "$log" | grep "total weight:" | awk '{print $NF}')
    total_time=$(echo "$log" | grep "Total render time:" | awk '{print $(NF-1)}')

    echo "$name,$image_size,$total_weight,$total_time" >> $Q1_CSV
    echo "  image_size=$image_size total_weight=$total_weight total_render_time=$total_time"
    echo ""
done

echo "=== Done. Results written to $Q1_CSV ==="

# --- Plot ---
python3 << 'EOF'
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

csv_path = os.path.expanduser("~/hw3/eva/logs/q1_serial/q1_serial.csv")

names, image_sizes, total_weights, render_times = [], [], [], []

with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        names.append(row["testcase"])
        image_sizes.append(int(row["image_size"]))
        total_weights.append(float(row["total_weight"]))
        render_times.append(float(row["total_render_time"]))

base_idx  = names.index("medium_c200000")
base_img  = image_sizes[base_idx]
base_wt   = total_weights[base_idx]
base_time = render_times[base_idx]

img_growth  = [s / base_img  for s in image_sizes]
wt_growth   = [w / base_wt   for w in total_weights]
time_growth = [t / base_time for t in render_times]

short_names = [n.replace("_", "\n") for n in names]

color_time = "#c0392b"
color_size = "#1a6faf"
color_wt   = "#27ae60"

x = np.arange(len(names))
fig, ax_wt = plt.subplots(figsize=(13, 7))
fig.suptitle("Serial Renderer: Image Size & Total Weight Growth vs Testcase", fontsize=14)

# left y-axis: total weight growth
ax_wt.set_ylabel("Total Weight Growth (xbase)", fontsize=11, color=color_wt)
ax_wt.tick_params(axis='y', labelcolor=color_wt)
ax_wt.set_ylim(0, max(wt_growth) * 1.3)

# right y-axis: image size growth
ax_img = ax_wt.twinx()
ax_img.set_ylabel("Image Size Growth (xbase)", fontsize=11, color=color_size)
ax_img.tick_params(axis='y', labelcolor=color_size)
ax_img.set_ylim(0, max(img_growth) * 1.3)

# hide ax_wt x ticks, use ax_wt for x
ax_wt.set_xticks(x)
ax_wt.set_xticklabels(short_names, fontsize=9)
ax_wt.set_xlabel("Testcase", fontsize=11)

# third axis for render time (invisible, just for scaling the line)
ax_time = ax_wt.twinx()
ax_time.spines["right"].set_visible(False)
ax_time.tick_params(axis='y', right=False, labelright=False)
ax_time.set_ylim(0, max(render_times) * 1.4)

# plot three lines
line_wt,   = ax_wt.plot(x,   wt_growth,   marker="s", color=color_wt,
                         linewidth=2, markersize=7, label="Total Weight Growth (xbase)", zorder=3)
line_img,  = ax_img.plot(x,  img_growth,  marker="o", color=color_size,
                          linewidth=2, markersize=7, label="Image Size Growth (xbase)", zorder=3)
line_time, = ax_time.plot(x, render_times, marker="^", color=color_time,
                           linewidth=2, markersize=7, label="Render Time (s)", zorder=3)

# labels on render time line nodes
time_offsets = [
    (0,     max(render_times) * 0.08),    # imbalance — default
    (0,     max(render_times) * 0.18),    # medium — higher above node
    (0,     max(render_times) * 0.08),    # large_c1000000 — default
    (0,     max(render_times) * 0.08),    # large_c2000000 — default
    (0,    -max(render_times) * 0.22),    # large_c4000000 — below node
]
time_va = ['bottom', 'bottom', 'bottom', 'bottom', 'top']

for i, (t, g) in enumerate(zip(render_times, time_growth)):
    dx, dy = time_offsets[i]
    ax_time.text(i + dx, t + dy,
                 f"{t:.6f}s\n{g:.2f}x",
                 ha='center', va=time_va[i], fontsize=8,
                 color=color_time, fontweight='bold', linespacing=1.5)

# labels on image size growth nodes
img_offsets = [
    (0, max(img_growth) * 0.03),    # imbalance — default
    (0, max(img_growth) * 0.03),    # medium — default
    (0, max(img_growth) * 0.03),    # large_c1000000 — default
    (0, max(img_growth) * 0.03),    # large_c2000000 — default
    (0, max(img_growth) * 0.03), # large_c4000000 — default
]
for i, g in enumerate(img_growth):
    dx, dy = img_offsets[i]
    ax_img.text(i + dx, g + dy, f"{g:.2f}x",
                ha='center', va='bottom', fontsize=9,
                color=color_size, fontweight='bold')

# labels on total weight growth nodes
wt_offsets = [
    (0, max(wt_growth) * 0.03),    # imbalance — default
    (0.35, max(wt_growth) * 0.03), # medium — shift right
    (0, max(wt_growth) * 0.03),    # large_c1000000 — default
    (0, max(wt_growth) * 0.03),    # large_c2000000 — default
    (0, max(wt_growth) * 0.03),    # large_c4000000 — default
]
for i, (g, w) in enumerate(zip(wt_growth, total_weights)):
    dx, dy = wt_offsets[i]
    ax_wt.text(i + dx, g + dy,
               f"{g:.2f}x\n{w:,.0f}",
               ha='center', va='bottom', fontsize=9,
               color=color_wt, fontweight='bold', linespacing=1.4)

ax_wt.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

lines_all  = [line_time, line_img, line_wt]
labels_all = ["Render Time (s)", "Image Size Growth (xbase)", "Total Weight Growth (xbase)"]
ax_wt.legend(lines_all, labels_all, fontsize=9,
             bbox_to_anchor=(0.5, -0.18), loc="lower center",
             ncol=3, framealpha=0.9)

plt.tight_layout()
outpath  = os.path.expanduser("~/hw3/eva/logs/q1_serial/q1_serial_plot.png")
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Plot saved to {outpath}")
EOF