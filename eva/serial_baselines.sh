#!/bin/bash

gcc renderer.c -o renderer -lm -march=native

mkdir -p ~/hw3/eva/output/serial
BASELINE_CSV=~/hw3/eva/logs/serial_baselines.csv
echo "testcase,total_render_time" > $BASELINE_CSV

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

    total=$(echo "$log" | grep "Total render time:" | awk '{print $(NF-1)}')

    echo "$name,$total" >> $BASELINE_CSV
    echo "  total_render_time=$total"
    echo ""
done

echo "=== Done. Baselines written to $BASELINE_CSV ==="