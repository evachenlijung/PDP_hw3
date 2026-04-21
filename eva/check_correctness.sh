#!/bin/bash

GOLDEN_DIR=~/hw3/testcases/golden

# collect all output PNGs from the various scripts
declare -A OUTPUT_PNGS

# q1_serial: one png per testcase
for name in imbalance_c100000 medium_c200000 large_c1000000 large_c2000000 large_c4000000; do
    OUTPUT_PNGS["q1_serial_${name}"]="$HOME/hw3/eva/output/serial/${name}.png"
done

# q1_mpi: one png per testcase per npernode
for name in imbalance_c100000 medium_c200000 large_c1000000 large_c2000000 large_c4000000; do
    for npernode in 1 2 4 8 12 16; do
        OUTPUT_PNGS["q1_mpi_${name}_npernode${npernode}"]="$HOME/hw3/eva/output/mpi/${name}_npernode${npernode}.png"
    done
done

# q1_mpi_new: same structure
for name in imbalance_c100000 medium_c200000 large_c1000000 large_c2000000 large_c4000000; do
    for npernode in 1 2 4 8 12 16; do
        OUTPUT_PNGS["q1_mpi_new_${name}_npernode${npernode}"]="$HOME/hw3/eva/output/mpi/${name}_npernode${npernode}.png"
    done
done

# q2: one png per testcase per npernode
for name in imbalance_c100000 medium_c200000 large_c1000000 large_c2000000 large_c4000000; do
    for npernode in 1 2 4 8 12 16; do
        OUTPUT_PNGS["q2_${name}_npernode${npernode}"]="$HOME/hw3/eva/output/q2/${name}_npernode${npernode}.png"
    done
done

# q2_new
for name in imbalance_c100000 medium_c200000 large_c1000000 large_c2000000 large_c4000000; do
    for npernode in 1 2 4 8 12 16; do
        OUTPUT_PNGS["q2_new_${name}_npernode${npernode}"]="$HOME/hw3/eva/output/q2_new/${name}_npernode${npernode}.png"
    done
done

echo "============================================"
echo " Correctness Check vs Golden"
echo "============================================"
echo ""

pass=0
fail=0
missing=0

for key in "${!OUTPUT_PNGS[@]}"; do
    output="${OUTPUT_PNGS[$key]}"

    # extract testcase name from key
    for name in imbalance_c100000 medium_c200000 large_c1000000 large_c2000000 large_c4000000; do
        if [[ "$key" == *"$name"* ]]; then
            golden="$GOLDEN_DIR/${name}.png"
            break
        fi
    done

    if [ ! -f "$output" ]; then
        echo "MISSING  [$key]"
        echo "         expected: $output"
        ((missing++))
        continue
    fi

    if [ ! -f "$golden" ]; then
        echo "NO_GOLDEN [$key] golden not found: $golden"
        continue
    fi

    diff_result=$(diff "$output" "$golden" 2>&1)
    if [ -z "$diff_result" ]; then
        echo "PASS     [$key]"
        ((pass++))
    else
        echo "FAIL     [$key]"
        echo "         output:  $output"
        echo "         golden:  $golden"
        ((fail++))
    fi
done

echo ""
echo "============================================"
echo " Summary: PASS=$pass  FAIL=$fail  MISSING=$missing"
echo "============================================"