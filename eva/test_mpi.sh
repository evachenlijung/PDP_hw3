#!/bin/bash

mkdir -p ~/hw3/eva/output

for bin in ../testcases/imbalance_c100000.bin \
           ../testcases/medium_c200000.bin \
           ../testcases/large_c1000000.bin \
           ../testcases/large_c2000000.bin \
           ../testcases/large_c4000000.bin; do

    name=$(basename $bin .bin)

    UCX_TLS=rc,sm,self \
    UCX_NET_DEVICES=rocep23s0:1 \
    mpirun \
      --hostfile hosts \
      -npernode 16 \
      --mca pml ucx \
      --mca btl ^tcp \
      ./renderer_mpi $bin ~/hw3/eva/output/${name}_output.png

done