python3 << 'EOF'
records = {
    "imbalance_c100000": 100000,
    "medium_c200000":    200000,
    "large_c1000000":   1000000,
    "large_c2000000":   2000000,
    "large_c4000000":   4000000,
}
results = {
    "imbalance_c100000": {1:"✗",2:"✗",4:"✗",8:"✗",12:"✗",16:"✗"},
    "medium_c200000":    {1:"✓",2:"✓",4:"✗",8:"✗",12:"✗",16:"✗"},
    "large_c1000000":    {1:"✓",2:"✓",4:"✓",8:"✓",12:"✓",16:"✗"},
    "large_c2000000":    {1:"✓",2:"✓",4:"✓",8:"✓",12:"✗",16:"✗"},
    "large_c4000000":    {1:"✓",2:"✓",4:"✓",8:"✓",12:"✓",16:"✓"},
}
RECSZ = 15
print(f"{'testcase':<25} {'npernode':>8} {'nprocs':>6} {'bytes/rank':>12} {'result':>8}")
print("-" * 65)
for name, npernode_results in results.items():
    for np_node, result in npernode_results.items():
        nprocs = np_node * 4
        bytes_per_rank = records[name] * RECSZ // nprocs
        print(f"{name:<25} {np_node:>8} {nprocs:>6} {bytes_per_rank:>12,} {result:>8}")
    print()
EOF