#!/bin/bash

# 建立獨立的 logs 資料夾存放分析報告
mkdir -p ~/hw3/eva/logs/verify_q1
mkdir -p ~/hw3/eva/output/mpi

TESTCASES=(
    "../testcases/imbalance_c100000.bin"
    "../testcases/medium_c200000.bin"
    "../testcases/large_c1000000.bin"
    "../testcases/large_c2000000.bin"
    "../testcases/large_c4000000.bin"
)

# 針對效能開始崩潰的區間進行高解析度測試
NPERNODE_LIST=(12 13 14 15 16)

# 定義 RDMA 網卡的硬體計數器路徑 (根據你的 UCX_NET_DEVICES 設定)
RDMA_COUNTERS="/sys/class/infiniband/rocep23s0/ports/1/counters"

echo "=== 🚀 開始進行 HPC 效能瓶頸深度驗證 ==="

for bin in "${TESTCASES[@]}"; do
    name=$(basename $bin .bin)

    for npernode in "${NPERNODE_LIST[@]}"; do
        nprocs=$((npernode * 4))
        outpng=~/hw3/eva/output/mpi/${name}_npernode${npernode}.png
        logfile=~/hw3/eva/logs/verify_q1/${name}_npernode${npernode}_run.log
        syslog=~/hw3/eva/logs/verify_q1/${name}_npernode${npernode}_sys.log
        
        echo "--------------------------------------------------------"
        echo "🔬 測試案例: $name | Npernode: $npernode | Nprocs: $nprocs"

        # ==========================================
        # 🟢 1. 啟動背景監控 (I/O Starvation & Memory)
        # 每 1 秒紀錄一次系統狀態 (b: 阻塞進程, wa: I/O等待, cs: 上下文切換)
        # ==========================================
        vmstat 1 > $syslog &
        VMSTAT_PID=$!

        # ==========================================
        # 🟢 2. 讀取執行前的 RDMA 網卡硬體計數 (NIC Bottleneck)
        # ==========================================
        if [ -f "$RDMA_COUNTERS/port_xmit_wait" ]; then
            rdma_wait_start=$(cat $RDMA_COUNTERS/port_xmit_wait)
            rdma_data_start=$(cat $RDMA_COUNTERS/port_xmit_data)
        else
            rdma_wait_start=0
            rdma_data_start=0
        fi

        # ==========================================
        # 🟢 3. 執行程式並捕捉 Context Switches (Core Oversubscription)
        # 使用 /usr/bin/time -v 捕捉作業系統層級的資源調度數據
        # ==========================================
        UCX_TLS=rc,sm,self \
        UCX_NET_DEVICES=rocep23s0:1 \
        UCX_LOG_LEVEL=error \
        /usr/bin/time -v mpirun \
        --hostfile hosts \
        -npernode $npernode \
        --mca pml ucx \
        --mca btl ^tcp \
        ./renderer_mpi_v4_reduce $bin $outpng \
        > $logfile 2>&1

        # ==========================================
        # 🟢 4. 停止背景監控並讀取執行後的網卡計數
        # ==========================================
        kill $VMSTAT_PID 2>/dev/null
        
        if [ -f "$RDMA_COUNTERS/port_xmit_wait" ]; then
            rdma_wait_end=$(cat $RDMA_COUNTERS/port_xmit_wait)
            rdma_data_end=$(cat $RDMA_COUNTERS/port_xmit_data)
            
            # 計算執行期間的網卡壅塞次數
            rdma_wait_diff=$((rdma_wait_end - rdma_wait_start))
            
            # port_xmit_data 單位是 4 Bytes (VL lanes)，換算成 MB
            rdma_data_diff=$(( (rdma_data_end - rdma_data_start) * 4 / 1024 / 1024 ))
        else
            rdma_wait_diff="N/A"
            rdma_data_diff="N/A"
        fi

        # ==========================================
        # 🟢 5. 萃取並印出關鍵瓶頸證據
        # ==========================================
        # 從 time -v 抓取上下文切換
        vol_cs=$(grep "Voluntary context switches" $logfile | awk '{print $NF}')
        invol_cs=$(grep "Involuntary context switches" $logfile | awk '{print $NF}')
        
        # 從 vmstat 計算平均 I/O 等待時間 (去掉前兩行標題，抓取第 16 欄 wa)
        avg_iowait=$(tail -n +3 $syslog | awk '{sum+=$16; count++} END {if(count>0) print sum/count; else print 0}')
        
        # 從程式輸出抓取時間
        scatter=$(grep "rank0: read+scatter time:" $logfile | awk '{print $(NF-1)}')
        total=$(grep "Total runtime" $logfile | awk '{print $(NF-1)}')

        echo "✅ Scatter Time : ${scatter}s"
        echo "✅ Total Time   : ${total}s"
        echo "🚨 [證據 1: Core Oversubscription] 核心強制搶佔次數 (Involuntary CS): $invol_cs"
        echo "🚨 [證據 2: I/O Starvation] CPU 平均等待 I/O 比例 (Avg %wa)       : ${avg_iowait}%"
        echo "🚨 [證據 3: NIC Fan-out] RDMA 網卡因硬體佇列滿載而等待次數      : $rdma_wait_diff"
        echo ""
    done
done

echo "=== 驗證結束 ==="