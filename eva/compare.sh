#!/bin/bash

# 🌟 設定預設的 CSV 檔案路徑
DEFAULT_A="../ericsson/q1_mpi_new_2.csv"
DEFAULT_B="./logs/q1_mpi/q1_mpi.csv"

# 判斷使用者傳入的參數數量
if [ "$#" -eq 0 ]; then
    A_CSV=$DEFAULT_A
    B_CSV=$DEFAULT_B
    echo "ℹ️  未提供參數，使用預設路徑進行比對..."
elif [ "$#" -eq 2 ]; then
    A_CSV=$1
    B_CSV=$2
else
    echo "❌ 錯誤：參數數量不對！"
    echo "💡 用法 1 (使用預設): ./compare.sh"
    echo "   預設 A: $DEFAULT_A"
    echo "   預設 B: $DEFAULT_B"
    echo "💡 用法 2 (自訂路徑): ./compare.sh <a.csv> <b.csv>"
    exit 1
fi

# 檢查檔案是否存在
if [ ! -f "$A_CSV" ]; then
    echo "❌ 找不到檔案 A：$A_CSV"
    exit 1
fi

if [ ! -f "$B_CSV" ]; then
    echo "❌ 找不到檔案 B：$B_CSV"
    exit 1
fi

echo "📊 效能深度比對 (時間單位: 秒 | Speedup = A / B)"
echo "📄 基準 (A): $A_CSV"
echo "📄 比較 (B): $B_CSV"
echo ""

python3 << EOF
import csv
import sys

a_csv = "$A_CSV"
b_csv = "$B_CSV"

# 1. 讀取 A_CSV 的資料
data_a = {}
try:
    with open(a_csv, 'r') as fa:
        reader = csv.DictReader(fa)
        for row in reader:
            key = (row['testcase'], row['npernode'])
            data_a[key] = {
                'scatter': float(row['scatter_time']),
                'composite': float(row['composite_time']),
                'total': float(row['total_time'])
            }
except Exception as e:
    print(f"讀取 {a_csv} 時發生錯誤: {e}")
    sys.exit(1)

# 2. 列印超寬排版的表格標題 (將倍率欄位稍微加寬以容納符號)
header = (
    f"{'Testcase':<18} | {'Node':<4} | "
    f"{'Sca(A)':>8} | {'Sca(B)':>8} | {'Sca_Spd':>9} | "
    f"{'Cmp(A)':>8} | {'Cmp(B)':>8} | {'Cmp_Spd':>9} | "
    f"{'Tot(A)':>8} | {'Tot(B)':>8} | {'Tot_Spd':>9}"
)
separator = "-" * 163
print(header)
print(separator)

# 3. 讀取 B_CSV 並印出所有原始時間與倍率
try:
    with open(b_csv, 'r') as fb:
        reader = csv.DictReader(fb)
        for row in reader:
            key = (row['testcase'], row['npernode'])
            
            if key in data_a:
                val_a = data_a[key]
                
                b_scatter = float(row['scatter_time'])
                b_composite = float(row['composite_time'])
                b_total = float(row['total_time'])

                # 🌟 全方位邏輯判定：針對四個階段分別比較 B 是否小於 A
                st_sca = "✅" if b_scatter < val_a['scatter'] else "❌"
                st_cmp = "✅" if b_composite < val_a['composite'] else "❌"
                st_tot = "✅" if b_total < val_a['total'] else "❌"

                sp_scatter   = (val_a['scatter'] / b_scatter)     if b_scatter > 0 else 0.0
                sp_composite = (val_a['composite'] / b_composite) if b_composite > 0 else 0.0
                sp_total     = (val_a['total'] / b_total)         if b_total > 0 else 0.0

                # 輸出格式：將符號緊緊黏在 Speedup 倍率旁邊
                print(f"{key[0]:<18} | {key[1]:<4} | "
                      f"{val_a['scatter']:>8.5f} | {b_scatter:>8.5f} | {sp_scatter:>5.2f}x {st_sca} | "
                      f"{val_a['composite']:>8.5f} | {b_composite:>8.5f} | {sp_composite:>5.2f}x {st_cmp} | "
                      f"{val_a['total']:>8.5f} | {b_total:>8.5f} | {sp_total:>5.2f}x {st_tot}")
except Exception as e:
    print(f"讀取 {b_csv} 時發生錯誤: {e}")
    sys.exit(1)

print(separator)
EOF