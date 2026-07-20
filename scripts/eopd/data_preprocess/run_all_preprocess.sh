#!/usr/bin/env bash
# ============================================================================
# 下载并预处理 EOPD 全规模复现所需的全部数据：
#   - 6 个评测基准：math500 / aime24 / aime25 / amc23 / minerva / olympiadbench
#   - 训练语料：open-r1/DAPO-Math-17k-Processed
#   - 冒烟切片：从 math500 取 64 行 -> data/smoke/smoke_train.parquet
#
# 用法: bash data/preprocess/run_all_preprocess.sh
# 前置: 需要能访问 HuggingFace CDN（本地已确认有网）。会自动 pip install datasets。
# ============================================================================
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH="$BASE_DIR/benchmarks"
TRAIN="$BASE_DIR/train"
SMOKE="$BASE_DIR/smoke"

mkdir -p "$BENCH" "$TRAIN" "$SMOKE"

# 确保 datasets 可用
python3 -c "import datasets" 2>/dev/null || pip install -q datasets

cd "$SCRIPT_DIR"

echo "===== 6 评测基准 ====="
python3 math500_test.py      --local_save_dir "$BENCH"
python3 aime24_test.py       --local_save_dir "$BENCH"
python3 aime25_test.py       --local_save_dir "$BENCH"
python3 amc23_test.py        --local_save_dir "$BENCH"
python3 minerva_test.py      --local_save_dir "$BENCH"
python3 olympiadbench_test.py --local_save_dir "$BENCH"

echo "===== 训练语料 (DAPO-Math-17k) ====="
python3 dapo_math17k.py      --local_save_dir "$TRAIN"

echo "===== 冒烟切片 (math500 前 64 行) ====="
export BENCH SMOKE
python3 - <<'PY'
import os
import pyarrow.parquet as pq
src = os.path.join(os.environ["BENCH"], "math500.parquet")
dst = os.path.join(os.environ["SMOKE"], "smoke_train.parquet")
t = pq.read_table(src)
n = min(64, t.num_rows)
pq.write_table(t.slice(0, n), dst)
print(f"smoke_train.parquet: {n} rows -> {dst}")
PY

echo "===== 校验产出 ====="
ls -lh "$BENCH"/*.parquet "$TRAIN"/*.parquet "$SMOKE"/*.parquet
echo "DONE"
