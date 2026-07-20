#!/usr/bin/env bash
# ============================================================================
# 下载并预处理 EOPD 全规模复现所需的全部数据，统一落盘到 DATA_DIR：
#   $DATA_DIR/<bench>/test.parquet       6 个评测基准 (math500/aime24/aime25/amc23/minerva/olympiadbench)
#   $DATA_DIR/dapo_math17k.parquet       训练语料 (open-r1/DAPO-Math-17k-Processed)
#   $DATA_DIR/smoke/smoke_train.parquet  冒烟切片 (math500 前 64 行)
#
# 这个布局与 examples/on_policy_distillation/eval_six_benchmarks.sh 直接对齐：
# 评测脚本读取 $DATA_DIR/<bench>/test.parquet，无需任何手动挪动。
# 训练脚本 (run_eopd.sh) 默认也从这个 DATA_DIR 取 train/val。
#
# 用法:
#   bash data/preprocess/run_all_preprocess.sh          # 默认 DATA_DIR=$HOME/data
#   DATA_DIR=/inspire/.../dk/data bash .../run_all_preprocess.sh
#
# 前置: 需要能访问 HuggingFace CDN（apex 上走 HF_HUB_OFFLINE 会失败，新 VM 有网）。
#        会自动 pip install datasets。
# 注意: 产出的 parquet 不进 git（体积大），切勿对仓库执行 `git clean -fdx`，
#        否则这些数据会被清掉，需要重跑本脚本。
# ============================================================================
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$HOME/data}"

echo "===== 数据根目录 DATA_DIR=$DATA_DIR ====="
mkdir -p "$DATA_DIR"

# 确保 datasets 可用
python3 -c "import datasets" 2>/dev/null || pip install -q datasets

cd "$SCRIPT_DIR"

echo "===== 6 评测基准 (-> $DATA_DIR/<bench>/test.parquet) ====="
for B in math500 aime24 aime25 amc23 minerva olympiadbench; do
  python3 ${B}_test.py --local_save_dir "$DATA_DIR/$B"
done

echo "===== 训练语料 (DAPO-Math-17k) ====="
python3 dapo_math17k.py --local_save_dir "$DATA_DIR"

echo "===== 冒烟切片 (math500 前 64 行) ====="
SMOKE_DIR="$DATA_DIR/smoke"
mkdir -p "$SMOKE_DIR"
python3 - "$DATA_DIR" "$SMOKE_DIR" <<'PY'
import sys, pyarrow.parquet as pq
data_dir, smoke_dir = sys.argv[1], sys.argv[2]
src = f"{data_dir}/math500/test.parquet"
dst = f"{smoke_dir}/smoke_train.parquet"
t = pq.read_table(src)
n = min(64, t.num_rows)
pq.write_table(t.slice(0, n), dst)
print(f"smoke_train.parquet: {n} rows -> {dst}")
PY

echo "===== 校验产出 ====="
ls -lh "$DATA_DIR"/*/test.parquet "$DATA_DIR/dapo_math17k.parquet" "$DATA_DIR/smoke"/*.parquet
echo "DONE. 所有数据已存到 DATA_DIR=$DATA_DIR"
