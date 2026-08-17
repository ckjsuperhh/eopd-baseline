#!/usr/bin/env bash
# EOPD 训练启动器
#   EOPD = clipped reverse KL (OPD) + entropy-gated forward-KL on high-entropy teacher tokens.
#
# 用法（在仓库根目录执行）：
#   # 最简：设好模型路径即可，train/val 默认从 DATA_DIR 取
#   STUDENT_MODEL_PATH=/path/to/Qwen3-1.7B-Base \
#   TEACHER_MODEL_PATH=/path/to/Qwen3-8B \
#     bash scripts/eopd/run_eopd.sh
#   # 或单独覆盖数据路径：
#   DATA_DIR=/inspire/.../dk/data STUDENT_MODEL_PATH=... TEACHER_MODEL_PATH=... \
#     bash scripts/eopd/run_eopd.sh
#
# 关键环境约定（详见 ../../setup_env.sh）：
#   - flash_attn：若机器 glibc < 2.32 无法装真 flash_attn，脚本会自动把本仓库自带的纯 torch shim
#     (scripts/eopd/flash_attn_shim) 注入 PYTHONPATH，让 use_remove_padding=True 走 sdpa 后端生效。
#     真实 flash_attn 可导入时则不会注入 shim（避免遮蔽真实安装）。
#   - 离线：本脚本固定使用本地模型/数据路径，并强制禁用 HuggingFace/Datasets 回源。
#   - 切勿设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments，与 vLLM memory pool 冲突会直接报错。
#
# 可调环境变量（含默认值）：
#   CONDA_ENV        eopd
#   DATA_DIR         本地数据目录；train/val 路径已在脚本内固定
#   CUDA_VISIBLE_DEVICES  0,1,2,3,4,5,6,7
#   N_GPUS_PER_NODE  8
#   GPU_MEM_UTIL     0.3
#   HF_HOME          $HOME/hf_cache
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHIM_DIR="$SCRIPT_DIR/flash_attn_shim"

# ---- conda ----
CONDA_BASE="$(conda info --base 2>/dev/null)" || true
# shellcheck disable=SC1091
[ -n "$CONDA_BASE" ] && source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null
# shellcheck disable=SC1091
[ -z "$CONDA_BASE" ] && source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "${CONDA_ENV:-eopd}"

# ---- flash_attn：真实缺失时才注入 shim ----
if ! python -c "import flash_attn.bert_padding" >/dev/null 2>&1; then
  echo "[run_eopd] real flash_attn not importable -> injecting pure-torch shim from $SHIM_DIR"
  export PYTHONPATH="$SHIM_DIR:${PYTHONPATH}"
else
  echo "[run_eopd] using installed flash_attn"
fi

# ---- HuggingFace / 离线 ----
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline
export VLLM_USE_V1=1
# 让训练日志实时刷新（写到文件/管道时也逐行输出，便于 tail -f 监控）
export PYTHONUNBUFFERED=1

# ---- 必需的模型 / 数据路径 ----
# 固定使用已准备好的本地路径；路径失效时直接失败，避免回源下载。
export STUDENT_MODEL_PATH="/inspire/hdd/project/multi-agent/zhangweinan-24046/dk/modelweights/Qwen3-1.7B-Base"
export TEACHER_MODEL_PATH="/inspire/hdd/project/multi-agent/zhangweinan-24046/dk/modelweights/Qwen3-8B"
export DATA_DIR="/inspire/hdd/project/multi-agent/zhangweinan-24046/kejiechen/CacheEOPD_fullscale_vm_bundle/data/math"
export TRAIN_FILE="$DATA_DIR/train.parquet"
export VAL_FILE="$DATA_DIR/test.parquet"

for required_path in "$STUDENT_MODEL_PATH/config.json" "$TEACHER_MODEL_PATH/config.json" "$TRAIN_FILE" "$VAL_FILE"; do
  if [ ! -e "$required_path" ]; then
    echo "Required local path is missing: $required_path" >&2
    exit 1
  fi
done

echo "[run_eopd] TRAIN_FILE=$TRAIN_FILE"
echo "[run_eopd] VAL_FILE=$VAL_FILE"

# 默认用满本机所有 GPU（单机 8 卡时即 0..7）。
# apex 等共享机器请显式覆盖，例如：
#   CUDA_VISIBLE_DEVICES=1,2,4,5 N_GPUS_PER_NODE=4 bash run_eopd.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.3}"
export METHOD="${METHOD:-eopd}"
# Checkpoint root passed through to on_policy_it.sh (CKPT_BASE="${CKPT_DIR:-/ckpts}").
# Override when /ckpts is not writable, e.g. CKPT_DIR=$HOME/ckpts bash scripts/eopd/run_eopd.sh
export CKPT_DIR="${CKPT_DIR:-/ckpts}"

cd "$REPO_ROOT"
bash examples/on_policy_distillation/on_policy_it.sh
