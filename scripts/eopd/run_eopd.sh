#!/usr/bin/env bash
# EOPD 训练启动器
#   EOPD = clipped reverse KL (OPD) + entropy-gated forward-KL on high-entropy teacher tokens.
#
# 用法（在仓库根目录执行）：
#   STUDENT_MODEL_PATH=/path/to/Qwen3-1.7B-Base \
#   TEACHER_MODEL_PATH=/path/to/Qwen3-8B \
#   TRAIN_FILE=/path/to/math/train.parquet \
#   VAL_FILE=/path/to/math500/test.parquet \
#     bash scripts/eopd/run_eopd.sh
#
# 关键环境约定（详见 ../../setup_env.sh）：
#   - flash_attn：若机器 glibc < 2.32 无法装真 flash_attn，脚本会自动把本仓库自带的纯 torch shim
#     (scripts/eopd/flash_attn_shim) 注入 PYTHONPATH，让 use_remove_padding=True 走 sdpa 后端生效。
#     真实 flash_attn 可导入时则不会注入 shim（避免遮蔽真实安装）。
#   - 离线：机器无法访问 HuggingFace 时设 OFFLINE=1（默认 0=在线，可从 HF 下载/缓存模型）。
#   - 切勿设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments，与 vLLM memory pool 冲突会直接报错。
#
# 可调环境变量（含默认值）：
#   CONDA_ENV        eopd
#   CUDA_VISIBLE_DEVICES  0,1,2,3
#   N_GPUS_PER_NODE  4
#   GPU_MEM_UTIL     0.3
#   HF_HOME          $HOME/hf_cache
#   HF_ENDPOINT      https://hf-mirror.com
#   OFFLINE          0
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
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
if [ "${OFFLINE:-0}" = "1" ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi
export VLLM_USE_V1=1

# ---- 必需的模型 / 数据路径 ----
: "${STUDENT_MODEL_PATH:?请设置 STUDENT_MODEL_PATH (Qwen3-1.7B-Base 快照目录)}"
: "${TEACHER_MODEL_PATH:?请设置 TEACHER_MODEL_PATH (Qwen3-8B 快照目录)}"
: "${TRAIN_FILE:?请设置 TRAIN_FILE (MATH 训练 parquet)}"
: "${VAL_FILE:?请设置 VAL_FILE (MATH500 验证 parquet)}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.3}"
export METHOD=eopd
# Checkpoint root passed through to on_policy_it.sh (CKPT_BASE="${CKPT_DIR:-/ckpts}").
# Override when /ckpts is not writable, e.g. CKPT_DIR=$HOME/ckpts bash scripts/eopd/run_eopd.sh
export CKPT_DIR="${CKPT_DIR:-/ckpts}"

cd "$REPO_ROOT"
bash examples/on_policy_distillation/on_policy_it.sh
