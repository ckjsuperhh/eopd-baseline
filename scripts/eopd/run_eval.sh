#!/usr/bin/env bash
# 六基准评测包装器（论文 Table 2：MATH500 / AMC23 / Minerva / OlympiadBench / AIME24 / AIME25）
# 调用前务必设置 MODEL_PATH 指向训练产出的 HF 格式 checkpoint（需含 config.json + model.safetensors + tokenizer）。
#
# 用法：
#   MODEL_PATH=/ckpts/EOPD/Qwen3-1.7B-Base-MATH-EOPD/global_step_XXX/actor/huggingface \
#     bash scripts/eopd/run_eval.sh
#
# 评测设定（论文 Table 2，脚本内默认已设，这里仅暴露覆盖项）：
#   8 samples/题, top_p=0.8, max_response_length=8192
# 结果由 examples/on_policy_distillation/score_avg_pass_at_k.py 汇总为 Avg@8 / Pass@8。
#
# 可调环境变量（含默认值）：
#   MODEL_PATH      （必需）HF 格式 checkpoint 目录
#   DATA_DIR        $HOME/data            六个基准 parquet 所在父目录
#   CUDA_VISIBLE_DEVICES  0,1,2,3
#   N_GPUS_PER_NODE  4
#   GPU_MEM_UTIL     0.5
#   OFFLINE          0
set -e

: "${MODEL_PATH:?请先设置 MODEL_PATH 指向 HF 格式 checkpoint（含 config.json + model.safetensors + tokenizer）}"

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
  echo "[run_eval] real flash_attn not importable -> injecting pure-torch shim from $SHIM_DIR"
  export PYTHONPATH="$SHIM_DIR:${PYTHONPATH}"
else
  echo "[run_eval] using installed flash_attn"
fi

# ---- HuggingFace / 离线 ----
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
if [ "${OFFLINE:-0}" = "1" ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi
export VLLM_USE_V1=1

export DATA_DIR="${DATA_DIR:-$HOME/data}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"

cd "$REPO_ROOT"
MODEL_PATH="${MODEL_PATH}" \
DATA_DIR="${DATA_DIR}" \
N_GPUS_PER_NODE="${N_GPUS_PER_NODE}" \
GPU_MEM_UTIL="${GPU_MEM_UTIL}" \
bash examples/on_policy_distillation/eval_six_benchmarks.sh
