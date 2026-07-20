#!/usr/bin/env bash
# ============================================================================
# EOPD 复现 —— 全环境一键配置脚本
# 目标：在另一台 Linux 虚拟机上配出与 apex-llm 完全一致的实验环境，
#       以便 clone 本仓库后直接跑 EOPD / OPD 训练与六基准评测。
#
# 用法：
#   bash setup_env.sh                # 默认创建 conda 环境 'eopd'
#   CONDA_ENV=myenv bash setup_env.sh
#
# 前置要求：
#   - NVIDIA 驱动（CUDA 12.6 兼容），nvidia-smi 可用
#   - 已安装 conda（miniconda/anaconda），且 conda 在 PATH 中
#   - 可访问 PyPI（pip）与 HuggingFace（下载模型/数据；离线机见下方 OFFLINE 段落）
#
# 本脚本会：
#   1) 创建 conda 环境（Python 3.11.15）
#   2) 安装 torch 2.7.1+cu126, vllm 0.10.0
#   3) 以 editable 方式安装本仓库（verl 0.7.0.dev0 fork）
#   4) 安装评测/数据预处理依赖（math-verify, pandas, pyarrow, datasets, ...）
#   5) 尝试安装真实 flash_attn（可选）；装不上则自动改用仓库自带纯 torch shim
#   6) （可选）下载模型权重、预处理六个评测基准 + 训练语料
#
# 验证通过的版本（apex-llm）：
#   python 3.11.15 | torch 2.7.1+cu126 | vllm 0.10.0 | transformers 4.57.6
#   ray 2.56.0 | datasets 5.0.0 | math-verify 0.9.0 | pyarrow 25.0.0
# ============================================================================
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-eopd}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHIM_DIR="$REPO_ROOT/scripts/eopd/flash_attn_shim"

# ---- 0. conda 可达性检查 ----
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 未找到 conda。请先安装 miniconda/anaconda 并将其加入 PATH。" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---- 1. 创建 conda 环境 ----
if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "[setup] conda 环境 '$CONDA_ENV' 已存在，跳过创建。"
else
  echo "[setup] 创建 conda 环境 '$CONDA_ENV' (python=3.11) ..."
  conda create -y -n "$CONDA_ENV" python=3.11
fi
conda activate "$CONDA_ENV"

# ---- 2. torch 2.7.1 + CUDA 12.6 ----
echo "[setup] 安装 torch 2.7.1+cu126 ..."
pip install --upgrade pip
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# ---- 3. vllm 0.10.0（与 torch 2.7 兼容）----
echo "[setup] 安装 vllm==0.10.0 ..."
pip install vllm==0.10.0

# ---- 4. 本仓库（verl fork）editable 安装 ----
echo "[setup] 以 editable 方式安装 verl (pip install -e .) ..."
cd "$REPO_ROOT"
# verl 的 setup 会一并拉取 ray / transformers / datasets 等依赖
pip install -e . --no-cache-dir

# ---- 5. 评测 / 数据预处理依赖 ----
echo "[setup] 安装评测与数据预处理依赖 ..."
pip install "math-verify" pandas pyarrow datasets tqdm regex

# ---- 6. flash_attn：优先真实安装，失败则依赖仓库 shim ----
echo "[setup] 尝试安装真实 flash_attn（失败也无妨，会改用仓库自带 shim）..."
if pip install flash_attn; then
  echo "[setup] 真实 flash_attn 安装成功。"
else
  echo "[setup] 真实 flash_attn 安装失败（常见：glibc < 2.32 或无法访问 GitHub 源码）。"
  echo "[setup] 训练/评测启动脚本会自动把以下 shim 注入 PYTHONPATH："
  echo "        $SHIM_DIR"
  echo "[setup] 该 shim 仅提供 verl remove-padding 所需的 bert_padding 打包函数，真实注意力走 sdpa 后端。"
fi

# ---- 7. 校验关键包 ----
echo "[setup] 校验关键包版本："
python - <<'PY'
import importlib.metadata as md
for p in ["torch", "vllm", "transformers", "ray", "datasets", "verl", "math_verify", "pyarrow", "pandas"]:
    try:
        print(f"  {p:14s} {md.version(p)}")
    except md.PackageNotFoundError:
        print(f"  {p:14s} MISSING")
import torch
print("  torch.cuda      ", torch.__version__, "cuda", torch.version.cuda)
PY

echo
echo "==================================================================="
echo " 环境配置完成。下一步："
echo
echo " (A) 准备模型权重（需要 HuggingFace 访问；Qwen3-8B 可能需先 huggingface-cli login）："
echo "       huggingface-cli download Qwen/Qwen3-1.7B-Base --local-dir \$HF_HOME/Qwen3-1.7B-Base"
echo "       huggingface-cli download Qwen/Qwen3-8B           --local-dir \$HF_HOME/Qwen3-8B"
echo "     （或在离线机上把已下载的快照目录放到 HF_HOME / 直接用本地路径）"
echo
echo " (B) 预处理评测基准 + 训练语料（需要 HuggingFace 数据集访问）："
echo "       bash scripts/eopd/data_preprocess/run_all_preprocess.sh"
echo "     产出：benchmarks/*.parquet（6 基准）、train/dapo_math17k.parquet"
echo
echo " (C) 启动训练（EOPD）："
echo "       STUDENT_MODEL_PATH=<...> TEACHER_MODEL_PATH=<...> \\"
echo "       TRAIN_FILE=<...> VAL_FILE=<...> \\"
echo "         bash scripts/eopd/run_eopd.sh"
echo "     对照 OPD：   bash scripts/eopd/run_opd.sh   （同上前导环境变量）"
echo
echo " (D) 训练完成后六基准评测："
echo "       MODEL_PATH=<checkpoint/huggingface> bash scripts/eopd/run_eval.sh"
echo
echo " (E) 后台监控训练日志（每 60s 扫描致命错误）："
echo "       tmux new -d -s mon 'bash scripts/eopd/eopd_monitor.sh'"
echo "==================================================================="
