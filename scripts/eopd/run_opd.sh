#!/usr/bin/env bash
# OPD 基线启动器
#   OPD = clipped reverse KL（即 EOPD 去掉高熵 token 的 forward-KL 项），作为 EOPD 的对照。
#   除 METHOD=opd 外，环境与 run_eopd.sh 完全一致（不设置 ref.topk_logits -> soft-KD 项不激活）。
#
# 用法：
#   STUDENT_MODEL_PATH=... TEACHER_MODEL_PATH=... TRAIN_FILE=... VAL_FILE=... \
#     bash scripts/eopd/run_opd.sh
#
# 所有环境约定与 run_eopd.sh 相同（flash_attn shim / 离线 / conda / GPU 等），
# 直接复用 run_eopd.sh，仅覆盖 METHOD=opd。
set -e

METHOD=opd exec bash "$(dirname "${BASH_SOURCE[0]}")/run_eopd.sh"
