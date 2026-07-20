#!/usr/bin/env bash
# On-Policy Distillation (OPD) / Entropy-Aware On-Policy Distillation (EOPD) 启动脚本
#
# 复现 EOPD 论文 (Entropy-Aware On-Policy Distillation of Language Models) Table 2：
#   - 学生模型：Qwen3-1.7B-Base
#   - 教师模型：Qwen3-8B （base 模型，无 thinking mode）
#   - 训练数据：MATH（3 epochs）
#   - 评测：MATH500 / AMC23 / Minerva / OlympiadBench / AIME24 / AIME25
#            【评测】temperature=1.0, top_p=0.8, max_response_length=8192, 8 samples/question
#
# 超参来源：论文 Appendix A, Table 9 (On-policy distillation and GRPO)
#   - 学习率 3e-6，cosine 调度，AdamW
#   - 训练 batch=128, mini-batch=32（4 步梯度/iter），每个 prompt 1 条 rollout
#   - 训练时 rollout：temperature=1.0, top_p=1.0（Qwen；Llama 才用 0.8），max response length=4096
#   - Top-k (FKL) = 16；τ=0.8, α=1.0（core_algos.py 默认，与论文一致）
#   - 硬件：4×A100-80GB（论文实测；GPU 数量不影响算法结果，只影响并行度）
#
# 该脚本驱动 verl.trainer.main_ppo + OnPolicyDistillTrainer。
# 教师模型作为 ref/teacher worker（FSDP）加载，由 actor_rollout_ref.ref.topk_logits 控制
# 是否启用 forward-KL（即 EOPD 的高熵 token 蒸馏项）：
#   - METHOD=eopd : 设置 ref.topk_logits=K  -> 启用 entropy-gated forward KL（EOPD）
#   - METHOD=opd  : 不设置 ref.topk_logits   -> 仅 clipped reverse KL（OPD baseline）
#
# 用法：
#   METHOD=eopd bash examples/on_policy_distillation/on_policy_it.sh
#   METHOD=opd  bash examples/on_policy_distillation/on_policy_it.sh
#
# 说明：soft_kd_entropy_threshold(τ) 与 soft_kd_loss_coef(α) 在 core_algos.py 中有
#       hasattr 兜底默认值 τ=0.8, α=1.0（与论文一致）。如需显式覆盖，
#       可在下方 EOPD 分支里追加对应 actor_rollout_ref.actor.policy_loss.* 覆盖项。
# 说明：PPO 裁剪 ε 论文未给出具体值，沿用 verl 默认 0.2。

set -x

# ========================== 路径（按需修改） ==========================
# 学生 / 教师模型本地路径
STUDENT_MODEL_PATH=${STUDENT_MODEL_PATH:-"/models/Qwen3-1.7B-Base"}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"/models/Qwen3-8B"}

# 训练数据（MATH，由 examples/data_preprocess/math_dataset.py 生成）
TRAIN_FILE=${TRAIN_FILE:-"/data/math/train.parquet"}
# 训练中验证集（可选，建议放 MATH500 test，data_source 可被 verl 识别）
VAL_FILE=${VAL_FILE:-"/data/math500/test.parquet"}

# ========================== 资源（按需修改） ==========================
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
PROJECT_NAME=${PROJECT_NAME:-"EOPD"}
EXP_NAME=${EXP_NAME:-"Qwen3-1.7B-Base-MATH"}

# ========================== 超参（论文 Appendix A, Table 9：On-policy distillation） ==========================
# 训练 batch：B=128 prompts, mini-batch=32 -> 4 步梯度/iter（每个 prompt 1 条 rollout）
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-1}   # 训练时每个 prompt 1 条 rollout（论文明确：on-policy 单样本）

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
# 注意：训练用 max response length = 4096（Table 9）；8192 是【评测】时的设定（Table 2）
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}

# 论文 Table 9：OPD/EOPD 学习率 3e-6，cosine 调度，AdamW
LR=${LR:-3e-6}
LR_SCHEDULER=${LR_SCHEDULER:-cosine}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}             # MATH 训练数据：3 个 epoch（DAPO 为 2，本次用 MATH）
WARMUP_STEPS=${WARMUP_STEPS:-10}            # 论文未给定 warmup，取小值；可按需调整
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}           # AdamW 默认 weight decay

# 训练时 rollout 采样：temperature=1.0, top_p=1.0（Qwen；Llama 才用 0.8）
TRAIN_TOP_P=${TRAIN_TOP_P:-1.0}

# EOPD 相关
TOPK=${TOPK:-16}                 # 论文 Appendix F：k=16（forward KL 仅在教师 top-16 上计算）
TAU=${TAU:-0.8}                  # 熵阈值 τ（高熵 token 才加 forward KL）
ALPHA=${ALPHA:-1.0}              # soft-KD 系数 α

# PPO 裁剪 ε（论文 Algorithm 1 要求 PPO clip，但未给出具体值；沿用 verl 默认 0.2）
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.2}

METHOD=${METHOD:-eopd}

# ========================== 构建 EOPD/OPD 差异项 ==========================
EXTRA_ARGS=()
if [ "${METHOD}" = "eopd" ]; then
    # 启用 entropy-gated forward KL（EOPD 的核心项）
    EXTRA_ARGS+=(
        actor_rollout_ref.ref.topk_logits=${TOPK}
        actor_rollout_ref.actor.policy_loss.soft_kd_student_full_vocab=True
        # 以下两项若需显式覆盖可取消注释（默认值已与论文一致 τ=0.8, α=1.0）
        # actor_rollout_ref.actor.policy_loss.soft_kd_entropy_threshold=${TAU}
        # actor_rollout_ref.actor.policy_loss.soft_kd_loss_coef=${ALPHA}
    )
    EXP_NAME="${EXP_NAME}-EOPD"
elif [ "${METHOD}" = "opd" ]; then
    # 不设置 ref.topk_logits -> soft-KD 项不激活，仅 clipped reverse KL（OPD baseline）
    EXP_NAME="${EXP_NAME}-OPD"
else
    echo "Unknown METHOD=${METHOD}, must be 'eopd' or 'opd'"
    exit 1
fi

# ========================== 启动 ==========================
python3 -m verl.trainer.main_ppo \
    trainer.trainer_class=OnPolicyDistillTrainer \
    algorithm.adv_estimator=on_policy \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    \
    actor_rollout_ref.model.path="${STUDENT_MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    \
    actor_rollout_ref.teacher_model.path="${TEACHER_MODEL_PATH}" \
    \
    actor_rollout_ref.actor.policy_loss.loss_mode=on_policy_distill \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_scheduler_type=${LR_SCHEDULER} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${WARMUP_STEPS} \
    actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    \
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=${TRAIN_TOP_P} \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.default_local_dir="/ckpts/${PROJECT_NAME}/${EXP_NAME}" \
    trainer.resume_mode=auto \
    "${EXTRA_ARGS[@]}" \
    $@
