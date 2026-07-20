#!/usr/bin/env bash
# 六基准评测脚本（对应论文 Table 2）：MATH500 / AMC23 / Minerva / OlympiadBench / AIME24 / AIME25
#
# 流程：
#   1) 对每个 benchmark，用 verl.trainer.main_generation 以
#      temperature=1.0, top_p=0.8, max response length=8192, n_samples=8 生成 8 条回答
#      （论文 Table 2 评测设定：8 samples/question）
#   2) 用 score_avg_pass_at_k.py 计算 Avg@8 与 Pass@8（math_verify 判定答案等价）
#
# 评测对象：训练后的学生模型。请把 MODEL_PATH 指向【训练产出的 HF 格式 checkpoint】
#          （verl 默认保存在 trainer.default_local_dir 下，需先导出/转为 HF 格式；
#            快速冒烟可临时指向原始 base 模型）。
#
# 前置：先跑 examples/data_preprocess/{math500_test,amc23_test,minerva_test,
#        olympiadbench_test,aime24_test,aime25_test}.py 生成各 test.parquet。
#
# 用法：
#   MODEL_PATH=/ckpts/.../global_step_xxx/actor bash examples/on_policy_distillation/eval_six_benchmarks.sh

set -x

# ========================== 评测模型 ==========================
MODEL_PATH=${MODEL_PATH:-"/models/Qwen3-1.7B-Base"}   # 指向训练后的 HF checkpoint

# ========================== 资源 ==========================
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
TP=${TP:-1}                              # tensor_model_parallel_size（1.7B 用 1 即可）
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5}

# ========================== 采样设定（论文 Table 2 评测） ==========================
N_SAMPLES=${N_SAMPLES:-8}               # 每题 8 条
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.8}                     # 评测 top-p = 0.8
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}   # 评测 max response = 8192
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-64}    # 每个生成 batch 的 prompt 数

# ========================== 数据路径（预处理输出） ==========================
# 与 run_all_preprocess.sh / run_eopd.sh 共用同一个 DATA_DIR：
#   基准输入: $DATA_DIR/<bench>/test.parquet
#   生成样本: $DATA_DIR/eval_gen/<bench>.parquet   (自动落盘，可复用)
#   评测分数: $DATA_DIR/eval_results/<MODEL_TAG>_scores.{txt,json}  (自动落盘)
DATA_DIR=${DATA_DIR:-"$HOME/data"}
GEN_DIR=${GEN_DIR:-"$DATA_DIR/eval_gen"}
RESULT_DIR=${RESULT_DIR:-"$DATA_DIR/eval_results"}
mkdir -p "${GEN_DIR}" "${RESULT_DIR}"
MODEL_TAG="$(basename "$(dirname "${MODEL_PATH}")")_$(basename "${MODEL_PATH}")"
SCORE_TXT="$RESULT_DIR/${MODEL_TAG}_scores.txt"
SCORE_JSON="$RESULT_DIR/${MODEL_TAG}_scores.json"

BENCHMARKS=(MATH500 AMC23 Minerva OlympiadBench AIME24 AIME25)
INPUT_PATHS=(
    "${DATA_DIR}/math500/test.parquet"
    "${DATA_DIR}/amc23/test.parquet"
    "${DATA_DIR}/minerva/test.parquet"
    "${DATA_DIR}/olympiadbench/test.parquet"
    "${DATA_DIR}/aime24/test.parquet"
    "${DATA_DIR}/aime25/test.parquet"
)

# ========================== Step 1: 逐基准生成 ==========================
SCORE_INPUTS=()
for i in "${!BENCHMARKS[@]}"; do
    B="${BENCHMARKS[$i]}"
    IN="${INPUT_PATHS[$i]}"
    OUT="${GEN_DIR}/${B}.parquet"

    echo "===== Generating ${B} (n_samples=${N_SAMPLES}) ====="
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=${NNODES} \
        trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
        data.path="${IN}" \
        data.prompt_key=prompt \
        data.batch_size=${GEN_BATCH_SIZE} \
        data.n_samples=${N_SAMPLES} \
        data.output_path="${OUT}" \
        model.path="${MODEL_PATH}" \
        +model.trust_remote_code=True \
        rollout.temperature=${TEMPERATURE} \
        rollout.top_p=${TOP_P} \
        rollout.top_k=-1 \
        rollout.prompt_length=${MAX_PROMPT_LENGTH} \
        rollout.response_length=${MAX_RESPONSE_LENGTH} \
        rollout.tensor_model_parallel_size=${TP} \
        rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
        rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}

    SCORE_INPUTS+=("${B}=${OUT}")
done

# ========================== Step 2: 计算 Avg@8 / Pass@8 ==========================
echo "===== Scoring all benchmarks ====="
echo "Scores will be saved to:"
echo "  $SCORE_TXT"
echo "  $SCORE_JSON"
python3 examples/on_policy_distillation/score_avg_pass_at_k.py \
    --output "$SCORE_JSON" \
    $(printf -- "--input %s " "${SCORE_INPUTS[@]}") | tee "$SCORE_TXT"
echo "DONE. 评测分数已存: $SCORE_TXT  (机器可读: $SCORE_JSON)"
