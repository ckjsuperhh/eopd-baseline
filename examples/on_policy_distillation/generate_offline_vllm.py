#!/usr/bin/env python3
# 离线 vLLM 批量生成（替代 verl.trainer.main_generation）。
#
# 为什么需要它：本仓库的 verl 版本里，main_generation 走 ActorRolloutRefWorker 的
# vLLMAsyncRollout，而 vLLM 同步 generate_sequences() 已在 PR #4411 被移除（generation.yaml
# 注释明说 main_generation 对 vLLM async 是 broken 的）。这里直接用 vLLM 离线 LLM 做同步
# 批量生成，绕开 verl 的 rollout  machinery。
#
# 输出格式与 main_generation 一致：保留输入 parquet 的全部列（含 reward_model.ground_truth），
# 并新增 `responses` 列（每个 prompt 一个 list，长度 = n_samples）。score_avg_pass_at_k.py 可直接读取。
#
# 【多卡加速：数据并行】
#   1.7B 这种小模型用 tensor_parallel_size>1 反而更慢（通信开销盖过计算），正确提速方式是
#   数据并行：把 prompts 切成 N 份，开 N 个独立 vLLM 实例（各占 1 卡、各 TP=1）同时生成。
#   用法：
#     --num_shards N --shard_id K    # 本进程只处理第 K 份（K in [0, N)），写到 <output>.shardK.parquet
#   评测脚本会对每个 benchmark 启动 N 个这样的进程（各绑一张卡），跑完用 --merge 拼回一份。
#
# 用法（参数与 eval_six_benchmarks.sh 对应）：
#   # 单进程（默认，num_shards=1，等价于只用 1 卡）
#   python3 generate_offline_vllm.py \
#     --model_path /path/to/hf_model --data_path /path/to/<bench>/test.parquet \
#     --output_path /path/to/eval_gen/<BENCH>.parquet \
#     --prompt_key prompt --n_samples 8 --temperature 1.0 --top_p 0.8 \
#     --max_prompt_length 1024 --max_tokens 8192 --batch_size 64 \
#     --tensor_parallel_size 1 --gpu_memory_utilization 0.5 --trust_remote_code
#
#   # 数据并行（8 卡）：每个 benchmark 启动 8 个进程，各自 CUDA_VISIBLE_DEVICES=i，传 --shard_id i
#   # 生成完后合并：
#   python3 generate_offline_vllm.py --merge \
#     --input_glob '/path/to/eval_gen/<BENCH>.shard*.parquet' \
#     --output_path /path/to/eval_gen/<BENCH>.parquet

import argparse
import glob
import os

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def build_prompt_texts(tokenizer, prompts, prompt_key):
    texts = []
    for p in prompts:
        if isinstance(p, str):
            texts.append(p)
        elif isinstance(p, (list, tuple)):
            # p 是 list of message dicts（[{role, content}, ...]）
            texts.append(
                tokenizer.apply_chat_template(
                    list(p), add_generation_prompt=True, tokenize=False
                )
            )
        else:
            # 兜底：当作原始文本
            texts.append(str(p))
    return texts


def merge_shards(input_glob, output_path):
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"no shards matched: {input_glob}")
    dfs = [pd.read_parquet(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)
    if "__idx__" in merged.columns:
        merged = merged.sort_values("__idx__").drop(columns="__idx__")
    merged = merged.reset_index(drop=True)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    merged.to_parquet(output_path)
    print(f"Merged {len(files)} shards -> {output_path}: {len(merged)} rows "
          f"x {len(merged['responses'].iloc[0]) if len(merged) else 0} responses.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--prompt_key", default="prompt")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--max_prompt_length", type=int, default=1024)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    ap.add_argument("--trust_remote_code", action="store_true")
    # ---- 数据并行分片 ----
    ap.add_argument("--num_shards", type=int, default=1,
                    help="数据并行份数（= 使用的 GPU 数）。1 表示单进程处理全部。")
    ap.add_argument("--shard_id", type=int, default=0,
                    help="本进程处理第几份（0 <= shard_id < num_shards）。")
    # ---- 合并模式 ----
    ap.add_argument("--merge", action="store_true",
                    help="合并模式：读取 --input_glob 匹配的若干 shard parquet，"
                         "拼成一份完整 parquet 写到 --output_path。")
    ap.add_argument("--input_glob", default=None,
                    help="合并模式用的 shard 文件通配符，如 'dir/BENCH.shard*.parquet'。")
    args = ap.parse_args()

    if args.merge:
        if not args.input_glob:
            ap.error("--merge 需要 --input_glob")
        merge_shards(args.input_glob, args.output_path)
        return

    if not (0 <= args.shard_id < args.num_shards):
        ap.error(f"--shard_id 必须 ∈ [0, {args.num_shards})，收到 {args.shard_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    df = pd.read_parquet(args.data_path)
    total = len(df)

    # 数据并行：本进程只取间隔为 num_shards、起点为 shard_id 的行（确定性分片）
    row_idx = list(range(args.shard_id, total, args.num_shards))
    sub = df.iloc[row_idx].copy()
    sub_prompts = sub[args.prompt_key].tolist()
    prompt_texts = build_prompt_texts(tokenizer, sub_prompts, args.prompt_key)

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        max_model_len=args.max_prompt_length + args.max_tokens,
        enforce_eager=True,
        disable_log_stats=True,
    )
    sampling = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        repetition_penalty=1.0,
    )

    responses = []
    n = len(prompt_texts)
    for i in range(0, n, args.batch_size):
        batch = prompt_texts[i : i + args.batch_size]
        outputs = llm.generate(batch, sampling)
        for o in outputs:
            responses.append([out.text for out in o.outputs])
        print(f"[shard {args.shard_id}] [{min(i + args.batch_size, n)}/{n}] generated")

    sub["responses"] = responses
    sub["__idx__"] = row_idx  # 合并时按原始顺序还原
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sub.to_parquet(args.output_path)
    print(f"Wrote {args.output_path}: {len(sub)} rows × {args.n_samples} responses "
          f"(shard {args.shard_id}/{args.num_shards}).")


if __name__ == "__main__":
    main()
