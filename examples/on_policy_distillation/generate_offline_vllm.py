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
# 用法（参数与 eval_six_benchmarks.sh 对应）：
#   python3 generate_offline_vllm.py \
#     --model_path /path/to/hf_model \
#     --data_path /path/to/<bench>/test.parquet \
#     --output_path /path/to/eval_gen/<BENCH>.parquet \
#     --prompt_key prompt --n_samples 8 --temperature 1.0 --top_p 0.8 \
#     --max_prompt_length 1024 --max_tokens 8192 --batch_size 64 \
#     --tensor_parallel_size 1 --gpu_memory_utilization 0.5 --trust_remote_code

import argparse
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
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    df = pd.read_parquet(args.data_path)
    prompt_texts = build_prompt_texts(tokenizer, df[args.prompt_key].tolist(), args.prompt_key)

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
    total = len(prompt_texts)
    for i in range(0, total, args.batch_size):
        batch = prompt_texts[i : i + args.batch_size]
        outputs = llm.generate(batch, sampling)
        for o in outputs:
            responses.append([out.text for out in o.outputs])
        print(f"[{min(i + args.batch_size, total)}/{total}] generated")

    df["responses"] = responses
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(args.output_path)
    print(f"Wrote {args.output_path}: {len(df)} rows × {args.n_samples} responses each.")


if __name__ == "__main__":
    main()
