# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Offline scoring for on-policy distillation evaluation.

Reads generation parquet(s) produced by `verl.trainer.main_generation`
(each row has a `responses` column = list of k generated strings, and a
`reward_model` column whose `ground_truth` field holds the reference answer),
then reports for each benchmark:

    Avg@k  = mean over prompts of (fraction of k responses correct)
    Pass@k = fraction of prompts with at least one correct response

k is taken from the actual number of responses per prompt (expects 8 for EOPD).

Usage:
    python score_avg_pass_at_k.py --input MATH500=/path/to/math500_gen.parquet \
                                  --input AMC23=/path/to/amc23_gen.parquet ...
"""

import argparse

import numpy as np
import pandas as pd

from verl.utils.reward_score.math_verify import compute_score

# 抑制 math_verify / verl 在评分时打印的 WARNING 噪声（如“未能提取预测”）。
# 这些只是单条样本无法解析答案的诊断信息，不影响最终结果；我们只保留末尾的
# 分数表（用 print 输出，不受 logging 级别影响）。
import logging

for _name in ("math_verify", "verl", "verl.utils.reward_score",
              "verl.utils.reward_score.math_verify"):
    logging.getLogger(_name).setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)  # root 兜底，确保 propagate 的 WARNING 也不落地

# 进度条（环境若未装 tqdm 则退化为无进度条的普通迭代，不影响评分）
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *a, **k):
        return iterable


def _get_ground_truth(reward_model):
    if isinstance(reward_model, dict):
        return reward_model.get("ground_truth", None)
    # Some parquets store reward_model as a stringified dict
    try:
        import json

        return json.loads(reward_model).get("ground_truth", None)
    except Exception:
        return None


def score_file(name, path):
    df = pd.read_parquet(path)
    if "responses" not in df.columns:
        raise ValueError(f"{path}: missing 'responses' column (run main_generation with n_samples>1 first)")

    k_per_prompt = []
    avg_list = []  # per-prompt fraction correct
    pass_list = []  # 1 if any correct else 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"[{name}]", leave=False):
        responses = row.get("responses", None)
        if responses is None:
            continue
        if isinstance(responses, str):
            # safety: might be stored as a string
            import ast

            try:
                responses = ast.literal_eval(responses)
            except Exception:
                responses = []
        try:
            responses = list(responses)
        except Exception:
            continue
        if not responses:
            continue

        gt = _get_ground_truth(row.get("reward_model", None))
        if gt is None:
            continue

        # 单条坏数据不应拖垮整个评测：compute_score 异常时记 0 分
        scores = []
        for r in responses:
            try:
                s = compute_score(str(r), str(gt))
                scores.append(float(s))
            except Exception:
                scores.append(0.0)
        if not scores:
            continue
        k = len(scores)
        k_per_prompt.append(k)
        avg_list.append(np.mean(scores))
        pass_list.append(1.0 if np.max(scores) > 0 else 0.0)

    avg_at_k = float(np.mean(avg_list)) * 100 if avg_list else 0.0
    pass_at_k = float(np.mean(pass_list)) * 100 if pass_list else 0.0
    n_prompts = len(avg_list)
    k = int(np.mean(k_per_prompt)) if k_per_prompt else 0
    return {"name": name, "n": n_prompts, "k": k, "avg": avg_at_k, "pass": pass_at_k}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Repeatable: BENCHMARK_NAME=/path/to/generation.parquet",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write a machine-readable JSON summary of results.",
    )
    args = parser.parse_args()

    if not args.input:
        parser.error("At least one --input BENCHMARK=/path/to/parquet is required")

    results = []
    for item in tqdm(args.input, desc="benchmarks", leave=True):
        if "=" not in item:
            parser.error(f"--input must be NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        results.append(score_file(name, path))

    # Print table
    print("\n" + "=" * 60)
    print(f"{'Benchmark':<16}{'N':>6}{'k':>4}{'Avg@k':>12}{'Pass@k':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<16}{r['n']:>6}{r['k']:>4}{r['avg']:>11.2f}%{r['pass']:>11.2f}%")
    print("=" * 60)

    # Micro-average across all prompts
    all_avg = []
    all_pass = []
    for r in results:
        # reload not needed; recompute micro from stored? we only stored aggregates.
        # Approximate micro-average using prompt-weighted means (each prompt counts once).
        pass
    if results:
        micro_avg = np.mean([r["avg"] for r in results])
        micro_pass = np.mean([r["pass"] for r in results])
        print(f"{'MEAN (simple)':<16}{'':>6}{'':>4}{micro_avg:>11.2f}%{micro_pass:>11.2f}%")
    print()

    if args.output:
        import json

        payload = {
            "per_benchmark": results,
            "mean_avg_at_k": float(micro_avg) if results else 0.0,
            "mean_pass_at_k": float(micro_pass) if results else 0.0,
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[score_avg_pass_at_k] JSON summary written -> {args.output}")


if __name__ == "__main__":
    main()
