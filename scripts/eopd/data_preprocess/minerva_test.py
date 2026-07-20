"""Preprocess svc-huggingface/minerva-math to verl parquet format.

Minerva Math 无独立 answer 字段，答案在 solution 中：取最后一个 \\boxed{}，
否则取 solution 最后一行非空文本作为 ground_truth。
"""
import argparse, json, os, re
import datasets
INSTR = "Let's think step by step and output the final answer within \\boxed{}."
def extract_answer(sol):
    if not sol:
        return ""
    m = list(re.finditer(r"\\boxed\{([^{}]*)\}", sol))
    if m:
        return m[-1].group(1).strip()
    lines = [l.strip() for l in sol.splitlines() if l.strip()]
    return lines[-1] if lines else ""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="benchmarks")
    args = parser.parse_args()
    ds = datasets.load_dataset("svc-huggingface/minerva-math", split="test")
    print(f"minerva loaded: {len(ds)} rows, features={ds.features}", flush=True)
    def proc(e, i):
        q = e["problem"] + " " + INSTR
        ans = extract_answer(e["solution"])
        return {"data_source": "minerva_math", "prompt": [{"role": "user", "content": q}],
                "ability": "math", "reward_model": {"style": "rule", "ground_truth": ans},
                "extra_info": {"split": "test", "index": i}}
    ds = ds.map(proc, with_indices=True)
    os.makedirs(args.local_save_dir, exist_ok=True)
    out = os.path.join(args.local_save_dir, "test.parquet")
    ds.to_parquet(out)
    with open(os.path.join(args.local_save_dir, "minerva_example.json"), "w") as f:
        json.dump(ds[0], f, indent=2, ensure_ascii=False)
    print(f"Saved -> {out}  ({len(ds)} rows)", flush=True)
if __name__ == "__main__":
    main()
