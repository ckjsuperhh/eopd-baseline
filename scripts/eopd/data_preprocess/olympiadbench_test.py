"""Preprocess lmms-lab/OlympiadBench (English split) to verl parquet format."""
import argparse, json, os
import datasets
INSTR = "Let's think step by step and output the final answer within \\boxed{}."
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="benchmarks")
    args = parser.parse_args()
    ds = datasets.load_dataset("lmms-lab/OlympiadBench", split="test_en")
    print(f"olympiadbench loaded: {len(ds)} rows, features={ds.features}", flush=True)
    def proc(e, i):
        q = e["question"] + " " + INSTR
        ans = e["final_answer"]
        # OlympiadBench `final_answer` is List[str]; verl reward expects a string
        # ground_truth, so join multiple acceptable answers into one string.
        if isinstance(ans, (list, tuple)):
            ans = " ".join(str(a) for a in ans)
        return {"data_source": "olympiadbench", "prompt": [{"role": "user", "content": q}],
                "ability": "math", "reward_model": {"style": "rule", "ground_truth": ans},
                "extra_info": {"split": "test_en", "index": i}}
    ds = ds.map(proc, with_indices=True)
    # Keep only the verl schema columns. The raw dataset carries an `images`
    # column (PIL objects) that is neither JSON- nor parquet-serializable and is
    # unused for text-only eval; dropping it makes to_parquet/json export work.
    keep = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    ds = ds.select_columns(keep)
    os.makedirs(args.local_save_dir, exist_ok=True)
    out = os.path.join(args.local_save_dir, "test.parquet")
    ds.to_parquet(out)
    with open(os.path.join(args.local_save_dir, "olympiadbench_example.json"), "w") as f:
        json.dump(ds[0], f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved -> {out}  ({len(ds)} rows)", flush=True)
if __name__ == "__main__":
    main()
