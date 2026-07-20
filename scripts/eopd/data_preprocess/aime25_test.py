"""Preprocess math-ai/aime25 to verl parquet format (standalone)."""
import argparse, json, os
import datasets
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="benchmarks")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    data_source = "math-ai/aime25"
    ds = datasets.load_dataset(data_source, split=args.split)
    print(f"{data_source} loaded: {len(ds)} rows, features={ds.features}", flush=True)
    def process_fn(example, idx):
        question = example["problem"] + " " + INSTRUCTION
        return {"data_source": data_source, "prompt": [{"role": "user", "content": question}],
                "ability": "math", "reward_model": {"style": "rule", "ground_truth": example["answer"]},
                "extra_info": {"split": args.split, "index": idx}}
    ds = ds.map(function=process_fn, with_indices=True)
    os.makedirs(args.local_save_dir, exist_ok=True)
    out = os.path.join(args.local_save_dir, "test.parquet")
    ds.to_parquet(out)
    with open(os.path.join(args.local_save_dir, "aime25_example.json"), "w") as f:
        json.dump(ds[0], f, indent=2, ensure_ascii=False)
    print(f"Saved -> {out}  ({len(ds)} rows)", flush=True)
if __name__ == "__main__":
    main()
