"""Preprocess math-ai/amc23 to verl parquet format."""
import argparse, json, os
import datasets
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
def _load(name, split=None):
    if split is not None:
        return datasets.load_dataset(name, split=split)
    d = datasets.load_dataset(name)
    if isinstance(d, datasets.DatasetDict):
        return d[list(d.keys())[0]]
    return d
def make_map_fn(split):
    def process_fn(example, idx):
        question = example["question"] + " " + INSTRUCTION
        return {"data_source": "amc23", "prompt": [{"role": "user", "content": question}],
                "ability": "math", "reward_model": {"style": "rule", "ground_truth": example["answer"]},
                "extra_info": {"split": split, "index": idx}}
    return process_fn
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="benchmarks")
    parser.add_argument("--split", default=None)
    args = parser.parse_args()
    ds = _load("math-ai/amc23", split=args.split)
    print(f"amc23 loaded: {len(ds)} rows, features={ds.features}", flush=True)
    ds = ds.map(function=make_map_fn("test"), with_indices=True)
    os.makedirs(args.local_save_dir, exist_ok=True)
    out = os.path.join(args.local_save_dir, "amc23.parquet")
    ds.to_parquet(out)
    with open(os.path.join(args.local_save_dir, "amc23_example.json"), "w") as f:
        json.dump(ds[0], f, indent=2, ensure_ascii=False)
    print(f"Saved -> {out}  ({len(ds)} rows)", flush=True)
