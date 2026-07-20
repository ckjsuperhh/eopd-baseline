"""Preprocess open-r1/DAPO-Math-17k-Processed as the EOPD training prompt corpus.

DAPO-Math-17k-Processed is already in verl RL format (prompt / reward_model /
data_source / ability / extra_info). We simply normalize to a stable schema and
re-save as a single parquet for on-policy distillation training.

To swap to another large corpus (e.g. OpenMathReasoning), replace DATASET and
adjust the field mapping below, or just point TRAIN_FILE at a different parquet.
"""
import argparse
import os

import datasets

DATASET = "open-r1/DAPO-Math-17k-Processed"
KEEP = ["prompt", "reward_model", "data_source", "ability", "extra_info"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="train")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--config", default="en",
                        help="DAPO-Math-17k-Processed 子配置：all / cn / en（论文用 en）")
    args = parser.parse_args()

    ds = datasets.load_dataset(args.dataset, args.config)
    if isinstance(ds, datasets.DatasetDict):
        ds = ds[list(ds.keys())[0]]
    print(f"{args.dataset} loaded: {len(ds)} rows, features={ds.features}", flush=True)

    # keep only the verl-training-relevant fields (ignore any extra columns)
    keep = [c for c in KEEP if c in ds.column_names]
    ds = ds.select_columns(keep) if hasattr(ds, "select_columns") else ds.remove_columns(
        [c for c in ds.column_names if c not in keep]
    )

    os.makedirs(args.local_save_dir, exist_ok=True)
    out = os.path.join(args.local_save_dir, "dapo_math17k.parquet")
    ds.to_parquet(out)
    print(f"Saved -> {out}  ({len(ds)} rows)", flush=True)


if __name__ == "__main__":
    main()
