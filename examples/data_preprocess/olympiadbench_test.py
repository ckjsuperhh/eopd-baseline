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
Preprocess the Hothan/OlympiadBench dataset to parquet format.
OlympiadBench (He et al., 2024) —— one of the six math reasoning benchmarks in EOPD (Table 2).

We use the text-only English math competition subset (OE_TO_maths_en_COMP).
If you want the Chinese subset as well, add "OE_TO_maths_zh_COMP" to OLYMPIAD_CONFIGS
and concatenate the resulting parquet files.
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


# Text-only (TO) maths (maths) competition (COMP) subsets. English by default.
OLYMPIAD_CONFIGS = ["OE_TO_maths_en_COMP"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir",
        default="/root/verl/data/olympiadbench",
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "Hothan/OlympiadBench"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    rows = []
    for config in OLYMPIAD_CONFIGS:
        if local_dataset_path is not None:
            dataset = datasets.load_dataset(local_dataset_path, config)
        else:
            dataset = datasets.load_dataset(data_source, config)

        # This subset uses a train split
        subset = dataset["train"]
        rows.append(subset)

    # Concatenate all selected subsets (typically just the EN text-only maths split)
    from datasets import concatenate_datasets

    test_dataset = concatenate_datasets(rows) if len(rows) > 1 else rows[0]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("question")

            question = question + " " + instruction_following

            # OlympiadBench stores the reference answer in `final_answer`
            solution = example.pop("final_answer")

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    hdfs_dir = args.hdfs_dir

    os.makedirs(local_dir, exist_ok=True)

    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Save one example as JSON for reference
    example = test_dataset[0]
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(example, f, indent=2)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
