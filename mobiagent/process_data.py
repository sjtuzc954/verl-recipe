import argparse
from pathlib import Path
from PIL import Image
import datasets

def make_map_fn(split, format, data_source="mobiagent"):
    def process_fn_sharegpt(example, idx):
        messages = example.pop("messages")
        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]
        output = messages[2]["content"]
        images = example.pop("images")
        ground_truth = output
        
        pil_img = Image.open(images[0]).convert("RGB")
        images = [pil_img]

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            "images": images,
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": split,
                "idx": idx,
            },
        }
        return data

    # alpaca format
    def process_fn_alpaca(example, idx):
        instruction = example.pop("instruction")
        output = example.pop("output")
        example.pop("input", None)
        images = example.pop("images")
        ground_truth = output

        pil_img = Image.open(images[0]).convert("RGB")
        images = [pil_img]

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": instruction,
                }
            ],
            "images": images,
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": split,
                "idx": idx,
            },
        }
        return data

    return process_fn_alpaca if format == "alpaca" else process_fn_sharegpt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--format", choices=["alpaca", "sharegpt"], default="sharegpt")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split_ratio", type=float, default=0.9)
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = datasets.load_dataset("json", data_files=str(args.train_file))["train"]

    # Random sampling based on sample_ratio
    if args.num_samples is not None:
        train_dataset = train_dataset.train_test_split(
            train_size=args.num_samples, 
            shuffle=True,
            seed=4
        )["train"]

    train_dataset = train_dataset.map(make_map_fn("train", args.format), with_indices=True, num_proc=16)
    
    # split train and test
    if args.split_ratio < 1.0:
        split_dataset = train_dataset.train_test_split(
            train_size=args.split_ratio,
            shuffle=True,
            seed=42
        )
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]
        
        train_dataset.to_parquet(str(args.output_dir / "train.parquet"))
        test_dataset.to_parquet(str(args.output_dir / "test.parquet"))
    else:
        train_dataset.to_parquet(str(args.output_dir / "train.parquet"))
