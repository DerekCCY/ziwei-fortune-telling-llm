import json
import random
from pathlib import Path
import argparse
import configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interp_model_tag", type=str, default="Qwen3-8B-FP8")
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()

    interp_dir = configs.INTERPRETATIONS_DIR / args.interp_model_tag
    eval_test_dir = configs.DATA_PATH / "eval" / "test"
    eval_test_dir.mkdir(parents=True, exist_ok=True)

    train_out = configs.SFT_DIR / "train.jsonl"
    train_out.parent.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # Collect all interpretation samples
    for file in sorted(interp_dir.glob("batch_*.jsonl")):
        with open(file, encoding="utf-8") as f:
            for line in f:
                all_samples.append(json.loads(line))

    random.shuffle(all_samples)
    test_count = int(len(all_samples) * args.test_ratio)

    test_samples = all_samples[:test_count]
    train_samples = all_samples[test_count:]

    # Save test set (reference data)
    test_file = eval_test_dir / f"test_{args.interp_model_tag}.jsonl"
    with open(test_file, "w", encoding="utf-8") as f:
        for s in test_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Save train set (for supervised tuning)
    with open(train_out, "w", encoding="utf-8") as f:
        for s in train_samples:
            sample = {
                "input": {
                    "birth": s["出生資料"],
                    "query": "請根據我的出生資料排出紫微斗數命盤，並依照固定格式做完整命理解讀。",
                },
                "output": {
                    "chart_json": s["命盤"],
                    "interpretation": s["解讀"]
                }
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[Split] Train samples: {len(train_samples)}")
    print(f"[Split] Test samples: {len(test_samples)} → saved to {test_file}")


if __name__ == "__main__":
    main()
