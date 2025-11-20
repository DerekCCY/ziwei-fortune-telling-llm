import json
import argparse
from pathlib import Path

import configs

# TODO: We can modify it later.
def make_default_query():
    """
    Build the default user query for training.
    You can later change this to multiple templates.
    """
    return "請根據我的出生資料排出紫微斗數命盤，並依照固定五段格式做完整命理解讀。"   


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument(
        "--interp_model_tag",
        type=str,
        default="Qwen3-8B-FP8",
        help="Folder name under interpretations/",
    )
    args = parser.parse_args()

    interp_dir = configs.INTERPRETATIONS_DIR / args.interp_model_tag
    output_file = configs.SFT_DIR / f"sft_{args.interp_model_tag}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Stage 3] Building SFT dataset from: {interp_dir}")
    fout = output_file.open("w", encoding="utf-8")

    num_samples = 0

    for idx in range(1, args.batches + 1):
        batch_file = interp_dir / configs.BATCH_TEMPLATE.format(idx)
        if not batch_file.exists():
            print(f"[WARN] Interpretation batch not found, skip: {batch_file}")
            continue

        with batch_file.open(encoding="utf-8") as fin:
            for line in fin:
                rec = json.loads(line)
                birth = rec["出生資料"]
                chart = rec["命盤"]
                interp = rec["解讀"]

                sample = {
                    "input": {
                        "birth": birth,
                        "query": make_default_query()
                    },
                    "output": {
                        "chart_json": chart,
                        "interpretation": interp
                    }
                }

                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                num_samples += 1

        print(f"[Stage 3] Processed batch {idx}")

    fout.close()
    print(f"[Stage 3] Done. Total samples: {num_samples}")
    print(f"[Stage 3] SFT dataset saved to: {output_file}")


if __name__ == "__main__":
    main()
