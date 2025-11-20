# scripts/generate_births.py

import json
import random
import argparse
from datetime import date, timedelta
from pathlib import Path
import configs


def random_birth_date(start_year=1975, end_year=2025):
    """Generate random birth date"""
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    delta = end - start
    rand = start + timedelta(days=random.randint(0, delta.days))
    return rand.strftime("%Y-%m-%d")


def random_birth_record():
    """Generate a single random birth info record"""
    return {
        "生日": random_birth_date(),
        "時辰_index": random.randint(0, 11),
        "性別": random.choice(["男", "女"])
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--batches", type=int, default=1)
    args = parser.parse_args()

    configs.BIRTHS_DIR.mkdir(parents=True, exist_ok=True)

    for batch_idx in range(1, args.batches + 1):
        filename = configs.BATCH_FILENAME.format(batch_idx)
        output_path = configs.BIRTHS_DIR / filename

        with open(output_path, "w", encoding="utf-8") as f:
            for _ in range(args.count):
                f.write(json.dumps(random_birth_record(), ensure_ascii=False) + "\n")

        print(f"[Births] Generated batch {batch_idx} → {output_path}")
