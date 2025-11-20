# scripts/generate_charts.py

import json
import argparse
from pathlib import Path
from py_iztro import Astro
import configs


def convert_chart(birth_date, hour_index, gender):
    """Convert py_iztro result into the project's hierarchical JSON schema."""
    astro = Astro()
    result = astro.by_solar(birth_date, hour_index, gender)
    chart = result.model_dump(by_alias=True)

    structured_chart = {
        "基本資料": {
            "出生日期": chart["solarDate"],
            "性別": chart["gender"],
            "生肖": chart["zodiac"],
            "命主": chart["soul"],
            "身主": chart["body"],
            "五行局": chart["fiveElementsClass"]
        },
        "命盤": {}
    }

    for p in chart["palaces"]:
        palace_name = p["name"]
        major_stars = [s["name"] for s in p["majorStars"]]
        huayao = [s["mutagen"] for s in p["majorStars"] if s["mutagen"]]

        structured_chart["命盤"][palace_name] = {
            "本命": {
                "主星": major_stars,
                "化曜": huayao
            },
            "大限": {
                "範圍": p["decadal"]["range"],
                "天干": p["decadal"]["heavenlyStem"],
                "地支": p["decadal"]["earthlyBranch"]
            },
            "流年": {
                "對應年齡": p["ages"][:3]
            }
        }

    return structured_chart


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=1)
    args = parser.parse_args()

    configs.CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    for batch_idx in range(1, args.batches + 1):
        input_file = configs.BIRTHS_DIR / configs.BATCH_FILENAME.format(batch_idx)
        output_file = configs.CHARTS_DIR / configs.BATCH_FILENAME.format(batch_idx)

        with open(input_file, encoding="utf-8") as fin, \
             open(output_file, "w", encoding="utf-8") as fout:

            for line in fin:
                birth = json.loads(line)
                chart = convert_chart(
                    birth_date=birth["生日"],
                    hour_index=birth["時辰_index"],
                    gender=birth["性別"]
                )

                fout.write(json.dumps({
                    "出生資料": birth,
                    "命盤": chart
                }, ensure_ascii=False) + "\n")

        print(f"[Charts] Batch {batch_idx} → {output_file}")
