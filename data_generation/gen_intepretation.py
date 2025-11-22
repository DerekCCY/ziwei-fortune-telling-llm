import json
import argparse
from pathlib import Path
import configs

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def load_prompt_template():
    prompt_path = Path(__file__).parent / 'data_generation' / "prompts" / "interpretation_prompt_advanced_zh.txt"
    return prompt_path.read_text(encoding="utf-8")

def build_prompt(template, chart_json):
    chart_str = json.dumps(chart_json, ensure_ascii=False)
    return template.replace("{{CHART_JSON}}", chart_str)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B-FP8")
    parser.add_argument("--max_new_tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model_tag = args.model.split("/")[-1]
    output_dir = configs.INTERPRETATIONS_DIR / model_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Stage 2] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    template = load_prompt_template()

    for idx in range(1, args.batches + 1):
        input_file = configs.CHARTS_DIR / configs.BATCH_FILENAME.format(idx)
        output_file = output_dir / configs.BATCH_FILENAME.format(idx)

        print(f"[Stage 2] Processing batch {idx}: {input_file}")

        with open(input_file, encoding="utf-8") as fin, \
            open(output_file, "w", encoding="utf-8") as fout:

            for line in fin:
                record = json.loads(line)
                chart = record["命盤"]

                prompt = build_prompt(template, chart)
                result = text_gen(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=True,
                    top_p=0.9,
                )[0]["generated_text"]

                interpretation = result[len(prompt):].strip()

                fout.write(json.dumps({
                    "出生資料": record["出生資料"],
                    "命盤": chart,
                    "解讀": interpretation
                }, ensure_ascii=False) + "\n")

        print(f"[Stage 2] Saved → {output_file}")


if __name__ == "__main__":
    main()
