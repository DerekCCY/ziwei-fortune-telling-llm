import json
import argparse
from pathlib import Path
import configs
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re
from tqdm.auto import tqdm 
import time
def load_prompt_template():
    prompt_path = Path(__file__).parent / "prompts" / "interpretation_prompt_advanced_zh.txt"
    return prompt_path.read_text(encoding="utf-8")

def build_prompt(template, chart_json):
    chart_str = json.dumps(chart_json, ensure_ascii=False)
    return template.replace("{{CHART_JSON}}", chart_str)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B-FP8")
    parser.add_argument("--max_new_tokens", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model_tag = args.model.split("/")[-1]
    output_dir = configs.INTERPRETATIONS_DIR / model_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Decide device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Stage 2] Using device: {device}")

    print(f"[Stage 2] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    
        # Use HF chat template for Qwen
    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        return_full_text=False   # will only output assistant reply
    )

    template = load_prompt_template()

    for idx in range(1, args.batches + 1):
        input_file = configs.CHARTS_DIR / configs.BATCH_FILENAME.format(idx)
        output_file = output_dir / configs.BATCH_FILENAME.format(idx)

        print(f"[Stage 2] Processing batch {idx}: {input_file}")

        with open(input_file, encoding="utf-8") as f_tmp:
            total_lines = sum(1 for _ in f_tmp)

        with open(input_file, encoding="utf-8") as fin:
    
            for line in tqdm(fin, total=total_lines,
                             desc=f"Batch {idx}", unit="chart"):
                record = json.loads(line)
                chart = record["命盤"]

                # Build chat messages
                messages = [
                    {"role": "system", "content": template},
                    {"role": "user", "content": json.dumps(chart, ensure_ascii=False)}
                ]

                # Convert to model prompt
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                start = time.time()
                # Generate
                result = text_gen(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=True,
                    top_p=0.9
                )[0]["generated_text"]
                
                # ---- 清掉推理區塊 ----'
                result = re.sub(r"<think>.*?</think>\s*", "", result, flags=re.S)
                #result = re.sub(r"<think>[\s\S]*?</think>", "", result, flags=re.IGNORECASE)
                #result = re.sub(r"&lt;think&gt;[\s\S]*?&lt;/think&gt;", "", result, flags=re.IGNORECASE)
#
                #lowered = result.lower()
                #for marker in ("<think", "&lt;think"):
                #    idx = lowered.find(marker)
                #    if idx != -1:
                #        result = result[:idx]
                #        lowered = result.lower()
                # No slicing needed — already clean
                interpretation = result.strip()
                
                with open(output_file, "a", encoding="utf-8") as fout:

                    fout.write(json.dumps({
                        "出生資料": record["出生資料"],
                        "命盤": chart,
                        "解讀": interpretation
                    }, ensure_ascii=False) + "\n")
                    
                end = time.time()
                print(f'One case processing time: {end-start}')

        print(f"[Stage 2] Saved → {output_file}")


if __name__ == "__main__":
    main()
