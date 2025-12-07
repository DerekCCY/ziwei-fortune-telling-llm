# python test2.py --data_file data/test.jsonl --output_file outputs/test_predictions.jsonl --max_new_tokens 2048
import argparse
import json
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM, PeftModel
from datasets import load_dataset

from helper import build_prompt  # 只用 Input prompt 那段
from tqdm import tqdm


BASE_MODEL_NAME = "Qwen/Qwen3-4b"
LORA_PATH = "./qwen_lora_output"


# -------------------- 1. 載入模型 & tokenizer -------------------- #
def load_model_and_tokenizer(
    base_model_name: str = BASE_MODEL_NAME,
    lora_path: str = LORA_PATH,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # 優先嘗試直接載 LoRA
    try:
        print("[INFO] Loading LoRA model with AutoPeftModelForCausalLM...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            lora_path,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    except Exception as e:
        print(f"[WARN] AutoPeftModelForCausalLM failed: {e}")
        print("[INFO] Fallback: load base model + PeftModel...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, lora_path)

    # pad_token 設定（盡量跟訓練一致）
    pad_id = None
    if getattr(model, "generation_config", None) and model.generation_config.pad_token_id is not None:
        pad_id = model.generation_config.pad_token_id
        print("[PAD] use model.generation_config.pad_token_id =", pad_id)
    elif tokenizer.pad_token_id is not None:
        pad_id = tokenizer.pad_token_id
        print("[PAD] use tokenizer.pad_token_id =", pad_id)
    else:
        pad_id = tokenizer.eos_token_id
        print("[PAD] use tokenizer.eos_token_id =", pad_id)

    tokenizer.pad_token_id = pad_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(pad_id)

    model.config.pad_token_id = pad_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = pad_id

    model.eval()
    return model, tokenizer


# -------------------- 2. 讀取 jsonl 測試資料 -------------------- #
def load_jsonl_dataset(path: str):
    """
    讀取和 training 相同的 jsonl 格式。
    例如：
      {"出生資料": {...}, "命盤": {...}, "解讀": "..."}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # datasets 會自動處理 json / jsonl
    ds = load_dataset("json", data_files=path, split="train")
    print(f"[INFO] Loaded dataset from {path}, size = {len(ds)}")
    return ds


# -------------------- 3. 從 sample 抽出出生資料 -------------------- #
def extract_birth_info(sample):
    """
    Training 時 `build_training_text(sample)` 用的是 sample["出生資料"]。
    這裡 inference 也沿用相同欄位。
    如果你的 testing jsonl 只有「出生資料」，
    一樣可以用這段。
    """
    if "出生資料" in sample:
        return sample["出生資料"]

    # 如果你的欄位名字不一樣，可以在這裡加其他邏輯
    # e.g. return sample["input"]["出生資料"]
    raise KeyError("sample 裡找不到 '出生資料' 欄位，請修改 extract_birth_info()。")


# -------------------- 4. 單筆推論：build_prompt + generate -------------------- #
def run_inference_one(
    model,
    tokenizer,
    birth_info: dict,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    # 只用 Input prompt：system + user + assistant + <think>...</think>
    prompt = build_prompt(birth_info)

    # token 化
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # 生成
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    # 把 prompt 部分切掉，只留模型新生成的 completion
    if full_text.startswith(prompt):
        completion = full_text[len(prompt):]
    else:
        idx = full_text.rfind(prompt)
        if idx != -1:
            completion = full_text[idx + len(prompt):]
        else:
            completion = full_text

    completion = completion.strip()

    # 把末尾的 <|im_end|> 去掉（如果有的話）
    end_token = "<|im_end|>"
    if completion.endswith(end_token):
        completion = completion[: -len(end_token)].rstrip()

    # 嘗試解析成 JSON（訓練時 target 是 {"命盤":..., "解讀":...}）
    parsed = None
    try:
        parsed = json.loads(completion)
    except Exception:
        pass

    return completion, parsed, full_text


# -------------------- 5. 主程式：整個 jsonl 跑 inference -------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        required=True,
        help="要做 inference 的 jsonl 檔，格式和 training 相同或至少要有 '出生資料' 欄位",
    )
    parser.add_argument(
        "--output_file",
        default="predictions.jsonl",
        help="輸出結果的 jsonl 檔路徑（每行一筆 prediction）",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    args = parser.parse_args()

    # 1) 載入模型
    model, tokenizer = load_model_and_tokenizer()

    # 2) 讀 jsonl
    dataset = load_jsonl_dataset(args.data_file)

    # 3) 跑 inference，逐行寫到 output_file
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    for i, sample in enumerate(tqdm(dataset)):
        try:
            birth_info = extract_birth_info(sample)
        except Exception as e:
            print(f"[WARN] 第 {i} 筆資料跳過（取不到出生資料）: {e}")
            continue

        completion, parsed, full_text = run_inference_one(
            model,
            tokenizer,
            birth_info,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        # 🔹 只存 model 的 string output
        # 如果你想用空行分隔，可以加一個多的 "\n"
        with open(args.output_file, "a", encoding="utf-8") as fout:
            fout.write(completion + "\n")

        # print(f"[INFO] 已完成 {i + 1} 筆 inference...")
    print(f"[DONE] 已將 inference 結果寫入 {args.output_file}")


if __name__ == "__main__":
    main()
