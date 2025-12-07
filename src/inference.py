# python src/evaluate_ziwei_rag.py --use_4bit --eval_file data/final_data/test.jsonl --max_samples 1 --max_new_tokens 2048 --debug_first
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from bert_score import score as bertscore_score
from safetensors.torch import load_file
from peft import PeftModel
from rag_retriever import RAGRetriever  # 需和 rag_retriever.py 放一起




# ====== System intro（跟 SFT 共用 sft_prompt.txt） ======
SYSTEM_INTRO = ""


def load_system_intro():
    """
    從專案下的 sft_prompt.txt 讀取 system intro，
    讓 evaluation 跟 SFT 使用同一套前置說明。
    """
    global SYSTEM_INTRO
    path = "src/prompt/sft_prompt.txt"

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        SYSTEM_INTRO = text + "\n\n"
        print(f"[System Intro] Loaded from {path}, length={len(SYSTEM_INTRO)} chars.")
    except FileNotFoundError:
        SYSTEM_INTRO = ""
        print(f"[System Intro] File not found at {path}, proceed without system intro.")
    except Exception as e:
        SYSTEM_INTRO = ""
        print(f"[System Intro] Failed to load from {path}: {e}")

# ====== 解析 arguments ======
def parse_args():
    parser = argparse.ArgumentParser(description="Ziwei model inference (with/without RAG)")

    # 模型相關
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HF base model name",
    )
    parser.add_argument(
        "--peft_model_path",
        type=str,
        default="results/best_model",
        help="Fine-tuned LoRA (SFT) folder",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Load base model in 4-bit",
    )

    # RAG 控制
    parser.add_argument(
        "--use_rag",
        action="store_true",
        help="Whether to use RAG for this run",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="RAG top-k",
    )
    parser.add_argument(
        "--include_chart",
        action="store_true",
        help="Whether RAG can return chart-type chunks",
    )

    # 資料與生成設定
    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
        help="Path to eval JSON / JSONL file",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Max number of samples to run (-1 = all)",
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for batched generation",
    )

    parser.add_argument(
        "--debug_first",
        action="store_true",
        help="Only run first sample and print prompt/output for debugging.",
    )
    parser.add_argument(
        "--save_predictions",
        type=str,
        default="results/eval/predictions.jsonl",
        help="Save per-sample ground_truth & prediction to this JSONL file",
    )
    return parser.parse_args()



# ====== 載模型 ======
def load_model_and_tokenizer(
    base_model_name: str,
    peft_model_path: str = None,  # 參數保留但不使用，避免改 main()
    use_4bit: bool = True,
    device_map: str = "auto",
):
    """
    只載入 base model 做推論，不掛載任何 LoRA adapter。
    """
    print(f"[Model] Loading base model only (no LoRA).")
    print(f"  base_model_name = {base_model_name}")
    if peft_model_path is not None:
        print(f"  (peft_model_path is ignored in this mode: {peft_model_path})")

    # 1) 載 base model
    if use_4bit:
        print("[Model] Loading base model in 4bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
        # 嚴格來說純推論不一定要這行；留著也沒關係
        print("[Model] (Optional) prepare_model_for_kbit_training for 4bit model...")
        model = prepare_model_for_kbit_training(model)
    else:
        print("[Model] Loading base model in full precision...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            trust_remote_code=True,
        )

    # 2) 載 tokenizer
    print("[Tokenizer] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 如果你有在 SFT 時「真的」加過 special token（例如 <END_ANSWER>），
    # 這裡再打開註解即可；現在先單純用原始 vocab。
    # SPECIAL_TOKENS = ["<END_ANSWER>"]
    # num_added = tokenizer.add_tokens(SPECIAL_TOKENS)
    # if num_added > 0:
    #     print(f"[Tokenizer] Added {num_added} special tokens: {SPECIAL_TOKENS}")
    #     print(f"[Model] Resizing token embeddings to {len(tokenizer)}")
    #     model.resize_token_embeddings(len(tokenizer))

    model.eval()
    return model, tokenizer



# def _try_load_adapter_file(peft_model_path):
#     safetensors_path = os.path.join(peft_model_path, "adapter_model.safetensors")
#     bin_path = os.path.join(peft_model_path, "adapter_model.bin")
#     if os.path.exists(safetensors_path):
#         try:
#             # safetensors loader (if installed)
#             from safetensors.torch import load_file as safetensors_load
#             return safetensors_load(safetensors_path)
#         except Exception:
#             # fallback to generic loader if safetensors not available
#             return torch.load(safetensors_path, map_location="cpu")
#     elif os.path.exists(bin_path):
#         return torch.load(bin_path, map_location="cpu")
#     else:
#         raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin found in {peft_model_path}")

# def load_model_and_tokenizer(
#     base_model_name: str,
#     peft_model_path: str,
#     use_4bit: bool = True,
#     device_map: str = "auto",
# ):
#     print(f"[Model] Loading PEFT config from: {peft_model_path}")
#     peft_config = PeftConfig.from_pretrained(peft_model_path)
#     print("  base_model_name_or_path from peft_config:", peft_config.base_model_name_or_path)
#     print("  target_modules:", peft_config.target_modules)

#     if peft_config.base_model_name_or_path != base_model_name:
#         print(
#             f"[WARN] base_model_name != peft_config.base_model_name_or_path "
#             f"({base_model_name} vs {peft_config.base_model_name_or_path})"
#         )

#     # 1) 載 base model
#     if use_4bit:
#         print("[Model] Loading base model in 4bit...")
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=False,
#         )
#         base_model = AutoModelForCausalLM.from_pretrained(
#             base_model_name,
#             quantization_config=bnb_config,
#             device_map=device_map,
#             trust_remote_code=True,
#         )
#         print("[Model] Running prepare_model_for_kbit_training...")
#         base_model = prepare_model_for_kbit_training(base_model)
#     else:
#         print("[Model] Loading base model in full precision...")
#         base_model = AutoModelForCausalLM.from_pretrained(
#             base_model_name,
#             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#             device_map=device_map,
#             trust_remote_code=True,
#         )

#     # 2) 載 tokenizer
#     print("[Tokenizer] Loading tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "left"

#     # 如果你在訓練時有加 special tokens（例如 <END_ANSWER>），
#     # 這裡要同樣加回並 resize embedding，確保 base_model embedding 大小與訓練一致。
#     SPECIAL_TOKENS = ["<END_ANSWER>"]
#     num_added = tokenizer.add_tokens(SPECIAL_TOKENS)
#     if num_added > 0:
#         print(f"[Tokenizer] Added {num_added} special tokens: {SPECIAL_TOKENS}")
#         print(f"[Model] Resizing token embeddings to {len(tokenizer)}")
#         base_model.resize_token_embeddings(len(tokenizer))

#     # 3) 用 get_peft_model 把 LoRA 結構掛到 base_model（但不要呼叫 PeftModel.from_pretrained）
#     print(f"[Model] Building PEFT wrapper (get_peft_model) from config at: {peft_model_path}")
#     model = get_peft_model(base_model, peft_config)

#     # 4) 讀 adapter 檔案（safetensors / bin），並過濾掉 embed_tokens / lm_head 等會造成 size mismatch 的 key
#     print(f"[Model] Loading adapter weights from: {peft_model_path}")
#     adapter_state = _try_load_adapter_file(peft_model_path)
#     print(f"[Model] Adapter state_dict params: {len(adapter_state)}")

#     # 這裡採用較寬鬆的 prefix 過濾，能涵蓋常見變種路徑
#     conflict_substrings = (
#         "embed_tokens.weight",
#         ".lm_head.weight",
#         "model.embed_tokens.weight",
#         "lm_head.weight",
#     )

#     filtered_state = {}
#     skipped = []
#     for k, v in adapter_state.items():
#         if any(sub in k for sub in conflict_substrings):
#             skipped.append((k, tuple(v.shape)))
#             continue
#         filtered_state[k] = v

#     print(f"[Model] Skipped {len(skipped)} conflict params (examples: {skipped[:3]})")
#     print(f"[Model] Remaining adapter params to load: {len(filtered_state)}")

#     # 5) 把剩下的 LoRA 權重載入 PEFT wrapper（strict=False：允許 missing keys，但我們已經過濾掉不想載的）
#     load_result = model.load_state_dict(filtered_state, strict=False)
#     print(
#         f"[Model] Adapter loaded. Missing keys: {len(load_result.missing_keys)}, "
#         f"Unexpected keys: {len(load_result.unexpected_keys)}"
#     )
#     if len(load_result.unexpected_keys) > 0:
#         print("[WARN] There are unexpected keys in adapter state_dict. Examples:", load_result.unexpected_keys[:10])
#     if len(load_result.missing_keys) > 0:
#         print("[INFO] Some keys missing (expected for LoRA):", load_result.missing_keys[:10])

#     # 6) final checks
#     print(model.print_trainable_parameters())
#     model.eval()
#     return model, tokenizer


# ====== 讀 eval 檔 ======
def load_eval_data(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    data: List[Dict[str, Any]] = []
    if p.suffix == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
    else:
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            data = obj
        else:
            data = [obj]

    print(f"[Data] Loaded {len(data)} samples from {path}")
    return data


# ====== 建 prompt（跟 SFT 的 build_prompt 對齊） ======
def build_prompt(birth_info: Dict[str, Any], rag_context: str) -> str:
    """
    - 前面接 SYSTEM_INTRO
    - <INPUT> 區塊只放出生 / 性別 / 時辰_index
    """
    global SYSTEM_INTRO

    birth_date = birth_info.get("生日", "Unknown")
    gender = birth_info.get("性別", "Unknown")
    time_index = birth_info.get("時辰_index", "Unknown")

    rag_block = ""
    if rag_context:
        rag_block = (
            "以下 <RAG_CONTEXT> 是從紫微斗數相關經典中擷取的內容，請作為參考輔助解讀，"
            "理解其中的規則與觀念，但不要逐字抄寫。\n\n"
            f"<RAG_CONTEXT>\n{rag_context}\n</RAG_CONTEXT>\n\n"
        )

    input_block = (
        "<|im_start|>user\n"
        f"出生: {birth_date}\n"
        f"性別: {gender}\n"
        f"時辰_index: {time_index}\n"
        "<|im_end|>\n"
    )

    SYSTEM_INTRO_ = (
        "<|im_start|>system\n"
        f"{SYSTEM_INTRO}"
        "<|im_end|>\n"
    )

    return (SYSTEM_INTRO_ or "") + rag_block + input_block + "<|im_start|>assistant\n" + "<think>\n\n</think>"



# ====== 主推論函式 ======
# @torch.no_grad()
# def generate_one(
#     model,
#     tokenizer,
#     retriever: Optional[RAGRetriever],s
#     sample: Dict[str, Any],
#     args,
# ):
#     birth_info = sample.get("出生資料", {}) or {}
#     question = sample.get("問題", sample.get("question", "")) or ""

#     # RAG query 只影響檢索，不進 <INPUT>
#     rag_context = ""
#     if retriever is not None:
#         rag_query = (
#             f"出生日期: {birth_info.get('生日', '')}；"
#             f"性別: {birth_info.get('性別', '')}；"
#             f"時辰_index: {birth_info.get('時辰_index', '')}；"
#             f"問題: {question}"
#         )
#         rag_context = retriever.build_context_block(
#             rag_query,
#             top_k=args.top_k,
#             include_chart=args.include_chart,
#         )

#     prompt = build_prompt(birth_info, rag_context)

#     inputs = tokenizer(prompt, return_tensors="pt")
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}
#     input_len = inputs["input_ids"].shape[1]
#     print(f"[DEBUG] input_len tokens: {input_len}, max_new_tokens: {args.max_new_tokens}")

#     # 把 <END_ANSWER> 也當成一個 EOS
#     end_answer_id = tokenizer.convert_tokens_to_ids("<END_ANSWER>")
#     eos_ids = [tokenizer.eos_token_id]
#     if end_answer_id is not None and end_answer_id != tokenizer.eos_token_id:
#         eos_ids.append(end_answer_id)
    
#     # end_answer_id = tokenizer.convert_tokens_to_ids("</OUTPUT>")
#     # eos_ids = [tokenizer.eos_token_id]
#     # if end_answer_id is not None and end_answer_id != tokenizer.eos_token_id:
#     #     eos_ids.append(end_answer_id)

#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=args.max_new_tokens,
#         do_sample=False,
#         temperature=args.temperature,
#         top_p=args.top_p,
#         eos_token_id=eos_ids,  # ✅ 同時遇到任一個 EOS 都會停
#     )
#     full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     if full_text.startswith(prompt):
#         model_output = full_text[len(prompt):].lstrip()
#     else:
#         model_output = full_text

#     pred_chart_str, pred_interp_str = split_chart_and_interpretation(model_output)

#     return model_output, pred_chart_str, pred_interp_str, rag_context, prompt


# ====== 從模型輸出中切出命盤 JSON & 解讀 ======
def split_chart_and_interpretation(model_output: str) -> Tuple[str, str]:
    """
    格式：
      <Natal Chart>
      {完整命盤 JSON}
      {後面直接接著文字解讀}
    """
    pos = model_output.find("<Natal Chart>")
    if pos != -1:
        tail = model_output[pos + len("<Natal Chart>"):].lstrip()
    else:
        tail = model_output

    first_brace = tail.find("{")
    last_brace = tail.rfind("}")

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        chart_str = tail[first_brace:last_brace + 1].strip()
        interp_str = tail[last_brace + 1:].strip()
    else:
        chart_str = ""
        interp_str = tail.strip()

    return chart_str, interp_str


def parse_chart_json(chart_str: str) -> Optional[Dict[str, Any]]:
    chart_str = chart_str.strip()
    if not chart_str:
        return None
    try:
        return json.loads(chart_str)
    except Exception:
        return None

@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    retriever: Optional[RAGRetriever],
    samples: List[Dict[str, Any]],
    args,
):
    """
    多筆一起做 inference，回傳和 generate_one 類似的結果列表。
    每個元素是:
      (model_output, pred_chart_str, pred_interp_str, rag_context, prompt)
    """
    prompts: List[str] = []
    rag_contexts: List[str] = []

    for sample in samples:
        birth_info = sample.get("出生資料", {}) or {}
        question = sample.get("問題", sample.get("question", "")) or ""

        # RAG query 只影響檢索，不進 <INPUT>
        rag_context = ""
        if retriever is not None:
            rag_query = (
                f"出生日期: {birth_info.get('生日', '')}；"
                f"性別: {birth_info.get('性別', '')}；"
                f"時辰_index: {birth_info.get('時辰_index', '')}；"
                f"問題: {question}"
            )
            rag_context = retriever.build_context_block(
                rag_query,
                top_k=args.top_k,
                include_chart=args.include_chart,
            )

        prompt = build_prompt(birth_info, rag_context)
        prompts.append(prompt)
        rag_contexts.append(rag_context)

    # tokenizer 對 list 做 batch encode
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
    )

    # 逐筆 decode & 切命盤 + 解讀
    results = []
    for i, prompt in enumerate(prompts):
        full_text = tokenizer.decode(outputs[i], skip_special_tokens=True)

        model_output = full_text[len(prompt):].lstrip()


        # if full_text.startswith(prompt):
        #     model_output = full_text[len(prompt):].lstrip()
        # else:
        #     model_output = full_text
        # print(full_text)

        pred_chart_str, pred_interp_str = split_chart_and_interpretation(model_output)
        results.append(
            (model_output, pred_chart_str, pred_interp_str, rag_contexts[i], prompt)
        )

    return results

# ====== 主程式 ======
def main():
    args = parse_args()

    load_system_intro()

    # 1. 資料
    data = load_eval_data(args.eval_file)
    if args.max_samples > 0:
        data = data[: args.max_samples]
    print(f"[Data] Running inference on {len(data)} samples")

    # 2. 模型
    model, tokenizer = load_model_and_tokenizer(
        base_model_name=args.base_model,
        peft_model_path=args.peft_model_path,
        use_4bit=args.use_4bit,
    )

    # 3. RAG retriever（可關閉）
    retriever = RAGRetriever() if args.use_rag else None
    if retriever is None:
        print("[RAG] Disabled for this run.")
    else:
        print("[RAG] Enabled.")

    # 4. 準備輸出檔 (jsonl，一筆一行，邊跑邊寫)
    f_out = None
    out_path = None
    if args.save_predictions:
        out_path = Path(args.save_predictions)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        f_out = out_path.open("w", encoding="utf-8")
        print(f"[Save] Writing predictions to {out_path} (streaming jsonl)")

    batch_size = max(1, args.batch_size)
    total = len(data)

    # 5. 逐 batch 推論
    for start in tqdm(range(0, total, batch_size), desc="Inference"):
        end = min(start + batch_size, total)
        batch_samples = data[start:end]

        # 實際推論
        batch_results = generate_batch(
            model, tokenizer, retriever, batch_samples, args
        )

        # 對 batch 內每一筆建立 ground_truth + prediction，立刻寫入一行 jsonl
        for sample, (model_output, pred_chart_str, pred_interp, rag_ctx, prompt) in zip(
            batch_samples, batch_results
        ):
            # Ground truth：保留命盤 + 解讀
            gold_chart = sample.get("命盤", None)
            gold_interp = sample.get(
                "解讀",
                sample.get("answer", sample.get("interpretation", ""))
            ) or ""

            # ✅ Prediction：只存「整段模型輸出」，先不區分命盤 / 解讀
            raw_output = (model_output or "").strip()

            record = {
                "ground_truth": {
                    "命盤": gold_chart,
                    "解讀": gold_interp,
                },
                "prediction": raw_output,  # ← 只是一個字串
            }

            if f_out is not None:
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()



if __name__ == "__main__":
    main()
