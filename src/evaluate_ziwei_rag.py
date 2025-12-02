
# python ./src/evaluate_ziwei_rag.py --peft_model_path results/checkpoint-477 --eval_file data/test_data/test_data.jsonl --max_samples 3 --save_predictions results/eval/test_results_norag.jsonl
import os
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from bert_score import score as bertscore_score

from rag_retriever import RAGRetriever  # 需和 rag_retriever.py 放一起


# ====== 紫微星曜名稱（用繁體作為「標準型」） ======
STAR_NAMES = [
    "紫微", "天機", "太陽", "武曲", "天同", "廉貞",
    "天府", "太陰", "貪狼", "巨門", "天相", "天梁", "七殺", "破軍",
    "文昌", "文曲", "左輔", "右弼", "天魁", "天鉞",
    "祿存", "擎羊", "陀羅", "火星", "鈴星",
]

# 簡體 -> 繁體的對照表（可再補）
SIM_TO_TRAD = {
    "太阳": "太陽",
    "太阴": "太陰",
    "贪狼": "貪狼",
    "廉贞": "廉貞",
    "七杀": "七殺",
    "天机": "天機",
    "巨门": "巨門",
    "破军": "破軍",
    "禄": "祿",
    "禄存": "祿存",
    "铃星": "鈴星",
}


def normalize_to_trad(text: str) -> str:
    """把常見星曜簡體轉成繁體，方便統一比對。"""
    for sim, trad in SIM_TO_TRAD.items():
        text = text.replace(sim, trad)
    return text


# ====== 解析 arguments ======
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Ziwei model with/without RAG")

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
        default="Ziwei-Doushu-0.5B-SFT",
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
        help="Whether to use RAG for this evaluation run",
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
        default="/home/ec2-user/ziwei-fortune-telling-llm/data/test_data/test_data.jsonl",
        required=True,
        help="Path to eval JSON / JSONL file",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Max number of samples to evaluate (-1 = all)",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument(
        "--save_predictions",
        type=str,
        default="results/eval/test_results.jsonl",
        help="If set, save per-sample predictions & metrics to this JSONL file",
    )
    return parser.parse_args()


# ====== 載模型 ======
def load_model_and_tokenizer(base_model_name: str, peft_model_path: str, use_4bit: bool):
    print(f"[Model] Loading base model: {base_model_name}")

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    print(f"[Model] Loading PEFT LoRA from: {peft_model_path}")
    model = PeftModel.from_pretrained(base, peft_model_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


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

def load_system_intro(path: str) -> str:
    """讀取 sft_prompt.txt 作為 system_intro。"""
    text = Path(path).read_text(encoding="utf-8")
    return text.strip()


# ====== 建 prompt（強制輸出格式） ======
def build_prompt(birth_info: Dict[str, Any], question: str, rag_context: str) -> str:
    birth_date = birth_info.get("生日", "Unknown")
    gender = birth_info.get("性別", "Unknown")
    time_index = birth_info.get("時辰_index", "Unknown")
    system_intro = load_system_intro("src/prompt/sft_prompt.txt")

    rag_block = ""
    if rag_context:
        rag_block = (
            "以下 <RAG_CONTEXT> 是從紫微斗數相關經典中擷取的內容，"
            "請作為參考輔助解讀，理解其中的規則與觀念，但不要逐字抄寫。\n\n"
            f"<RAG_CONTEXT>\n{rag_context}\n</RAG_CONTEXT>\n\n"
        )

    input_block = (
        "<INPUT>\n"
        f"出生: {birth_date}\n"
        f"性別: {gender}\n"
        f"時辰_index: {time_index}\n"
        f"問題: {question}\n"
        "</INPUT>\n\n"
    )

    output_instruction = (
        "請依照上述規則，只輸出 <Natal Chart> 和 <Interpretation> 兩個區塊。\n"
    )

    return system_intro + rag_block + input_block + output_instruction


# ====== 從模型輸出中抓出兩個部分 ======
def extract_tag_block(text: str, tag: str) -> str:
    pattern = rf"<{tag}>\s*(.+?)\s*</{tag}>"
    m = re.search(pattern, text, flags=re.S)
    if not m:
        return ""
    return m.group(1).strip()


def clean_json_block(s: str) -> str:
    """去掉 ```json / ``` 包皮，回傳乾淨 JSON 字串。"""
    s = s.strip()
    # 開頭的 ```json 或 ```
    s = re.sub(r"^```json\s*", "", s)
    s = re.sub(r"^```\s*", "", s)
    # 結尾的 ```
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def extract_first_blocks(model_output: str):
    text = model_output or ""

    natal_blocks = list(re.finditer(
        r"<Natal Chart>\s*(\{.*?\})\s*</Natal Chart>",
        text,
        flags=re.S
    ))
    interp_blocks = list(re.finditer(
        r"<Interpretation>\s*(.*?)\s*</Interpretation>",
        text,
        flags=re.S
    ))

    format_ok = (len(natal_blocks) == 1 and len(interp_blocks) == 1)

    natal_json_str = natal_blocks[0].group(1) if natal_blocks else ""
    interp_text = interp_blocks[0].group(1) if interp_blocks else ""

    return format_ok, natal_json_str, interp_text

def parse_chart_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    s = raw.strip()

    # 找第一個 { 到最後一個 }
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    s = s[start:end+1]

    try:
        data = json.loads(s)
    except Exception:
        return None

    return data


# ====== 星曜提及集合（從「文字」中抓） ======
def extract_star_set(text: str) -> set:
    text = normalize_to_trad(text or "")
    stars_found = set()
    for name in STAR_NAMES:
        if name in text:
            stars_found.add(name)
    return stars_found


# ====== 從「命盤 JSON」收集星曜集合 ======
def collect_chart_star_set(chart: Dict[str, Any]) -> set:
    if not chart:
        return set()

    stars = set()
    palaces = chart.get("命盤", {})
    for _, palace_data in palaces.items():
        # 訓練資料有的是 {"本命": {...}}，有的是直接 {...}
        node = palace_data.get("本命", palace_data)
        main_stars = node.get("主星", []) or []
        for s in main_stars:
            s = normalize_to_trad(str(s))
            stars.add(s)
    return stars

def normalize_chart_schema(chart: Dict[str, Any]) -> Dict[str, Any]:
    """
    把 model 的命盤 JSON 整理成接近 gold 的樣子：
    - 只保留 12 宮
    - 宮位內若是 {主星, 化曜}，包進 "本命"
    - 缺的欄位補上空結構
    """
    if not chart or "命盤" not in chart:
        return chart

    raw_mp = chart["命盤"]
    norm_mp = {}

    for palace, val in raw_mp.items():
        if palace not in CANON_PALACES:
            # 直接忽略 宗族/朋友/任權 等奇怪宮位
            continue

        node = val
        # case 1: 已經有 本命
        if "本命" in node:
            benming = node["本命"] or {}
        else:
            # case 2: 直接是 {主星, 化曜}
            stars = node.get("主星", [])
            huayao = node.get("化曜", [])
            benming = {
                "主星": stars,
                "化曜": huayao,
            }

        daxian = node.get("大限", {
            "範圍": [],
            "天干": "",
            "地支": "",
        })
        liunian = node.get("流年", {
            "對應年齡": [],
        })

        norm_mp[palace] = {
            "本命": {
                "主星": benming.get("主星", []),
                "化曜": benming.get("化曜", []),
            },
            "大限": {
                "範圍": daxian.get("範圍", []),
                "天干": daxian.get("天干", ""),
                "地支": daxian.get("地支", ""),
            },
            "流年": {
                "對應年齡": liunian.get("對應年齡", []),
            },
        }

    chart["命盤"] = norm_mp
    return chart

# ====== 計算基本資料正確率 ======
def compute_basic_info_acc(
    gold_chart: Dict[str, Any],
    pred_chart: Dict[str, Any],
) -> float:
    if not gold_chart or not pred_chart:
        return 0.0

    gold = gold_chart.get("基本資料", {})
    pred = pred_chart.get("基本資料", {})

    # ⚠️ 這裡欄位名要跟你資料集一致
    keys = ["出生日期", "性別", "生肖", "命主", "身主", "五行局"]

    correct = 0
    total = 0
    for k in keys:
        if k in gold:
            total += 1
            if k in pred and gold[k] == pred[k]:
                correct += 1

    return correct / total if total > 0 else 0.0


# ====== 主推論函式 ======
@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    retriever: Optional[RAGRetriever],
    sample: Dict[str, Any],
    args,
):
    birth_info = sample.get("出生資料", {}) or {}
    question = sample.get("問題", sample.get("question", "")) or ""

    # RAG
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

    prompt = build_prompt(birth_info, question, rag_context)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=getattr(args, "max_input_length", 4096),  # 有就用，沒有可以拿掉
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

    # ---- 關鍵：只取新生成的 token ----
    input_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[0][input_len:]

    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    # 如果你想保留整條做 debug，也可以：
    # full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 用「生成出的文字」來抽 tag，而不是整個 prompt
    pred_chart_str = extract_tag_block(gen_text, "Natal Chart")
    pred_interp_str = extract_tag_block(gen_text, "Interpretation")

    # 回傳也建議改成 gen_text，而不是 full_text
    return gen_text, pred_chart_str, pred_interp_str, rag_context, prompt


# ====== 主程式 ======
def main():
    args = parse_args()

    # 1. 資料
    data = load_eval_data(args.eval_file)
    if args.max_samples > 0:
        data = data[: args.max_samples]

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

    # 4. 評估統計
    chart_exact_flags = []
    basic_info_scores = []
    chart_star_jaccards = []
    text_star_jaccards = []
    format_ok_flags = []

    gold_interps: List[str] = []
    pred_interps: List[str] = []

    per_sample_records: List[Dict[str, Any]] = []

    # 5. 遍歷樣本推論
    for idx, sample in enumerate(tqdm(data, desc="Evaluating")):
        gen_text, pred_chart_str, pred_interp_str, rag_context, prompt = generate_one(
            model, tokenizer, retriever, sample, args
        )
        
        birth_info = sample.get("出生資料", {}) or {}
        question = sample.get("問題", sample.get("question", "")) or ""

        # Ground truth
        gold_chart = sample.get("命盤", None)
        gold_interp = sample.get("解讀", sample.get("answer", sample.get("interpretation", ""))) or ""

        # 方便 debug：有沒有兩個區塊都產出
        format_ok = bool(pred_chart_str and pred_interp_str)
        format_ok_flags.append(1 if format_ok else 0)

        # 解析命盤 JSON
        pred_chart = parse_chart_json(pred_chart_str)

        # ---- 1. chart exact match + basic info + chart 星曜 Jaccard ----
        chart_exact = 0
        basic_info_acc = 0.0
        chart_star_jacc = 0.0

        gold_chart_star_set = set()
        pred_chart_star_set = set()

        if gold_chart is not None:
            gold_chart_star_set = collect_chart_star_set(gold_chart)
        if pred_chart is not None:
            pred_chart_star_set = collect_chart_star_set(pred_chart)

        if gold_chart is not None and pred_chart is not None:
            chart_exact = 1 if pred_chart == gold_chart else 0
            basic_info_acc = compute_basic_info_acc(gold_chart, pred_chart)

            if gold_chart_star_set or pred_chart_star_set:
                inter = len(gold_chart_star_set & pred_chart_star_set)
                union = len(gold_chart_star_set | pred_chart_star_set)
                chart_star_jacc = inter / union if union > 0 else 0.0
            else:
                chart_star_jacc = 1.0  # 兩邊都沒星，就當作 1
        else:
            chart_exact = 0
            basic_info_acc = 0.0
            chart_star_jacc = 0.0

        chart_exact_flags.append(chart_exact)
        basic_info_scores.append(basic_info_acc)
        chart_star_jaccards.append(chart_star_jacc)

        # ---- 2. star mention accuracy（用「解讀文字」的星曜集合做 Jaccard）----
        gold_text_star_set = extract_star_set(gold_interp)
        pred_text_star_set = extract_star_set(pred_interp_str)

        if gold_text_star_set or pred_text_star_set:
            inter = len(gold_text_star_set & pred_text_star_set)
            union = len(gold_text_star_set | pred_text_star_set)
            text_star_jacc = inter / union if union > 0 else 0.0
        else:
            text_star_jacc = 1.0
        text_star_jaccards.append(text_star_jacc)

        gold_interps.append(gold_interp)
        pred_interps.append(pred_interp_str)

        # ---- 每筆紀錄存成易讀 JSON 結構 ----
        per_sample_records.append({
            "index": idx,
            "birth_info": birth_info,
            "question": question,
            "gold": {
                "chart": gold_chart,
                "interpretation": gold_interp,
                "chart_star_set": sorted(list(gold_chart_star_set)),
                "text_star_set": sorted(list(gold_text_star_set)),
            },
            "prediction": {
                "chart_parsed": pred_chart,
                "chart_text": pred_chart_str,
                "interpretation": pred_interp_str,
                "chart_star_set": sorted(list(pred_chart_star_set)),
                "text_star_set": sorted(list(pred_text_star_set)),
                "model_output": gen_text,     # ★ 這裡改成純生成的文本
                # "prompt": prompt,           # 要 debug 再打開
            },
            "metrics": {
                "format_ok": bool(format_ok),
                "chart_exact_match": bool(chart_exact),
                "basic_info_acc": basic_info_acc,
                "chart_star_jaccard": chart_star_jacc,
                "text_star_jaccard": text_star_jacc,
            },
            "rag": {
                "used": retriever is not None,
                "top_k": args.top_k if retriever is not None else 0,
                "context": rag_context if retriever is not None else None,
            },
        })

    n = len(data)
    chart_exact_mean = float(np.mean(chart_exact_flags)) if chart_exact_flags else 0.0
    basic_info_mean = float(np.mean(basic_info_scores)) if basic_info_scores else 0.0
    chart_star_jacc_mean = float(np.mean(chart_star_jaccards)) if chart_star_jaccards else 0.0
    text_star_jacc_mean = float(np.mean(text_star_jaccards)) if text_star_jaccards else 0.0
    format_ok_mean = float(np.mean(format_ok_flags)) if format_ok_flags else 0.0

    print("\n===== Basic metrics =====")
    print(f"Samples                   : {n}")
    print(f"Format OK ratio           : {format_ok_mean:.4f}")
    print(f"Chart exact match         : {chart_exact_mean:.4f}")
    print(f"Basic info accuracy       : {basic_info_mean:.4f}")
    print(f"Chart star Jaccard (JSON) : {chart_star_jacc_mean:.4f}")
    print(f"Star Jaccard (from text)  : {text_star_jacc_mean:.4f}")

    # ---- 3. Cosine similarity over interpretations ----
    print("\n[Embedding] Computing cosine similarity with sentence-transformers ...")
    emb_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    emb_model = SentenceTransformer(emb_model_name)

    gold_emb = emb_model.encode(gold_interps, convert_to_numpy=True, batch_size=16, show_progress_bar=True)
    pred_emb = emb_model.encode(pred_interps, convert_to_numpy=True, batch_size=16, show_progress_bar=True)

    # normalize
    gold_norm = gold_emb / np.linalg.norm(gold_emb, axis=1, keepdims=True)
    pred_norm = pred_emb / np.linalg.norm(pred_emb, axis=1, keepdims=True)
    cos_sims = np.sum(gold_norm * pred_norm, axis=1)
    cos_sim_mean = float(np.mean(cos_sims))

    print(f"\nCosine similarity (interpretation embeddings): {cos_sim_mean:.4f}")

    # ---- 4. BERTScore ----
    print("\n[BERTScore] Computing BERTScore F1 (lang='zh') ...")
    P, R, F1 = bertscore_score(pred_interps, gold_interps, lang="zh")
    bert_f1 = float(F1.mean())
    print(f"BERTScore F1: {bert_f1:.4f}")

    # 把 cosine / BERTScore 填回每筆紀錄
    for i, rec in enumerate(per_sample_records):
        rec["cosine_similarity"] = float(cos_sims[i])
        rec["bertscore_f1"] = float(F1[i])
        # 也塞進 metrics 裡，方便一次看
        rec["metrics"]["cosine_similarity"] = float(cos_sims[i])
        rec["metrics"]["bertscore_f1"] = float(F1[i])

    print("\n===== Summary =====")
    print(f"Format OK ratio           : {format_ok_mean:.4f}")
    print(f"Chart exact match         : {chart_exact_mean:.4f}")
    print(f"Basic info accuracy       : {basic_info_mean:.4f}")
    print(f"Chart star Jaccard (JSON) : {chart_star_jacc_mean:.4f}")
    print(f"Star Jaccard (from text)  : {text_star_jacc_mean:.4f}")
    print(f"Cosine similarity         : {cos_sim_mean:.4f}")
    print(f"BERTScore F1              : {bert_f1:.4f}")

    # ---- 存成 JSONL ----
    if args.save_predictions:
        out_path = Path(args.save_predictions)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for rec in per_sample_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n[Save] Per-sample predictions saved to: {out_path}")


if __name__ == "__main__":
    main()