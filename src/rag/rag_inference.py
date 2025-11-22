import json
from pathlib import Path
import argparse

import configs

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from py_iztro import Astro

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def load_rag_index(emb_model_name):
    """Load FAISS index and metadata for RAG."""
    index = faiss.read_index(str(configs.RAG_INDEX_PATH))
    meta_path = configs.RAG_DIR / "docs_meta.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    emb_model = SentenceTransformer(emb_model_name)
    return index, meta, emb_model


def retrieve_docs(query_text, index, meta, emb_model, top_k=5):
    """Retrieve top-k relevant docs."""
    q_emb = emb_model.encode([query_text], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, top_k)
    docs = []
    for i in idxs[0]:
        docs.append(meta[i]["text"])
    return docs


def compute_chart(birth_date, hour_index, gender):
    """Compute Ziwei chart using py_iztro, return project schema."""
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


def build_inference_prompt(retrieved_docs, chart_json, user_question):
    """Build final prompt for the fine-tuned model."""
    refs = "\n\n".join(retrieved_docs)
    chart_str = json.dumps(chart_json, ensure_ascii=False)

    # System-style context in natural language (Chinese)
    prompt = f"""你是一位資深的紫微斗數老師，以下是可供參考的資料與規則說明（不需要逐條解釋，只要在解讀時適度參考）：

【紫微斗數參考資料】
{refs}

【此人的紫微斗數命盤資料（JSON 結構）】
{chart_str}

請根據以上命盤與參考資料，回答下面的問題，並依照你在訓練中學到的固定五段式格式輸出完整解讀。

【問題】：
{user_question}

請直接輸出解讀，不要解釋你使用了哪些規則，也不要重複貼出 JSON。
"""
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--birth", type=str, required=True, help="Birth date, e.g. 2001-01-02")
    parser.add_argument("--hour_index", type=int, required=True, help="Hour index (0-11)")
    parser.add_argument("--gender", type=str, required=True, help="性別：男 or 女")
    parser.add_argument("--question", type=str, default="請為我做一份完整的命理解讀，並特別說明未來十年的事業與感情。")
    parser.add_argument("--ft_model", type=str, required=True, help="Fine-tuned model name/path.")
    parser.add_argument("--emb_model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--max_new_tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print("[RAG-Infer] Computing Ziwei chart...")
    chart_json = compute_chart(args.birth, args.hour_index, args.gender)

    print("[RAG-Infer] Loading RAG index...")
    index, meta, emb_model = load_rag_index(args.emb_model)

    rag_query = f"紫微斗數 事業 感情 大限 官祿宮 夫妻宮 {args.question}"
    retrieved = retrieve_docs(rag_query, index, meta, emb_model, top_k=6)

    prompt = build_inference_prompt(retrieved, chart_json, args.question)

    print(f"[RAG-Infer] Loading fine-tuned model: {args.ft_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.ft_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.ft_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    print("[RAG-Infer] Generating interpretation...")
    out = gen(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=True,
        top_p=0.9
    )[0]["generated_text"]

    completion = out[len(prompt):].strip()
    print("\n================= 命理解讀 =================\n")
    print(completion)


if __name__ == "__main__":
    main()
