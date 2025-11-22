import json
from pathlib import Path
import argparse

import configs

import faiss
from sentence_transformers import SentenceTransformer
import numpy as np


def main():
    '''
    1. Step 1 — Build index (once)
        - Load your knowledge corpus (docs.jsonl)
        - Convert each doc into embeddings
        - Build FAISS index
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-m3",
        help="Embedding model name.",
    )
    args = parser.parse_args()

    docs_path = configs.RAG_DOCS_PATH
    index_path = configs.RAG_INDEX_PATH

    print(f"[RAG] Loading docs from {docs_path}")
    texts = []
    meta = []

    with open(docs_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            meta.append(obj)

    print(f"[RAG] Loaded {len(texts)} docs.")

    print(f"[RAG] Loading embedding model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    print("[RAG] Encoding documents...")
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    dim = emb.shape[1]

    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(emb)
    index.add(emb)

    faiss.write_index(index, str(index_path))

    meta_path = configs.RAG_DIR / "docs_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[RAG] Index saved to {index_path}")
    print(f"[RAG] Meta saved to {meta_path}")


if __name__ == "__main__":
    main()
