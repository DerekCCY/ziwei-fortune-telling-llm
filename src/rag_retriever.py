# rag_retriever.py
import os
import json
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = "rag_index"
INDEX_PATH = os.path.join(INDEX_DIR, "rag_index.faiss")
META_PATH = os.path.join(INDEX_DIR, "rag_index_meta.json")


class RAGRetriever:
    def __init__(self, index_path=INDEX_PATH, meta_path=META_PATH):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file not found at {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.docs = meta["docs"]
        embed_model_name = meta.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

        print(f"[RAG] Loading embedding model: {embed_model_name}")
        self.embedder = SentenceTransformer(embed_model_name)

        print(f"[RAG] Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)

    def retrieve(self, query: str, top_k: int = 5, include_chart: bool = True):
        """給 query，回傳 top_k docs。可以選擇是否包含 chart。"""
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(q_emb, top_k * 2)  # 多拿一點再過濾
        idxs = I[0]
        dists = D[0]

        results = []
        for idx, dist in zip(idxs, dists):
            if idx < 0:
                continue
            doc = self.docs[int(idx)]
            if not include_chart and doc.get("block_type") == "chart":
                continue
            results.append({
                "score": float(dist),
                "id": doc.get("id"),
                "big_title": doc.get("big_title"),
                "mid_title": doc.get("mid_title"),
                "small_title": doc.get("small_title"),
                "block_type": doc.get("block_type"),
                "source_file": doc.get("source_file"),
                "text": doc.get("text"),
            })
            if len(results) >= top_k:
                break
        return results

    def build_context_block(self, query: str, top_k: int = 5, include_chart: bool = False) -> str:
        """
        把檢索結果變成可以塞進 prompt 的文字區塊。
        預設不帶 chart（只用文字規則），需要時可以開 include_chart=True。
        """
        hits = self.retrieve(query, top_k=top_k, include_chart=include_chart)
        if not hits:
            return ""

        lines = []
        for h in hits:
            title_parts = []
            if h["big_title"]:
                title_parts.append(h["big_title"])
            if h["mid_title"]:
                title_parts.append(h["mid_title"])
            if h["small_title"]:
                title_parts.append(h["small_title"])

            header = " / ".join(title_parts)
            header = f"[{h['block_type']}] {header}" if header else f"[{h['block_type']}]"

            lines.append(header)
            lines.append(h["text"])
            lines.append("")

        return "\n".join(lines).strip()
