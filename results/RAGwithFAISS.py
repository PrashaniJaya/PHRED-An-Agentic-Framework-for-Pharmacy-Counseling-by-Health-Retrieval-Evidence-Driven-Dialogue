# rag_json_faiss.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError as e:
    raise RuntimeError(
        "faiss not installed. Run `pip install faiss-cpu` (or conda install -c pytorch faiss-cpu)."
    ) from e


class JSONRAGFaiss:
    def __init__(
        self,
        kb_file: str,
        answer_field: str = "answer_chatgpt",
        model_name: str = "multi-qa-mpnet-base-dot-v1",
        top_k: int = 3,
        threshold: float = 0.30,
        batch_size: int = 64,
        normalize: bool = True,
        index_path: str | None = None,  # optional save/load path
    ):
        """
        Builds a FAISS Inner-Product (IP) index. If normalize=True, IP == cosine similarity.
        """
        # 1) Load & format KB
        with open(kb_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.records = []
        for row in raw:
            q, a = (row.get("input", "").strip(), row.get(answer_field, "").strip())
            if q and a:
                self.records.append({"q": q, "a": a})

        self.texts = [f"Q: {r['q']}\nA: {r['a']}" for r in self.records]

        # 2) Embed
        self.model = SentenceTransformer(model_name)
        emb = self.model.encode(
            self.texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        ).astype("float32")

        # 3) FAISS index (IP). With L2-normalized vectors, IP ≈ cosine similarity.
        self.normalize = normalize
        if normalize:
            faiss.normalize_L2(emb)

        dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # exact search, inner product
        self.index.add(emb)

        self.embeddings = emb  # keep if you ever need it directly
        self.top_k = top_k
        self.threshold = threshold

        # Optional: save index
        self.index_path = index_path
        if index_path:
            faiss.write_index(self.index, index_path)

    def retrieve(self, query: str):
        # 4) Encode query (float32) and normalize if using cosine/IP
        q_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        if self.normalize:
            faiss.normalize_L2(q_vec)

        # 5) Search in FAISS (returns distances D and indices I)
        D, I = self.index.search(q_vec, self.top_k * 5)  # overfetch then filter by threshold
        D, I = D[0], I[0]  # unwrap single-query results

        # 6) Build results list with thresholding
        hits = []
        for idx, score in zip(I, D):
            if idx == -1:
                continue
            # with normalized vectors & IP, score ∈ [-1, 1] ~ cosine similarity
            if score >= self.threshold:
                hits.append((self.texts[idx], float(score)))

        # keep top_k after thresholding
        hits = sorted(hits, key=lambda x: x[1], reverse=True)[: self.top_k]
        return hits

    def answer(self, query: str):
        hits = self.retrieve(query)
        if not hits:
            return "No relevant entries found."
        return "\n---\n".join(f"{t}\n(score: {s:.2f})" for t, s in hits)

    @classmethod
    def load_from_index(
        cls,
        kb_file: str,
        index_path: str,
        answer_field: str = "answer_chatgpt",
        model_name: str = "multi-qa-mpnet-base-dot-v1",
        top_k: int = 3,
        threshold: float = 0.30,
        normalize: bool = True,
    ):
        """
        Load an existing FAISS index from disk and reuse the textual KB.
        (You still need the same embedding model to encode queries.)
        """
        with open(kb_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        obj = object.__new__(cls)
        obj.records = [
            {"q": r.get("input", "").strip(), "a": r.get(answer_field, "").strip()}
            for r in raw
            if r.get("input") and r.get(answer_field)
        ]
        obj.texts = [f"Q: {r['q']}\nA: {r['a']}" for r in obj.records]

        obj.model = SentenceTransformer(model_name)
        obj.index = faiss.read_index(index_path)
        obj.embeddings = None  # not needed when loaded
        obj.top_k = top_k
        obj.threshold = threshold
        obj.normalize = normalize
        obj.index_path = index_path
        return obj


if __name__ == "__main__":
    agent = JSONRAGFaiss(
        "meddialog.json",
        answer_field="answer_chatgpt",
        top_k=3,
        threshold=0.35,
        batch_size=64,
        normalize=True,
        index_path=None,  # set a path to persist, e.g. "kb.index"
    )
    q = "Difference between regular contraceptive pills and i-pill?"
    print(agent.answer(q))
