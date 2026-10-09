import math
import re
from typing import List, Dict, Any
from config import Config

class BM25:
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self.doc_len = [len(self._tokenize(doc)) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(corpus) if corpus else 1.0
        self.doc_freqs = []
        self.idf = {}
        self._build()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def _build(self):
        df = {}
        for doc in self.corpus:
            freqs = {}
            for t in self._tokenize(doc):
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_freqs.append(freqs)
            for t in set(freqs.keys()):
                df[t] = df.get(t, 0) + 1
        num_docs = len(self.corpus)
        for t, f in df.items():
            self.idf[t] = math.log((num_docs - f + 0.5) / (f + 0.5) + 1)

    def get_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> List[float]:
        tokens = self._tokenize(query)
        scores = [0.0] * len(self.corpus)
        for i, freqs in enumerate(self.doc_freqs):
            doc_l = self.doc_len[i]
            s = 0.0
            for t in tokens:
                if t in freqs:
                    f = freqs[t]
                    idf = self.idf.get(t, 0.0)
                    num = f * (k1 + 1)
                    den = f + k1 * (1 - b + b * (doc_l / self.avgdl))
                    s += idf * (num / den)
            scores[i] = s
        max_s = max(scores) if scores and max(scores) > 0 else 1.0
        return [s / max_s for s in scores]

class Embedder:
    def __init__(self):
        self.provider = Config.EMBEDDING_PROVIDER
        self.st_model = None
        if self.provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                self.st_model = SentenceTransformer(Config.SENTENCE_TRANSFORMER_MODEL)
            except Exception:
                self.provider = "fallback"

    def encode(self, text: str) -> List[float]:
        if self.provider == "sentence-transformers" and self.st_model:
            return self.st_model.encode(text).tolist()
        
        dim = 128
        tokens = re.findall(r'\w+', text.lower())
        vec = [0.0] * dim
        if not tokens:
            return vec
        for token in tokens:
            idx = abs(hash(token)) % dim
            vec[idx] += 1.0
            for i in range(len(token) - 1):
                ngram_idx = abs(hash(token[i:i+2])) % dim
                vec[ngram_idx] += 0.5
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm > 0 else vec

class HybridRetrievalEngine:
    def __init__(self, storage, embedder: Embedder):
        self.storage = storage
        self.embedder = embedder

    def recall(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        active_memories = self.storage.get_active_memories()
        if not active_memories:
            return []

        query_emb = self.embedder.encode(query)
        corpus = [m["text"] for m in active_memories]
        
        bm25 = BM25(corpus)
        bm25_scores = bm25.get_scores(query)

        scored_results = []
        for idx, mem in enumerate(active_memories):
            dot = sum(a * b for a, b in zip(query_emb, mem["embedding"]))
            norm_q = math.sqrt(sum(a * a for a in query_emb))
            norm_m = math.sqrt(sum(b * b for b in mem["embedding"]))
            cosine_score = (dot / (norm_q * norm_m)) if norm_q > 0 and norm_m > 0 else 0.0

            bm_s = bm25_scores[idx]
            hybrid_score = (Config.DENSE_WEIGHT * cosine_score) + (Config.BM25_WEIGHT * bm_s)
            
            importance_boost = 1.0 + (mem.get("importance_score", 3.0) - 3.0) * 0.05
            final_score = round(min(1.0, hybrid_score * importance_boost), 4)

            if final_score >= Config.SIMILARITY_THRESHOLD:
                scored_results.append({
                    "id": mem["id"],
                    "text": mem["text"],
                    "type": mem["type"],
                    "score": final_score,
                    "importance_score": mem["importance_score"],
                    "timestamp": mem["timestamp"]
                })

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]