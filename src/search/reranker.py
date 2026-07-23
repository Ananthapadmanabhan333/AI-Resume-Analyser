import numpy as np

class BGEReranker:
    def __init__(self, model_name="BAAI/bge-reranker-large", device="cpu", use_mock=True):
        self.model_name = model_name
        self.device = device
        self.use_mock = use_mock
        self.model = None

        if not use_mock:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(model_name, device=device)
            except Exception as e:
                print(f"Failed to load sentence-transformers reranker: {e}. Falling back to mock Reranker.")
                self.use_mock = True

    def rerank(self, query: str, candidates: list, top_k: int = 5) -> list:
        # candidates should be a list of dicts with {"doc_id": ..., "text": ...}
        if not candidates:
            return []

        scored_candidates = []
        
        if self.use_mock:
            # Generate deterministic mock relevance scores based on query and text keyword overlaps
            q_words = set(query.lower().split())
            for c in candidates:
                c_words = set(c["text"].lower().split())
                overlap = len(q_words.intersection(c_words))
                # Add some random noise for realism
                noise = (hash(c["doc_id"]) % 100) / 1000.0
                score = float(overlap / len(q_words)) + noise if q_words else 0.0 + noise
                scored_candidates.append((c, score))
        else:
            # Evaluate using actual cross encoder
            pairs = [[query, c["text"]] for c in candidates]
            scores = self.model.predict(pairs)
            for idx, score in enumerate(scores):
                scored_candidates.append((candidates[idx], float(score)))

        # Sort by score descending
        sorted_candidates = sorted(scored_candidates, key=lambda x: x[1], reverse=True)
        
        # Build return objects with updated score
        results = []
        for c, score in sorted_candidates[:top_k]:
            c_copy = c.copy()
            c_copy["rerank_score"] = score
            results.append(c_copy)
            
        return results

if __name__ == "__main__":
    reranker = BGEReranker(use_mock=True)
    cands = [
        {"doc_id": "1", "text": "Machine Learning Engineer with PyTorch experience"},
        {"doc_id": "2", "text": "React Web Developer using TailwindCSS"}
    ]
    res = reranker.rerank("Query: PyTorch expert", cands)
    print("Reranked results:", res)
