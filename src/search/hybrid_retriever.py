import numpy as np
from collections import defaultdict
from src.search.embedder import BGEM3Embedder

class QdrantHybridRetriever:
    def __init__(self, embedder: BGEM3Embedder, connection_str=":memory:"):
        self.embedder = embedder
        self.connection_str = connection_str
        self.use_mock_db = (connection_str == ":memory:")
        self.documents = {}  # doc_id -> metadata & text
        self.dense_index = {}  # doc_id -> dense_vec
        self.sparse_index = {}  # doc_id -> sparse_weights

    def index_document(self, doc_id: str, text: str, metadata: dict):
        dense_vec = self.embedder.embed_dense(text)
        sparse_weights = self.embedder.embed_sparse(text)
        
        self.documents[doc_id] = {
            "text": text,
            "metadata": metadata
        }
        self.dense_index[doc_id] = dense_vec
        self.sparse_index[doc_id] = sparse_weights

    def rrf(self, dense_results: list, sparse_results: list, k=60) -> list:
        # Reciprocal Rank Fusion
        scores = defaultdict(float)
        
        # Rank by dense scores (higher is better)
        dense_ranked = sorted(dense_results, key=lambda x: x[1], reverse=True)
        for rank, (doc_id, _) in enumerate(dense_ranked):
            scores[doc_id] += 1.0 / (k + (rank + 1))
            
        # Rank by sparse scores
        sparse_ranked = sorted(sparse_results, key=lambda x: x[1], reverse=True)
        for rank, (doc_id, _) in enumerate(sparse_ranked):
            scores[doc_id] += 1.0 / (k + (rank + 1))
            
        # Sorted final scores
        final_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return final_scores

    def search_dense(self, query_vec: np.ndarray, filters: dict) -> list:
        results = []
        for doc_id, doc_vec in self.dense_index.items():
            # Apply metadata filters
            doc_metadata = self.documents[doc_id]["metadata"]
            matched = True
            for k, v in filters.items():
                if doc_metadata.get(k) != v:
                    matched = False
                    break
            
            if not matched:
                continue
                
            # Cosine similarity
            dot = np.dot(query_vec, doc_vec)
            norm_q = np.linalg.norm(query_vec)
            norm_d = np.linalg.norm(doc_vec)
            sim = dot / (norm_q * norm_d) if norm_q > 0 and norm_d > 0 else 0.0
            results.append((doc_id, float(sim)))
            
        return results

    def search_sparse(self, query_weights: dict, filters: dict) -> list:
        results = []
        for doc_id, doc_weights in self.sparse_index.items():
            doc_metadata = self.documents[doc_id]["metadata"]
            matched = True
            for k, v in filters.items():
                if doc_metadata.get(k) != v:
                    matched = False
                    break
            
            if not matched:
                continue
                
            # Dot product of sparse overlap
            score = 0.0
            for term, q_weight in query_weights.items():
                if term in doc_weights:
                    score += q_weight * doc_weights[term]
            results.append((doc_id, score))
            
        return results

    def query(self, query_text: str, filters: dict = {}, limit: int = 5) -> list:
        query_vec = self.embedder.embed_dense(query_text)
        query_weights = self.embedder.embed_sparse(query_text)
        
        dense_res = self.search_dense(query_vec, filters)
        sparse_res = self.search_sparse(query_weights, filters)
        
        fused = self.rrf(dense_res, sparse_res)
        
        # Map back to documents metadata and raw text
        output = []
        for doc_id, score in fused[:limit]:
            output.append({
                "doc_id": doc_id,
                "text": self.documents[doc_id]["text"],
                "metadata": self.documents[doc_id]["metadata"],
                "rrf_score": score
            })
        return output

if __name__ == "__main__":
    embedder = BGEM3Embedder(use_mock=True)
    retriever = QdrantHybridRetriever(embedder)
    retriever.index_document("doc1", "Senior PyTorch developer located in SF", {"location": "SF"})
    retriever.index_document("doc2", "Frontend React JS developer located in NY", {"location": "NY"})
    
    res = retriever.query("Need PyTorch expert", {"location": "SF"})
    print("Search Result:", res)
