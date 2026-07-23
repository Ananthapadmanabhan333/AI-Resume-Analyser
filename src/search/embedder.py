import numpy as np

class BGEM3Embedder:
    def __init__(self, model_name="BAAI/bge-m3", device="cpu", use_mock=True):
        self.model_name = model_name
        self.device = device
        self.use_mock = use_mock
        self.model = None
        
        if not use_mock:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name, device=device)
            except Exception as e:
                print(f"Failed to load sentence-transformers model: {e}. Falling back to Mock Embeddings.")
                self.use_mock = True

    def embed_dense(self, text: str) -> np.ndarray:
        if self.use_mock:
            # Generate a reproducible mock 1024-dim normalized vector
            state = np.random.RandomState(hash(text) % (2**32 - 1))
            vec = state.randn(1024)
            return vec / np.linalg.norm(vec)
        
        # Real sentence transformer dense embedding
        return self.model.encode(text, convert_to_numpy=True)

    def embed_sparse(self, text: str) -> dict:
        if self.use_mock:
            # Generate dummy lexical tokens and weight associations
            tokens = text.lower().split()
            lex_weights = {}
            for t in tokens:
                clean_t = "".join(ch for ch in t if ch.isalnum())
                if clean_t:
                    lex_weights[clean_t] = round(float(np.abs(np.sin(hash(clean_t)))), 4)
            return lex_weights
            
        # Return vocabulary weight associations using model tokenizers if required
        # For bge-m3 sentence transformers, sparse embeddings are returned under token embeddings
        # We simulate/simplify BGE sparse output dictionary
        tokens = self.model.tokenizer.tokenize(text)
        lex_weights = {}
        for token in tokens:
            lex_weights[token] = 1.0 # default weight
        return lex_weights

if __name__ == "__main__":
    embedder = BGEM3Embedder(use_mock=True)
    dense_vec = embedder.embed_dense("Sample Machine Learning Resume")
    sparse_map = embedder.embed_sparse("Sample Machine Learning Resume")
    print(f"Dense vector length: {len(dense_vec)}")
    print(f"Sparse map keys: {list(sparse_map.keys())[:5]}")
