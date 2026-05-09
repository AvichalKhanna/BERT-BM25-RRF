import json
import pickle
import os
from rank_bm25 import BM25Okapi
from tqdm import tqdm

class BM25Retriever:
    def __init__(self, passages_path="data/passages.json"):
        self.passages = self._load_passages(passages_path)
        self.bm25 = None
        self.index_path = "data/bm25_index.pkl"

    def _load_passages(self, path):
        print("Loading passages...")
        with open(path, "r") as f:
            return json.load(f)

    def _tokenize(self, text):
        return text.lower().split()

    def build_index(self):
        print("Building BM25 index...")
        
        # Process in chunks to avoid MemoryError
        tokenized = []
        for i, p in enumerate(tqdm(self.passages)):
            tokenized.append(self._tokenize(p["text"]))
            if i % 50000 == 0 and i > 0:
                print(f"  Tokenized {i}/{len(self.passages)}...")
        
        self.bm25 = BM25Okapi(tokenized)
        
        with open(self.index_path, "wb") as f:
            pickle.dump(self.bm25, f)
        print(f"✅ BM25 index saved to {self.index_path}")
        
    def load_index(self):
        print("Loading BM25 index from disk...")
        with open(self.index_path, "rb") as f:
            self.bm25 = pickle.load(f)
        print("✅ BM25 index loaded")

    def search(self, query, top_k=100):
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top_k indices
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "id": self.passages[idx]["id"],
                "text": self.passages[idx]["text"],
                "score": float(scores[idx]),
                "rank": len(results) + 1
            })
        return results


if __name__ == "__main__":
    retriever = BM25Retriever()
    
    # Build index first time
    if not os.path.exists("data/bm25_index.pkl"):
        retriever.build_index()
    else:
        retriever.load_index()

    # Test it
    query = "what is machine learning"
    results = retriever.search(query, top_k=5)
    
    print(f"\n🔍 Query: '{query}'")
    print("-" * 60)
    for r in results:
        print(f"Rank {r['rank']} (score: {r['score']:.3f})")
        print(f"  {r['text'][:150]}...")
        print()