import json
import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class DenseRetriever:
    def __init__(self, passages_path="data/passages.json"):
        print("Loading model...")
        self.model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        self.dimension = 384
        self.passages = self._load_passages(passages_path)
        self.index = None
        self.index_path = "data/faiss_index.bin"
        self.embeddings_path = "data/embeddings.npy"

    def _load_passages(self, path):
        print("Loading passages...")
        with open(path, "r") as f:
            return json.load(f)

    def build_index(self, batch_size=512):
        print("Encoding passages with MiniLM...")
        texts = [p["text"] for p in self.passages]
        
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            all_embeddings.append(embeddings)

        all_embeddings = np.vstack(all_embeddings).astype(np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(all_embeddings)
        
        # Save embeddings
        np.save(self.embeddings_path, all_embeddings)
        print(f"✅ Embeddings saved: {all_embeddings.shape}")

        # Build FAISS index
        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product = cosine
        self.index.add(all_embeddings)
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        print(f"✅ FAISS index saved: {self.index.ntotal} vectors")

    def load_index(self):
        print("Loading FAISS index from disk...")
        self.index = faiss.read_index(self.index_path)
        print(f"✅ FAISS index loaded: {self.index.ntotal} vectors")

    def search(self, query, top_k=100):
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype(np.float32)
        
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for rank, (idx, score) in enumerate(
            zip(indices[0], scores[0])
        ):
            results.append({
                "id": self.passages[idx]["id"],
                "text": self.passages[idx]["text"],
                "score": float(score),
                "rank": rank + 1
            })
        return results


if __name__ == "__main__":
    retriever = DenseRetriever()

    if not os.path.exists("data/faiss_index.bin"):
        retriever.build_index(batch_size=64)
    else:
        retriever.load_index()

    # Test it
    query = "what is machine learning"
    results = retriever.search(query, top_k=5)

    print(f"\n🔍 Query: '{query}'")
    print("-" * 60)
    for r in results:
        print(f"Rank {r['rank']} (score: {r['score']:.4f})")
        print(f"  {r['text'][:150]}...")
        print()