from retrieval.bm25_retreiver import BM25Retriever
from retrieval.dense_retriever_sample import DenseRetrieverSample  # changed

class HybridRetriever:
    def __init__(self):
        self.bm25 = BM25Retriever()
        self.bm25.load_index()
        
        self.dense = DenseRetrieverSample()  # changed
        self.dense.load_index()


    def _reciprocal_rank_fusion(
        self, 
        bm25_results, 
        dense_results, 
        k=60,                # RRF constant, 60 is standard
        bm25_weight=0.6,     # tunable
        dense_weight=0.4     # BERT weighted higher
    ):
        scores = {}
        texts = {}

        # Score BM25 results
        for result in bm25_results:
            doc_id = result["id"]
            rank = result["rank"]
            rrf_score = bm25_weight * (1 / (k + rank))
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            texts[doc_id] = result["text"]

        # Score Dense results
        for result in dense_results:
            doc_id = result["id"]
            rank = result["rank"]
            rrf_score = dense_weight * (1 / (k + rank))
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            texts[doc_id] = result["text"]

        # Sort by combined score
        sorted_docs = sorted(
            scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )

        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs):
            results.append({
                "id": doc_id,
                "text": texts[doc_id],
                "score": score,
                "rank": rank + 1
            })
        return results

    def search(self, query, top_k=10):
        # Get top 100 from each retriever
        bm25_results = self.bm25.search(query, top_k=100)
        dense_results = self.dense.search(query, top_k=100)

        # Fuse with RRF
        fused = self._reciprocal_rank_fusion(
            bm25_results, 
            dense_results
        )
        return fused[:top_k]


if __name__ == "__main__":
    retriever = HybridRetriever()

    # Test queries
    queries = [
        "what is machine learning",
        "symptoms of diabetes",
        "how does photosynthesis work",
        "best programming languages 2024"
    ]

    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 60)
        results = retriever.search(query, top_k=3)
        for r in results:
            print(f"Rank {r['rank']} (RRF score: {r['score']:.6f})")
            print(f"  {r['text'][:200]}...")
            print()