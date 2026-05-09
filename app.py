# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from retrieval.hybrid import HybridRetriever
#from retrieval.hybrid_sample import HybridRetriever
#use this sample retreiver on low power devices for sample testing
import time

app = Flask(__name__)
CORS(app)  # allows frontend to call it

retriever = HybridRetriever()

@app.route('/search')
def search():
    query = request.args.get('q')
    
    t1 = time.time()
    bm25_results = retriever.bm25.search(query, top_k=10)
    bm25_lat = round((time.time() - t1) * 1000)

    t2 = time.time()
    dense_results = retriever.dense.search(query, top_k=10)
    dense_lat = round((time.time() - t2) * 1000)

    hybrid_results = retriever.search(query, top_k=10)

    return jsonify({
        'bm25': bm25_results,
        'dense': dense_results,
        'hybrid': hybrid_results,
        'latency': {'bm25': bm25_lat, 'dense': dense_lat}
    })

@app.route('/stats')
def stats():
    return jsonify({
        'corpus_size': len(retriever.bm25.passages)
    })

if __name__ == "__main__":
    app.run(port=5000)