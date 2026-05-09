# Neural Information Retrieval Engine

Hybrid dense-sparse retrieval system combining BM25 
with a fine-tuned MiniLM bi-encoder and FAISS ANN search.

## Results
| Method | MRR@10 | Latency |
|--------|--------|---------|
| BM25 | 0.29 | ~18ms |
| BERT Dense | 0.34 | ~36ms |
| Hybrid RRF | 0.38 | ~48ms |

**31% improvement over BM25 baseline.**

## Setup
pip install -r requirements.txt
python data/download.py
python retrieval/bm25_retriever.py
python retrieval/dense_retriever.py
python app.py
start index.html with live server

## Live Demo
[link coming]"# BERT-BM25-RRF" 
