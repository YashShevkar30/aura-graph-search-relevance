# Aura: Deep Graph Search Relevance Engine

[![CI](https://github.com/YashShevkar30/aura-graph-search-relevance/actions/workflows/ci.yml/badge.svg)](https://github.com/YashShevkar30/aura-graph-search-relevance/actions)

A production-grade graph-powered search engine combining **Node2Vec** structural
embeddings with **TF-IDF** text relevance and **SVM** noise filtering. Built to be
discussed in ML systems and search engineering interviews.

## Architecture

```
Query ──┬──▶ TF-IDF Retrieval (text relevance)  ──┐
        │                                          ├──▶ Hybrid Scorer ──▶ SVM Filter ──▶ Results
        └──▶ Node2Vec Retrieval (graph proximity) ──┘
               (biased random walks + Word2Vec)
```

## Key Results (Local Demo — Cora Citation Network)

| Metric | Value |
|--------|-------|
| Dataset | 2,708 nodes / 5,429 edges |
| SVM F1 (macro, CV) | Measured at training time |
| Memory reduction (sparse TF-IDF) | 15.2% |
| Node2Vec training | ~30s (200 walks x 30 length) |

> All metrics are local-demo on Cora (2,708 papers). See
> `docs/memory_analysis.md` for production scaling notes.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Graph Embeddings** | Node2Vec (Gensim Word2Vec) |
| **Text Relevance** | TF-IDF (scikit-learn, sparse CSR) |
| **Noise Filter** | SVM with RBF kernel |
| **Hybrid Scoring** | Weighted linear fusion |
| **API** | FastAPI |
| **CI** | GitHub Actions |

## Quick Start

```bash
pip install -r requirements.txt
make ingest     # Download Cora dataset
make index      # Build Node2Vec + TF-IDF indices
make train      # Train SVM classifier
make evaluate   # Run evaluation (P@K, MRR, NDCG)
make serve      # Start search API on :8000
```

## Project Structure

```
aura-graph-search-relevance/
├── aura/
│   ├── api/              # FastAPI search interface
│   ├── data/             # Ingestion (Cora dataset)
│   ├── evaluation/       # P@K, MRR, NDCG metrics
│   ├── indexing/         # Node2Vec + TF-IDF
│   └── ranking/          # SVM filter + hybrid ranker
├── tests/
├── docs/                 # Interview notes + memory analysis
├── reports/
├── Dockerfile
└── Makefile
```

## License
MIT License — Yash Shevkar
