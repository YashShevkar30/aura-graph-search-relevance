# Memory Optimization Report

## Measurements (Cora — 2,708 nodes)

| Component | Dense (MB) | Sparse/Optimized (MB) | Reduction |
|-----------|-----------|----------------------|-----------|
| TF-IDF Matrix | ~15.2 | ~12.9 | 15.2% |
| Node2Vec Embeddings | 0.66 | 0.66 (float32) | — |
| Graph (edgelist) | 0.12 | 0.12 | — |

## Production Scaling Estimates (50K nodes)

| Component | Dense | Sparse | Reduction |
|-----------|-------|--------|-----------|
| TF-IDF | ~1.9 GB | ~190 MB | ~90% |
| Embeddings | 12.2 MB | 3.1 MB (int8) | ~75% |

> Note: Local demo measurements. Production numbers are extrapolated
> based on sparsity patterns and data characteristics.
