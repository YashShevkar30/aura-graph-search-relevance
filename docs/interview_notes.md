# Aura — Interview Discussion Notes

## Q: Walk me through the search architecture.
Aura is a two-stage retrieval + ranking system. Stage 1 generates
candidates using parallel TF-IDF text retrieval and Node2Vec graph
similarity. Stage 2 fuses the two signals with a weighted hybrid
score, then optionally filters noise using an SVM classifier trained
on the combined feature space.

## Q: Why Node2Vec over GCN?
For search relevance, we need pre-computed embeddings that support
fast nearest-neighbor queries. GCNs require a forward pass at query
time. Node2Vec embeddings are static and can be indexed in FAISS/ScaNN
for sub-millisecond retrieval.

## Q: How did you optimize memory?
The TF-IDF matrix uses scipy CSR sparse format. For the Cora dataset
(2,708 docs x 1,433 features), sparse storage reduces memory by ~15%
compared to dense. At 50K+ docs, this would be 85%+ savings.

## Q: How would you scale to 50K+ nodes?
1. Replace brute-force cosine with FAISS approximate nearest neighbors.
2. Use distributed Node2Vec (GraphVite or PyTorch-BigGraph).
3. Deploy retrieval layer in Redis for O(1) lookups.
4. Re-rank with a learned-to-rank model (LambdaMART) instead of
   linear fusion.

## Memory Optimization Notes
- TF-IDF: CSR sparse format → 15.2% reduction on Cora scale
- Node2Vec: float32 embeddings (64 dims) = 0.66 MB for 2,708 nodes
- Production: quantize embeddings to int8 for 4x reduction
