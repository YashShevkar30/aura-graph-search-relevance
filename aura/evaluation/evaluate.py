"""
Full Evaluation Pipeline
=========================
Evaluates the hybrid ranker on held-out queries using
Precision@K, Recall@K, MRR, and NDCG@K.
"""
import json
import numpy as np
import networkx as nx
import pandas as pd
from loguru import logger
from aura.config import config
from aura.indexing.node2vec import Node2VecEmbedder
from aura.indexing.tfidf import TFIDFIndex
from aura.ranking.hybrid_ranker import HybridRanker
from aura.evaluation.metrics import precision_at_k, recall_at_k, mrr, ndcg_at_k


def run_evaluation():
    proc = config.DATA_PROCESSED
    labels = np.load(proc / "labels.npy")
    meta = pd.read_csv(proc / "metadata.csv")
    titles = meta["title"].tolist()

    # Load graph and rebuild embedder
    G = nx.read_edgelist(str(proc / "graph.edgelist"), nodetype=int)
    n2v = Node2VecEmbedder()
    embeddings = np.load(proc / "node2vec_embeddings.npy")

    # Rebuild TF-IDF
    tfidf = TFIDFIndex.load(config.MODEL_DIR / "tfidf_index.pkl")

    ranker = HybridRanker(n2v=n2v, tfidf=tfidf)

    # Evaluate: for each query node, ground truth = same-class nodes
    k_values = [5, 10, 20]
    n_queries = min(200, len(labels))
    query_nodes = np.random.RandomState(42).choice(len(labels), n_queries, replace=False)

    results = {}
    for k in k_values:
        p_scores, r_scores, mrr_scores, ndcg_scores = [], [], [], []

        for qnode in query_nodes:
            query_label = labels[qnode]
            relevant = set(np.where(labels == query_label)[0]) - {qnode}
            query_text = titles[qnode]

            search_results = ranker.search(query_text, query_node=int(qnode), top_k=k)
            retrieved = [r["node_id"] for r in search_results]

            p_scores.append(precision_at_k(retrieved, relevant, k))
            r_scores.append(recall_at_k(retrieved, relevant, k))
            mrr_scores.append(mrr(retrieved, relevant))
            ndcg_scores.append(ndcg_at_k(retrieved, relevant, k))

        results[f"precision@{k}"] = round(float(np.mean(p_scores)), 4)
        results[f"recall@{k}"] = round(float(np.mean(r_scores)), 4)
        results[f"mrr@{k}"] = round(float(np.mean(mrr_scores)), 4)
        results[f"ndcg@{k}"] = round(float(np.mean(ndcg_scores)), 4)

    results["n_queries"] = n_queries

    report_dir = config.PROJECT_ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    with open(report_dir / "evaluation_report.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info("SEARCH EVALUATION REPORT")
    logger.info(f"{'='*50}")
    for m, v in results.items():
        logger.info(f"  {m}: {v}")

    return results


if __name__ == "__main__":
    run_evaluation()
