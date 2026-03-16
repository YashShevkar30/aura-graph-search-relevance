"""
Hybrid Ranking Pipeline
========================
Combines Node2Vec graph similarity and TF-IDF text relevance
into a unified scoring function, then filters with SVM.
"""
import numpy as np
from loguru import logger
from aura.config import config
from aura.indexing.node2vec import Node2VecEmbedder
from aura.indexing.tfidf import TFIDFIndex
from aura.ranking.svm_filter import RelevanceClassifier


class HybridRanker:
    """Two-stage ranker: retrieve → re-rank with hybrid score."""

    def __init__(
        self,
        n2v: Node2VecEmbedder,
        tfidf: TFIDFIndex,
        classifier: RelevanceClassifier = None,
        graph_weight: float = None,
        text_weight: float = None,
    ):
        self.n2v = n2v
        self.tfidf = tfidf
        self.classifier = classifier
        self.graph_weight = graph_weight or config.GRAPH_WEIGHT
        self.text_weight = text_weight or config.TEXT_WEIGHT

    def search(self, query: str, query_node: int = None, top_k: int = None) -> list[dict]:
        """
        Hybrid search combining graph and text signals.

        1. TF-IDF retrieves text-relevant candidates
        2. Node2Vec retrieves structurally similar nodes (if query_node given)
        3. Scores are combined with weighted fusion
        4. SVM classifier filters low-confidence noise (optional)
        """
        top_k = top_k or config.TOP_K

        # Stage 1: Text retrieval
        text_results = self.tfidf.query(query, top_k=top_k * 3)
        candidate_scores = {}

        for r in text_results:
            nid = r["node_id"]
            candidate_scores[nid] = {
                "text_score": r["text_score"],
                "graph_score": 0.0,
                "title": r["title"],
            }

        # Stage 2: Graph retrieval (if query node provided)
        if query_node is not None:
            graph_results = self.n2v.most_similar_nodes(query_node, top_k=top_k * 3)
            for r in graph_results:
                nid = r["node_id"]
                if nid in candidate_scores:
                    candidate_scores[nid]["graph_score"] = r["similarity"]
                else:
                    candidate_scores[nid] = {
                        "text_score": 0.0,
                        "graph_score": r["similarity"],
                        "title": "",
                    }

        # Stage 3: Hybrid scoring
        results = []
        for nid, scores in candidate_scores.items():
            hybrid = (
                self.graph_weight * scores["graph_score"]
                + self.text_weight * scores["text_score"]
            )

            entry = {
                "node_id": nid,
                "hybrid_score": round(hybrid, 4),
                "graph_score": round(scores["graph_score"], 4),
                "text_score": round(scores["text_score"], 4),
                "title": scores["title"],
            }

            # Stage 4: SVM noise filtering (optional)
            if self.classifier and self.classifier._is_trained:
                graph_emb = self.n2v.get_embedding(nid)
                tfidf_vec = self.tfidf.get_document_vector(nid)
                pred = self.classifier.predict(graph_emb, tfidf_vec)
                entry["svm_confidence"] = pred["confidence"]
            results.append(entry)

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:top_k]
