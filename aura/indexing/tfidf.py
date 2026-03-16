"""
TF-IDF Text Relevance Index
============================
Builds a sparse TF-IDF index over node text metadata for
keyword-based retrieval. Uses sublinear TF and n-gram features.
"""
import numpy as np
import joblib
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
from aura.config import config


class TFIDFIndex:
    """Sparse TF-IDF index with cosine similarity retrieval."""

    def __init__(self, max_features: int = None, ngram_range: tuple = None):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features or config.TFIDF_MAX_FEATURES,
            ngram_range=ngram_range or config.TFIDF_NGRAM_RANGE,
            sublinear_tf=True,
            stop_words="english",
            min_df=2,
        )
        self.tfidf_matrix = None
        self._documents = []

    def fit(self, documents: list[str]):
        """Build TF-IDF index from document corpus."""
        self._documents = documents
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        logger.info(
            f"TF-IDF index built: {self.tfidf_matrix.shape[0]} docs, "
            f"{self.tfidf_matrix.shape[1]} features, "
            f"density={self.tfidf_matrix.nnz / np.prod(self.tfidf_matrix.shape):.4%}"
        )
        return self

    def query(self, query_text: str, top_k: int = 10) -> list[dict]:
        """Retrieve top-K documents by TF-IDF cosine similarity."""
        if self.tfidf_matrix is None:
            raise RuntimeError("Index not built. Call fit() first.")

        query_vec = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append({
                    "node_id": int(idx),
                    "text_score": round(float(similarities[idx]), 4),
                    "title": self._documents[idx][:100],
                })
        return results

    def get_document_vector(self, doc_idx: int) -> np.ndarray:
        """Get sparse TF-IDF vector for a document."""
        if self.tfidf_matrix is None:
            return np.zeros(1)
        return self.tfidf_matrix[doc_idx].toarray().flatten()

    @property
    def memory_usage_mb(self) -> float:
        """Memory footprint of the sparse index."""
        if self.tfidf_matrix is None:
            return 0.0
        data_bytes = (
            self.tfidf_matrix.data.nbytes +
            self.tfidf_matrix.indices.nbytes +
            self.tfidf_matrix.indptr.nbytes
        )
        return round(data_bytes / (1024 * 1024), 2)

    def save(self, path=None):
        path = path or config.MODEL_DIR / "tfidf_index.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path) -> "TFIDFIndex":
        return joblib.load(path)
