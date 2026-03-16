"""
SVM Relevance Classifier
==========================
Binary SVM classifier that filters irrelevant/noisy candidates
from the retrieval pipeline. Uses combined graph + text features.
"""
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from loguru import logger
from aura.config import config


class RelevanceClassifier:
    """SVM-based binary relevance classifier for noise filtering."""

    def __init__(self, C: float = None, kernel: str = None):
        self.scaler = StandardScaler()
        self.model = SVC(
            C=C or config.SVM_C,
            kernel=kernel or config.SVM_KERNEL,
            probability=True,
            random_state=42,
        )
        self._is_trained = False
        self.cv_results = None

    def _build_features(
        self,
        graph_embeddings: np.ndarray,
        tfidf_vectors: np.ndarray,
    ) -> np.ndarray:
        """Concatenate graph and text features."""
        return np.hstack([graph_embeddings, tfidf_vectors])

    def fit(
        self,
        graph_embeddings: np.ndarray,
        tfidf_vectors: np.ndarray,
        labels: np.ndarray,
    ):
        """Train SVM on combined feature space."""
        X = self._build_features(graph_embeddings, tfidf_vectors)
        X_scaled = self.scaler.fit_transform(X)

        logger.info(f"Training SVM classifier: {X_scaled.shape[0]} samples, "
                   f"{X_scaled.shape[1]} features")

        # Cross-validate
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X_scaled, labels, cv=cv, scoring="f1_macro")
        self.cv_results = {
            "f1_mean": round(float(scores.mean()), 4),
            "f1_std": round(float(scores.std()), 4),
        }
        logger.info(f"CV F1 (macro): {self.cv_results['f1_mean']} ± {self.cv_results['f1_std']}")

        # Full training
        self.model.fit(X_scaled, labels)
        self._is_trained = True
        return self

    def predict(
        self,
        graph_embedding: np.ndarray,
        tfidf_vector: np.ndarray,
    ) -> dict:
        """Predict relevance for a single candidate."""
        if not self._is_trained:
            raise RuntimeError("Classifier not trained.")
        X = np.hstack([graph_embedding, tfidf_vector]).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        prob = self.model.predict_proba(X_scaled)[0]
        pred = self.model.predict(X_scaled)[0]
        return {"prediction": int(pred), "confidence": round(float(max(prob)), 4)}

    def evaluate(
        self,
        graph_embeddings: np.ndarray,
        tfidf_vectors: np.ndarray,
        labels: np.ndarray,
    ) -> dict:
        X = self._build_features(graph_embeddings, tfidf_vectors)
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return {
            "accuracy": round(accuracy_score(labels, preds), 4),
            "f1_macro": round(f1_score(labels, preds, average="macro"), 4),
            "f1_weighted": round(f1_score(labels, preds, average="weighted"), 4),
            "report": classification_report(labels, preds, output_dict=True),
        }

    def save(self, path=None):
        path = path or config.MODEL_DIR / "svm_classifier.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path) -> "RelevanceClassifier":
        return joblib.load(path)
