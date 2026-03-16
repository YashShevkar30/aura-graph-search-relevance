"""
SVM Training Script
====================
Loads precomputed embeddings, builds combined features, trains
the relevance classifier, and saves artifacts.
"""
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from loguru import logger
from aura.config import config
from aura.ranking.svm_filter import RelevanceClassifier


def train_pipeline():
    proc = config.DATA_PROCESSED
    embeddings = np.load(proc / "node2vec_embeddings.npy")
    tfidf_dense = np.load(proc / "tfidf_dense.npy")
    labels = np.load(proc / "labels.npy")

    # Create binary relevance labels (threshold: same class = relevant)
    # For training, we use the multi-class labels directly
    logger.info(f"Training data: {embeddings.shape[0]} samples")

    X_graph_train, X_graph_test, X_text_train, X_text_test, y_train, y_test = \
        train_test_split(embeddings, tfidf_dense, labels, test_size=0.2,
                        random_state=42, stratify=labels)

    clf = RelevanceClassifier()
    clf.fit(X_graph_train, X_text_train, y_train)

    results = clf.evaluate(X_graph_test, X_text_test, y_test)
    logger.info(f"Test results: {results}")

    clf.save()
    return clf, results

if __name__ == "__main__":
    train_pipeline()
