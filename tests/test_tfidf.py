import pytest
from aura.indexing.tfidf import TFIDFIndex

@pytest.fixture
def index():
    docs = [
        "neural network deep learning classification",
        "bayesian probability statistical inference model",
        "genetic algorithm optimization evolutionary",
        "reinforcement learning policy reward agent",
        "neural network deep learning classification model",
    ]
    idx = TFIDFIndex(max_features=100, ngram_range=(1,1))
    idx.fit(docs)
    return idx

def test_index_shape(index):
    assert index.tfidf_matrix.shape[0] == 5

def test_query_returns_results(index):
    results = index.query("neural network", top_k=3)
    assert len(results) > 0
    assert results[0]["text_score"] > 0

def test_memory_usage(index):
    assert index.memory_usage_mb >= 0
