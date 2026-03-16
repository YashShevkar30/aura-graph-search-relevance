import pytest
from aura.evaluation.metrics import precision_at_k, recall_at_k, mrr, ndcg_at_k

class TestPrecision:
    def test_perfect(self):
        assert precision_at_k([1,2,3], {1,2,3}, 3) == 1.0
    def test_zero(self):
        assert precision_at_k([4,5,6], {1,2,3}, 3) == 0.0
    def test_partial(self):
        assert precision_at_k([1,4,2], {1,2,3}, 3) == pytest.approx(2/3)

class TestRecall:
    def test_full(self):
        assert recall_at_k([1,2,3], {1,2}, 3) == 1.0
    def test_partial(self):
        assert recall_at_k([1,4,5], {1,2}, 3) == 0.5

class TestMRR:
    def test_first(self):
        assert mrr([1,2,3], {1}) == 1.0
    def test_second(self):
        assert mrr([2,1,3], {1}) == 0.5
    def test_miss(self):
        assert mrr([4,5,6], {1}) == 0.0

class TestNDCG:
    def test_perfect(self):
        assert ndcg_at_k([1,2,3], {1,2,3}, 3) == 1.0
    def test_zero(self):
        assert ndcg_at_k([4,5,6], {1,2,3}, 3) == 0.0
