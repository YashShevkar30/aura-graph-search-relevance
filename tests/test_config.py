import pytest
from aura.config import config

def test_defaults():
    assert config.N2V_DIMENSIONS == 64
    assert config.TOP_K == 10
    assert config.GRAPH_WEIGHT + config.TEXT_WEIGHT == pytest.approx(1.0)
