from pathlib import Path

class AuraConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_RAW = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    MODEL_DIR = PROJECT_ROOT / "models"

    # Node2Vec hyperparameters
    N2V_DIMENSIONS: int = 64
    N2V_WALK_LENGTH: int = 30
    N2V_NUM_WALKS: int = 200
    N2V_P: float = 1.0  # return parameter
    N2V_Q: float = 1.0  # in-out parameter
    N2V_WORKERS: int = 4
    N2V_WINDOW: int = 10

    # TF-IDF
    TFIDF_MAX_FEATURES: int = 10000
    TFIDF_NGRAM_RANGE: tuple = (1, 2)

    # SVM
    SVM_C: float = 1.0
    SVM_KERNEL: str = "rbf"

    # Ranking
    GRAPH_WEIGHT: float = 0.6
    TEXT_WEIGHT: float = 0.4
    TOP_K: int = 10

config = AuraConfig()
