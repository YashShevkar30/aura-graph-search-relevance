"""
Index Building Orchestrator
============================
Orchestrates the full indexing pipeline: load graph, train
Node2Vec, build TF-IDF index, and persist all artifacts.
"""
import numpy as np
import networkx as nx
import pandas as pd
from loguru import logger
from aura.config import config
from aura.indexing.node2vec import Node2VecEmbedder
from aura.indexing.tfidf import TFIDFIndex


def build_full_index():
    proc = config.DATA_PROCESSED

    # Load graph
    G = nx.read_edgelist(str(proc / "graph.edgelist"), nodetype=int)
    meta = pd.read_csv(proc / "metadata.csv")
    titles = meta["title"].tolist()
    labels = np.load(proc / "labels.npy")

    logger.info(f"Index build: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Node2Vec embeddings
    n2v = Node2VecEmbedder()
    embeddings = n2v.fit(G)
    np.save(proc / "node2vec_embeddings.npy", embeddings)
    logger.info(f"Node2Vec embeddings: {embeddings.shape}")

    # TF-IDF index
    tfidf = TFIDFIndex()
    tfidf.fit(titles)
    tfidf.save()

    # Save dense TF-IDF for SVM training
    tfidf_dense = tfidf.tfidf_matrix.toarray()
    np.save(proc / "tfidf_dense.npy", tfidf_dense)

    logger.info(f"TF-IDF memory: {tfidf.memory_usage_mb} MB (sparse)")
    logger.info(f"Dense equivalent would be: "
               f"{round(tfidf_dense.nbytes / (1024*1024), 2)} MB")
    logger.info(f"Memory reduction via sparse: "
               f"{round((1 - tfidf.memory_usage_mb / (tfidf_dense.nbytes/(1024*1024))) * 100, 1)}%")

    return n2v, tfidf


if __name__ == "__main__":
    build_full_index()
