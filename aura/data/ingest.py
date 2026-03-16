"""
Data Ingestion Pipeline — Cora Citation Network
=================================================
Loads the Cora dataset: a citation network of 2,708 scientific papers
across 7 classes, with 5,429 citation edges and 1,433-dimensional
binary word vectors.

This is a standard benchmark for graph-based ML and search tasks.
"""
import os
import urllib.request
import tarfile
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from loguru import logger
from aura.config import config

CORA_URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
CLASSES = [
    "Case_Based", "Genetic_Algorithms", "Neural_Networks",
    "Probabilistic_Methods", "Reinforcement_Learning",
    "Rule_Learning", "Theory",
]


def download_cora(dest_dir: Path = None) -> Path:
    dest_dir = dest_dir or config.DATA_RAW
    dest_dir.mkdir(parents=True, exist_ok=True)
    tgz_path = dest_dir / "cora.tgz"
    extract_dir = dest_dir / "cora"

    if extract_dir.exists():
        logger.info(f"Cora already exists at {extract_dir}")
        return extract_dir

    logger.info(f"Downloading Cora dataset from {CORA_URL}")
    urllib.request.urlretrieve(CORA_URL, tgz_path)
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(dest_dir)
    tgz_path.unlink()
    logger.info(f"Extracted to {extract_dir}")
    return extract_dir


def generate_synthetic_cora(n_nodes=2708, n_edges=5429, n_features=1433, n_classes=7, seed=42):
    """Generate synthetic Cora-like data for environments where download fails."""
    rng = np.random.default_rng(seed)
    logger.info("Generating synthetic Cora-like dataset")

    # Node features (sparse binary)
    features = rng.choice([0, 1], size=(n_nodes, n_features), p=[0.98, 0.02])
    labels = rng.integers(0, n_classes, size=n_nodes)

    # Citation edges (ensure connected graph)
    edges = set()
    for i in range(1, n_nodes):
        j = rng.integers(0, i)
        edges.add((i, j))
    while len(edges) < n_edges:
        i, j = rng.integers(0, n_nodes, size=2)
        if i != j:
            edges.add((min(i, j), max(i, j)))

    # Generate text titles
    topics = ["neural", "bayesian", "genetic", "reinforcement",
              "classification", "clustering", "kernel", "optimization",
              "deep", "learning", "network", "graph", "model", "algorithm"]
    titles = []
    for i in range(n_nodes):
        n_words = rng.integers(4, 10)
        title = " ".join(rng.choice(topics, size=n_words))
        titles.append(f"Paper {i}: {title}")

    return features, labels, list(edges), titles


def load_cora(data_dir: Path = None):
    """Load Cora and construct graph + feature matrix."""
    try:
        data_dir = download_cora(data_dir)
        content_file = data_dir / "cora.content"
        cites_file = data_dir / "cora.cites"

        if not content_file.exists() or not cites_file.exists():
            raise FileNotFoundError("Cora files not found")

        # Parse content (node_id, features..., label)
        content = pd.read_csv(content_file, sep="\t", header=None)
        node_ids = content.iloc[:, 0].values
        features = content.iloc[:, 1:-1].values.astype(np.float32)
        labels_raw = content.iloc[:, -1].values
        label_map = {l: i for i, l in enumerate(sorted(set(labels_raw)))}
        labels = np.array([label_map[l] for l in labels_raw])

        # Parse citations
        cites = pd.read_csv(cites_file, sep="\t", header=None, names=["src", "dst"])
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        edges = [(id_to_idx[s], id_to_idx[d])
                 for s, d in zip(cites["src"], cites["dst"])
                 if s in id_to_idx and d in id_to_idx]

        titles = [f"Paper {nid}" for nid in node_ids]

        logger.info(f"Loaded Cora: {len(node_ids)} nodes, {len(edges)} edges, "
                   f"{features.shape[1]} features, {len(label_map)} classes")
    except Exception as e:
        logger.warning(f"Failed to load real Cora: {e}. Using synthetic data.")
        features, labels, edges, titles = generate_synthetic_cora()

    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(len(labels)))
    G.add_edges_from(edges)
    for i in range(len(labels)):
        G.nodes[i]["label"] = int(labels[i])
        G.nodes[i]["title"] = titles[i]

    logger.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
               f"connected={nx.is_connected(G)}")

    # Persist
    out_dir = config.DATA_PROCESSED
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "features.npy", features)
    np.save(out_dir / "labels.npy", labels)
    nx.write_edgelist(G, out_dir / "graph.edgelist")
    pd.DataFrame({"node_id": range(len(titles)), "title": titles}).to_csv(
        out_dir / "metadata.csv", index=False
    )

    return G, features, labels, titles


if __name__ == "__main__":
    load_cora()
