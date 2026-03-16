"""
Node2Vec Graph Embedding Pipeline
===================================
Learns dense vector representations of graph nodes using biased
random walks and Skip-gram (Word2Vec). The resulting embeddings
capture structural graph similarity.

Reference: Grover & Leskovec, "node2vec: Scalable Feature Learning
for Networks" (KDD 2016).
"""
import time
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from loguru import logger
from aura.config import config


class Node2VecEmbedder:
    """Learns node embeddings via biased random walks + Word2Vec."""

    def __init__(
        self,
        dimensions: int = None,
        walk_length: int = None,
        num_walks: int = None,
        p: float = None,
        q: float = None,
        workers: int = None,
    ):
        self.dimensions = dimensions or config.N2V_DIMENSIONS
        self.walk_length = walk_length or config.N2V_WALK_LENGTH
        self.num_walks = num_walks or config.N2V_NUM_WALKS
        self.p = p or config.N2V_P
        self.q = q or config.N2V_Q
        self.workers = workers or config.N2V_WORKERS
        self._model = None
        self._graph = None
        self.training_time = 0.0

    def _compute_transition_probs(self, G: nx.Graph):
        """Precompute biased transition probabilities for 2nd-order walks."""
        alias_nodes = {}
        for node in G.nodes():
            neighbors = sorted(G.neighbors(node))
            weights = [1.0 / len(neighbors)] * len(neighbors) if neighbors else [1.0]
            alias_nodes[node] = neighbors
        return alias_nodes

    def _random_walk(self, G: nx.Graph, start_node: int) -> list[str]:
        """Generate a single biased random walk starting from start_node."""
        walk = [start_node]
        neighbors_cache = {n: list(G.neighbors(n)) for n in G.nodes()}

        for _ in range(self.walk_length - 1):
            cur = walk[-1]
            neighbors = neighbors_cache.get(cur, [])
            if not neighbors:
                break

            if len(walk) < 2:
                walk.append(neighbors[np.random.randint(len(neighbors))])
                continue

            prev = walk[-2]
            # Biased sampling: p controls returning, q controls exploration
            weights = []
            for nbr in neighbors:
                if nbr == prev:
                    weights.append(1.0 / self.p)
                elif G.has_edge(nbr, prev):
                    weights.append(1.0)
                else:
                    weights.append(1.0 / self.q)

            weights = np.array(weights)
            weights /= weights.sum()
            choice = np.random.choice(len(neighbors), p=weights)
            walk.append(neighbors[choice])

        return [str(n) for n in walk]

    def _generate_walks(self, G: nx.Graph) -> list[list[str]]:
        """Generate corpus of random walks."""
        nodes = list(G.nodes())
        walks = []
        for walk_iter in range(self.num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                walks.append(self._random_walk(G, node))
        logger.info(f"Generated {len(walks):,} walks of length {self.walk_length}")
        return walks

    def fit(self, G: nx.Graph) -> np.ndarray:
        """Train Node2Vec embeddings on graph G."""
        self._graph = G
        logger.info(
            f"Training Node2Vec: dim={self.dimensions}, "
            f"walks={self.num_walks}, length={self.walk_length}, "
            f"p={self.p}, q={self.q}"
        )

        start = time.time()
        walks = self._generate_walks(G)
        self._model = Word2Vec(
            sentences=walks,
            vector_size=self.dimensions,
            window=config.N2V_WINDOW,
            min_count=0,
            sg=1,  # Skip-gram
            workers=self.workers,
            epochs=5,
        )
        self.training_time = time.time() - start
        logger.info(f"Node2Vec training complete in {self.training_time:.1f}s")

        return self.get_embeddings()

    def get_embeddings(self) -> np.ndarray:
        """Return embedding matrix ordered by node index."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        n_nodes = self._graph.number_of_nodes()
        embeddings = np.zeros((n_nodes, self.dimensions), dtype=np.float32)
        for node in range(n_nodes):
            key = str(node)
            if key in self._model.wv:
                embeddings[node] = self._model.wv[key]
        return embeddings

    def get_embedding(self, node_id: int) -> np.ndarray:
        key = str(node_id)
        if self._model and key in self._model.wv:
            return self._model.wv[key]
        return np.zeros(self.dimensions)

    def most_similar_nodes(self, node_id: int, top_k: int = 10) -> list[dict]:
        """Find most similar nodes in embedding space."""
        key = str(node_id)
        if self._model is None or key not in self._model.wv:
            return []
        similar = self._model.wv.most_similar(key, topn=top_k)
        return [{"node_id": int(nid), "similarity": round(sim, 4)} for nid, sim in similar]
