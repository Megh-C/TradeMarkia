import numpy as np
import pickle
import os
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:
    """
    Semantic query cache.

    Stores previous queries and their embeddings.
    When a new query arrives, we check if a similar query
    exists and reuse the stored result.

    Features:
    - cosine similarity lookup
    - LRU eviction
    - disk persistence
    """

    def __init__(
        self,
        capacity=500,
        similarity_threshold=0.75,
        cache_path="data/cache/cache.pkl"
    ):

        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.cache_path = cache_path

        # OrderedDict allows LRU behaviour
        self.cache = OrderedDict()

        # cache statistics
        self.total_queries = 0
        self.cache_hits = 0

        self._load_cache()

    # ----------------------------
    # Cache lookup
    # ----------------------------
    def get(self, query_embedding):

        self.total_queries += 1

        if len(self.cache) == 0:
            return None

        keys = list(self.cache.keys())

        embeddings = np.array([
            self.cache[key]["embedding"]
            for key in keys
        ])

        similarities = cosine_similarity(
            [query_embedding],
            embeddings
        )[0]
        print("MAX SIMILARITY:", np.max(similarities))

        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score >= self.similarity_threshold:

            self.cache_hits += 1

            key = keys[best_idx]
            entry = self.cache[key]

            # move entry to end (LRU update)
            self.cache.move_to_end(key)

            return {
                "cache_hit": True,
                "matched_query": key,
                "similarity_score": float(best_score),
                "result": entry["result"],
                "cluster": entry["cluster"]
            }

        return None

    # ----------------------------
    # Add new entry
    # ----------------------------
    def put(self, query, embedding, result, cluster):

        if query in self.cache:
            self.cache.move_to_end(query)

        self.cache[query] = {
            "embedding": embedding,
            "result": result,
            "cluster": cluster
        }

        # evict LRU if capacity exceeded
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    # ----------------------------
    # Save cache to disk
    # ----------------------------
    def save(self):

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        with open(self.cache_path, "wb") as f:
            pickle.dump(self.cache, f)

    # ----------------------------
    # Load cache from disk
    # ----------------------------
    def _load_cache(self):

        if os.path.exists(self.cache_path):

            with open(self.cache_path, "rb") as f:
                self.cache = pickle.load(f)

    # ----------------------------
    # Clear cache
    # ----------------------------
    def clear(self):

        self.cache.clear()

        self.total_queries = 0
        self.cache_hits = 0

        self.save()