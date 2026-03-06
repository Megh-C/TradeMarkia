from fastapi import APIRouter
from pydantic import BaseModel

from app.embeddings.embedder import QueryEmbedder
from app.cache.semantic_cache import SemanticCache
from app.search.vector_search import VectorSearch
from app.clustering.cluster_predictor import ClusterPredictor


router = APIRouter()


# request schema
class QueryRequest(BaseModel):
    query: str


# initialize components
embedder = QueryEmbedder()
cache = SemanticCache()
search_engine = VectorSearch()
cluster_predictor = ClusterPredictor()


# -----------------------------
# POST /query
# -----------------------------
@router.post("/query")
def query_system(request: QueryRequest):

    query = request.query

    # embed query
    query_embedding = embedder.embed(query)

    # cache lookup
    cached = cache.get(query_embedding)

    if cached:

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": cached["matched_query"],
            "similarity": cached["similarity_score"],
            "result": cached["result"],
            "cluster": cached["cluster"]
        }

    # vector search
    result = search_engine.search(query_embedding)

    document = result["document"]

    # cluster prediction
    cluster_info = cluster_predictor.predict(query_embedding)

    cluster = cluster_info["dominant_cluster"]

    # store in cache
    cache.put(query, query_embedding, document, cluster)

    cache.save()

    return {
        "query": query,
        "cache_hit": False,
        "result": document,
        "cluster": cluster
    }


# -----------------------------
# GET /cache/stats
# -----------------------------
@router.get("/cache/stats")
def cache_stats():

    size = len(cache.cache)

    hit_rate = 0

    if cache.total_queries > 0:
        hit_rate = cache.cache_hits / cache.total_queries

    return {
        "cache_size": size,
        "cache_capacity": cache.capacity,
        "hit_rate": hit_rate
    }


# -----------------------------
# DELETE /cache
# -----------------------------
@router.delete("/cache")
def clear_cache():

    cache.clear()

    return {
        "message": "Cache cleared successfully"
    }