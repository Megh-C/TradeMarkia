from fastapi import FastAPI
from app.api.routes import router


app = FastAPI(
    title="Semantic Search System",
    description="Semantic search with clustering and semantic caching",
    version="1.0"
)


# register routes
app.include_router(router)