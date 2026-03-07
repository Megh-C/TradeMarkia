```markdown
# Semantic Search System with Fuzzy Clustering and Semantic Caching

## Overview

This project implements a full semantic search system that retrieves documents based on **meaning rather than keyword matching**. Traditional keyword-based search fails when queries and documents use different wording to express the same concept. This system addresses that limitation by using **sentence embeddings and vector similarity** to identify semantically related content.

The project demonstrates the design and implementation of a complete machine learning powered search system including:

- Semantic document embeddings
- Vector similarity search
- Probabilistic topic clustering
- Query-level semantic caching
- REST API architecture
- Interactive frontend
- Containerized deployment

The dataset used is the **20 Newsgroups dataset**, a widely used benchmark dataset consisting of approximately 20,000 Usenet discussion posts spanning multiple topical domains such as politics, religion, technology, and sports.

The goal of this project was not only to implement semantic retrieval but to design a **scalable system architecture that combines machine learning, information retrieval, and backend engineering principles**.

---

# System Architecture

The system is structured as a modular pipeline that separates offline model building from online query serving.

High-level architecture:

```

User Query
│
▼
Streamlit Frontend
│
▼
FastAPI Backend
│
├── Query Embedding Generation
│
├── Semantic Cache Lookup
│        │
│        ├── Cache Hit → Return cached result
│        │
│        └── Cache Miss
│
▼
Vector Similarity Search (ChromaDB)
│
▼
Dimensionality Reduction (PCA)
│
▼
Fuzzy Cluster Prediction (GMM)
│
▼
Return Result + Cluster Information

```

The system consists of three main layers:

**Data Processing Layer**

Responsible for preparing the dataset, generating embeddings, and training clustering models.

**Search Infrastructure Layer**

Handles vector indexing, similarity search, and cluster prediction.

**Application Layer**

Provides a REST API and an interactive frontend for user interaction.

---

# Dataset

The project uses the **20 Newsgroups dataset**, which contains Usenet discussion posts categorized into multiple topical groups.

Although widely used for research, the dataset contains significant noise because it originates from email-like discussion threads. Examples of noise include:

- Email headers
- Reply chains
- ASCII signatures
- Email addresses
- Metadata such as "In article ... writes"

Cleaning this data was a necessary preprocessing step before building the semantic search index.

Despite cleaning efforts, some formatting artifacts remain due to inconsistent structure in the original dataset.

---

# Document Representation using Embeddings

Each document is converted into a dense vector representation using the **SentenceTransformer model `all-MiniLM-L6-v2`**.

This model produces **384-dimensional sentence embeddings** that capture semantic meaning.

The model was chosen because it provides a strong balance between:

- semantic accuracy
- inference speed
- computational efficiency

Embedding the dataset allows documents that discuss the same concept using different wording to appear close to each other in vector space.

---

# Vector Search

The system uses **ChromaDB** as the vector database for storing and retrieving embeddings.

Each document entry includes:

- Document ID
- Text content
- Metadata
- Embedding vector

When a query is submitted, the system:

1. Converts the query into an embedding using the same embedding model.
2. Performs nearest-neighbor search against stored document embeddings.
3. Returns the most semantically similar document.

The similarity metric used for retrieval is **cosine similarity**, which measures the angular similarity between vectors and is widely used in embedding-based retrieval systems.

---

# Dimensionality Reduction

High-dimensional embeddings can degrade clustering performance due to noise and sparsity.

To address this, **Principal Component Analysis (PCA)** is applied to reduce the dimensionality of the embeddings.

```

Original embedding size: 384 dimensions
Reduced embedding size: 50 dimensions

```

Dimensionality reduction improves clustering performance by:

- removing redundant information
- reducing noise
- lowering computational complexity

The PCA model is trained offline and reused during query-time cluster prediction.

---

# Determining the Optimal Number of Clusters

To determine the appropriate number of clusters for the dataset, the **Bayesian Information Criterion (BIC)** was used.

BIC evaluates model quality while penalizing model complexity. Lower BIC scores indicate a better balance between fit and simplicity.

The evaluation suggested approximately **80 clusters** as the mathematically optimal solution.

However, this result produced clusters with extremely high confidence values and very small document groups, indicating **overfitting**.

To achieve better generalization and more interpretable clusters, the final cluster count was manually adjusted to:

```

33 clusters

```

This provided a better balance between:

- cluster interpretability
- topic generalization
- model stability

---

# Fuzzy Clustering using Gaussian Mixture Models

Instead of using hard clustering methods such as K-Means, this system uses **Gaussian Mixture Models (GMM)**.

GMM performs **probabilistic clustering**, meaning each document can belong to multiple clusters with different probabilities.

Example cluster membership:

```

Cluster 12 → 0.72
Cluster 5  → 0.21
Cluster 3  → 0.07

```

This approach better reflects real-world text data where documents often discuss multiple topics simultaneously.

For example, a discussion about **space policy** could belong to:

- science
- politics
- technology

This fuzzy clustering approach enables more flexible semantic grouping compared to rigid clustering techniques.

---

# Query Processing Pipeline

When a user submits a search query, the following steps occur:

1. The query is embedded using the same SentenceTransformer model used during indexing.
2. The system checks whether a semantically similar query already exists in the cache.
3. If a cache hit occurs, the cached result is returned immediately.
4. If no cache match is found, vector search is performed against the document embeddings.
5. The query embedding is transformed using the trained PCA model.
6. The Gaussian Mixture Model predicts cluster membership probabilities.
7. The system returns the most relevant document along with its predicted cluster.

---

# Semantic Cache Design

To improve query performance, a **semantic caching system** was implemented.

Unlike traditional caching which relies on exact query matching, this system caches results based on **semantic similarity between query embeddings**.

Each cache entry stores:

- query text
- query embedding
- search result
- predicted cluster

When a new query arrives:

1. The query embedding is computed.
2. Cosine similarity is calculated against cached embeddings.
3. If similarity exceeds a predefined threshold, the cached result is reused.

Cache configuration:

```

Cache capacity: 500 queries
Similarity threshold: 0.85
Replacement policy: LRU

```

This approach significantly reduces repeated computation when similar queries are submitted.

---

# API Design

The backend is implemented using **FastAPI**, which provides a high-performance asynchronous API framework.

Available endpoints:

```

POST /query
Executes semantic search and returns result + cluster prediction

GET /cache/stats
Returns cache usage statistics

DELETE /cache
Clears the semantic cache

```

FastAPI automatically generates API documentation using Swagger.

---

# Frontend Interface

A lightweight user interface was implemented using **Streamlit**.

The frontend allows users to:

- enter search queries
- view retrieved documents
- see predicted cluster assignments
- observe whether results were served from cache
- inspect cache statistics
- clear cached results

Some additional text filtering is applied in the frontend to hide residual artifacts such as signatures or email addresses that remain after dataset cleaning.

---

# Containerized Deployment

The system is fully containerized using **Docker**.

Two services are defined:

```

backend  → FastAPI server
frontend → Streamlit application

```

The services are orchestrated using **Docker Compose**.

To run the entire system:

```

docker compose up

```

Access points:

```

Frontend UI
[http://localhost:8501](http://localhost:8501)

Backend API
[http://localhost:8000/docs](http://localhost:8000/docs)

```

This setup allows the entire system to run in a reproducible environment without requiring manual dependency installation.

---

# Key Design Decisions

Embedding model  
SentenceTransformer `all-MiniLM-L6-v2`

Vector database  
ChromaDB

Similarity metric  
Cosine similarity

Dimensionality reduction  
PCA (384 → 50)

Clustering method  
Gaussian Mixture Model (fuzzy clustering)

Cluster selection method  
Bayesian Information Criterion with manual adjustment

Cache strategy  
Semantic similarity + LRU eviction

Deployment  
Docker + Docker Compose

---

# Current Limitations

Some limitations remain in the current implementation:

- Dataset noise remains due to imperfect cleaning of Usenet formatting.
- Some ASCII signatures and disclaimers still appear in certain documents.
- Clustering quality is affected by the noisy source text.
- The system retrieves entire documents rather than smaller semantic passages.
- The embedding model used is lightweight; larger models could improve retrieval accuracy.

---

# Future Improvements

Potential improvements include:

- Advanced text preprocessing for signature and reply detection
- Document chunking to enable passage-level retrieval
- Approximate nearest neighbor indexing for faster large-scale retrieval
- Automatic cluster labeling using topic modeling
- Retrieval-Augmented Generation integration with large language models
- More advanced cache policies such as LFU or adaptive eviction
- Quantitative evaluation metrics for retrieval quality
- Visualization of cluster structure

---

# Summary

This project demonstrates the design and implementation of an end-to-end semantic search system that integrates:

- embedding-based retrieval
- probabilistic topic clustering
- semantic query caching
- API-based backend services
- interactive frontend
- containerized deployment

The system combines machine learning, information retrieval, and backend engineering principles to create a deployable semantic search platform.
```

---

