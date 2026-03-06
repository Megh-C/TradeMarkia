import chromadb


class VectorSearch:
    """
    Handles semantic search over the vector database.

    Uses ChromaDB to retrieve the most similar document
    given a query embedding.
    """

    def __init__(self, db_path="chroma_db", collection_name="newsgroups"):

        # connect to persistent chroma database
        self.client = chromadb.PersistentClient(path=db_path)

        # access collection
        self.collection = self.client.get_collection(name=collection_name)

    def search(self, query_embedding, k=1):
        """
        Perform semantic search.

        Returns top-k documents most similar to the query.
        """

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        document = results["documents"][0][0]
        metadata = results["metadatas"][0][0]

        return {
            "document": document,
            "metadata": metadata
        }