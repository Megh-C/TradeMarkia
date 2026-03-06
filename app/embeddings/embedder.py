from sentence_transformers import SentenceTransformer


class QueryEmbedder:
    """
    Handles query embedding generation.

    Uses the same embedding model that was used to
    generate document embeddings during indexing.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):

        # load embedding model
        self.model = SentenceTransformer(model_name)

    def embed(self, text):
        """
        Convert query text into embedding vector.
        """

        embedding = self.model.encode(text)

        return embedding