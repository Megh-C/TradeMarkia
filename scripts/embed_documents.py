from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Load the embedding model.
# using the SentenceTransformers model "all-MiniLM-L6-v2".
# the model generates 384-dimensional embeddings and is widely used for semantic similarity tasks due to its
# good balance between speed and semantic quality.

def load_model():

    model = SentenceTransformer("all-MiniLM-L6-v2")

    return model



# generate embeddings for all the docs.
#
# each document is converted into a vector representation capturing its semantic meaning. Documents with similar
# meaning will have embeddings that are close in vector space

def embed_documents(model, documents):

    embeddings = model.encode(
        documents,
        batch_size=64,
        show_progress_bar=True
    )

    return embeddings