import chromadb
import numpy as np
import pickle



# extracts all stored embeddings from the Chroma vector database and saves them locally.
#
# Why we do this:
# Clustering algorithms like Gaussian Mixture Models require direct access to the embedding matrix.
# Since our embeddings are currently stored inside ChromaDB, we must retrieve them first.



# Load the persistent Chroma database
client = chromadb.PersistentClient(
    path="chroma_db"
)

# Access the collection we created earlier
collection = client.get_collection(
    name="newsgroups"
)



# Retrieve all stored vectors
#
# include=["embeddings"] ensures we retrieve the vector
# representations for each document.

data = collection.get(
    include=["embeddings"]
)

# embeddings to numpy array
embeddings = np.array(data["embeddings"])



# Save embeddings to disk
#
# save them as a pickle file so future scripts can
# load them quickly without querying the database again.

with open("data/processed/embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)


print("Embeddings extracted successfully.")
print("Shape of embedding matrix:", embeddings.shape)