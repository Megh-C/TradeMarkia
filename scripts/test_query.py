import chromadb
from sentence_transformers import SentenceTransformer



# to test whether the vector database is working


# load persistent Chroma database
client = chromadb.PersistentClient(
    path="chroma_db"
)

# access the collection created earlier
collection = client.get_collection(
    name="newsgroups"
)


# load embedding model
# this must be the same model used during indexing
model = SentenceTransformer("all-MiniLM-L6-v2")


# example semantic query
query = "space shuttle launch NASA mission"


# convert query into embedding vector
query_embedding = model.encode(query).tolist()


# perform similarity search
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)


# print result 
print("\nQuery:", query)
print("\nTop 5 most similar documents:\n")

for i, doc in enumerate(results["documents"][0]):
    print(f"Result {i+1}:")
    print(doc[:400])  # print first 400 characters
    print("\n----------------------------\n")