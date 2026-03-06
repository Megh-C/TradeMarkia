import pickle
import json
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer


# load cleaned documents
with open("data/processed/cleaned_documents.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

texts = [doc["text"] for doc in docs]


# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)


print("Training PCA...")
pca = PCA(n_components=50)
pca.fit(embeddings)


# save PCA model
with open("data/processed/pca_model.pkl", "wb") as f:
    pickle.dump(pca, f)

print("PCA model saved successfully.")