import pickle
import numpy as np
from sklearn.decomposition import PCA


# ---------------------------------------------------------
# STEP 2A
#
# Reduce embedding dimensionality using PCA before
# clustering. High-dimensional embeddings can cause
# Gaussian Mixture Models to over-penalize complex models.
#
# PCA keeps the most informative components while
# removing noise.
# ---------------------------------------------------------


# Load embeddings
with open("data/processed/embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

print("Original embedding shape:", embeddings.shape)


# Apply PCA
pca = PCA(n_components=50, random_state=42)

reduced_embeddings = pca.fit_transform(embeddings)

print("Reduced embedding shape:", reduced_embeddings.shape)


# Save reduced embeddings
with open("data/processed/reduced_embeddings.pkl", "wb") as f:
    pickle.dump(reduced_embeddings, f)

print("Reduced embeddings saved successfully.")