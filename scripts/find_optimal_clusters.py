import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------
# STEP 2B
#
# Determine the optimal number of clusters using the
# Bayesian Information Criterion (BIC).
#
# This version uses PCA-reduced embeddings (50 dimensions)
# which improves stability of Gaussian Mixture Models
# compared to the original 384-dimensional embeddings.
# ---------------------------------------------------------


# Load reduced embeddings
with open("data/processed/reduced_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

print("Embedding matrix shape:", embeddings.shape)


# Range of cluster numbers to evaluate
cluster_range = range(2, 81)

bic_scores = []


# ---------------------------------------------------------
# Train Gaussian Mixture Model for each K
# ---------------------------------------------------------
for k in cluster_range:

    print(f"Training GMM with {k} clusters...")

    gmm = GaussianMixture(
        n_components=k,
        covariance_type="diag",   # faster + more stable
        random_state=42
    )

    gmm.fit(embeddings)

    bic = gmm.bic(embeddings)

    bic_scores.append(bic)


# ---------------------------------------------------------
# Find best cluster count
# ---------------------------------------------------------
best_k = cluster_range[np.argmin(bic_scores)]

print("\nOptimal number of clusters:", best_k)


# ---------------------------------------------------------
# Plot BIC curve
# ---------------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(cluster_range, bic_scores, marker="o")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("BIC Score")
plt.title("BIC Score vs Number of Clusters")

plt.show()