import pickle
import numpy as np
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------
# STEP 3
#
# Train the final Gaussian Mixture Model using the
# optimal number of clusters determined earlier.
#
# This model produces fuzzy cluster memberships
# (probability distribution over clusters).
# ---------------------------------------------------------


# number of clusters chosen from BIC analysis
BEST_K = 33


# ---------------------------------------------------------
# Load PCA-reduced embeddings
# ---------------------------------------------------------
with open("data/processed/reduced_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

print("Embedding matrix shape:", embeddings.shape)


# ---------------------------------------------------------
# Train Gaussian Mixture Model
# ---------------------------------------------------------
print("Training Gaussian Mixture Model...")

gmm = GaussianMixture(
    n_components=BEST_K,
    covariance_type="tied",
    reg_covar=1e-2,
    random_state=42
)

gmm.fit(embeddings)

print("Model training complete.")


# ---------------------------------------------------------
# Compute fuzzy cluster memberships
# ---------------------------------------------------------
print("Computing cluster probabilities...")

cluster_probabilities = gmm.predict_proba(embeddings)

print("Cluster probability matrix shape:", cluster_probabilities.shape)


# ---------------------------------------------------------
# Save trained model
# ---------------------------------------------------------
with open("data/processed/gmm_model.pkl", "wb") as f:
    pickle.dump(gmm, f)


# ---------------------------------------------------------
# Save cluster probabilities
# ---------------------------------------------------------
with open("data/processed/cluster_probabilities.pkl", "wb") as f:
    pickle.dump(cluster_probabilities, f)


print("Fuzzy clustering results saved successfully.")