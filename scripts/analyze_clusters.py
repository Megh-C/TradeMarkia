import pickle
import json
import numpy as np


# ---------------------------------------------------------
# STEP 4
#
# This script analyzes fuzzy clusters by identifying
# the documents with the highest membership probability
# for each cluster.
#
# This helps interpret what semantic theme each cluster
# represents.
# ---------------------------------------------------------


# load cluster probabilities
with open("data/processed/cluster_probabilities.pkl", "rb") as f:
    cluster_probs = pickle.load(f)


# load cleaned documents
with open("data/processed/cleaned_documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)


cluster_probs = np.array(cluster_probs)

num_clusters = cluster_probs.shape[1]


print("Total clusters:", num_clusters)


# ---------------------------------------------------------
# For each cluster:
# find the top documents with highest probability
# ---------------------------------------------------------
for cluster_id in range(num_clusters):

    print("\n===============================")
    print(f"CLUSTER {cluster_id}")
    print("===============================\n")


    # probabilities for this cluster
    probs = cluster_probs[:, cluster_id]


    # get indices of top documents
    top_indices = np.argsort(probs)[-5:][::-1]


    for idx in top_indices:

        prob = probs[idx]
        text = documents[idx]["text"][:300]

        print(f"Probability: {prob:.3f}")
        print(text)
        print("\n---\n")