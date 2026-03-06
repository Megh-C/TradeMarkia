import pickle
import numpy as np


class ClusterPredictor:
    """
    Predicts cluster membership for a query.

    Uses the trained PCA model and Gaussian Mixture Model
    from the offline clustering pipeline.
    """

    def __init__(
        self,
        pca_path="data/processed/pca_model.pkl",
        gmm_path="data/processed/gmm_model.pkl"
    ):

        # load PCA transformer
        with open(pca_path, "rb") as f:
            self.pca = pickle.load(f)

        # load GMM clustering model
        with open(gmm_path, "rb") as f:
            self.gmm = pickle.load(f)

    def predict(self, embedding):
        """
        Predict cluster probabilities for a query embedding.
        """

        # reshape to 2D array
        embedding = np.array(embedding).reshape(1, -1)

        # reduce dimension using PCA
        reduced = self.pca.transform(embedding)

        # get cluster probability distribution
        probs = self.gmm.predict_proba(reduced)[0]

        # dominant cluster
        cluster = int(np.argmax(probs))

        return {
            "dominant_cluster": cluster,
            "probabilities": probs.tolist()
        }