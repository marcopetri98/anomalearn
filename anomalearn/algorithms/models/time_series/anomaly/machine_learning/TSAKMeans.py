import numpy as np
from sklearn.cluster import KMeans

from .TSAWindowWrapper import TSAWindowWrapper


class TSAKMeans(TSAWindowWrapper):
    """It is a concrete class wrapping k means to perform anomaly detection."""

    def __init__(self, window: int = 5,
                 stride: int = 1,
                 scaling: str = "minmax",
                 scoring: str = "average",
                 classification: str = "voting",
                 threshold: float = None,
                 anomaly_portion: float = 0.01,
                 anomaly_threshold: float = 1.0,
                 kmeans_params: dict = None):
        super().__init__(window=window,
                         stride=stride,
                         scaling=scaling,
                         scoring=scoring,
                         classification=classification,
                         threshold=threshold,
                         anomaly_portion=anomaly_portion)

        self.anomaly_threshold = anomaly_threshold
        self.kmeans_params = kmeans_params

    def _compute_window_labels(self, vector_data: np.ndarray) -> np.ndarray:
        # Builds the model and fits it to the vector data
        anomaly_scores = self._compute_window_scores(vector_data)

        anomalies = np.argwhere(anomaly_scores >= self.anomaly_threshold)
        anomalies = anomalies.reshape(anomalies.shape[0])

        window_anomalies = np.zeros(vector_data.shape[0])
        window_anomalies[anomalies] = 1

        return window_anomalies

    def _compute_window_scores(self, vector_data: np.ndarray) -> np.ndarray:
        # Builds the model and fits it to the vector data
        self._build_wrapped()
        clusters = self._wrapped_model.fit_predict(vector_data)

        # Compute centroids of clusters
        centroids = []
        for cluster in set(clusters):
            cluster_points = np.argwhere(clusters == cluster)
            centroids.append(np.mean(vector_data[cluster_points]))
        centroids = np.array(centroids)

        # Compute the anomaly scores as distance from the closest centroid
        anomaly_scores = np.zeros(vector_data.shape[0])
        for i in range(vector_data.shape[0]):
            distance = np.linalg.norm(vector_data[i] - centroids[clusters[i]])
            anomaly_scores[i] = distance

        return anomaly_scores

    def _build_wrapped(self) -> None:
        self._wrapped_model = KMeans(**self.kmeans_params)
