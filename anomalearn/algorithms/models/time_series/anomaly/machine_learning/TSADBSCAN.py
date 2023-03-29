import numpy as np
from sklearn.cluster import DBSCAN

from .TSAWindowWrapper import TSAWindowWrapper


class TSADBSCAN(TSAWindowWrapper):
    """Concrete class representing the application of DBSCAN approach to time series.

    It is a wrapper of the scikit-learn DBSCAN approach. It uses the
    TimeSeriesProjector to project the time series onto a vector space. Then,
    it uses DBSCAN to find all the anomalies and compute the score of an anomaly
    as described in the fit_predict method. Please, note that the vanilla
    DBSCAN implementation does not produce anomaly scores.

    Parameters
    ----------
    window_scoring: {"z-score", "centroid"}, default="centroid"
        It defines the method with the point anomalies are computed. With
        "centroid" the anomaly is computed as euclidean distance from the
        closest centroid. Then, all the scores are normalized using min-max.
        With z-score, the anomalies are computed using the z-score.

    See Also
    --------
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    """
    WINDOW_SCORING_METHODS = ["z-score", "centroid"]

    def __init__(self, window: int = 5,
                 stride: int = 1,
                 scaling: str = "minmax",
                 scoring: str = "average",
                 classification: str = "voting",
                 threshold: float = None,
                 anomaly_portion: float = 0.01,
                 window_scoring: str = "centroid",
                 eps: float = 0.5,
                 min_samples: int = 5,
                 metric: str = "euclidean",
                 metric_params: dict = None,
                 algorithm: str = "auto",
                 leaf_size: int = 30,
                 p: float = None,
                 n_jobs: int = None):
        super().__init__(window=window,
                         stride=stride,
                         scaling=scaling,
                         scoring=scoring,
                         classification=classification,
                         threshold=threshold,
                         anomaly_portion=anomaly_portion)

        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

        self.window_scoring = window_scoring

        self.__check_parameters()

    def set_params(self, **params):
        super().set_params(**params)
        self.__check_parameters()

    def _compute_window_labels(self, vector_data: np.ndarray) -> np.ndarray:
        # Builds the model and fits it to the vector data
        self._build_wrapped()
        self._wrapped_model.fit(vector_data)

        anomalies = np.argwhere(self._wrapped_model.labels_ == -1)
        anomalies = anomalies.reshape(anomalies.shape[0])

        window_anomalies = np.zeros(vector_data.shape[0])
        window_anomalies[anomalies] = 1

        return window_anomalies

    def _compute_window_scores(self, vector_data: np.ndarray) -> np.ndarray:
        # Builds the model and fits it to the vector data
        self._build_wrapped()
        self._wrapped_model.fit(vector_data)

        clusters = set(self._wrapped_model.labels_).difference({-1})

        # Compute centroids to be able to compute the anomaly score
        centroids = []
        for cluster in clusters:
            cluster_points = np.argwhere(self._wrapped_model.labels_ == cluster)
            centroids.append(np.mean(vector_data[cluster_points]))
        centroids = np.array(centroids)

        # Computes the anomaly score
        anomaly_scores = np.zeros(vector_data.shape[0])

        if self.window_scoring == "z-score":
            mean = np.average(vector_data, axis=0)
            std = np.std(vector_data, axis=0, ddof=1)

            # Compute the anomaly scores using z-score
            for i in range(vector_data.shape[0]):
                deviated_point = (vector_data[i] - mean) / std
                anomaly_scores[i] = np.linalg.norm(deviated_point)
        else:
            # Compute the anomaly scores using distance from the closest centroid
            for i in range(vector_data.shape[0]):
                min_distance = np.inf

                for j in range(centroids.shape[0]):
                    distance = np.linalg.norm(vector_data[i] - centroids[j])
                    if distance < min_distance:
                        min_distance = distance

                anomaly_scores[i] = min_distance

        return anomaly_scores

    def _build_wrapped(self) -> None:
        self._wrapped_model = DBSCAN(self.eps,
                                     min_samples=self.min_samples,
                                     metric=self.metric,
                                     metric_params=self.metric_params,
                                     algorithm=self.algorithm,
                                     leaf_size=self.leaf_size,
                                     p=self.p,
                                     n_jobs=self.n_jobs)

    def __check_parameters(self):
        """Checks that the objects parameters are correct.

        Returns
        -------
        None
        """
        if self.window_scoring not in self.WINDOW_SCORING_METHODS:
            raise ValueError("The score method must be one of",
                             self.WINDOW_SCORING_METHODS)
