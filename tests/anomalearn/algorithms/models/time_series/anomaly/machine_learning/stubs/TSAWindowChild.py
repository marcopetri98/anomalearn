import numpy as np

from anomalearn.algorithms.models.time_series.anomaly.machine_learning import TSAWindow


class TSAWindowChild(TSAWindow):
    def __init__(self, window: int = 5,
                 stride: int = 1,
                 scaling: str = "minmax",
                 scoring: str = "average",
                 classification: str = "voting",
                 threshold: float = None,
                 anomaly_portion: float = 0.01):
        super().__init__(window=window,
                         stride=stride,
                         scaling=scaling,
                         scoring=scoring,
                         classification=classification,
                         threshold=threshold,
                         anomaly_portion=anomaly_portion)

    def anomaly_score(self, x, *args, **kwargs) -> np.ndarray:
        x_new, windows_per_point = self._project_time_series(x)

        window_scores = self._compute_window_scores(x_new)
        anomaly_scores = self._compute_point_scores(window_scores,
                                                    windows_per_point)
        
        return anomaly_scores

    def classify(self, x, *args, **kwargs) -> np.ndarray:
        x_new, windows_per_point = self._project_time_series(x)

        window_scores = self._compute_window_scores(x_new)
        window_anomalies = self._compute_window_labels(x_new)
        anomaly_scores = self._compute_point_scores(window_scores,
                                                    windows_per_point)
        labels, _ = self._compute_point_labels(window_anomalies,
                                               windows_per_point,
                                               anomaly_scores)
        
        return labels
    
    def _compute_window_labels(self, vector_data: np.ndarray) -> np.ndarray:
        window_scores = self._compute_window_scores(vector_data)
        mean_score = np.mean(window_scores)
        return window_scores > mean_score

    def _compute_window_scores(self, vector_data: np.ndarray) -> np.ndarray:
        return np.sum(vector_data, axis=1)
