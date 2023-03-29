import numpy as np
from sklearn.svm import OneClassSVM

from .TSAOCSVM import TSAOCSVM


class TSAOCSVMPhase(TSAOCSVM):
    """OSVM adaptation to time series using scikit-learn.

    Parameters
    ----------
    windows : list[int]
        The list of all the windows to try in the phase space.

    phase_agreement : float
        The agreement level between different phases to classify a point as
        anomalous.

    Attributes
    ----------
    models : list[OneClassSVM]
        A list of OneClassSVM models for each window computed in the phase
        space.

    See Also
    --------
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
    """

    def __init__(self, windows: list[int] = None,
                 stride: int = 1,
                 scaling: str = "minmax",
                 scoring: str = "average",
                 classification: str = "voting",
                 threshold: float = None,
                 anomaly_portion: float = 0.01,
                 phase_agreement: float = 0.5,
                 kernel: str = "rbf",
                 degree: int = 3,
                 gamma: str = "scale",
                 coef0: float = 0.0,
                 tol: float = 1e-3,
                 nu: float = 0.5,
                 shrinking: bool = True,
                 cache_size: float = 200,
                 verbose: bool = False,
                 max_iter: int = -1):
        if windows is None:
            windows = [5]

        super().__init__(window=windows[0],
                         stride=stride,
                         scaling=scaling,
                         scoring=scoring,
                         classification=classification,
                         threshold=threshold,
                         anomaly_portion=anomaly_portion)

        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter

        self.windows = windows
        self.phase_agreement = phase_agreement
        self.models = None

    def fit(self, x, y=None, *args, **kwargs) -> None:
        self.models = []
        for embedding in self.windows:
            self.window = embedding
            super().fit(x, y)
            self.models.append(self._wrapped_model)

    def classify(self, x, *args, **kwargs) -> np.ndarray:
        anomaly_votes = np.zeros(x.shape[0])
        for i in range(len(self.windows)):
            self.window = self.windows[i]
            self._wrapped_model = self.models[i]

            phase_anomalies = super().classify(x) / len(self.windows)
            anomaly_votes += phase_anomalies

        final_anomalies = np.argwhere(anomaly_votes > self.phase_agreement)
        anomalies = np.zeros(anomaly_votes.shape)
        anomalies[final_anomalies] = 1

        return anomalies

    def anomaly_score(self, x, *args, **kwargs) -> np.ndarray:
        scores = np.zeros(x.shape[0])
        for i in range(len(self.windows)):
            self.window = self.windows[i]
            self._wrapped_model = self.models[i]

            phase_scores = super().anomaly_score(x) / len(self.windows)
            scores += phase_scores

        return scores
