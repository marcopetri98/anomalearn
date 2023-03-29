from abc import ABC
from typing import Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array

from . import ITimeSeriesAnomalyWindow
from .....BaseModel import BaseModel
from ......input_validation.array_checks import check_x_y_smaller_1d
from ......utils import print_warning


class TSAWindow(ITimeSeriesAnomalyWindow, BaseModel, ABC):
    """Abstract class implementing a sliding window approach.

    Parameters
    ----------
    window : int, default=5
        The length of the window to consider performing anomaly detection.

    stride : int, default=1
        The offset at which the window is moved when computing the anomalies.

    scaling: {"none", "minmax"}, default="minmax"
        The scaling method to scale the anomaly scores.

    scoring: {"left", "centre", "right", "min", "max", "average"}, default="average"
        The scoring method used compute the anomaly scores. When "centre" the
        score of a point is computed as the score of the window centred on that
        point (i.e., windows must be odd), if there is no window centred on the
        point the score is NaN. If "min" the score is the minimum anomaly score
        of the window that contains the point. If "max" the score is the maximum
        anomaly score of the window that contains the point. If "average" the
        score is the average score of all windows containing the point.

    classification: {"left", "centre", "right", "voting", "majority_voting", "unanimity", "points_score"}, default="voting"
        It defines the way in which a point is declared as anomaly. With voting,
        a point is an anomaly if at least `threshold` percentage of windows
        containing the point agree in saying it is an anomaly. With
        points_score, the points are considered anomalies if their score is
        above anomaly_threshold. `majority_voting` is a specific case of voting
        in which the `threshold` is always 0.5. The methods `left`, `centre` and
        `right` states that the classification of the points is propagated from
        the windows containing the point respectively at the left, at the centre
        and at the right. `unanimity` voting means that all the windows
        containing a point must agree to say that a point is anomalous.

    threshold: float, default=None
        The threshold used to compute if a point is an anomaly or not. In case
        classification is majority_voting, left, centre or right it is ignored.

    anomaly_portion: float, default=0.01
        The percentage of anomaly points in the dataset.
    """
    ACCEPTED_SCORING_METHODS = ["left", "centre", "right", "min", "max", "average"]
    ACCEPTED_SCALING_METHODS = ["none", "minmax"]
    ACCEPTED_LABELLING_METHODS = ["left", "centre", "right", "voting", "unanimity", "majority_voting", "points_score"]

    def __init__(self, window: int = 5,
                 stride: int = 1,
                 scaling: str = "minmax",
                 scoring: str = "average",
                 classification: str = "voting",
                 threshold: float = None,
                 anomaly_portion: float = 0.01):
        super().__init__()

        self.window = window
        self.stride = stride
        self.scaling = scaling
        self.scoring = scoring
        self.classification = classification
        self.threshold = threshold if threshold is not None else 0.5
        self.anomaly_portion = anomaly_portion

        self.__check_parameters()

    def set_params(self, **params) -> None:
        super().set_params(**params)
        self.__check_parameters()

    def _project_time_series(self, time_series: np.ndarray) -> Tuple[np.ndarray,
                                                                     np.ndarray]:
        # Input validation
        check_array(time_series)
        data = np.array(time_series)

        if self.window > data.shape[0]:
            raise ValueError("Window cannot be larger than data size.")
        elif data.shape[1] > 1:
            raise ValueError("Only univariate time series is currently "
                             "supported.")
        elif (data.shape[0] - self.window) % self.stride != 0:
            print_warning("Stride does not divide data.shape[0] - window, "
                          "points not included at the end will be discarded.")

        # Number of times a point is considered in a window
        num_windows = np.zeros(data.shape[0])
        x_new = []

        # Transform univariate time series into spatial data
        for i in range(0, data.shape[0] - self.window + 1, self.stride):
            num_windows[i:i + self.window] += 1
            current_data: np.ndarray = data[i:i + self.window]
            current_data = current_data.reshape(-1)
            x_new.append(current_data.tolist())

        x_new = np.array(x_new)

        return x_new, num_windows

    def _compute_point_scores(self, window_scores,
                              windows_per_point) -> np.ndarray:
        check_x_y_smaller_1d(window_scores, windows_per_point, force_all_finite=False)

        window_scores = np.array(window_scores)
        windows_per_point = np.array(windows_per_point)

        scores = np.zeros(windows_per_point.shape[0])

        # Compute score of each point
        if self.scoring in ["min", "max"]:
            scores_list = [[] for _ in range(scores.shape[0])]
            for i in range(window_scores.shape[0]):
                idx = i * self.stride
                # handle the possibility of having infinite or nan values as
                # scores for windows
                if not (np.isinf(window_scores[i]) or np.isnan(window_scores[i])):
                    for j in range(idx, idx + self.window):
                        scores_list[j].append(window_scores[i])

            for i in range(scores.shape[0]):
                if len(scores_list[i]) != 0:
                    if self.scoring == "min":
                        scores[i] = min(scores_list[i]) if len(scores_list[i]) != 0 else np.nan
                    else:
                        scores[i] = max(scores_list[i]) if len(scores_list[i]) != 0 else np.nan
                else:
                    scores[i] = np.nan

        elif self.scoring == "average":
            # handle points without window to avoid division by zero
            no_windows = np.where(windows_per_point == 0)
            windows_per_point[no_windows] = 1

            for i in range(window_scores.shape[0]):
                idx = i * self.stride
                if not (np.isinf(window_scores[i]) or np.isnan(window_scores[i])):
                    scores[idx:idx + self.window] += window_scores[i]
                else:
                    windows_per_point[idx:idx + self.window] -= 1

            # give score and restore nan when there are no windows
            scores = scores / windows_per_point
            scores[no_windows] = np.nan

        elif self.scoring == "left":
            scores[:] = np.nan
            for i in range(window_scores.shape[0]):
                idx = i * self.stride
                scores[idx] = window_scores[i]

        elif self.scoring == "right":
            scores[:] = np.nan
            for i in range(window_scores.shape[0]):
                idx = i * self.stride
                scores[idx + self.window - 1] = window_scores[i]

        elif self.scoring == "centre":
            scores[:] = np.nan
            half_window = int((self.window - 1) / 2)
            for i in range(window_scores.shape[0]):
                idx = i * self.stride
                scores[idx + half_window] = window_scores[i]

        if (np.isnan(scores).any() or np.isinf(scores).any()) and self.scaling == "none":
            print_warning("Cannot perform scale since there are nan or infinite"
                          " values in scores.")
        else:
            # Scale the scores if requested
            match self.scaling:
                case "minmax":
                    # Min-max normalization
                    scores = scores.reshape((scores.shape[0], 1))
                    scores = MinMaxScaler().fit_transform(scores)
                    scores = scores.reshape(scores.shape[0])

        return scores

    def _compute_point_labels(self, window_labels,
                              windows_per_point,
                              point_scores=None) -> Tuple[np.ndarray, float]:
        check_x_y_smaller_1d(window_labels, windows_per_point)

        window_labels = np.array(window_labels)
        windows_per_point = np.array(windows_per_point)

        threshold = self.threshold
        labels = np.zeros(windows_per_point.shape[0])

        if self.classification in ["voting", "majority_voting", "unanimity"]:
            # compute the percentage of windows that agree that ith point is
            # an anomalous point
            for i in range(window_labels.shape[0]):
                if window_labels[i] == 1:
                    idx = i * self.stride
                    labels[idx:idx + self.window] += 1
            labels = labels / windows_per_point

            if self.classification == "unanimity":
                true_anomalies = np.argwhere(labels == 1)
                labels = np.zeros(labels.shape)
                labels[true_anomalies] = 1
            else:
                # compute anomalous points based on an agreement threshold
                tau = 0.5 if self.classification == "majority_voting" else threshold
                true_anomalies = np.argwhere(labels > tau)
                labels = np.zeros(labels.shape)
                labels[true_anomalies] = 1

        elif self.classification == "points_score":
            # anomalies are points with a score higher than the threshold
            labels[np.argwhere(point_scores > threshold)] = 1

        elif self.classification == "left":
            # the window for which the point is at the left defines the label
            labels[:] = np.nan
            for i in range(window_labels.shape[0]):
                idx = i * self.stride
                labels[idx] = window_labels[i]

        elif self.classification == "right":
            # the window for which the point is at the right defines the label
            labels[:] = np.nan
            for i in range(window_labels.shape[0]):
                idx = i * self.stride
                labels[idx + self.window - 1] = window_labels[i]

        elif self.classification == "centre":
            # the window centred on the point decides the label
            labels[:] = np.nan
            half_window = int((self.window - 1) / 2)
            for i in range(window_labels.shape[0]):
                idx = i * self.stride
                labels[idx + half_window] = window_labels[i]

        return labels, threshold

    def __check_parameters(self) -> None:
        """Checks that the class parameters are correct.

        Returns
        -------
        None
        """
        if self.scoring not in self.ACCEPTED_SCORING_METHODS:
            raise ValueError("Scoring method must be one of the following: " +
                             str(self.ACCEPTED_SCORING_METHODS))
        elif self.scaling not in self.ACCEPTED_SCALING_METHODS:
            raise ValueError("Scoring method must be one of the following: " +
                             str(self.ACCEPTED_SCALING_METHODS))
        elif self.classification not in self.ACCEPTED_LABELLING_METHODS:
            raise ValueError("Scoring method must be one of the following: " +
                             str(self.ACCEPTED_LABELLING_METHODS))
        elif self.window <= 0 or self.stride <= 0:
            raise ValueError("Stride and window must be positive.")
        elif self.threshold is not None and self.classification == "voting" and not 0 <= self.threshold <= 1:
            raise ValueError("Threshold must be None or 0 <= threshold <= 1")
        elif not 0 < self.anomaly_portion <= 0.5:
            raise ValueError("The contamination must be inside (0,0.5]")

        if self.scoring in ["centre"] and self.window % 2 == 0:
            raise ValueError("If scoring is {}, the window must be odd".format(self.scoring))
        elif self.scoring in ["left", "centre", "right"] and self.stride != 1:
            raise ValueError("If scoring is {}, the stride must be 1, otherwise points will be missed".format(self.scoring))

        if self.classification in ["centre"] and self.window % 2 == 0:
            raise ValueError("If classification is {}, the window must be odd".format(self.classification))
        elif self.classification in ["left", "centre", "right"] and self.stride != 1:
            raise ValueError("If classification is {}, the stride must be 1, otherwise points will be missed".format(self.classification))
