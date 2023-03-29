import abc

import numpy as np

from .... import IAnomalyScorer, IAnomalyClassifier


class ITimeSeriesAnomalyWindow(IAnomalyScorer, IAnomalyClassifier):
    """Interface for sliding window univariate time series anomaly detection.
    """
    
    @abc.abstractmethod
    def _project_time_series(self, time_series: np.ndarray) -> np.ndarray:
        """Compute the new space.

        Parameters
        ----------
        time_series : array-like of shape (n_samples, n_features)
            The input data to be transformed.

        Returns
        -------
        X_new : ndarray of shape (n_windows, window)
            The transformed data.

        num_windows : ndarray of shape (n_samples,)
            The number of windows containing the point at index i.
        """
        pass
    
    @abc.abstractmethod
    def _compute_point_scores(self, window_scores,
                              windows_per_point) -> np.ndarray:
        """Computes the scoring of the points for the time series.

        Parameters
        ----------
        window_scores : array-like of shape (n_windows,)
            The scores of the windows for the time series to be used to compute
            the scores of the points.
        windows_per_point : array-like of shape (n_points,)
            The number of windows containing the point at that specific index.

        Returns
        -------
        point_scores : ndarray
            The scores of the points.
        """
        pass
    
    @abc.abstractmethod
    def _compute_point_labels(self, window_labels,
                              windows_per_point,
                              point_scores=None) -> np.ndarray:
        """Computes the scoring of the points for the time series.
        
        Receives as input the labels of the windows and an array with the
        dimension of the time series from which windows have been obtained with
        the number of windows for each point that contains it.

        Parameters
        ----------
        window_labels : array-like of shape (n_windows,)
            The labels of the windows for the time series to be used to compute
            the labels of the points.

        windows_per_point : array-like of shape (n_points,)
            The number of windows containing the point at that specific index.

        point_scores : array-like of shape (n_points,), default=None
            The scores of the points in range [0,1].

        Returns
        -------
        point_labels : ndarray of shape (n_points,)
            The labels of the points.
        threshold : float
            The threshold above which points are considered as anomalies.
        """
        pass
    
    @abc.abstractmethod
    def _compute_window_scores(self, vector_data: np.ndarray) -> np.ndarray:
        """Compute the score of the passed windows.
        
        Parameters
        ----------
        vector_data : ndarray of shape (n_samples, n_features)
            The vector data on which we need to compute the anomaly score.

        Returns
        -------
        window_scores : ndarray of shape (n_samples,)
            The anomaly score of each window.
        """
        pass
    
    @abc.abstractmethod
    def _compute_window_labels(self, vector_data: np.ndarray) -> np.ndarray:
        """Compute the labels of the passed windows.
        
        Parameters
        ----------
        vector_data : ndarray of shape (n_samples, n_features)
            The vector data on which we need to compute the anomaly score.

        Returns
        -------
        window_anomalies : ndarray of shape (n_samples,)
            The anomaly label of each window.
        """
        pass
