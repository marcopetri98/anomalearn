from pathlib import Path
from typing import Callable

import numpy as np
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from . import TSAStatistical


class TSAARIMA(TSAStatistical):
    """ARIMA model to perform anomaly detection on time series.

    This method uses ARIMA to predict points for the time series and uses the
    prediction error as a measure of abnormality. The class is a wrapper around
    the ARIMA model, the fit parameters must not contain `endog` since it is
    automatically set by the fit function. All the other parameters taken during
    model's build and/or fit must be passed though `fit_params` or
    `build_params`.

    Notes
    -----
    For all the other parameters that are not included in the documentation, see
    the `statsmodels <https://www.statsmodels.org/stable/api.html>`
    documentation for `ARIMA models <https://www.statsmodels.org/stable/generate
    d/statsmodels.tsa.arima.model.ARIMA.html>`.
    """
    __arima_file = "tsa_arima_model.pickle"

    def __init__(self, prediction_horizon: int = 1,
                 validation_split: float = 0.1,
                 mean_cov_sets: str = "training",
                 threshold_sets: str = "training",
                 error_method: str = "difference",
                 error_function: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
                 threshold_computation: str = "gaussian",
                 threshold_function: Callable[[np.ndarray], np.ndarray] | None = None,
                 scoring_function: str | Callable[[np.ndarray], np.ndarray] = "gaussian"):
        super().__init__(prediction_horizon=prediction_horizon,
                         validation_split=validation_split,
                         mean_cov_sets=mean_cov_sets,
                         threshold_sets=threshold_sets,
                         error_method=error_method,
                         error_function=error_function,
                         threshold_computation=threshold_computation,
                         threshold_function=threshold_function,
                         scoring_function=scoring_function)

    def save(self, path: str,
             *args,
             **kwargs) -> None:
        super().save(path=path)

        path_obj = Path(path)

        self._fitted_model.save(str(path_obj / self.__arima_file))

    def load(self, path: str,
             *args,
             **kwargs) -> None:
        super().load(path=path)

        path_obj = Path(path)

        self._fitted_model = ARIMAResults.load(str(path_obj / self.__arima_file))

    def fit(self, x=None,
            y=None,
            verbose: bool = True,
            fit_params: dict = None,
            build_params: dict = None,
            *args,
            **kwargs):
        num_validation = round(x.shape[0] * self.validation_split)
        if build_params is not None:
            build_params["endog"] = x[:-num_validation]
        else:
            build_params = {"endog": x[:-num_validation]}

        super().fit(x=x,
                    y=y,
                    verbose=verbose,
                    fit_params=fit_params,
                    build_params=build_params)

    def _model_predict(self, x: np.ndarray,
                       *args,
                       **kwargs):
        if self.prediction_horizon == 1:
            pred_model: ARIMAResults = self._fitted_model.apply(x, refit=False)
            prediction_results = pred_model.predict(0)
            predictions = prediction_results
        else:
            predictions = np.full((x.shape[0], self.prediction_horizon), np.nan)

            for i in range(x.shape[0] - self.prediction_horizon):
                data = x[:i + 1]
                pred_model: ARIMAResults = self._fitted_model.apply(data, refit=False)
                prediction_results = pred_model.forecast(self.prediction_horizon)
                np.fill_diagonal(predictions[i + 1:i + 1 + self.prediction_horizon], prediction_results)

        return predictions.reshape((-1, x.shape[1], self.prediction_horizon))

    def _model_build(self, build_params: dict = None,
                     *args,
                     **kwargs) -> None:
        if build_params is None:
            raise ValueError("arima needs endog in order to be created")

        self._model = ARIMA(**build_params)
