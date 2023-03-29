from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from . import TSAStatistical
from ......utils import save_py_json, load_py_json


class TSAES(TSAStatistical):
    """ES model to perform anomaly detection on time series.

    When points are predicted using SES, it is important to know that the points
    to be predicted must be the whole sequence immediately after the training.
    It is not possible to predicted from the 50th point to the 100th with the
    current implementation.

    Notes
    -----
    For all the other parameters that are not included in the documentation, see
    the `statsmodels <https://www.statsmodels.org/stable/api.html>`
    documentation for `ARIMA models <https://www.statsmodels.org/stable/generate
    d/statsmodels.tsa.holtwinters.SimpleExpSmoothing.html>`."""
    __es_params = "tsa_es_parameters.csv"
    __es_build_params = "tsa_es_build_params.json"

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

        self._build_params = None

    def save(self, path: str,
             *args,
             **kwargs) -> None:
        super().save(path=path)

        path_obj = Path(path)

        params_formatted = self._fitted_model.params_formatted.reset_index()
        params_formatted.to_csv(str(path_obj / self.__es_params), index=False)

        if self._build_params is not None:
            new_build_params = self._build_params.copy()
            for key, value in new_build_params.items():
                if isinstance(value, np.ndarray):
                    new_build_params[key] = value.tolist()
                elif isinstance(value, np.ma.MaskedArray):
                    new_build_params[key] = value.tolist(fill_value=None)
        else:
            new_build_params = None
        save_py_json(new_build_params, str(path_obj / self.__es_build_params))

    def load(self, path: str,
             *args,
             **kwargs) -> None:
        super().load(path=path)

        path_obj = Path(path)

        params_formatted = pd.read_csv(str(path_obj / self.__es_params), index_col="index")

        self._build_params = load_py_json(str(path_obj / self.__es_build_params))
        self._build_params["endog"] = np.random.rand(100)
        self._model_build(build_params=self._build_params)
        self._model_fit()
        self._fitted_model.params_formatted = params_formatted

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

        self._build_params = build_params

        super().fit(x=x,
                    y=y,
                    verbose=verbose,
                    fit_params=fit_params,
                    build_params=build_params)

    def _model_predict(self, x: np.ndarray,
                       *args,
                       **kwargs):
        if self.prediction_horizon == 1:
            build_params = self._build_params.copy()
            build_params["endog"] = x
            self._model_build(build_params=build_params)
            new_df = self._fitted_model.params_formatted.reset_index()
            params = {row["index"]: row["param"] for idx, row in new_df.iterrows()}
            self._fitted_model.initialize(self._model, params)
            prediction_results = self._fitted_model.predict(0)
            predictions = prediction_results
        else:
            predictions = np.full((x.shape[0], self.prediction_horizon), np.nan)

            for i in range(x.shape[0] - self.prediction_horizon - 1):
                build_params = self._build_params.copy()
                build_params["endog"] = x[:i + 2]
                self._model_build(build_params=build_params)
                new_df = self._fitted_model.params_formatted.reset_index()
                params = {row["index"]: row["param"] for idx, row in new_df.iterrows()}
                self._fitted_model.initialize(self._model, params)
                prediction_results = self._fitted_model.forecast(self.prediction_horizon)
                np.fill_diagonal(predictions[i + 2:i + 2 + self.prediction_horizon], prediction_results)

        return predictions.reshape((-1, x.shape[1], self.prediction_horizon))

    def _model_build(self, build_params: dict = None,
                     *args,
                     **kwargs) -> None:
        if build_params is None:
            raise ValueError("exponential smoothing needs endog in order to be created")

        self._model = ExponentialSmoothing(**build_params)
