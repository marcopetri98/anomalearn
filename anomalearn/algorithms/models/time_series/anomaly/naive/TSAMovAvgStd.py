from numbers import Number

import numpy as np
from scipy.optimize import brute
from sklearn.metrics import f1_score

from . import TSAConstant
from ......utils import print_header, print_step, mov_avg, mov_std


class TSAMovAvgStd(TSAConstant):
    """This class learns a moving average or moving standard deviation classifier.

    This class is a learner for the rule `y = 1(movxxx(series, w) > c)` or for
    the rule `y = 1(movxxx(series, w) < c)` where `1(param)` is the indicator
    function of param and `xxx` can either be `avg` or `std`. The rule can be
    learned either in a semi-supervised or in a supervised fashion.

    This model has two parameters, the window of the mov_avg or mov_std and the
    constant that must be used to decide whether a point is anomalous or not.
    Ones the derived series `movxxx(series, w)` has been computed, the constant
    `c` can be found by means of `TSAConstant` naive classifier.

    If self-supervised, the constant is learnt as the one for which all normal
    data are under or over it. If supervised, the constant and the window are
    learnt as the one maximizing the F1 on the training set.

    Parameters
    ----------
    comparison : ["less", "greater", "auto"], default="auto"
        The comparison that the constant must do. With "greater" the learned
        function is `y = 1(movavg(series, w) > c)`, with "less" it is
        `y = 1(movavg(series, w) < c)`. With "auto" and supervised  learning it
        chooses the better. Otherwise, it is identical to "greater".

    learning : ["supervised"], default="supervised"
        States the type of "learning" that the function must perform. With
        "semi-supervised" it learns the constant from normal data only. With
        "supervised" it learns the constant from labeled data.
        
    max_window : int, default=100
        It is the maximum window of the time series to try to find the model.
        The model will search from windows within 3 and `max_window`.
        
    method : ["movavg", "movstd"], default="movavg"
        It is the moving window method to be used. It can either be "movavg" for
        moving average or "movstd" for moving standard deviation.

    Attributes
    ----------
    _window : int
        The sliding window to be used to compute the moving average or the
        moving standard deviation.

    _mov_avg_series : ndarray
        The moving average series computed when classifying a time series. This
        field is updated each time a time series is classified.

    _mov_std_series : ndarray
        The moving standard deviation series computed when classifying a time
        series. This field is updated each time a time series is classified.

    Notes
    -----
    The class inherits from `TSAConstant` since it is conceptually the same
    model. It must learn a constant to compare with the values of movavg or
    mov_std to decide whether a point is anomalous. The only difference is that
    the constant is learnt over a moving average series and not on the original
    series.
    """
    def __init__(self, comparison: str = "auto",
                 learning: str = "supervised",
                 max_window: int = 100,
                 method: str = "movavg"):
        super().__init__(comparison=comparison, learning=learning)

        self.max_window = max_window
        self.method = method

        self._window = 0
        self._mov_avg_series = None
        self._mov_std_series = None
        
        self.__check_parameters()

    def get_parameters(self) -> dict:
        """Gets all the parameters of the model.

        Returns
        -------
        parameters : dict
            A dictionary with all the parameters of the model, including the
            "private" parameters of the model. "comparison" is the `comparison`
            parameter of the model, "learning" is the `learning` parameter of
            the model, "constant" is the constant or the list of constants
            learnt by the model during fit, "learnt_comparison" is the learnt
            comparison or comparisons by the model during fit, "multivariate"
            is the flag learnt during fit by the model to specify whether the
            series is multivariate or univariate, "max_window" is the
            `max_window` parameter of the model, "method" is the `method`
            parameter of the model and "window" is the window learnt by the
            model during fit.
        """
        all_params = super().get_parameters()
        all_params["max_window"] = self.max_window
        all_params["method"] = self.method
        all_params["window"] = self._window
        return all_params

    def get_moving_series(self) -> np.ndarray | None:
        """Gets the moving average or standard deviation series depending on `method`.

        Returns
        -------
        moving_series : ndarray
            It is the moving average or moving standard deviation series
            computed by the model after the `classify` method has been called on
            a time series.
        """
        return self._mov_avg_series if self._mov_avg_series is not None else self._mov_std_series

    def set_parameters(self, comparison: str = None,
                       learning: str = None,
                       constant: Number | list[Number] = None,
                       learnt_comparison: str | list[str] = None,
                       multivariate: bool = None,
                       max_window: int = None,
                       method: str = None,
                       window: Number | list[Number] = None,
                       *args,
                       **kwargs) -> None:
        """
        Parameters
        ----------
        max_window : int, default=None
            The maximum window to be searched during fit.

        method : str, default=None
            The method to be used.

        window : Number or list of Number, default=None
            The window learnt during fit.
        """
        self.max_window = max_window if max_window is not None else self.max_window
        self.method = method if method is not None else self.method
        self._window = window if window is not None else self._window
        super().set_parameters(comparison=comparison,
                               learning=learning,
                               constant=constant,
                               learnt_comparison=learnt_comparison,
                               multivariate=multivariate)

    def classify(self, x, verbose: bool = True, *args, **kwargs) -> np.ndarray:
        self._mov_avg_series = None
        self._mov_std_series = None
        
        if verbose:
            print_header("Started samples' classification")
            print_step(f"Computing {self.method} series")
        
        if self.method == "movavg":
            mov_series = mov_avg(x, self._window)
            self._mov_avg_series = mov_series
        else:
            mov_series = mov_std(x, self._window)
            self._mov_std_series = mov_series
        
        middle = super().classify(mov_series, verbose=verbose if verbose != 2 else True)
        half = int((self._window - 1) / 2)
        all_predictions = np.full(x.shape[0], np.nan)
        all_predictions[half:-half] = middle
        
        if verbose:
            print_header("Ended samples' classification")
        
        return all_predictions

    def fit(self, x, y=None, verbose: bool = True, *args, **kwargs) -> None:
        x = np.array(x)
        y = np.array(y)
        
        if verbose:
            print_header("Started learning of window and constant")

        self._multivariate = x.ndim != 1 and x.shape[1] != 1

        def compute_1_minus_f1(window, *params):
            try:
                window = round(window[0])
            except Exception:
                window = round(window)

            if window < 3 or window % 2 == 0:
                return 1
            else:
                half = int((window - 1) / 2)
                mov_series = mov_avg(x, window) if self.method == "movavg" else mov_std(x, window)
                targets = y[half:-half]
                super(TSAMovAvgStd, self).fit(mov_series, targets, verbose=verbose if verbose != 2 else True)
                
                return 1 - f1_score(targets, super(TSAMovAvgStd, self).classify(mov_series, verbose=False))

        if self.learning == "supervised":
            optimal_window = brute(compute_1_minus_f1, [(3, self.max_window)])
            self._window = round(optimal_window[0])
            _ = compute_1_minus_f1(self._window)
        
        if verbose:
            print_header("Ended learning of window and constant")
            
    def __check_parameters(self):
        if self.learning != "supervised":
            raise ValueError("moving average method only accepts supervised "
                             "training")
