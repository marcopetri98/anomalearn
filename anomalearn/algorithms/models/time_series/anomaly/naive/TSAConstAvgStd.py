from numbers import Number

import numpy as np
from scipy.optimize import brute
from sklearn.metrics import f1_score

from .... import IAnomalyClassifier
from ..... import IParametric
from ......utils import mov_avg, mov_std, print_header, print_step


class TSAConstAvgStd(IAnomalyClassifier, IParametric):
    """This class learns a simple classifier.
    
    This class is a learner for the rule `y = 1(a * movavg(x, w) + b * movstd(x, w)
    + c < x) OR 1(a * movavg(x, w) - b * movstd(x, w) + c > x)` where `w` is an
    odd number representing the moving window, `movavg` stands for moving
    average, `movstd` stands for moving standard deviation, and `a`, `b`, `c`
    are real numbers. The symbol `x` stands for the input time series. The
    function `1(param)` is the indicator function. The rule can be learned
    either in a semi-supervised fashion or in a fully supervised fashion.
    
    The model parameters are learnt over all the channel of the series, they are
    not learnt channel-wise for the moment.
    
    Parameters
    ----------
    learning : ["supervised"], default="supervised"
        States the type of "learning" that the function must perform. With
        "semi-supervised" it learns the constant from normal data only. With
        "supervised" it learns the constant from labeled data.
        
    max_window : int, default=100
        It is the maximum window of the time series to try to find the model.
        The model will search from windows within 3 and `max_window`.

    Attributes
    ----------
    _anomaly_label : int
        The label of the anomalies.

    _multivariate : bool
        A flag learnt during fit stating if the time series is multivariate.

    _lower_series : ndarray
        A series representing the lower series obtained from the one classified
        using `classify` method.

    _upper_series : ndarray
        A series representing the upper series obtained from the one classified
        using `classify` method.

    _a : ndarray or Number
        It is the parameter `a` of the model.

    _b : ndarray or Number
        It is the parameter `b` of the model.

    _c : ndarray or Number
        It is the parameter `c` of the model.

    _w : Number
        It is the window used by the method to classify the time series.
    """
    def __init__(self, learning: str = "supervised",
                 max_window: int = 100):
        super().__init__()
        
        self.learning = learning
        self.max_window = max_window

        self._anomaly_label = 1
        self._multivariate = None
        self._lower_series = None
        self._upper_series = None
        self._a = None
        self._b = None
        self._c = None
        self._w = None
        
    def get_upper_series(self) -> np.ndarray | None:
        """Gets the lower bound series.

        Returns
        -------
        lower_series : ndarray
            It is the lower bound series computed by the model after the
            `classify` method has been called on a time series.
        """
        return self._upper_series
        
    def get_lower_series(self) -> np.ndarray | None:
        """Gets the upper bound series.

        Returns
        -------
        upper_series : ndarray
            It is the upper bound series computed by the model after the
            `classify` method has been called on a time series.
        """
        return self._lower_series

    def get_parameters(self) -> dict:
        """Gets all the parameters of the model.

        Returns
        -------
        parameters : dict
            A dictionary with all the parameters of the model, including the
            "private" parameters of the model. "learning" is the `learning`
            parameter of the model, "max_window" is the `max_window` parameter
            of the model, "multivariate" is the flag learnt during fit by the
            model to specify whether the series is multivariate or univariate,
            "a" is the parameter `a` of the model or the list of parameters `a`
            learnt, "b" is the parameter `b` of the model or the list of
            parameters `b` learnt, "c" is the parameter `c` of the model or the
            list of parameters `c` learnt and "w" is the window learnt by the
            model during fit.
        """
        all_params = dict()
        all_params["learning"] = self.learning
        all_params["max_window"] = self.max_window
        all_params["multivariate"] = self._multivariate
        try:
            all_params["a"] = self._a.tolist()
            all_params["b"] = self._b.tolist()
            all_params["c"] = self._c.tolist()
        except:
            all_params["a"] = self._a
            all_params["b"] = self._b
            all_params["c"] = self._c
        all_params["w"] = self._w
        return all_params

    def set_parameters(self, learning: str = None,
                       max_window: int = None,
                       a: Number | list[Number] = None,
                       b: Number | list[Number] = None,
                       c: Number | list[Number] = None,
                       w: Number | list[Number] = None,
                       multivariate: bool = None,
                       *args,
                       **kwargs) -> None:
        """Sets the parameters of the model.

        Parameters
        ----------
        learning : str, default=None
            The learning method.

        max_window : int, default=None
            The maximum window to be searched during fit.

        a : Number or list of Number, default=None
            It is the parameter `a` learnt by the model during fit or the list
            of parameters `a` learnt by the model during fit.

        b : Number or list of Number, default=None
            It is the parameter `b` learnt by the model during fit or the list
            of parameters  `b` learnt by the model during fit.

        c : Number or list of Number, default=None
            It is the parameter `c` learnt by the model during fit or the list
            of parameters `c` learnt by the model during fit.

        w : Number or list of Number, default=None

        multivariate : bool, default=None
            The flag learnt during fit stating if the series is multivariate or
            not.

        args
            Not used, present for signature change or multiple inheritance.

        kwargs
            Not used, present for signature change or multiple inheritance.

        Returns
        -------
        None
        """
        self.learning = learning if learning is not None else self.learning
        self.max_window = max_window if max_window is not None else self.max_window
        self._a = a if a is not None else self._a
        self._b = b if b is not None else self._b
        self._c = c if c is not None else self._c
        self._w = w if w is not None else self._w
        self._multivariate = multivariate if multivariate is not None else self._multivariate

        if isinstance(self._a, list):
            self._a = np.array(self._a)
        if isinstance(self._b, list):
            self._b = np.array(self._b)
        if isinstance(self._c, list):
            self._c = np.array(self._c)

    def classify(self, x, verbose: bool = True, *args, **kwargs) -> np.ndarray:
        """
        Parameters
        ----------
        verbose : bool or int, default=True
            States if verbose printing must be done. With False very little
            printing is performed. With True detailed printing is done.
        """
        if verbose:
            print_header("Started samples' classification")

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if verbose:
            print_step("Computing the moving average and moving std series")

        moving_avg = mov_avg(x, self._w)
        moving_std = mov_std(x, self._w)
        half = int((self._w - 1) / 2)

        if verbose:
            print_step("Computing the lower and upper series")

        self._upper_series = self._a * moving_avg + self._b * moving_std + self._c
        self._lower_series = self._a * moving_avg - self._b * moving_std + self._c

        if self._upper_series.ndim == 1:
            self._upper_series = self._upper_series.reshape(-1, 1)
            self._lower_series = self._lower_series.reshape(-1, 1)

        if verbose:
            print_step("Building the predictions")

        pred = (x[half:-half] > self._upper_series) | (x[half:-half] < self._lower_series)
        middle = np.array(list(map(lambda row: 1 if np.min(row) == 1 else 0, pred)))
        all_predictions = np.full(x.shape[0], np.nan)
        all_predictions[half:-half] = middle

        if verbose:
            print_header("Ended samples' classification")

        return all_predictions
    
    def fit(self, x, y=None, verbose: bool = True, *args, **kwargs) -> None:
        """
        Parameters
        ----------
        verbose : bool or int, default=True
            States if verbose printing must be done. With False very little
            printing is performed. With True detailed printing is done.
        """
        x = np.array(x)
        y = np.array(y)

        if verbose:
            print_header("Started learning")
            print_step("Checking if the time series is multivariate")

        self._multivariate = x.ndim != 1 and x.shape[1] != 1

        def compute_1_minus_f1(points, *params):
            a, b, c, w = points
            series, targets = params

            w = round(w)
            if w % 2 == 0:
                return 1
            
            moving_avg = mov_avg(series, w)
            moving_std = mov_std(series, w)
            half = int((w - 1) / 2)
            
            upper_boundary = a * moving_avg + b * moving_std + c
            lower_boundary = a * moving_avg - b * moving_std + c
            series = series[half:-half]

            if upper_boundary.ndim == 1:
                upper_boundary = upper_boundary.reshape(-1, 1)
                lower_boundary = lower_boundary.reshape(-1, 1)
            
            pred = (series > upper_boundary) | (series < lower_boundary)
            return 1 - f1_score(targets[half:-half], pred)

        def globally_optimize(extra_args):
            return brute(compute_1_minus_f1,
                         (slice(0, 2, 1), slice(-4, 4.1, 0.1), (np.min(extra_args[0]), np.max(extra_args[0])), (3, self.max_window)),
                         args=extra_args)

        if self.learning == "supervised":
            y = np.array(y)
            
            if y.shape[0] != x.shape[0]:
                raise ValueError("x and y must have the same number of points")
            elif self._anomaly_label not in y:
                raise ValueError("supervised training requires at least one anomaly")
            
            if self._multivariate:
                # TODO: implement the same procedure using a simple neural network to optimize the
                #  parameters using backpropagation since it supports nonlinear discontinuous functions
                if verbose:
                    print_step("Getting the optimal parameters")

                extra_params = (x, y)
                optimum = globally_optimize(extra_params)
                self._a, self._b, self._c, self._w = optimum
                self._w = round(self._w)
            else:
                if verbose:
                    print_step("Getting the optimal parameters")

                extra_params = (x, y)
                optimum = globally_optimize(extra_params)
                self._a, self._b, self._c, self._w = optimum
                self._w = round(self._w)

        if verbose:
            print_header("Ended learning")

    def __check_parameters(self):
        if not isinstance(self.learning, str):
            raise TypeError("learning must be a str")

        learnings = ["supervised"]

        if self.learning not in learnings:
            raise ValueError(f"learning must be one of {learnings}")
