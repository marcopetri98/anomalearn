from numbers import Number

import numpy as np
from scipy.optimize import brute
from sklearn.metrics import f1_score

from .... import IAnomalyClassifier
from ..... import IParametric
from ......utils import print_header, print_step


class TSAConstant(IAnomalyClassifier, IParametric):
    """This class learns a constant to classify anomalies.

    This class is a learner of the rule `y = 1(x > c)` or `y = 1(x < c)` where
    `1(param)` is the indicator function of param. This rule can be learned
    either in self-supervised or supervised fashion.

    If self-supervised, the constant is learned as the one for which all normal
    data are under or over it. If supervised, the constant is learned as the one
    maximizing the F1 on the training set.

    Parameters
    ----------
    comparison : ["less", "greater", "auto"], default="auto"
        The comparison that the constant must do. With "greater" the learned
        function is `y = 1(x > c)`, with "less" it is `y = 1(x < c)`. With
        "auto" and supervised learning it chooses the better. Otherwise, it
        is identical to "greater".

    learning : ["semi-supervised", "supervised"], default="supervised"
        States the type of "learning" that the function must perform. With
        "semi-supervised" it learns the constant from normal data only. With
        "supervised" it learns the constant from labeled data.

    Attributes
    ----------
    _anomaly_label : int
        The label of an anomaly.

    _multivariate : bool
        A flag learnt during fit stating if the time series is multivariate.

    _constant : Number or list of Number
        The constant or constants learnt during model fit to classify anomalies.

    _comparison : str or list of str
        The comparison or comparisons learnt during model fit to classify
        anomalies.
    """
    def __init__(self, comparison: str = "auto",
                 learning: str = "supervised"):
        super().__init__()
        
        self.comparison = comparison
        self.learning = learning
        
        self._anomaly_label = 1
        self._multivariate = None
        self._constant = None
        self._comparison = self.comparison
        
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
            comparison or comparisons by the model during fit and "multivariate"
            is the flag learnt during fit by the model to specify whether the
            series is multivariate or univariate.
        """
        all_params = dict()
        all_params["comparison"] = self.comparison
        all_params["learning"] = self.learning
        try:
            all_params["constant"] = self._constant.tolist()
            all_params["learnt_comparison"] = self._comparison.copy()
        except:
            all_params["constant"] = self._constant
            all_params["learnt_comparison"] = self._comparison
        all_params["multivariate"] = self._multivariate
        return all_params

    def set_parameters(self, comparison: str = None,
                       learning: str = None,
                       constant: Number | list[Number] = None,
                       learnt_comparison: str | list[str] = None,
                       multivariate: bool = None,
                       *args,
                       **kwargs) -> None:
        """Sets the parameters of the model.

        Parameters
        ----------
        comparison : str, default=None
            The way in which to learn comparisons.

        learning : str, default=None
            The learning method.

        constant : Number or list of Number, default=None
            The learnt constant or constants during fit.

        learnt_comparison : str or list of str, default=None
            The learnt comparison or comparisons during fit.

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
        self.comparison = comparison if comparison is not None else self.comparison
        self.learning = learning if learning is not None else self.learning
        self._constant = constant if constant is not None else self._constant
        self._comparison = learnt_comparison if learnt_comparison is not None else self._comparison
        self._multivariate = multivariate if multivariate is not None else self._multivariate

        if isinstance(self._constant, list):
            self._constant = np.array(self._constant)

    def classify(self, x, verbose: bool = True, *args, **kwargs) -> np.ndarray:
        """
        Parameters
        ----------
        verbose : bool or int, default=True
            States if verbose printing must be done. With False very little
            printing is performed. With True some printing is done. With 2,
            extremely detailed printing is performed.
        """
        if self._multivariate is None or self._constant is None:
            raise ValueError("fit the model before classifying samples")

        if verbose == 2:
            print_header("Started samples' classification")
        if verbose:
            print_step(f"Classifying using rule: {self._comparison}")

        x = np.array(x)
        compared = np.zeros(x.shape)
        
        if self._multivariate:
            for dim in range(compared.shape[1]):
                if self._comparison[dim] == "less":
                    compared[:, dim] = x[:, dim] < self._constant[dim]
                else:
                    compared[:, dim] = x[:, dim] > self._constant[dim]
        else:
            if self._comparison == "less":
                compared = x < self._constant
            else:
                compared = x > self._constant

        if verbose == 2:
            print_header("Ended samples' classification")

        if self._multivariate:
            return np.array(list(map(lambda row: np.min(row), compared)))
        else:
            return compared.flatten()

    def fit(self, x, y=None, verbose: bool | int = 2, *args, **kwargs) -> None:
        """
        Parameters
        ----------
        verbose : bool or int, default=True
            States if verbose printing must be done. With False very little
            printing is performed. With True some printing is done. With 2,
            extremely detailed printing is performed.
        """
        x = np.array(x)
        
        if verbose == 2:
            print_header("Started constant's learning")
            print_step("Check if time series is multivariate")
        
        self._multivariate = x.ndim != 1 and x.shape[1] != 1
        
        if self.learning == "semi-supervised":
            if verbose:
                print_step("Get minimum or maximum values with semi-supervised learning")
            
            if self._multivariate:
                self._constant = np.zeros(x.shape[1])
                for i in range(x.shape[1]):
                    self._constant[i] = np.min(x[:, i]) if self.comparison == "less" else np.max(x[:, i])
            else:
                self._constant = np.min(x) if self.comparison == "less" else np.max(x)
        else:
            y = np.array(y)
            
            if y.shape[0] != x.shape[0]:
                raise ValueError("x and y must have the same number of points")
            elif self._anomaly_label not in y:
                raise ValueError("supervised training requires at least one anomaly")

            def get_comp_cons(optimize_res_lt, optimize_res_gt):
                match self.comparison:
                    case "less":
                        return "less", optimize_res_lt[0]

                    case "greater":
                        return "greater", optimize_res_gt[0]

                    case "auto":
                        if optimize_res_lt[1] < optimize_res_gt[1]:
                            return "less", optimize_res_lt[0]
                        else:
                            return "greater", optimize_res_gt[0]

            def maximize_f1_lt(c, *params):
                try:
                    c = c[0]
                except Exception:
                    c = c
                return 1 - f1_score(y, params[0] < c)

            def maximize_f1_gt(c, *params):
                try:
                    c = c[0]
                except Exception:
                    c = c
                return 1 - f1_score(y, params[0] > c)

            def globally_optimize(series):
                def call_obj_func1(points):
                    return maximize_f1_lt(points, series)

                def call_obj_func2(points):
                    return maximize_f1_gt(points, series)

                optimal_lt = brute(call_obj_func1, [(int(np.min(series)), int(np.max(series)))], full_output=True)
                optimal_gt = brute(call_obj_func2, [(int(np.min(series)), int(np.max(series)))], full_output=True)
                return optimal_lt, optimal_gt

            if self._multivariate:
                if verbose:
                    print_step("Get optimal constant and comparison using supervised learning for each channel")
                
                self._constant = np.zeros(x.shape[1])
                self._comparison = [None for _ in range(x.shape[1])]
                for i in range(x.shape[1]):
                    if verbose:
                        print_step(f"Learning on channel {i}")
                    
                    optimal_lt, optimal_gt = globally_optimize(x[:, i])
                    self._comparison[i], self._constant[i] = get_comp_cons(optimal_lt, optimal_gt)
            else:
                if verbose:
                    print_step("Get optimal constant and comparison using supervised learning")
                
                optimal_lt, optimal_gt = globally_optimize(x)
                self._comparison, self._constant = get_comp_cons(optimal_lt, optimal_gt)
                if isinstance(self._constant, np.ndarray):
                    self._constant = self._constant[0]
                
        if verbose == 2:
            print_header("Ended constant's learning")

    def __check_parameters(self):
        if not isinstance(self.comparison, str):
            raise TypeError("comparison must be a str")
        elif not isinstance(self.learning, str):
            raise TypeError("learning must be a str")

        comparisons = ["less", "greater", "auto"]
        learnings = ["semi-supervised", "supervised"]

        if self.comparison not in comparisons:
            raise ValueError(f"comparison must be one of {comparisons}")
        elif self.learning not in learnings:
            raise ValueError(f"learning must be one of {learnings}")
