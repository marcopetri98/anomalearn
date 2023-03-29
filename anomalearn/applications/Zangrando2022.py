import os.path
from numbers import Number
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from ..reader.time_series import TSMultipleReader, ODINTSReader
from ..utils import print_header, print_step


class Zangrando2022Loader(object):
    """Data loader for the paper Zangrando 2022 (https://doi.org/10.1186/s42162-022-00230-7).

    The paper implemented in this class is "Anomaly detection in quasi-periodic
    energy consumption data series: a comparison of algorithms" by the authors
    Niccolò Zangrando, Piero Fraternali, Marco Petri, Nicolò Oreste Pinciroli
    Vago and Sergio Luis Herrera González.

    This class implements the data loading procedures implemented in the paper
    to train the models. It the methods used to extract the training sequences,
    the validation sequences and the testing sequence.
    
    Parameters
    ----------
    datasets : list[str]
        The datasets that can be used by this instance.
    
    training_lengths : list[str]
        The training lengths that can be loaded by this instance.
    
    window_sizes : list[float | int]
        The window sizes that can be loaded by this instance.
        
    datasets_path : str
        The path that contains the dataset.
    """
    DATASETS = ["fridge1", "fridge2", "fridge3"]
    TRAIN_LENGTH = ["1m", "3w", "2w", "1w", "6d", "5d", "4d", "3d", "2d", "1d"]
    WINDOW_SIZE = [3, 2, 1, 0.5]
    DATASET_WINDOW = {
        "fridge1": {3: 305, 2: 205, 1: 100, 0.5: 50},
        "fridge2": {3: 240, 2: 160, 1: 80, 0.5: 40},
        "fridge3": {3: 135, 2: 90, 1: 45, 0.5: 25}
    }
    
    def __init__(self, datasets: list[str],
                 training_lengths: list[str],
                 window_sizes: list[float | int] = None,
                 datasets_path: str = "data/anomaly_detection/private_fridge"):
        super().__init__()
        
        self.datasets = datasets
        self.training_lengths = training_lengths
        self.window_sizes = window_sizes if window_sizes is not None else [3, 2, 1, 0.5]
        self.datasets_path = datasets_path
        
        self.__check_parameters()
        
    def __len__(self):
        return len(self.datasets) * len(self.training_lengths) * len(self.window_sizes)
    
    def __getitem__(self, item):
        if not isinstance(item, int):
            raise ValueError("use __getitem__ only to iterate")
        
        train_len = len(self.training_lengths)
        window_len = len(self.window_sizes)
        
        dataset_idx = item % (train_len * window_len)
        training_idx = (item - dataset_idx * train_len * window_len) % window_len
        window_idx = item - dataset_idx * train_len * window_len - training_idx * window_len
        
        ds = self.datasets[dataset_idx]
        train = self.training_lengths[training_idx]
        window = self.window_sizes[window_idx]
        
        return self.get_train_valid_test(ds, train, window, verbose=True)
    
    def get_train_valid_test(self, dataset: str,
                             training_length: str,
                             window_size: int | float,
                             window_multiple_of_period: bool = True,
                             verbose: bool = True,
                             *args,
                             **kwargs) -> Tuple[list[pd.DataFrame],
                                                np.ndarray,
                                                pd.DataFrame,
                                                pd.DataFrame]:
        """Get training, validation and testing for the specified dataset.
        
        Parameters
        ----------
        dataset : str
            The dataset to load.
        
        training_length : str
            The training length of the dataset to load.
            
        window_size : int | float
            The length of the window size for the algorithms. If the parameter
            `window_multiple_of_period` is `True`, it must be one of the
            available training lengths. Otherwise, it can be any integer number
            specifying a used-defined window.
        
        window_multiple_of_period : bool, default=True
            If `True` the window is expressed as a multiple of the period,
            otherwise, it is directly passed as an integer.
            
        verbose : bool, default=True
            States if detailed printing must be performed.

        Returns
        -------
        train_sequences, train_concatenated, validation, testing : list, ndarray, ndarray, ndarray
            Four sequences representing all the training sequences, the
            concatenated training sequences, the validation sequence and the
            testing sequence.
            
        Raises
        ------
        TypeError
            When a parameter has wrong type.
            
        ValueError
            When a parameter has wrong value.
        """
        if not isinstance(window_multiple_of_period, bool):
            raise TypeError("window_multiple_of_period must be bool")
        elif not isinstance(verbose, bool):
            raise TypeError("verbose must be bool")
        
        if dataset not in self.datasets:
            raise ValueError("dataset must be one of self.datasets")
        elif training_length not in self.training_lengths:
            raise ValueError("training_length must be one of self.training_lengths")
        elif window_size not in self.window_sizes and window_multiple_of_period:
            raise ValueError("dataset must be one of self.window_sizes")
        
        if verbose:
            print_header("Dataset extraction started")
        
        # get the directory with the training data sequences
        dir_path = os.path.join(self.datasets_path, "{}/train/clean_data/{}".format(dataset, training_length))
        dir_path = str(os.path.relpath(dir_path))
        
        anom_path = os.path.join(self.datasets_path, "{}/anomalies_{}.csv".format(dataset, dataset))
        anom_path = str(os.path.relpath(anom_path))
        
        val_path = os.path.join(self.datasets_path, "{}/val/val_{}.csv".format(dataset, dataset))
        val_path = str(os.path.relpath(val_path))
        
        test_path = os.path.join(self.datasets_path, "{}/test/test_{}.csv".format(dataset, dataset))
        test_path = str(os.path.relpath(test_path))
        
        dir_contents_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
        dir_files = [f for f in dir_contents_paths if os.path.isfile(f)]

        if verbose:
            print_step("Reading the training data from clean directories")
            print_step("Directory {} is being read".format(dir_path))

        # read all dataframes and eliminate null csv
        multiple_reader = TSMultipleReader()
        dataframes = multiple_reader.read_multiple(dir_files, verbose=False).get_all_dataframes()
        all_data = None
        nonempty_df = list()
        for df in dataframes:
            if df.shape[0] != 0:
                df = df.rename(columns={"ctime": "timestamp", "device_consumption": "value"})
                df.insert(len(df.columns), "target", 0)
                nonempty_df.append(df.copy())
        
                if all_data is None:
                    all_data = df["value"].values.reshape(-1)
                else:
                    all_data = np.append(all_data, df["value"].values.reshape(-1))

        if verbose:
            print_step("Reading the validation data")
            print_step("File {} is being read".format(val_path))

        # read the validation dataframe
        odin_reader = ODINTSReader(anom_path, "ctime", "device_consumption")
        validation_frame = odin_reader.read(val_path, verbose=False).get_dataframe()
        
        if verbose:
            print_step("Reading the testing data")
            print_step("File {} is being read".format(test_path))

        # read the testing dataframe
        odin_reader = ODINTSReader(anom_path, "ctime", "device_consumption")
        testing_frame = odin_reader.read(test_path, verbose=False).get_dataframe()
        
        if verbose:
            print_step("Computing training mean and std")

        # normalize data in input to the model and validation on the basis of training
        scaler = StandardScaler()
        scaler.fit(all_data.reshape((-1, 1)))
        
        if verbose:
            print_step("Standardizing the sets")

        for i in range(len(nonempty_df)):
            values_tr = nonempty_df[i]["value"].values.reshape(-1, 1)
            nonempty_df[i]["value"] = scaler.transform(values_tr).reshape(-1)

        values_tr = all_data.reshape(-1, 1)
        all_data = scaler.transform(values_tr).reshape(-1)

        values_tr = validation_frame["value"].values.reshape(-1, 1)
        validation_frame["value"] = scaler.transform(values_tr).reshape(-1)

        values_tr = testing_frame["value"].values.reshape(-1, 1)
        testing_frame["value"] = scaler.transform(values_tr).reshape(-1)
        
        if window_multiple_of_period:
            training_sequences = [e for e in nonempty_df if e.shape[0] >= self.DATASET_WINDOW[dataset][window_size]]
        else:
            training_sequences = [e for e in nonempty_df if e.shape[0] >= window_size]
        
        if verbose:
            print_header("Dataset extraction ended")
            
        return training_sequences, all_data, validation_frame, testing_frame
    
    def __check_parameters(self):
        """Checks if the attributes are ok.
        
        Returns
        -------
        None
        
        Raises
        ------
        TypeError
            If one of the attributes has wrong type.
            
        ValueError
            If one of the attributes has wrong value.
        """
        if not isinstance(self.datasets, list):
            raise TypeError("datasets must be a list of strings")
        elif not isinstance(self.training_lengths, list):
            raise TypeError("training_lengths must be a list of strings")
        elif not isinstance(self.window_sizes, list):
            raise TypeError("window_sizes must be a list of numbers")
        elif not isinstance(self.datasets_path, str):
            raise TypeError("datasets_path must be a string")
        
        if len(set(self.datasets).difference(self.DATASETS)) != 0:
            raise ValueError("datasets elements must be contained in {}".format(self.DATASETS))
        elif len(set(self.training_lengths).difference(self.TRAIN_LENGTH)) != 0:
            raise ValueError("training_lengths elements must be contained in {}".format(self.TRAIN_LENGTH))
        elif len(set(self.window_sizes).difference(self.WINDOW_SIZE)) != 0:
            raise ValueError("window_sizes elements must be contained in {}".format(self.WINDOW_SIZE))
        elif not os.path.isdir(os.path.realpath(self.datasets_path)):
            raise ValueError("datasets_path must point to a valid directory")
        elif len(self.datasets) == 0:
            raise ValueError("datasets must not be an empty list")
        elif len(self.training_lengths) == 0:
            raise ValueError("training_lengths must no be an empty list")
        elif len(self.window_sizes) == 0:
            raise ValueError("window_sizes must not be an empty list")


class Zangrando2022Threshold(object):
    """Threshold computation class.

    The paper implemented in this class is "Anomaly detection in quasi-periodic
    energy consumption data series: a comparison of algorithms" by the authors
    Niccolò Zangrando, Piero Fraternali, Marco Petri, Nicolò Oreste Pinciroli
    Vago and Sergio Luis Herrera González.

    This class implements the methods used to compute the optimal threshold for
    the implemented models in the paper.
    
    Parameters
    ----------
    bounded_scores : bool
        `True` if the scores of the model are bounded, i.e., always contained
        inside a certain range (a,b). `False` if the scores are not bounded.
    
    min_bound : float
        The minimum value a score can have if the scores are bounded.
    
    max_bound : float
        The maximum value a score can have if the scores are bounded.
    
    num_thresholds : int
        The number of thresholds to try when searching for the optimal one.
    """
    def __init__(self, bounded_scores: bool,
                 min_bound: float = 0,
                 max_bound: float = 1,
                 num_thresholds: int = 41):
        super().__init__()
        
        self.bounded_scores = bounded_scores
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.num_thresholds = num_thresholds
        
        self.__check_parameters()
        
    def compute_best_threshold(self, validation_scores,
                               labels,
                               verbose: bool = True) -> Tuple[float, float]:
        """Computes the best threhsold on the validation set.
        
        Parameters
        ----------
        validation_scores : array-like
            The scores of the method on the validation set.
            
        labels : array-like
            The labels for the points.
            
        verbose : bool, default=True
            States if detailed printing must be done.

        Returns
        -------
        best_f1, best_threshold : float, float
            The best f1 obtained with the threshold and the best threshold on
            the validation set.
            
        Raises
        ------
        TypeError
            When a parameter has wrong type.
        """
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be bool")
        
        scores = np.array(validation_scores)
        true_labels = np.array(labels)
        
        best_threshold = -50000
        best_f1 = -1
        
        if self.bounded_scores:
            thresholds_to_try = np.linspace(self.min_bound, self.max_bound, self.num_thresholds)
        else:
            low_min = np.min(scores[np.argwhere(scores < 200).squeeze()])
            low_max = int(np.max(scores[np.argwhere(scores < 200).squeeze()]))
            
            thresholds_to_try = np.linspace(low_min, low_max, low_max * self.num_thresholds + 1 + self.num_thresholds)
        
        for threshold in thresholds_to_try:
            pred_labels = scores > threshold
            f1 = metrics.f1_score(true_labels, pred_labels)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                
        return best_f1, best_threshold

    def __check_parameters(self):
        """Checks if the attributes are ok.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If one of the attributes has wrong type.

        ValueError
            If one of the attributes has wrong value.
        """
        if not isinstance(self.bounded_scores, bool):
            raise TypeError("bounded_scores must be bool")
        elif not isinstance(self.min_bound, Number):
            raise TypeError("min_bound must be bool")
        elif not isinstance(self.max_bound, Number):
            raise TypeError("max_bound must be bool")
        elif not isinstance(self.num_thresholds, int):
            raise TypeError("num_thresholds must be int")
        
        if self.min_bound >= self.max_bound:
            raise ValueError("min_bound must be less (<) than max_bound")
        elif self.num_thresholds < 2:
            raise ValueError("num_thresholds must be at least 2")
