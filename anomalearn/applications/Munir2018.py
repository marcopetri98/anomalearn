import os.path
from numbers import Number
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..reader.time_series import TSReader, YahooS5Reader


class Munir2018Loader(object):
    """Data loader for DeepAnT model.
    
    This class implements the loading of the datasets used in the paper "DeepAnT
    A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series"
    available at https://doi.org/10.1109/ACCESS.2018.2886457. The Data Loader
    can be used to load any of the datasets that has been used to evaluate
    DeepAnT performance. For each of them it is possible to retrieve the
    training, validation and testing sets.
    
    Parameters
    ----------
    window_size : int
        The size of the window.
    
    setting : str, default="yahoo_s5"
        The experimental setting that we want to consider for the loader.

    dataset_location : str, default="data/anomaly_detection/yahoo_s5"
        The location in the file system of the dataset.

    train_perc : float, default=0.36
        The percentage of data points to be used for training. The default value
        is the one used in Munir et al. for the training of the model.

    valid_perc : float, default=0.04
        The percentage of data points to be used for validation. The default
        value is the one used in Munir et al. for the training of the model.

    test_perc : float, default=0.6
        The percentage of data points to be used for testing. The default value
        is the one used in Munir et al. for the training of the model.
        
    Attributes
    ----------
    _reader : TSReader
        The reader for the dataset that is used by the loader.
    """
    DATASETS = ["yahoo_s5",
                "nab",
                "shuttle",
                "pima",
                "ForestCover",
                "Ionosphere",
                "HTTP",
                "SMTP",
                "Mulcross",
                "Mammography"]
    TRAINING_PERC = 0.36
    VALIDATION_PERC = 0.04
    TESTING_PERC = 0.6
    
    __IMPLEMENTED_DS = ["yahoo_s5"]
    
    def __init__(self, window_size: int,
                 setting: str = "yahoo_s5",
                 dataset_location: str = "data/anomaly_detection/yahoo_s5",
                 train_perc: float = TRAINING_PERC,
                 valid_perc: float = VALIDATION_PERC,
                 test_perc: float = TESTING_PERC):
        super().__init__()
        
        self.dataset_location = dataset_location
        self.window_size = window_size
        self.setting = setting
        self.train_perc = train_perc
        self.valid_perc = valid_perc
        self.test_perc = test_perc
        self._reader = None
        self._swap_setting = None
        
        self.__check_parameters()
        self._set_reader()
        
    def __len__(self):
        match self.setting:
            case "yahoo_s5":
                return len(self._reader)
        
            case _:
                # FIXME: implement all datasets
                raise NotImplementedError("not implemented dataset")
        
    def __getitem__(self, item):
        if not isinstance(item, int):
            raise ValueError("item must be an integer")
        elif not 0 <= item < len(self):
            raise IndexError(f"the dataset {self.setting} has {len(self)} series")
        
        return self.get_train_valid_test(item)
        
    def set_setting(self, setting: str) -> None:
        """Change the current setting of the loader.
        
        Parameters
        ----------
        setting : str
            The experimental setting that we want to consider for the loader.

        Returns
        -------
        None
        """
        self.setting = setting
        self._set_reader()
        
        self.__check_parameters()
        
    def set_window_size(self, window_size: int) -> None:
        """Change the current value for the window size.
        
        Parameters
        ----------
        window_size : int
            The size of the window.

        Returns
        -------
        None
        """
        self.window_size = window_size
        
        self.__check_parameters()

    def get_train_valid_test(self, series: int,
                             setting: str = None,
                             window_size: int = None,
                             *args,
                             **kwargs) -> Tuple[list[pd.DataFrame],
                                                pd.DataFrame,
                                                pd.DataFrame]:
        """Get the training, validation and testing
        
        Parameters
        ----------
        series : int
            The series to load from the setting

        setting : str, default=None
            The experimental setting that we want to consider for the loader.
            The default value is the `setting` member of the object.

        window_size : int, default=None
            The size of the window. The default value is the `window_size`
            member of the object.

        args
            Not used, present for inheritance change of signature.

        kwargs
            Not used, present for inheritance change of signature.

        Returns
        -------
        training_sequences, validation, testing : DataFrame, DataFrame, DataFrame
            The list of the training sequences of normal points that are at
            least as long as one window. The validation sequence and the testing
            sequence.
        """
        # check types and values
        if not isinstance(series, int):
            raise ValueError("item must be an integer")
        elif not 0 <= series < len(self):
            raise IndexError(f"the dataset {self.setting} has {len(self)} series")
        elif setting is not None and not isinstance(setting, str):
            raise TypeError("setting must be a string")
        elif window_size is not None and not isinstance(window_size, int):
            raise TypeError("window_size must be an integer")
        elif setting is not None and setting not in self.DATASETS:
            raise ValueError(f"setting must be one of {self.DATASETS}")
        elif window_size is not None and window_size <= 0:
            raise ValueError("window_size must be strictly positive")

        setting_is_changed = False
        if setting is not None and setting != self.setting:
            setting_is_changed = True
            self._swap_setting = self.setting
            self.set_setting(setting)

        window_size = window_size if window_size is not None else self.window_size
        dataset = self._reader[series]
        train_valid, test = train_test_split(dataset,
                                             test_size=self.test_perc,
                                             shuffle=False)
        train, valid = train_test_split(dataset,
                                        test_size=self.valid_perc,
                                        shuffle=False)
        train_gt = train["target"].values

        if setting_is_changed:
            self.set_setting(self._swap_setting)

        anomalies = np.argwhere(train_gt == 1).reshape(-1)
        training_sequences = list()

        if anomalies.shape[0] != 0:
            start_indexes = np.append([0], anomalies[:-1] + 1)
            sequences = list(zip(start_indexes, anomalies))
            sequences.append((anomalies[-1] + 1, train_gt.shape[0]))
            sequences = list(filter(lambda tup: tup[0] != tup[1], sequences))
            sequences = [slice(*el) for el in sequences]

            for sl in sequences:
                seq = train[sl]
                if seq.shape[0] >= window_size:
                    training_sequences.append(seq)
        else:
            if train.shape[0] < window_size:
                raise ValueError("The window size is so big that the time "
                                 "time series cannot fit in one window.")
            else:
                training_sequences.append(train)

        return training_sequences, valid, test
    
    def _set_reader(self):
        """Given the chosen setting, the reader is defined.
        
        Returns
        -------
        None
        """
        match self.setting:
            case "yahoo_s5":
                self._reader = YahooS5Reader(self.dataset_location)
            
            case _:
                # FIXME: implement all datasets
                raise NotImplementedError("not implemented dataset")
    
    def __check_parameters(self):
        """Check values and types of the parameters.
        
        Returns
        -------
        None
        
        Raises
        ------
        TypeError
            If a parameter has the wrong type.
            
        ValueError
            If a parameter has the wrong type.
        """
        if not isinstance(self.setting, str):
            raise TypeError("setting must be a string")
        elif not isinstance(self.dataset_location, str):
            raise TypeError("dataset_location must be a string")
        elif not isinstance(self.window_size, int):
            raise TypeError("window_size must be an integer")
        elif not isinstance(self.train_perc, Number):
            raise TypeError("train_perc must be a number")
        elif not isinstance(self.valid_perc, Number):
            raise TypeError("valid_perc must be a number")
        elif not isinstance(self.test_perc, Number):
            raise TypeError("test_perc must be a number")
        
        if self.setting not in self.DATASETS:
            raise ValueError(f"setting must be one of {self.DATASETS}")
        elif not os.path.isdir(self.dataset_location):
            raise ValueError("dataset_location must be the directory containing"
                             " the dataset")
        elif self.window_size < 1:
            raise ValueError("window_size must be strictly positive")
        elif not 0 < self.train_perc < 1:
            raise ValueError("train_perc must be between 0 and 1")
        elif not 0 < self.valid_perc < 1:
            raise ValueError("valid_perc must be between 0 and 1")
        elif not 0 < self.test_perc < 1:
            raise ValueError("test_perc must be between 0 and 1")
        elif self.train_perc + self.valid_perc + self.test_perc != 1:
            raise ValueError("train_perc+valid_perc+test_perc must be 1")
        
        # FIXME: implement all datasets
        if self.setting not in self.__IMPLEMENTED_DS:
            raise NotImplementedError(f"{self.setting} is still not implemented")
