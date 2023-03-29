from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from .. import TSReader, rts_config
from ... import MissingStrategy


class ODINTSReader(TSReader):
    """A reader for ODIN TS annotated datasets.
    
    Parameters
    ----------
    anomalies_path : str
        It is the file path in which the anomalies are stored as csv.
    
    timestamp_col : str
        It is the column with the timestamps of data.
    
    univariate_col : str
        It is the column on which the dataset has been labelled.
    
    start_col : str, default="start_date"
        It is the column of the anomalies file stating the start of an anomaly
        window.
    
    end_col : str, default="end_date"
        It is the column of the anomalies file stating the end of an anomaly
        window.
    """
    _DAY_COL = "day_of_the_week"
    _ANOMALY_TYPE = "anomaly_type"
    
    def __init__(self, anomalies_path: str,
                 timestamp_col: str,
                 univariate_col: str,
                 start_col: str = "start_date",
                 end_col: str = "end_date",
                 anomaly_type_col: str = "anomaly_type"):
        super().__init__()

        self.__logger = logging.getLogger(__name__)
        self.anomalies_path = anomalies_path
        self.timestamp_col = timestamp_col
        self.univariate_col = univariate_col
        self.start_col = start_col
        self.end_col = end_col
        self.anomaly_type_col = anomaly_type_col
        
        self._unmodified_dataset: pd.DataFrame | None = None
        
    def read(self, path,
             file_format: str = "csv",
             pandas_args: dict | None = None,
             resample: bool = False,
             resampling_granularity: str = "1min",
             missing_strategy: MissingStrategy = MissingStrategy.DROP,
             missing_fixed_value: float = 0.0,
             *args,
             **kwargs) -> ODINTSReader:
        # TODO: implement interpolation imputation
        if missing_strategy not in [MissingStrategy.NOTHING, MissingStrategy.DROP, MissingStrategy.FIXED_VALUE]:
            raise NotImplementedError("Interpolation still not implemented")

        super().read(path, file_format, verbose=False)

        self._unmodified_dataset = self._dataset.copy()
        dataset_cp = self._add_information()
        
        self.__logger.info("renaming columns with standard names")
        # add anomaly labels to original dataset and drop useless columns
        self._dataset.insert(len(self._dataset.columns),
                             rts_config["Univariate"]["target_column"],
                             dataset_cp[rts_config["Univariate"]["target_column"]].values)
        self._dataset.rename(columns={
                                self.timestamp_col: rts_config["Univariate"]["index_column"],
                                self.univariate_col: rts_config["Univariate"]["value_column"]
                            },
                            inplace=True)
        self._dataset.drop(columns=self._dataset.columns.difference([rts_config["Univariate"]["index_column"],
                                                                     rts_config["Univariate"]["value_column"],
                                                                     rts_config["Univariate"]["target_column"]]),
                           inplace=True)
        
        if resample:
            self.__logger.info("resampling")
            self._dataset[rts_config["Univariate"]["index_column"]] = pd.to_datetime(self._dataset[rts_config["Univariate"]["index_column"]])
            self._dataset.index = pd.to_datetime(self._dataset[rts_config["Univariate"]["index_column"]])
            self._dataset = self._dataset.resample(resampling_granularity).agg({rts_config["Univariate"]["value_column"]: np.mean, rts_config["Univariate"]["target_column"]: np.max})
            self._dataset.reset_index(inplace=True)
            self._dataset[rts_config["Univariate"]["index_column"]] = self._dataset[rts_config["Univariate"]["index_column"]].dt.strftime("%Y-%m-%d %H:%M:%S")
        
        if missing_strategy == MissingStrategy.DROP:
            self.__logger.info("dropping missing values")
            self._dataset.dropna(inplace=True)
        elif missing_strategy == MissingStrategy.FIXED_VALUE:
            self.__logger.info("placing fixed value for missing values")
            self._dataset.fillna(missing_fixed_value, inplace=True)
        
        return self
    
    def get_complete_dataframe(self) -> pd.DataFrame:
        """Same as reading, but all properties are returned (not only target).

        Returns
        -------
        pd.DataFrame
            Dataset with complete information.
        """
        if self._unmodified_dataset is None:
            raise RuntimeError("You must first read the dataset before being "
                               "able to get it.")
        
        enhanced_dataset = self._add_information(complete=True)
        
        new_dataset = self._unmodified_dataset.copy()
        new_dataset.insert(len(new_dataset.columns),
                           rts_config["Univariate"]["target_column"],
                           enhanced_dataset[rts_config["Univariate"]["target_column"]].values)
        new_dataset.insert(len(new_dataset.columns),
                           self._ANOMALY_TYPE,
                           enhanced_dataset[self._ANOMALY_TYPE].values)
        new_dataset.insert(len(new_dataset.columns),
                           self._DAY_COL,
                           enhanced_dataset[self._DAY_COL].values)
        new_dataset = new_dataset.rename(columns={
            self.timestamp_col: rts_config["Univariate"]["index_column"],
            self.univariate_col: rts_config["Univariate"]["value_column"]})
        new_dataset = new_dataset.drop(columns=new_dataset.columns.difference([rts_config["Univariate"]["index_column"],
                                                                               rts_config["Univariate"]["value_column"],
                                                                               rts_config["Univariate"]["target_column"],
                                                                               self._ANOMALY_TYPE,
                                                                               self._DAY_COL]))
        
        return new_dataset
    
    def _add_information(self, complete: bool = False) -> pd.DataFrame:
        """Add information contained in the anomalies' path.
        
        Parameters
        ----------
        complete : bool, default=False
            States if anomaly columns must be added beside the labels.
        
        Returns
        -------
        enhanced_dataset : pd.DataFrame
            The dataset enhanced with the information contained in the json file
            of the anomalies.
        """
        self.__logger.info("reading odin annotations")
        # read the file with anomalies annotations
        anomalies_df = pd.read_csv(self.anomalies_path)

        # translate the dataset in a more accessible format for modification
        dataset_cp: pd.DataFrame = self._unmodified_dataset.copy()
        dataset_cp[self.timestamp_col] = pd.to_datetime(dataset_cp[self.timestamp_col],
                                                        format="%Y-%m-%d %H:%M:%S")
        dataset_cp = dataset_cp.set_index(self.timestamp_col)

        # add the columns with anomaly labels
        anomalies = np.zeros(dataset_cp.shape[0])
        day = np.ones(dataset_cp.shape[0]) * -1
        anomaly_type = ["No"] * dataset_cp.shape[0]
        dataset_cp.insert(len(dataset_cp.columns), rts_config["Univariate"]["target_column"], anomalies)
        
        if complete:
            dataset_cp.insert(len(dataset_cp.columns), self._DAY_COL, day)
            dataset_cp.insert(len(dataset_cp.columns), self._ANOMALY_TYPE, anomaly_type)

        self.__logger.info("computing anomaly intervals from file")
        # get the anomaly intervals
        anomaly_intervals = [(datetime.strptime(el[0], "%Y-%m-%d %H:%M:%S"),
                              datetime.strptime(el[1], "%Y-%m-%d %H:%M:%S"))
                             for el in zip(anomalies_df[self.start_col].tolist(),
                                           anomalies_df[self.end_col].tolist())]
        if complete:
            anomaly_type_dict = {datetime.strptime(el[0], "%Y-%m-%d %H:%M:%S"): el[1]
                                 for el in zip(anomalies_df[self.start_col].tolist(),
                                               anomalies_df[self.anomaly_type_col].tolist())}

        self.__logger.info("converting intervals to point labels")
        # build the anomaly labels on original dataset
        for start, end in anomaly_intervals:
            dataset_cp.loc[start:end, rts_config["Univariate"]["target_column"]] = 1
            if complete:
                dataset_cp.loc[start:end, self._ANOMALY_TYPE] = anomaly_type_dict[start]
                for idx, row in dataset_cp.loc[start:end].iterrows():
                    dataset_cp.loc[idx, self._DAY_COL] = idx.to_pydatetime().weekday()
        
        return dataset_cp
    