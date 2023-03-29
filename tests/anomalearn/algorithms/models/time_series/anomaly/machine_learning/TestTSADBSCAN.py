import os.path
import pathlib
import unittest

import numpy as np
import pandas as pd

from anomalearn.algorithms.models.time_series.anomaly.machine_learning import TSADBSCAN


class TestTSADBSCAN(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sinusoidal_path = "test_data/sinusoidal_wave.csv"
        dir_path = pathlib.Path(__file__).parent.resolve()

        cls.sinusoidal_path = os.path.join(dir_path, sinusoidal_path)
        cls.sinusoidal_df = pd.read_csv(cls.sinusoidal_path)

    def setUp(self) -> None:
        self.sinusoidal_series = self.sinusoidal_df["value"].copy()
        self.sinusoidal_series_len = self.sinusoidal_series.values.shape[0]
        self.sinusoidal_targets = self.sinusoidal_df["target"].copy()

    def test_reproducibility(self):
        configurations = [
            {"window": 10, "eps": 0.5, "min_samples": 5},
            {"window": 5, "eps": 0.5, "min_samples": 5},
            {"window": 15, "eps": 0.5, "min_samples": 5},
            {"window": 20, "eps": 0.5, "min_samples": 5},
            {"window": 10, "eps": 1, "min_samples": 5},
            {"window": 10, "eps": 0.1, "min_samples": 15},
            {"window": 10, "eps": 1, "min_samples": 15}
        ]
        reproductions = 10

        for config in configurations:
            scores = None
            model = TSADBSCAN(classification="points_score",
                              scaling="none",
                              **config)

            for _ in range(reproductions):
                model_scores = model.anomaly_score(self.sinusoidal_series.values.reshape(-1, 1))

                if scores is None:
                    scores = model_scores
                else:
                    comparison = (scores == model_scores) | (np.isnan(scores) & np.isnan(model_scores))
                    self.assertTrue(comparison.all())
