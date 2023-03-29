import logging
import unittest
from pathlib import Path

from anomalearn.reader.time_series import MGABReader
from tests.anomalearn.reader.time_series.implementations.common_tests import assert_get_item_behaviour, \
    assert_not_equal_numpy


class TestMGABReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        logging.basicConfig()
        series_logger = logging.getLogger("anomalearn.reader.time_series")
        series_logger.setLevel(logging.DEBUG)
        cls.series_logger = series_logger
        cls.dataset_length = 10

    def setUp(self) -> None:
        self.reader = MGABReader(Path(__file__).parent / "../../../../../data/anomaly_detection/mgab")
        self.benchmark_folder = Path(__file__).parent / "../../../../../data/anomaly_detection/mgab"

    def test_len(self):
        self.assertEqual(self.dataset_length, len(self.reader))

    def test_get_item(self):
        tmp_path = Path(__file__).parent / (str(Path(__file__).name).split(".")[0] + "_temp_.txt")
        assert_get_item_behaviour(self,
                                  tmp_path,
                                  self.series_logger,
                                  self.reader,
                                  "TSReader:dataset read from ",
                                  self.benchmark_folder,
                                  self.dataset_length)

    def test_read(self):
        obj1 = self.reader.read(self.benchmark_folder / "1.csv").get_dataframe().values
        obj2 = self.reader.read(self.benchmark_folder / "3.csv").get_dataframe().values
        obj3 = self.reader.read(self.benchmark_folder / "5.csv").get_dataframe().values
        obj4 = self.reader.read(self.benchmark_folder / "7.csv").get_dataframe().values

        assert_not_equal_numpy(self, obj1, obj2, obj3, obj4)
        self.assertRaises(ValueError, self.reader.read, "no_path")
