import logging
import unittest
from pathlib import Path

from anomalearn.reader.time_series import UCRReader
from tests.anomalearn.reader.time_series.implementations.common_tests import assert_get_item_behaviour, \
    assert_not_equal_numpy


class TestUCRReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        logging.basicConfig()
        series_logger = logging.getLogger("anomalearn.reader.time_series")
        series_logger.setLevel(logging.DEBUG)
        cls.series_logger = series_logger
        cls.dataset_length = 250

    def setUp(self) -> None:
        self.reader = UCRReader(Path(__file__).parent / "../../../../../data/anomaly_detection/ucr")
        self.benchmark_folder = Path(__file__).parent / "../../../../../data/anomaly_detection/ucr"

    def test_len(self):
        self.assertEqual(self.dataset_length, len(self.reader))

    def test_get_item(self):
        tmp_path = Path(__file__).parent / (str(Path(__file__).name).split(".")[0] + "_temp_.txt")
        assert_get_item_behaviour(self,
                                  tmp_path,
                                  self.series_logger,
                                  self.reader,
                                  "reading dataset from ",
                                  self.benchmark_folder,
                                  self.dataset_length)

    def test_read(self):
        obj1 = self.reader.read(self.benchmark_folder / "001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt").get_dataframe().values
        obj2 = self.reader.read(self.benchmark_folder / "002_UCR_Anomaly_DISTORTED2sddb40_35000_56600_56900.txt").get_dataframe().values
        obj3 = self.reader.read(self.benchmark_folder / "003_UCR_Anomaly_DISTORTED3sddb40_35000_46600_46900.txt").get_dataframe().values
        obj4 = self.reader.read(self.benchmark_folder / "004_UCR_Anomaly_DISTORTEDBIDMC1_2500_5400_5600.txt").get_dataframe().values

        assert_not_equal_numpy(self, obj1, obj2, obj3, obj4)
        self.assertRaises(ValueError, self.reader.read, "no_path")
