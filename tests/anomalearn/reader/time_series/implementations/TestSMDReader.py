import logging
import unittest
from pathlib import Path

from anomalearn.reader.time_series import SMDReader
from tests.anomalearn.reader.time_series.implementations.common_tests import assert_get_item_behaviour, \
    assert_not_equal_numpy


class TestSMDReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        logging.basicConfig()
        series_logger = logging.getLogger("anomalearn.reader.time_series")
        series_logger.setLevel(logging.DEBUG)
        cls.series_logger = series_logger
        cls.dataset_length = 28

    def setUp(self) -> None:
        self.reader = SMDReader(Path(__file__).parent / "../../../../../data/anomaly_detection/smd")
        self.benchmark_folder = Path(__file__).parent / "../../../../../data/anomaly_detection/smd"

    def test_len(self):
        self.assertEqual(self.dataset_length, len(self.reader))

    def test_get_item(self):
        tmp_path = Path(__file__).parent / (str(Path(__file__).name).split(".")[0] + "_temp_.txt")
        assert_get_item_behaviour(self,
                                  tmp_path,
                                  self.series_logger,
                                  self.reader,
                                  "reading training file at ",
                                  self.benchmark_folder,
                                  self.dataset_length)

    def test_read(self):
        obj1 = self.reader.read("machine-1-1").get_dataframe().values
        obj2 = self.reader.read("machine-2-1").get_dataframe().values
        obj3 = self.reader.read("machine-3-1").get_dataframe().values
        obj4 = self.reader.read("machine-1-8").get_dataframe().values

        assert_not_equal_numpy(self, obj1, obj2, obj3, obj4)
        self.assertRaises(ValueError, self.reader.read, "machine doesn't exist")
