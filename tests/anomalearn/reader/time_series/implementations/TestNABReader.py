import logging
import unittest
from pathlib import Path

from anomalearn.reader.time_series import NABReader
from tests.anomalearn.reader.time_series.implementations.common_tests import assert_get_item_behaviour, \
    assert_not_equal_numpy


class TestNABReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        logging.basicConfig()
        series_logger = logging.getLogger("anomalearn.reader.time_series")
        series_logger.setLevel(logging.DEBUG)
        cls.series_logger = series_logger
        cls.dataset_length = 58

    def setUp(self) -> None:
        self.reader = NABReader(Path(__file__).parent / "../../../../../data/anomaly_detection/nab")
        self.benchmark_folder = Path(__file__).parent / "../../../../../data/anomaly_detection/nab"

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
        obj1 = self.reader.read("art_daily_no_noise").get_dataframe().values
        obj2 = self.reader.read("exchange-2_cpc_results").get_dataframe().values
        obj3 = self.reader.read("cpu_utilization_asg_misconfiguration").get_dataframe().values
        obj4 = self.reader.read("Twitter_volume_CRM").get_dataframe().values

        assert_not_equal_numpy(self, obj1, obj2, obj3, obj4)
        self.assertRaises(ValueError, self.reader.read, "series doesn't exist")
