import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from anomalearn.reader.time_series import TSMultipleReader


class TestTSMultipleReader(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng()
        self.series1 = rng.integers(0, 1000, (100, 3))
        self.series2 = rng.integers(0, 1000, (100, 3))
        
    def create_df_and_load(self, reader):
        with TemporaryDirectory() as tmp_dir:
            dir_path = Path(tmp_dir)
            df1 = pd.DataFrame(self.series1, columns=["A", "B", "C"])
            df2 = pd.DataFrame(self.series2, columns=["A", "B", "C"])
            df1.to_csv(dir_path / "series1.csv", index=False)
            df2.to_csv(dir_path / "series2.csv", index=False)
            
            reader.read_multiple([dir_path / "series1.csv", dir_path / "series2.csv"], files_format="csv", verbose=False)
        
    def test_multiple_read_and_select(self):
        reader = TSMultipleReader()
        
        with TemporaryDirectory() as tmp_dir:
            dir_path = Path(tmp_dir)
            df1 = pd.DataFrame(self.series1, columns=["A", "B", "C"])
            df2 = pd.DataFrame(self.series2, columns=["A", "B", "C"])
            df1.to_csv(dir_path / "series1.csv", index=False)
            df2.to_csv(dir_path / "series2.csv", index=False)
            df1.to_json(dir_path / "series1.json", orient="columns")
            df2.to_json(dir_path / "series2.json", orient="columns")
            df1.to_xml(dir_path / "series1.xml", index=False, parser="etree")
            df2.to_xml(dir_path / "series2.xml", index=False, parser="etree")
            df1.to_html(dir_path / "series1.html", index=False)
            df2.to_html(dir_path / "series2.html", index=False)
            
            reader.read_multiple([dir_path / "series1.csv", dir_path / "series2.csv"], files_format="csv", verbose=False)
            ser1, ser2 = reader.select_dataframe(0).get_dataframe().values, reader.select_dataframe(1).get_dataframe().values
            np.testing.assert_array_equal(self.series1, ser1)
            np.testing.assert_array_equal(self.series2, ser2)
            
            reader.read_multiple([dir_path / "series1.json", dir_path / "series2.json"], files_format="json", verbose=False, pandas_args={"orient": "columns"})
            ser1, ser2 = reader.select_dataframe(0).get_dataframe().values, reader.select_dataframe(1).get_dataframe().values
            np.testing.assert_array_equal(self.series1, ser1)
            np.testing.assert_array_equal(self.series2, ser2)
            
            reader.read_multiple([dir_path / "series1.xml", dir_path / "series2.xml"], files_format="xml", verbose=False, pandas_args={"parser": "etree"})
            ser1, ser2 = reader.select_dataframe(0).get_dataframe().values, reader.select_dataframe(1).get_dataframe().values
            np.testing.assert_array_equal(self.series1, ser1)
            np.testing.assert_array_equal(self.series2, ser2)
            
            self.assertRaises(NotImplementedError, reader.read_multiple, [dir_path / "series1.html", dir_path / "series2.html"], files_format="html", verbose=False)
        
    def test_select_dataframe(self):
        reader = TSMultipleReader()
        
        self.assertRaises(ValueError, reader.select_dataframe, 0)
        self.assertRaises(ValueError, reader.select_dataframe, 1000)
        
        self.create_df_and_load(reader)
        
        self.assertRaises(IndexError, reader.select_dataframe, 100)
        
        reader.select_dataframe(0)
        reader.select_dataframe(1)
        
    def test_get_all_dataframes(self):
        reader = TSMultipleReader()
        
        self.assertRaises(ValueError, reader.get_all_dataframes)
        
        self.create_df_and_load(reader)
        
        datasets = reader.get_all_dataframes()
        self.assertEqual(2, len(datasets))
        self.assertIsNot(datasets[0], reader._all_dataframes[0])
        self.assertIsNot(datasets[0], reader._all_dataframes[1])
        self.assertIsNot(datasets[1], reader._all_dataframes[0])
        self.assertIsNot(datasets[1], reader._all_dataframes[1])
        self.assertTrue(datasets[0].equals(reader._all_dataframes[0]))
        self.assertTrue(datasets[1].equals(reader._all_dataframes[1]))
        
    def test_get_ith_dataframe(self):
        reader = TSMultipleReader()
        
        self.assertRaises(ValueError, reader.get_ith_dataframe, 0)
        self.assertRaises(ValueError, reader.get_ith_dataframe, 1000)

        self.create_df_and_load(reader)
        
        self.assertRaises(IndexError, reader.get_ith_dataframe, 100)
        
        dataset = reader.get_ith_dataframe(0)
        self.assertTrue(dataset.equals(reader._all_dataframes[0]))
