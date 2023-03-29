import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from anomalearn.reader.time_series import TSReader


class TestTSReader(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng()
        self.series = rng.integers(0, 1000, (100, 3))
    
    def test_read_and_get_dataframe(self):
        reader = TSReader()
        
        with TemporaryDirectory() as tmp_dir:
            dir_path = Path(tmp_dir)
            df = pd.DataFrame(self.series, columns=["A", "B", "C"])
            df.to_csv(dir_path / "series.csv", index=False)
            df.to_json(dir_path / "series.json", orient="columns")
            df.to_xml(dir_path / "series.xml", index=False, parser="etree")
            df.to_html(dir_path / "series.html", index=False)
            
            df_csv = reader.read(dir_path / "series.csv", file_format="csv", verbose=False).get_dataframe()
            np.testing.assert_array_equal(self.series, df_csv.values)
            
            df_json = reader.read(dir_path / "series.json", file_format="json", verbose=False, pandas_args={"orient": "columns"}).get_dataframe()
            np.testing.assert_array_equal(self.series, df_json.values)
            
            df_xml = reader.read(dir_path / "series.xml", file_format="xml", verbose=False, pandas_args={"parser": "etree"}).get_dataframe()
            np.testing.assert_array_equal(self.series, df_xml.values)
            
            self.assertRaises(NotImplementedError, reader.read, dir_path / "series.html", file_format="html", verbose=False)
    
    def test_get_dataframe(self):
        reader = TSReader()
        
        self.assertRaises(ValueError, reader.get_dataframe)
