import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from anomalearn.utils import save_py_json, load_py_json


class TestJsonFunctions(unittest.TestCase):
    def test_save_and_load_py_json(self):
        with TemporaryDirectory() as tmp_dir:
            string = "Dr. Halsey"
            integer = 2511
            list_ = list(range(100))
            tuple_ = (2511, 2517, 2519)
            dictionary = {"John-117": 2511, "Dr. Halsey": 2492}
            invalid = None
            bool_1 = True
            bool_2 = False
            real = 24.02
            
            for i, obj in enumerate([string, integer, list_, tuple_, dictionary, invalid, bool_1, bool_2, real]):
                json_file = str(Path(tmp_dir) / "file.json")
                save_py_json(obj, json_file)
                self.assertEqual(1, len(os.listdir(tmp_dir)))
                self.assertEqual("file.json", os.listdir(tmp_dir)[0])
                loaded = load_py_json(json_file)

                if i in [0, 1, 6, 7, 8]:
                    self.assertEqual(obj, loaded)
                if i == 2:
                    self.assertListEqual(obj, loaded)
                if i == 3:
                    self.assertListEqual(list(obj), loaded)
                if i == 4:
                    self.assertDictEqual(obj, loaded)
                if i == 5:
                    self.assertIsNone(loaded)
