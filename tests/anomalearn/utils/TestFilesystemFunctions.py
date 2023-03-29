import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from anomalearn.utils import find_or_create_dir


class TestFilesystemFunctions(unittest.TestCase):
    def test_find_or_create_dir(self):
        with TemporaryDirectory() as tmp_dir:
            # should do nothing
            find_or_create_dir(tmp_dir)
            
            txt = Path(tmp_dir) / "newfile.txt"
            txt.touch()
            self.assertRaises(ValueError, find_or_create_dir, str(txt))
            
        new_dir = Path("./0123456abcdefgh_temp_test_dir_for_func")
        self.assertFalse(new_dir.exists())
        find_or_create_dir(str(new_dir))
        self.assertTrue(new_dir.exists())
        new_dir.rmdir()
