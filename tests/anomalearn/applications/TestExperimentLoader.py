import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from anomalearn.applications import ExperimentLoader
from anomalearn.reader.time_series import YahooS5Reader, MGABReader, NABReader, \
    SMDReader, rts_config


class TestExperimentLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark_folder = Path(__file__).parent / "../../../data/anomaly_detection"

    def test_create(self):
        _ = ExperimentLoader()
        _ = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5")])
        _ = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5"), MGABReader(self.benchmark_folder / "mgab")])
        _ = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5")], [(0.3, 0.7)])
        _ = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5"), MGABReader(self.benchmark_folder / "mgab")],
                             [(0.3, 0.7), None])
        _ = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5")], series_to_use=[[0, 10, 20, 50, 100]])
        _ = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5"), MGABReader(self.benchmark_folder / "mgab")],
                             series_to_use=[[0, 10, 20, 50, 100], None])
        _ = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5")], [(0.7, 0.3)], [[0, 10, 20, 50, 100]])
        _ = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5"), MGABReader(self.benchmark_folder / "mgab")],
                             [(0.3, 0.7), None],
                             [[0, 10, 20, 50, 100], None])
        _ = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5")], default_split=(0.5, 0.5))
        _ = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5"), MGABReader(self.benchmark_folder / "mgab")],
                             default_split=(0.5, 0.5))

        self.assertRaises(ValueError, ExperimentLoader, [YahooS5Reader(self.benchmark_folder / "yahoo_s5")], [(0, 0)])
        self.assertRaises(ValueError, ExperimentLoader, [YahooS5Reader(self.benchmark_folder / "yahoo_s5")], [(0, 0, 1)])
        self.assertRaises(ValueError, ExperimentLoader, [YahooS5Reader(self.benchmark_folder / "yahoo_s5"), MGABReader(self.benchmark_folder / "mgab")],
                          [None, (0, 0, 1)])
        self.assertRaises(ValueError, ExperimentLoader, [YahooS5Reader(self.benchmark_folder / "yahoo_s5")], default_split=(1, 1)),
        self.assertRaises(ValueError, ExperimentLoader, [YahooS5Reader(self.benchmark_folder / "yahoo_s5"), MGABReader(self.benchmark_folder / "mgab")],
                          default_split=(1, 1))
        self.assertRaises(ValueError, ExperimentLoader, [YahooS5Reader(self.benchmark_folder / "yahoo_s5")], series_to_use=[[0, 10, 2000]])
        self.assertRaises(ValueError, ExperimentLoader, [YahooS5Reader(self.benchmark_folder / "yahoo_s5"), MGABReader(self.benchmark_folder / "mgab")],
                          series_to_use=[None, [0, 9, 2000]])

    def test_equal(self):
        exp1 = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5")])
        exp2 = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5")])
        self.assertEqual(exp1, exp2)

        exp2 = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5")], [[0.3, 0.7]])
        self.assertNotEqual(exp1, exp2)

        exp2 = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5")], series_to_use=[[0, 10, 20, 50, 100]])
        self.assertNotEqual(exp1, exp2)

        exp2 = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5"), MGABReader(self.benchmark_folder / "mgab")])
        self.assertNotEqual(exp1, exp2)

        self.assertNotEqual(exp1, 1900)
        self.assertNotEqual(exp1, "Halo Reach")

    def test_len(self):
        exp1 = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5")])
        self.assertEqual(1, len(exp1))

        exp1 = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5")], [[0.3, 0.7]])
        self.assertEqual(1, len(exp1))

        exp1 = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5")], series_to_use=[[0, 10, 20, 50, 100]])
        self.assertEqual(1, len(exp1))

        exp1 = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5")], [[0.3, 0.7]], [[0, 10, 20, 50, 100]])
        self.assertEqual(1, len(exp1))

        exp1 = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5"),
                                 MGABReader(self.benchmark_folder / "mgab")])
        self.assertEqual(2, len(exp1))

        exp1 = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5"),
                                 YahooS5Reader(self.benchmark_folder / "yahoo_s5"),
                                 MGABReader(self.benchmark_folder / "mgab"),
                                 NABReader(self.benchmark_folder / "nab")])
        self.assertEqual(4, len(exp1))

    def test_contains(self):
        exp1 = ExperimentLoader()
        self.assertNotIn(YahooS5Reader, exp1)

        exp1 = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5")])
        self.assertIn(YahooS5Reader, exp1)

        yahoo_reader = YahooS5Reader(self.benchmark_folder / "yahoo_s5")
        self.assertIn(yahoo_reader, exp1)

        self.assertNotIn(MGABReader, exp1)

    def test_get_item(self):
        exp1 = ExperimentLoader()
        self.assertRaises(IndexError, exp1.__getitem__, 0)

        exp1 = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5")])
        self.assertRaises(IndexError, exp1.__getitem__, 1)
        reader, split, series = exp1[0]
        self.assertIsInstance(reader, YahooS5Reader)
        self.assertTupleEqual((0.8, 0.2), split)
        self.assertIsNone(series)

        exp1 = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5")], [(0.9, 0.1)])
        reader, split, series = exp1[0]
        self.assertIsInstance(reader, YahooS5Reader)
        self.assertTupleEqual((0.9, 0.1), split)
        self.assertIsNone(series)

        exp1 = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5")], [(0.9, 0.1)], [[1, 2, 3]])
        reader, split, series = exp1[0]
        self.assertIsInstance(reader, YahooS5Reader)
        self.assertTupleEqual((0.9, 0.1), split)
        self.assertListEqual([1, 2, 3], series)

    def test_set_item(self):
        exp1 = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5"), MGABReader(self.benchmark_folder / "mgab")],
                                [(0.9, 0.1), None],
                                [None, [1, 2, 3]])
        self.assertRaises(TypeError, exp1.__setitem__, 0, "The Force Awakens")
        self.assertRaises(TypeError, exp1.__setitem__, 0, [YahooS5Reader(self.benchmark_folder / "yahoo_s5"), "The Force Awakens", [0, 1, 2, 3]])
        self.assertRaises(TypeError, exp1.__setitem__, 0, [YahooS5Reader(self.benchmark_folder / "yahoo_s5"), object(), [0, 1, 2, 3]])
        self.assertRaises(TypeError, exp1.__setitem__, 0, [YahooS5Reader(self.benchmark_folder / "yahoo_s5"), (0.8, 0.2), "Return of the Jedi"])
        self.assertRaises(TypeError, exp1.__setitem__, 0, ["Star Wars", (0.8, 0.2), [0, 1, 2, 3]])
        self.assertRaises(IndexError, exp1.__setitem__, 3, "The Force Awakens")
        self.assertRaises(IndexError, exp1.__setitem__, -10, "The Force Awakens")

        exp1[0] = [YahooS5Reader(self.benchmark_folder / "yahoo_s5"), (0.8, 0.2), [0, 1, 2, 3]]
        reader, split, series = exp1[0]
        self.assertIsInstance(reader, YahooS5Reader)
        self.assertTupleEqual((0.8, 0.2), split)
        self.assertListEqual([0, 1, 2, 3], series)

    def test_del_item(self):
        exp1 = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5"), MGABReader(self.benchmark_folder / "mgab")],
                                [(0.9, 0.1), None],
                                [None, [1, 2, 3]])
        self.assertRaises(IndexError, exp1.__delitem__, -5)
        self.assertRaises(IndexError, exp1.__delitem__, 5)

        del exp1[0]
        reader, split, series = exp1[0]
        self.assertIsInstance(reader, MGABReader)
        self.assertIsNone(split)
        self.assertListEqual([1, 2, 3], series)

    def test_index(self):
        exp1 = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5"), MGABReader(self.benchmark_folder / "mgab")],
                                [(0.9, 0.1), None],
                                [None, [1, 2, 3]])
        self.assertRaises(ValueError, exp1.index, NABReader)
        self.assertRaises(ValueError, exp1.index, MGABReader, stop=1)
        self.assertRaises(ValueError, exp1.index, YahooS5Reader, start=1)

        self.assertEqual(0, exp1.index(YahooS5Reader))
        self.assertEqual(1, exp1.index(MGABReader))

    def test_count(self):
        exp1 = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5"), MGABReader(self.benchmark_folder / "mgab")],
                                [(0.9, 0.1), None],
                                [None, [1, 2, 3]])
        self.assertEqual(1, exp1.count(YahooS5Reader))
        self.assertEqual(1, exp1.count(MGABReader))
        self.assertEqual(0, exp1.count(NABReader))

        exp1 = ExperimentLoader([YahooS5Reader(self.benchmark_folder / "yahoo_s5"), YahooS5Reader(self.benchmark_folder / "yahoo_s5")],
                                [(0.9, 0.1), None],
                                [None, [1, 2, 3]])
        self.assertEqual(2, exp1.count(YahooS5Reader))
        self.assertEqual(0, exp1.count(MGABReader))
        self.assertEqual(0, exp1.count(NABReader))

    def test_insert(self):
        for first_index in [0, -100, 100]:
            exp1 = ExperimentLoader()
            exp1.insert(first_index, YahooS5Reader(self.benchmark_folder / "yahoo_s5"))
            self.assertEqual(1, len(exp1))
            self.assertIn(YahooS5Reader, exp1)
            self.assertEqual(0, exp1.index(YahooS5Reader))

        for last_index in [1, 100]:
            exp1 = ExperimentLoader()
            exp1.insert(0, YahooS5Reader(self.benchmark_folder / "yahoo_s5"))
            exp1.insert(last_index, MGABReader(self.benchmark_folder / "yahoo_s5"))
            self.assertEqual(2, len(exp1))
            self.assertIn(MGABReader, exp1)
            self.assertEqual(1, exp1.index(MGABReader))

        exp1 = ExperimentLoader()
        exp1.insert(0, (YahooS5Reader(self.benchmark_folder / "yahoo_s5"), (0.8, 0.2), [0, 1, 2]))
        self.assertEqual(1, len(exp1))
        self.assertIn(YahooS5Reader, exp1)
        self.assertEqual(0, exp1.index(YahooS5Reader))
        reader, split, series = exp1[0]
        self.assertIsInstance(reader, YahooS5Reader)
        self.assertTupleEqual((0.8, 0.2), split)
        self.assertListEqual([0, 1, 2], series)

    def test_num_series(self):
        yahoo = YahooS5Reader(self.benchmark_folder / "yahoo_s5")
        mgab = MGABReader(self.benchmark_folder / "mgab")
        exp1 = ExperimentLoader([yahoo, mgab])
        exp2 = ExperimentLoader([yahoo, mgab], [(0.9, 0.1), None], [None, [1, 2, 3]])
        exp3 = ExperimentLoader([yahoo, mgab], [(0.9, 0.1), None], [[1, 2, 3], [1, 2, 3]])

        self.assertEqual(len(yahoo) + len(mgab), exp1.num_series)
        self.assertEqual(len(yahoo) + 3, exp2.num_series)
        self.assertEqual(6, exp3.num_series)

    def test_get_series(self):
        def get_train_test(df: pd.DataFrame, train_pr: float) -> tuple[pd.DataFrame, pd.DataFrame]:
            length = df.shape[0]
            last_train = round(length * train_pr)
            return df.iloc[:last_train], df.iloc[last_train:]
        
        def numpy_assert_train_test(real, got, train_pr: float):
            got_train, got_test = got[0], got[1]
            real_train, real_test = get_train_test(real, train_pr)
            np.testing.assert_array_equal(real_train.values, got_train.values)
            np.testing.assert_array_equal(real_test.values, got_test.values)
            
        # check usage of default split for dataset without column is_training
        yahoo = YahooS5Reader(self.benchmark_folder / "yahoo_s5")
        mgab = MGABReader(self.benchmark_folder / "mgab")
        smd = SMDReader(self.benchmark_folder / "smd")
        exp1 = ExperimentLoader([yahoo, mgab])
        exp2 = ExperimentLoader([yahoo, mgab], [(0.9, 0.1), (0.9, 0.1)], [None, [1, 2, 3]])
        exp3 = ExperimentLoader([yahoo, mgab], [(0.5, 0.5), (0.5, 0.5)], [[1, 2, 3], [1, 2, 3]])

        numpy_assert_train_test(yahoo[0], exp1.get_series(0), 0.8)
        numpy_assert_train_test(yahoo[150], exp1.get_series(150), 0.8)
        numpy_assert_train_test(mgab[0], exp1.get_series(367), 0.8)
        numpy_assert_train_test(mgab[9], exp1.get_series(-1), 0.8)
        numpy_assert_train_test(yahoo[366], exp1.get_series(-11), 0.8)
        self.assertRaises(IndexError, exp1.get_series, 500)
        self.assertRaises(IndexError, exp1.get_series, -501)

        numpy_assert_train_test(yahoo[0], exp2.get_series(0), 0.9)
        numpy_assert_train_test(mgab[1], exp2.get_series(367), 0.9)
        numpy_assert_train_test(mgab[3], exp2.get_series(-1), 0.9)
        self.assertRaises(IndexError, exp2.get_series, 370)
        self.assertRaises(IndexError, exp2.get_series, -371)

        numpy_assert_train_test(yahoo[1], exp3.get_series(0), 0.5)
        numpy_assert_train_test(yahoo[3], exp3.get_series(2), 0.5)
        numpy_assert_train_test(mgab[1], exp3.get_series(3), 0.5)
        numpy_assert_train_test(mgab[3], exp3.get_series(-1), 0.5)
        self.assertRaises(IndexError, exp3.get_series, 7)
        self.assertRaises(IndexError, exp3.get_series, -7)
        
        # check that dataset with column is_training really use their default split
        exp1 = ExperimentLoader([smd, mgab], [(0.5, 0.5), (0.5, 0.5)])
        numpy_assert_train_test(smd[0], exp1.get_series(0), 0.5)
        numpy_assert_train_test(mgab[9], exp1.get_series(-1), 0.5)
        
        exp1 = ExperimentLoader([smd, mgab], [None, (0.5, 0.5)])
        numpy_assert_train_test(mgab[9], exp1.get_series(-1), 0.5)
        real_smd = smd[0]
        train, test = real_smd[real_smd[rts_config["DEFAULT"]["is_training"]] == 1], real_smd[real_smd[rts_config["DEFAULT"]["is_training"]] == 0]
        got_train, got_test = exp1.get_series(0)
        np.testing.assert_array_equal(train.values, got_train.values)
        np.testing.assert_array_equal(test.values, got_test.values)

    def test_get_train_test_split(self):
        mgab = MGABReader(self.benchmark_folder / "mgab")
        exp1 = ExperimentLoader([mgab, mgab])
        self.assertRaises(IndexError, exp1.get_train_test_split, -100)
        self.assertRaises(IndexError, exp1.get_train_test_split, 100)
        self.assertTupleEqual((0.8, 0.2), exp1.get_train_test_split(0))
        self.assertTupleEqual((0.8, 0.2), exp1.get_train_test_split(1))
        
        exp1 = ExperimentLoader([mgab, mgab], [(0.5, 0.5), None])
        self.assertTupleEqual((0.5, 0.5), exp1.get_train_test_split(0))
        self.assertIsNone(exp1.get_train_test_split(1))
        
        exp1 = ExperimentLoader([mgab, mgab], [None, (0.5, 0.5)])
        self.assertIsNone(exp1.get_train_test_split(0))
        self.assertTupleEqual((0.5, 0.5), exp1.get_train_test_split(1))
        
        exp1 = ExperimentLoader([mgab, mgab], [(0.5, 0.5), (0.5, 0.5)])
        self.assertTupleEqual((0.5, 0.5), exp1.get_train_test_split(0))
        self.assertTupleEqual((0.5, 0.5), exp1.get_train_test_split(1))
        
    def test_get_series_to_use(self):
        mgab = MGABReader(self.benchmark_folder / "mgab")
        exp1 = ExperimentLoader([mgab, mgab])
        self.assertRaises(IndexError, exp1.get_series_to_use, -100)
        self.assertRaises(IndexError, exp1.get_series_to_use, 100)
        self.assertIsNone(exp1.get_series_to_use(0))
        self.assertIsNone(exp1.get_series_to_use(1))
        
        exp1 = ExperimentLoader([mgab, mgab], series_to_use=[None, [0, 1, 2]])
        self.assertIsNone(exp1.get_series_to_use(0))
        self.assertListEqual([0, 1, 2], exp1.get_series_to_use(1))
        
        exp1 = ExperimentLoader([mgab, mgab], series_to_use=[[0, 1, 2], None])
        self.assertListEqual([0, 1, 2], exp1.get_series_to_use(0))
        self.assertIsNone(exp1.get_series_to_use(1))
        
        exp1 = ExperimentLoader([mgab, mgab], series_to_use=[[0, 1, 2], [0, 1, 2]])
        self.assertListEqual([0, 1, 2], exp1.get_series_to_use(0))
        self.assertListEqual([0, 1, 2], exp1.get_series_to_use(1))

    def test_series_iterator(self):
        mgab = MGABReader(self.benchmark_folder / "mgab")
        for version in [0, 1, 2]:
            if version == 0:
                exp1 = ExperimentLoader([mgab, mgab])
            elif version == 1:
                exp1 = ExperimentLoader([mgab, mgab], [(0.9, 0.1), None], [None, [1, 2, 3]])
            else:
                exp1 = ExperimentLoader([mgab, mgab], [(0.9, 0.1), None], [[1, 2, 3], [1, 2, 3]])

            for i, (train, test) in enumerate(exp1.series_iterator()):
                train_df, test_df = exp1.get_series(i)
                np.testing.assert_array_equal(train.values, train_df.values)
                np.testing.assert_array_equal(test.values, test_df.values)

    def test_set_train_test_split(self):
        yahoo = YahooS5Reader(self.benchmark_folder / "yahoo_s5")
        mgab = MGABReader(self.benchmark_folder / "mgab")
        exp1 = ExperimentLoader([yahoo, mgab])

        exp1.set_train_test_split(0, (0.1, 0.9))
        _, split, _ = exp1[0]
        self.assertEqual((0.1, 0.9), split)

        exp1.set_train_test_split(1, (0.5, 0.5))
        _, split, _ = exp1[1]
        self.assertEqual((0.5, 0.5), split)

    def test_set_series_to_use(self):
        yahoo = YahooS5Reader(self.benchmark_folder / "yahoo_s5")
        mgab = MGABReader(self.benchmark_folder / "mgab")
        exp1 = ExperimentLoader([yahoo, mgab])

        exp1.set_series_to_use(0, [1, 2, 3])
        _, _, series = exp1[0]
        self.assertListEqual([1, 2, 3], series)

        exp1.set_series_to_use(1, [1, 2, 3])
        _, _, series = exp1[1]
        self.assertListEqual([1, 2, 3], series)
