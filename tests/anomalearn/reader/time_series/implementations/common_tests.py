import logging
import re
from logging import Logger
from pathlib import Path
from unittest import TestCase

import numpy as np


def assert_get_item_behaviour(tester: TestCase,
                              output_path: Path,
                              logger: Logger,
                              reader,
                              split_string: str,
                              benchmark_folder: Path,
                              benchmark_length: int) -> None:
    """Checks that __getitem__ of readers work fine.

    Parameters
    ----------
    tester : TestCase
        TestCase instance to assert.

    output_path : Path
        The temporary path in which to store logging output.

    logger : Logger
        Object that is being used for logging.

    reader
        Object that is being tested.

    split_string : str
        The string before the path saying which file is being loaded.

    benchmark_folder : Path
        The folder of the dataset.

    benchmark_length : int
        The number of series in the dataset.

    Returns
    -------
    None
    """
    if output_path.exists():
        output_path.unlink()

    logging.basicConfig(filename=output_path, force=True)
    logger.setLevel(logging.DEBUG)

    for i in range(len(reader)):
        _ = reader[i]

    output = output_path.read_text()
    read_files = re.findall(".*" + split_string + ".*\n", output)
    filenames = set([e.split(split_string)[1] for e in read_files])
    tester.assertEqual(benchmark_length, len(filenames))
    for filename in filenames:
        filename_path = Path(filename).parent.resolve()
        tester.assertTrue(str(filename_path).startswith(str(benchmark_folder.resolve())))

    logging.basicConfig(force=True)
    output_path.unlink()

    obj1 = reader[0].values
    obj2 = reader[1].values
    obj3 = reader[2].values
    obj4 = reader[3].values

    assert_not_equal_numpy(tester, obj1, obj2, obj3, obj4)
    tester.assertRaises(IndexError, reader.__getitem__, 100000)


def assert_not_equal_numpy(tester: TestCase,
                           *args) -> None:
    """Asserts that every combination of class 2 is composed of two different arrays.

    Parameters
    ----------
    tester : TestCase
        Object used to assert.

    args
        List of numpy arrays to test that are different from each other.

    Returns
    -------
    None
    """
    for i in range(len(args) - 1):
        for j in range(i + 1, len(args)):
            tester.assertFalse(np.array_equal(args[i], args[j]))
