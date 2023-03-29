from pathlib import Path

import numpy as np

from anomalearn.utils import save_py_json


def reset_series_uni():
    return np.arange(100).reshape(-1, 1), np.zeros(100)


def reset_series_multi():
    return np.concatenate((np.arange(100).reshape(-1, 1), np.arange(100).reshape(-1, 1)), axis=1), np.zeros(100)


this = Path(__file__).parent / "test_data" / "mov_std_simplicity"

# case 0: score 0 (therefore diff=0, lower_bound=None, upper_bound=None, window=2)
uni_series, uni_labels = reset_series_uni()
uni_labels[20] = 1
case_num = 0
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 0,
              "diff_order": 0,
              "lower_bound": [None],
              "upper_bound": [None],
              "window": 2},
             this / f"mov_std_case_{case_num}_result.json")

# case 1: score 1, diff 0, lower bound, window 2
uni_series, uni_labels = reset_series_uni()
uni_labels[0] = 1
case_num = 1
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 1,
              "diff_order": 0,
              "lower_bound": [0],
              "upper_bound": [None],
              "window": 2},
             this / f"mov_std_case_{case_num}_result.json")

# case 2: score 1, diff 0, upper bound, window 2
uni_series, uni_labels = reset_series_uni()
uni_series[20] = 21
uni_labels[20] = 1
case_num = 2
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 1,
              "diff_order": 0,
              "lower_bound": [None],
              "upper_bound": [1],
              "window": 2},
             this / f"mov_std_case_{case_num}_result.json")

# case 3: score 1, diff 0, both bounds, window 2
uni_series, uni_labels = reset_series_uni()
uni_labels[0] = 1
uni_series[40] = 41
uni_labels[41] = 1
uni_labels[40] = 1
case_num = 3
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 1,
              "diff_order": 0,
              "lower_bound": [0],
              "upper_bound": [1],
              "window": 2},
             this / f"mov_std_case_{case_num}_result.json")

# case 4: score 1, diff 1, lower bound, window 2
uni_series = np.sin(np.linspace(0, 2 * np.pi, 100)).reshape((-1, 1))
uni_labels = np.zeros(100)
uni_labels[1] = 1
case_num = 4
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 1,
              "diff_order": 1,
              "lower_bound": [0],
              "upper_bound": [None],
              "window": 2},
             this / f"mov_std_case_{case_num}_result.json")

# case 5: score 1, diff 1, upper bound, window 2
uni_series, uni_labels = reset_series_uni()
uni_series[9] = 6
uni_series[10] = 8
uni_labels[9] = 1
uni_labels[10] = 1
case_num = 5
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 1,
              "diff_order": 1,
              "lower_bound": [None],
              "upper_bound": [1.5],
              "window": 2},
             this / f"mov_std_case_{case_num}_result.json")

# case 6: score 1, diff 1, both bounds, window 2
uni_series, uni_labels = reset_series_uni()
noise = np.random.rand(*uni_series.shape) * 0.0001
uni_series = uni_series + noise
uni_series[9] = 6
uni_series[10] = 8
uni_labels[1] = 1
uni_labels[9] = 1
uni_labels[10] = 1
case_num = 6
diff = np.diff(uni_series, 1, axis=0)
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 1,
              "diff_order": 1,
              "lower_bound": [0],
              "upper_bound": [np.std([diff[7], diff[8]])],
              "window": 2},
             this / f"mov_std_case_{case_num}_result.json")

# case 7: 0 < score < 1, diff 0, lower bound, window 2
uni_series, uni_labels = reset_series_uni()
uni_labels[0] = 1
uni_labels[20] = 1
case_num = 7
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 0.5,
              "diff_order": 0,
              "lower_bound": [0],
              "upper_bound": [None],
              "window": 2},
             this / f"mov_std_case_{case_num}_result.json")

# case 8: 0 < score < 1, diff 0, upper bound, window 2
uni_series, uni_labels = reset_series_uni()
uni_labels[20] = 1
uni_series[40] = 41
uni_labels[40] = 1
case_num = 8
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 0.5,
              "diff_order": 0,
              "lower_bound": [None],
              "upper_bound": [1],
              "window": 2},
             this / f"mov_std_case_{case_num}_result.json")

# case 9: 0 < score < 1, diff 0, both bounds, window 2
uni_series, uni_labels = reset_series_uni()
uni_labels[0] = 1
uni_labels[20] = 1
uni_series[40] = 41
uni_labels[40] = 1
uni_labels[41] = 1
case_num = 9
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 3/4,
              "diff_order": 0,
              "lower_bound": [0],
              "upper_bound": [1],
              "window": 2},
             this / f"mov_std_case_{case_num}_result.json")

# case 10: 0 < score < 1, diff 1, lower bound, window 2
uni_series = np.sin(np.linspace(0, 2 * np.pi, 100)).reshape((-1, 1))
uni_labels = np.zeros(100)
uni_labels[1] = 1
uni_labels[35] = 1
case_num = 10
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 0.5,
              "diff_order": 1,
              "lower_bound": [0],
              "upper_bound": [None],
              "window": 2},
             this / f"mov_std_case_{case_num}_result.json")

# case 11: 0 < score < 1, diff 1, upper bound, window 2
uni_series, uni_labels = reset_series_uni()
uni_series[9] = 6
uni_series[10] = 8
uni_labels[9] = 1
uni_labels[10] = 1
uni_labels[70] = 1
case_num = 11
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 2/3,
              "diff_order": 1,
              "lower_bound": [None],
              "upper_bound": [1.5],
              "window": 2},
             this / f"mov_std_case_{case_num}_result.json")

# case 12: 0 < score < 1, diff 1, both bounds, window 2
uni_series, uni_labels = reset_series_uni()
noise = np.random.rand(*uni_series.shape) * 0.0001
uni_series = uni_series + noise
uni_series[9] = 6
uni_series[10] = 8
uni_labels[1] = 1
uni_labels[9] = 1
uni_labels[10] = 1
uni_labels[70] = 1
case_num = 12
diff = np.diff(uni_series, 1, axis=0)
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 3/4,
              "diff_order": 1,
              "lower_bound": [0],
              "upper_bound": [np.std([diff[7], diff[8]])],
              "window": 2},
             this / f"mov_std_case_{case_num}_result.json")

# case 13: score 1, diff 0, lower bound, window >2
uni_series, uni_labels = reset_series_uni()
uni_series[18] = 19
uni_series[20] = 19
uni_labels[19] = 1
case_num = 13
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 1,
              "diff_order": 0,
              "lower_bound": [0],
              "upper_bound": [None],
              "window": 3},
             this / f"mov_std_case_{case_num}_result.json")

# case 14: score 1, diff 0, upper bound, window >2
uni_series, uni_labels = reset_series_uni()
uni_series[18] = 19
uni_series[20] = 19
uni_labels[17] = 1
uni_labels[21] = 1
case_num = 14
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 1,
              "diff_order": 0,
              "lower_bound": [None],
              "upper_bound": [np.std([uni_series[16], uni_series[17], uni_series[18]])],
              "window": 3},
             this / f"mov_std_case_{case_num}_result.json")

# case 15: score 1, diff 0, both bounds, window >2
uni_series, uni_labels = reset_series_uni()
uni_series[18] = 19
uni_series[20] = 19
uni_labels[17] = 1
uni_labels[19] = 1
uni_labels[21] = 1
case_num = 15
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 1,
              "diff_order": 0,
              "lower_bound": [0],
              "upper_bound": [np.std([uni_series[16], uni_series[17], uni_series[18]])],
              "window": 3},
             this / f"mov_std_case_{case_num}_result.json")

# case 16: score 1, diff 1, lower bound, window >2
uni_series, uni_labels = reset_series_uni()
noise = np.random.rand(*uni_series.shape) * 0.0001
uni_series = uni_series + noise
uni_series[0] = 1
uni_series[1] = 1
uni_series[2] = 1
uni_labels[1] = 1
case_num = 16
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 1,
              "diff_order": 1,
              "lower_bound": [0],
              "upper_bound": [None],
              "window": 3},
             this / f"mov_std_case_{case_num}_result.json")

# case 17: score 1, diff 1, upper bound, window >2
uni_series, uni_labels = reset_series_uni()
uni_series[10] = 8
uni_labels[9] = 1
uni_labels[10] = 1
uni_labels[11] = 1
uni_labels[12] = 1
diff = np.diff(uni_series, 1, axis=0)
case_num = 17
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 1,
              "diff_order": 1,
              "lower_bound": [None],
              "upper_bound": [np.std([diff[7], diff[8], diff[9]])],
              "window": 3},
             this / f"mov_std_case_{case_num}_result.json")

# case 18: score 1, diff 1, both bounds, window >2
uni_series, uni_labels = reset_series_uni()
noise = np.random.rand(*uni_series.shape) * 0.0001
uni_series = uni_series + noise
uni_series[10] = 15
uni_labels[9] = 1
uni_labels[10] = 1
uni_labels[11] = 1
uni_labels[12] = 1
uni_series[0] = 1
uni_series[1] = 1
uni_series[2] = 1
uni_labels[1] = 1
diff = np.diff(uni_series, 1, axis=0)
case_num = 18
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 1,
              "diff_order": 1,
              "lower_bound": [0],
              "upper_bound": [min(np.std([diff[7], diff[8], diff[9]]), np.std([diff[10], diff[11], diff[12]]))],
              "window": 3},
             this / f"mov_std_case_{case_num}_result.json")

# case 19: 0 < score < 1, diff 0, lower bound, window >2
uni_series, uni_labels = reset_series_uni()
uni_series[18] = 19
uni_series[20] = 19
uni_labels[19] = 1
uni_labels[70] = 1
case_num = 13
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 0.5,
              "diff_order": 0,
              "lower_bound": [0],
              "upper_bound": [None],
              "window": 3},
             this / f"mov_std_case_{case_num}_result.json")

# case 20: 0 < score < 1, diff 0, upper bound, window >2
uni_series, uni_labels = reset_series_uni()
uni_series[18] = 19
uni_series[20] = 19
uni_labels[17] = 1
uni_labels[21] = 1
uni_labels[70] = 1
case_num = 20
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 2/3,
              "diff_order": 0,
              "lower_bound": [None],
              "upper_bound": [np.std([uni_series[16], uni_series[17], uni_series[18]])],
              "window": 3},
             this / f"mov_std_case_{case_num}_result.json")

# case 21: 0 < score < 1, diff 0, both bounds, window >2
uni_series, uni_labels = reset_series_uni()
uni_series[18] = 19
uni_series[20] = 19
uni_labels[17] = 1
uni_labels[19] = 1
uni_labels[21] = 1
uni_labels[70] = 1
case_num = 21
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 3/4,
              "diff_order": 0,
              "lower_bound": [0],
              "upper_bound": [np.std([uni_series[16], uni_series[17], uni_series[18]])],
              "window": 3},
             this / f"mov_std_case_{case_num}_result.json")

# case 22: 0 < score < 1, diff 1, lower bound, window >2
uni_series, uni_labels = reset_series_uni()
noise = np.random.rand(*uni_series.shape) * 0.0001
uni_series = uni_series + noise
uni_series[0] = 1
uni_series[1] = 1
uni_series[2] = 1
uni_labels[1] = 1
uni_labels[30] = 1
case_num = 22
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 0.5,
              "diff_order": 1,
              "lower_bound": [0],
              "upper_bound": [None],
              "window": 3},
             this / f"mov_std_case_{case_num}_result.json")

# case 23: 0 < score < 1, diff 1, upper bound, window >2
uni_series, uni_labels = reset_series_uni()
uni_series[10] = 8
uni_labels[9] = 1
uni_labels[10] = 1
uni_labels[11] = 1
uni_labels[12] = 1
uni_labels[70] = 1
diff = np.diff(uni_series, 1, axis=0)
case_num = 23
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 4/5,
              "diff_order": 1,
              "lower_bound": [None],
              "upper_bound": [np.std([diff[7], diff[8], diff[9]])],
              "window": 3},
             this / f"mov_std_case_{case_num}_result.json")

# case 24: 0 < score < 1, diff 1, both bounds, window >2
uni_series, uni_labels = reset_series_uni()
noise = np.random.rand(*uni_series.shape) * 0.0001
uni_series = uni_series + noise
uni_series[10] = 15
uni_labels[9] = 1
uni_labels[10] = 1
uni_labels[11] = 1
uni_labels[12] = 1
uni_series[0] = 1
uni_series[1] = 1
uni_series[2] = 1
uni_labels[1] = 1
uni_labels[70] = 1
diff = np.diff(uni_series, 1, axis=0)
case_num = 24
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 5/6,
              "diff_order": 1,
              "lower_bound": [0],
              "upper_bound": [min(np.std([diff[7], diff[8], diff[9]]), np.std([diff[10], diff[11], diff[12]]))],
              "window": 3},
             this / f"mov_std_case_{case_num}_result.json")

# case 25: case 0 but multivariate
uni_series, uni_labels = reset_series_multi()
uni_labels[20] = 1
case_num = 25
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 0,
              "diff_order": 0,
              "lower_bound": [None, None],
              "upper_bound": [None, None],
              "window": 2},
             this / f"mov_std_case_{case_num}_result.json")

# case 26: case 15 but multivariate on first dim
uni_series, uni_labels = reset_series_multi()
uni_series[18, 0] = 19
uni_series[20, 0] = 19
uni_labels[17] = 1
uni_labels[19] = 1
uni_labels[21] = 1
case_num = 26
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 1,
              "diff_order": 0,
              "lower_bound": [0, None],
              "upper_bound": [np.std([uni_series[16, 0], uni_series[17, 0], uni_series[18, 0]]), None],
              "window": 3},
             this / f"mov_std_case_{case_num}_result.json")

# case 27: case 15 but multivariate on second dim
uni_series, uni_labels = reset_series_multi()
uni_series[18, 1] = 19
uni_series[20, 1] = 19
uni_labels[17] = 1
uni_labels[19] = 1
uni_labels[21] = 1
case_num = 27
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 1,
              "diff_order": 0,
              "lower_bound": [None, 0],
              "upper_bound": [None, np.std([uni_series[16, 1], uni_series[17, 1], uni_series[18, 1]])],
              "window": 3},
             this / f"mov_std_case_{case_num}_result.json")

# case 28: case 15 but multivariate on both dim
uni_series, uni_labels = reset_series_multi()
uni_series[18, 0] = 19
uni_series[20, 0] = 19
uni_series[18, 1] = 19
uni_series[20, 1] = 19
uni_labels[17] = 1
uni_labels[19] = 1
uni_labels[21] = 1
case_num = 28
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 1,
              "diff_order": 0,
              "lower_bound": [0, 0],
              "upper_bound": [np.std([uni_series[16, 0], uni_series[17, 0], uni_series[18, 0]]), np.std([uni_series[16, 1], uni_series[17, 1], uni_series[18, 1]])],
              "window": 3},
             this / f"mov_std_case_{case_num}_result.json")

# case 29: case 13 but multivariate on first dim
uni_series, uni_labels = reset_series_multi()
uni_series[18, 0] = 19
uni_series[20, 0] = 19
uni_labels[19] = 1
case_num = 29
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 1,
              "diff_order": 0,
              "lower_bound": [0, None],
              "upper_bound": [None, None],
              "window": 3},
             this / f"mov_std_case_{case_num}_result.json")

# case 30: case 14 but multivariate on second dim
uni_series, uni_labels = reset_series_multi()
uni_series[18, 1] = 19
uni_series[20, 1] = 19
uni_labels[17] = 1
uni_labels[21] = 1
case_num = 30
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 1,
              "diff_order": 0,
              "lower_bound": [None, None],
              "upper_bound": [None, np.std([uni_series[16, 1], uni_series[17, 1], uni_series[18, 1]])],
              "window": 3},
             this / f"mov_std_case_{case_num}_result.json")
