from pathlib import Path

import numpy as np

from anomalearn.utils import save_py_json


def reset_series_uni():
    return np.arange(100).reshape(-1, 1), np.zeros(100)


def reset_series_multi():
    return np.concatenate((np.arange(100).reshape(-1, 1), np.arange(100).reshape(-1, 1)), axis=1), np.zeros(100)


this = Path(__file__).parent / "test_data" / "constant_simplicity"

# case 0: score 0 (therefore diff=0, lower_bound=None, upper_bound=None)
uni_series, uni_labels = reset_series_uni()
uni_labels[20] = 1
case_num = 0
np.savetxt(str(this / f"const_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"const_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"constant_score": 0, "diff_order": 0, "lower_bound": [None], "upper_bound": [None]}, this / f"const_case_{case_num}_result.json")

# case 1: score 1, diff 0, lower bound
case_num = 1
uni_series, uni_labels = reset_series_uni()
uni_series[70] = -1000
uni_labels[70] = 1
np.savetxt(str(this / f"const_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"const_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"constant_score": 1, "diff_order": 0, "lower_bound": [-1000], "upper_bound": [None]}, this / f"const_case_{case_num}_result.json")

# case 2: score 1, diff 0, upper bound
case_num = 2
uni_series, uni_labels = reset_series_uni()
uni_series[20] = 1000
uni_labels[20] = 1
np.savetxt(str(this / f"const_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"const_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"constant_score": 1, "diff_order": 0, "lower_bound": [None], "upper_bound": [1000]}, this / f"const_case_{case_num}_result.json")

# case 3: score 1, diff 0, both bounds
case_num = 3
uni_series, uni_labels = reset_series_uni()
uni_series[20] = 1000
uni_labels[20] = 1
uni_series[70] = -1000
uni_labels[70] = 1
np.savetxt(str(this / f"const_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"const_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"constant_score": 1, "diff_order": 0, "lower_bound": [-1000], "upper_bound": [1000]}, this / f"const_case_{case_num}_result.json")

# case 4: score 1, diff 1, lower bound
case_num = 4
uni_series, uni_labels = reset_series_uni()
uni_series[20] = 19
uni_labels[20] = 1
np.savetxt(str(this / f"const_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"const_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"constant_score": 1, "diff_order": 1, "lower_bound": [0], "upper_bound": [None]}, this / f"const_case_{case_num}_result.json")

# case 5: score 1, diff 1, upper bound
case_num = 5
uni_series, uni_labels = reset_series_uni()
uni_series[20] = 21
uni_labels[20] = 1
np.savetxt(str(this / f"const_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"const_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"constant_score": 1, "diff_order": 1, "lower_bound": [None], "upper_bound": [2]}, this / f"const_case_{case_num}_result.json")

# case 6: score 1, diff 1, both bounds
case_num = 6
uni_series, uni_labels = reset_series_uni()
uni_series[20] = 23
uni_labels[20] = 1
uni_series[21] = 23
uni_series[22] = 23
uni_series[70] = 68
uni_labels[70] = 1
np.savetxt(str(this / f"const_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"const_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"constant_score": 1, "diff_order": 1, "lower_bound": [-1], "upper_bound": [4]}, this / f"const_case_{case_num}_result.json")

# case 7: 0 < score < 1, diff 0, lower bound
case_num = 7
uni_series, uni_labels = reset_series_uni()
uni_labels[50] = 1
uni_series[70] = -1000
uni_labels[70] = 1
np.savetxt(str(this / f"const_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"const_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"constant_score": 0.5, "diff_order": 0, "lower_bound": [-1000], "upper_bound": [None]}, this / f"const_case_{case_num}_result.json")

# case 8: 0 < score < 1, diff 0, upper bound
case_num = 8
uni_series, uni_labels = reset_series_uni()
uni_labels[50] = 1
uni_series[20] = 1000
uni_labels[20] = 1
np.savetxt(str(this / f"const_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"const_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"constant_score": 0.5, "diff_order": 0, "lower_bound": [None], "upper_bound": [1000]}, this / f"const_case_{case_num}_result.json")

# case 9: 0 < score < 1, diff 0, both bounds
case_num = 9
uni_series, uni_labels = reset_series_uni()
uni_labels[50] = 1
uni_series[20] = 1000
uni_labels[20] = 1
uni_series[70] = -1000
uni_labels[70] = 1
np.savetxt(str(this / f"const_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"const_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"constant_score": 2/3, "diff_order": 0, "lower_bound": [-1000], "upper_bound": [1000]}, this / f"const_case_{case_num}_result.json")

# case 10: 0 < score < 1, diff 1, lower bound
case_num = 10
uni_series, uni_labels = reset_series_uni()
uni_labels[50] = 1
uni_series[20] = 19
uni_labels[20] = 1
np.savetxt(str(this / f"const_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"const_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"constant_score": 0.5, "diff_order": 1, "lower_bound": [0], "upper_bound": [None]}, this / f"const_case_{case_num}_result.json")

# case 11: 0 < score < 1, diff 1, upper bound
case_num = 11
uni_series, uni_labels = reset_series_uni()
uni_labels[50] = 1
uni_series[20] = 21
uni_labels[20] = 1
np.savetxt(str(this / f"const_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"const_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"constant_score": 0.5, "diff_order": 1, "lower_bound": [None], "upper_bound": [2]}, this / f"const_case_{case_num}_result.json")

# case 12: 0 < score < 1, diff 1, both bounds
case_num = 12
uni_series, uni_labels = reset_series_uni()
uni_labels[50] = 1
uni_series[20] = 18
uni_labels[20] = 1
uni_series[40] = 43
uni_series[41] = 43
uni_series[42] = 43
uni_labels[40] = 1
np.savetxt(str(this / f"const_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"const_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"constant_score": 2/3, "diff_order": 1, "lower_bound": [-1], "upper_bound": [4]}, this / f"const_case_{case_num}_result.json")

# case 13: case 0 but multivariate
uni_series, uni_labels = reset_series_multi()
uni_labels[20] = 1
case_num = 13
np.savetxt(str(this / f"const_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"const_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"constant_score": 0, "diff_order": 0, "lower_bound": [None, None], "upper_bound": [None, None]}, this / f"const_case_{case_num}_result.json")

# case 14: case 4 but multivariate on first dim
uni_series, uni_labels = reset_series_multi()
uni_series[20, 0] = 19
uni_labels[20] = 1
case_num = 14
np.savetxt(str(this / f"const_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"const_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"constant_score": 1, "diff_order": 1, "lower_bound": [0, None], "upper_bound": [None, None]}, this / f"const_case_{case_num}_result.json")

# case 15: case 4 but multivariate on second dim
uni_series, uni_labels = reset_series_multi()
uni_series[20, 1] = 19
uni_labels[20] = 1
case_num = 15
np.savetxt(str(this / f"const_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"const_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"constant_score": 1, "diff_order": 1, "lower_bound": [None, 0], "upper_bound": [None, None]}, this / f"const_case_{case_num}_result.json")

# case 16: case 2 but multivariate on first dim
uni_series, uni_labels = reset_series_multi()
uni_series[20, 0] = 1000
uni_labels[20] = 1
case_num = 16
np.savetxt(str(this / f"const_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"const_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"constant_score": 1, "diff_order": 0, "lower_bound": [None, None], "upper_bound": [1000, None]}, this / f"const_case_{case_num}_result.json")

# case 17: case 2 but multivariate on second dim
uni_series, uni_labels = reset_series_multi()
uni_series[20, 1] = 1000
uni_labels[20] = 1
case_num = 17
np.savetxt(str(this / f"const_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"const_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"constant_score": 1, "diff_order": 0, "lower_bound": [None, None], "upper_bound": [None, 1000]}, this / f"const_case_{case_num}_result.json")

# case 18: case 3 but multivariate on first and second dim
uni_series, uni_labels = reset_series_multi()
uni_series[20, 1] = 1000
uni_labels[20] = 1
uni_series[70, 0] = -1000
uni_labels[70] = 1
case_num = 18
np.savetxt(str(this / f"const_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"const_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"constant_score": 1, "diff_order": 0, "lower_bound": [-1000, None], "upper_bound": [None, 1000]}, this / f"const_case_{case_num}_result.json")
