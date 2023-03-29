from pathlib import Path

import numpy as np

from anomalearn.utils import save_py_json


def reset_series_uni():
    return np.arange(100).reshape(-1, 1), np.zeros(100)


def reset_series_multi():
    return np.concatenate((np.arange(100).reshape(-1, 1), np.arange(100).reshape(-1, 1)), axis=1), np.zeros(100)


this = Path(__file__).parent / "test_data" / "mov_avg_simplicity"

# case 0: score 0 (therefore diff=0, lower_bound=None, upper_bound=None, window=2)
uni_series, uni_labels = reset_series_uni()
uni_labels[20] = 1
case_num = 0
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 0,
              "diff_order": 0,
              "lower_bound": [None],
              "upper_bound": [None],
              "window": 2},
             this / f"mov_avg_case_{case_num}_result.json")

# case 1: score 1, diff 0, lower bound, window 2
uni_series, uni_labels = reset_series_uni()
uni_series[19] = -100
uni_labels[19] = 1
uni_series[20] = -100
uni_labels[20] = 1
case_num = 1
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 1,
              "diff_order": 0,
              "lower_bound": [-41],
              "upper_bound": [None],
              "window": 2},
             this / f"mov_avg_case_{case_num}_result.json")

# case 2: score 1, diff 0, upper bound, window 2
uni_series, uni_labels = reset_series_uni()
uni_series[20] = 1000
uni_labels[21] = 1
case_num = 2
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 1,
              "diff_order": 0,
              "lower_bound": [None],
              "upper_bound": [510.5],
              "window": 2},
             this / f"mov_avg_case_{case_num}_result.json")

# case 3: score 1, diff 0, both bounds, window 2
uni_series, uni_labels = reset_series_uni()
uni_series[10] = -100
uni_labels[10] = 1
uni_series[20] = 1000
uni_labels[21] = 1
case_num = 3
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 1,
              "diff_order": 0,
              "lower_bound": [-45.5],
              "upper_bound": [510.5],
              "window": 2},
             this / f"mov_avg_case_{case_num}_result.json")

# case 4: score 1, diff 1, lower bound, window 2
uni_series, uni_labels = reset_series_uni()
uni_series[20] = 0
uni_labels[20] = 1
case_num = 4
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 1,
              "diff_order": 1,
              "lower_bound": [-9],
              "upper_bound": [None],
              "window": 2},
             this / f"mov_avg_case_{case_num}_result.json")

# case 5: score 1, diff 1, upper bound, window 2
uni_series, uni_labels = reset_series_uni()
uni_series[20] = 40
uni_labels[20] = 1
case_num = 5
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 1,
              "diff_order": 1,
              "lower_bound": [None],
              "upper_bound": [11],
              "window": 2},
             this / f"mov_avg_case_{case_num}_result.json")

# case 6: score 1, diff 1, both bounds, window 2
uni_series, uni_labels = reset_series_uni()
uni_labels[35] = 1
uni_labels[20] = 1
uni_series[20] = 0
uni_series[35] = 65
uni_series[36] = 60
uni_series[37] = 55
uni_series[38] = 50
uni_series[39] = 45
case_num = 6
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 1,
              "diff_order": 1,
              "lower_bound": [-9],
              "upper_bound": [16],
              "window": 2},
             this / f"mov_avg_case_{case_num}_result.json")

# case 7: 0 < score < 1, diff 0, lower bound, window 2
uni_series, uni_labels = reset_series_uni()
uni_series[19] = -100
uni_labels[19] = 1
uni_series[20] = -100
uni_labels[20] = 1
uni_labels[70] = 1
case_num = 7
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 2/3,
              "diff_order": 0,
              "lower_bound": [-41],
              "upper_bound": [None],
              "window": 2},
             this / f"mov_avg_case_{case_num}_result.json")

# case 8: 0 < score < 1, diff 0, upper bound, window 2
uni_series, uni_labels = reset_series_uni()
uni_series[20] = 1000
uni_labels[21] = 1
uni_labels[70] = 1
case_num = 8
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 0.5,
              "diff_order": 0,
              "lower_bound": [None],
              "upper_bound": [510.5],
              "window": 2},
             this / f"mov_avg_case_{case_num}_result.json")

# case 9: 0 < score < 1, diff 0, both bounds, window 2
uni_series, uni_labels = reset_series_uni()
uni_series[10] = -100
uni_labels[10] = 1
uni_series[20] = 1000
uni_labels[21] = 1
uni_labels[70] = 1
case_num = 9
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 2/3,
              "diff_order": 0,
              "lower_bound": [-45.5],
              "upper_bound": [510.5],
              "window": 2},
             this / f"mov_avg_case_{case_num}_result.json")

# case 10: 0 < score < 1, diff 1, lower bound, window 2
uni_series, uni_labels = reset_series_uni()
uni_series[20] = 0
uni_labels[20] = 1
uni_labels[70] = 1
case_num = 10
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 0.5,
              "diff_order": 1,
              "lower_bound": [-9],
              "upper_bound": [None],
              "window": 2},
             this / f"mov_avg_case_{case_num}_result.json")

# case 11: 0 < score < 1, diff 1, upper bound, window 2
uni_series, uni_labels = reset_series_uni()
uni_labels[20] = 1
uni_labels[35] = 1
uni_series[35] = 65
case_num = 11
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 0.5,
              "diff_order": 1,
              "lower_bound": [None],
              "upper_bound": [16],
              "window": 2},
             this / f"mov_avg_case_{case_num}_result.json")

# case 12: 0 < score < 1, diff 1, both bounds, window 2
uni_series, uni_labels = reset_series_uni()
uni_labels[35] = 1
uni_labels[20] = 1
uni_series[20] = 0
uni_series[35] = 65
uni_series[36] = 60
uni_series[37] = 55
uni_series[38] = 50
uni_series[39] = 45
uni_labels[70] = 1
case_num = 12
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 2/3,
              "diff_order": 1,
              "lower_bound": [-9],
              "upper_bound": [16],
              "window": 2},
             this / f"mov_avg_case_{case_num}_result.json")

# case 13: score 1, diff 0, lower bound, window >2
uni_series, uni_labels = reset_series_uni()
uni_labels[18] = 1
uni_labels[19] = 1
uni_labels[20] = 1
uni_series[19] = -100
case_num = 13
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 1,
              "diff_order": 0,
              "lower_bound": [-59/3],
              "upper_bound": [None],
              "window": 3},
             this / f"mov_avg_case_{case_num}_result.json")

# case 14: score 1, diff 0, upper bound, window >2
uni_series, uni_labels = reset_series_uni()
uni_labels[38] = 1
uni_labels[39] = 1
uni_labels[40] = 1
uni_series[39] = 1000
case_num = 14
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 1,
              "diff_order": 0,
              "lower_bound": [None],
              "upper_bound": [1075/3],
              "window": 3},
             this / f"mov_avg_case_{case_num}_result.json")

# case 15: score 1, diff 0, both bounds, window >2
uni_series, uni_labels = reset_series_uni()
uni_labels[18] = 1
uni_labels[19] = 1
uni_labels[20] = 1
uni_series[19] = -100
uni_labels[38] = 1
uni_labels[39] = 1
uni_labels[40] = 1
uni_series[39] = 1000
case_num = 15
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 1,
              "diff_order": 0,
              "lower_bound": [-59/3],
              "upper_bound": [1075/3],
              "window": 3},
             this / f"mov_avg_case_{case_num}_result.json")

# case 16: score 1, diff 1, lower bound, window >2
uni_series, uni_labels = reset_series_uni()
uni_series[10] = 0
uni_labels[9] = 1
case_num = 16
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 1,
              "diff_order": 1,
              "lower_bound": [-7/3],
              "upper_bound": [None],
              "window": 3},
             this / f"mov_avg_case_{case_num}_result.json")

# case 17: score 1, diff 1, upper bound, window >2
uni_series, uni_labels = reset_series_uni()
uni_series[10] = 20
uni_labels[9] = 1
case_num = 17
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 1,
              "diff_order": 1,
              "lower_bound": [None],
              "upper_bound": [13/3],
              "window": 3},
             this / f"mov_avg_case_{case_num}_result.json")

# case 18: score 1, diff 1, both bounds, window >2
uni_series, uni_labels = reset_series_uni()
uni_series[10] = 0
uni_labels[9] = 1
uni_labels[12] = 1
case_num = 18
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 1,
              "diff_order": 1,
              "lower_bound": [-7/3],
              "upper_bound": [13/3],
              "window": 3},
             this / f"mov_avg_case_{case_num}_result.json")

# case 19: 0 < score < 1, diff 0, lower bound, window >2
uni_series, uni_labels = reset_series_uni()
uni_labels[18] = 1
uni_labels[19] = 1
uni_labels[20] = 1
uni_series[19] = -100
uni_labels[70] = 1
case_num = 19
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 3/4,
              "diff_order": 0,
              "lower_bound": [-59/3],
              "upper_bound": [None],
              "window": 3},
             this / f"mov_avg_case_{case_num}_result.json")

# case 20: 0 < score < 1, diff 0, upper bound, window >2
uni_series, uni_labels = reset_series_uni()
uni_labels[38] = 1
uni_labels[39] = 1
uni_labels[40] = 1
uni_series[39] = 1000
uni_labels[80] = 1
case_num = 20
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 3/4,
              "diff_order": 0,
              "lower_bound": [None],
              "upper_bound": [1075/3],
              "window": 3},
             this / f"mov_avg_case_{case_num}_result.json")

# case 21: 0 < score < 1, diff 0, both bounds, window >2
uni_series, uni_labels = reset_series_uni()
uni_labels[18] = 1
uni_labels[19] = 1
uni_labels[20] = 1
uni_series[19] = -100
uni_labels[38] = 1
uni_labels[39] = 1
uni_labels[40] = 1
uni_series[39] = 1000
uni_labels[80] = 1
case_num = 21
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 6/7,
              "diff_order": 0,
              "lower_bound": [-59/3],
              "upper_bound": [1075/3],
              "window": 3},
             this / f"mov_avg_case_{case_num}_result.json")

# case 22: 0 < score < 1, diff 1, lower bound, window >2
uni_series, uni_labels = reset_series_uni()
uni_series[10] = 0
uni_labels[9] = 1
uni_labels[70] = 1
case_num = 22
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 0.5,
              "diff_order": 1,
              "lower_bound": [-7/3],
              "upper_bound": [None],
              "window": 3},
             this / f"mov_avg_case_{case_num}_result.json")

# case 23: 0 < score < 1, diff 1, upper bound, window >2
uni_series, uni_labels = reset_series_uni()
uni_series[10] = 20
uni_labels[9] = 1
uni_labels[80] = 1
case_num = 23
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 0.5,
              "diff_order": 1,
              "lower_bound": [None],
              "upper_bound": [13/3],
              "window": 3},
             this / f"mov_avg_case_{case_num}_result.json")

# case 24: 0 < score < 1, diff 1, both bounds, window >2
uni_series, uni_labels = reset_series_uni()
uni_series[10] = 0
uni_labels[9] = 1
uni_labels[12] = 1
uni_labels[80] = 1
case_num = 24
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 2/3,
              "diff_order": 1,
              "lower_bound": [-7/3],
              "upper_bound": [13/3],
              "window": 3},
             this / f"mov_avg_case_{case_num}_result.json")

# case 25: case 0 but multivariate
uni_series, uni_labels = reset_series_multi()
uni_labels[20] = 1
case_num = 25
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 0,
              "diff_order": 0,
              "lower_bound": [None, None],
              "upper_bound": [None, None],
              "window": 2},
             this / f"mov_avg_case_{case_num}_result.json")

# case 26: case 15 but multivariate on first and second dim
uni_series, uni_labels = reset_series_multi()
uni_labels[18] = 1
uni_labels[19] = 1
uni_labels[20] = 1
uni_series[19, 1] = -100
uni_labels[38] = 1
uni_labels[39] = 1
uni_labels[40] = 1
uni_series[39, 0] = 1000
case_num = 26
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 1,
              "diff_order": 0,
              "lower_bound": [None, -59/3],
              "upper_bound": [1075/3, None],
              "window": 3},
             this / f"mov_avg_case_{case_num}_result.json")

# case 27: case 13 but multivariate on first dim
uni_series, uni_labels = reset_series_multi()
uni_labels[18] = 1
uni_labels[19] = 1
uni_labels[20] = 1
uni_series[19, 0] = -100
case_num = 27
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 1,
              "diff_order": 0,
              "lower_bound": [-59/3, None],
              "upper_bound": [None, None],
              "window": 3},
             this / f"mov_avg_case_{case_num}_result.json")

# case 28: case 14 but multivariate on second dim
uni_series, uni_labels = reset_series_multi()
uni_labels[38] = 1
uni_labels[39] = 1
uni_labels[40] = 1
uni_series[39, 1] = 1000
case_num = 28
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 1,
              "diff_order": 0,
              "lower_bound": [None, None],
              "upper_bound": [None, 1075/3],
              "window": 3},
             this / f"mov_avg_case_{case_num}_result.json")

# case 29: case 15 but multivariate on all dim
uni_series, uni_labels = reset_series_multi()
uni_labels[18] = 1
uni_labels[19] = 1
uni_labels[20] = 1
uni_series[19, 1] = -100
uni_labels[18] = 1
uni_labels[19] = 1
uni_labels[20] = 1
uni_series[19, 0] = -100
uni_labels[38] = 1
uni_labels[39] = 1
uni_labels[40] = 1
uni_series[39, 1] = 1000
uni_labels[38] = 1
uni_labels[39] = 1
uni_labels[40] = 1
uni_series[39, 0] = 1000
case_num = 29
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 1,
              "diff_order": 0,
              "lower_bound": [-59/3, -59/3],
              "upper_bound": [1075/3, 1075/3],
              "window": 3},
             this / f"mov_avg_case_{case_num}_result.json")
