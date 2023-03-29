from pathlib import Path

import numpy as np

from anomalearn.utils import save_py_json


def reset_series_uni():
    series, labels = np.zeros(100).reshape(-1, 1), np.zeros(100)
    for i in range(series.shape[0]):
        if i % 2 != 0:
            series[i, 0] = 1
    return series, labels


def reset_series_multi():
    series, labels = reset_series_uni()
    return np.concatenate((series, series.copy()), axis=1), np.zeros(100)


this = Path(__file__).parent / "test_data" / "mixed_simplicity"

# case 0: score 1
uni_series, uni_labels = reset_series_uni()
# fix to solve problem with window 2
uni_labels[0] = 1
# only std detected and std + avg detected
uni_series[10] = 0.25
uni_series[11] = 0.75
uni_series[12] = 0.25
uni_labels[10] = 1
uni_labels[11] = 1
uni_labels[12] = 1
uni_labels[13] = 1
# avg + const detected, avg + std detected and all detected
uni_series[20] = 10
uni_series[21] = 11
uni_series[22] = 10
uni_labels[20] = 1
uni_labels[21] = 1
uni_labels[22] = 1
uni_labels[23] = 1
# only avg detected and avg + const detected
uni_series[30] = 2
uni_labels[30] = 1
uni_labels[31] = 1

case_num = 0
np.savetxt(str(this / f"mixed_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mixed_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mixed_score": 1}, this / f"mixed_case_{case_num}_result.json")

# case 1: 0 < score < 1
uni_series, uni_labels = reset_series_uni()
# fix to solve problem with window 2
uni_labels[0] = 1
# only std detected and std + avg detected
uni_series[10] = 0.25
uni_series[11] = 0.75
uni_series[12] = 0.25
uni_labels[10] = 1
uni_labels[11] = 1
uni_labels[12] = 1
uni_labels[13] = 1
# avg + const detected, avg + std detected and all detected
uni_series[20] = 10
uni_series[21] = 11
uni_series[22] = 12
uni_labels[20] = 1
uni_labels[21] = 1
uni_labels[22] = 1
uni_labels[23] = 1
# only avg detected and avg + const detected
uni_series[31] = 2
uni_labels[31] = 1
uni_labels[32] = 1
# undetectable anomalies
uni_labels[50] = 1
uni_labels[51] = 1
uni_labels[52] = 1

case_num = 1
np.savetxt(str(this / f"mixed_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mixed_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mixed_score": 11/14}, this / f"mixed_case_{case_num}_result.json")

# case 2: case 0 but with two dimensions
uni_series, uni_labels = reset_series_multi()
# fix to solve problem with window 2
uni_labels[0] = 1
# only std detected and std + avg detected
uni_series[10, 0] = 0.25
uni_series[11, 0] = 0.75
uni_series[12, 0] = 0.25
uni_labels[10] = 1
uni_labels[11] = 1
uni_labels[12] = 1
uni_labels[13] = 1
# avg + const detected, avg + std detected and all detected
uni_series[20, 1] = 10
uni_series[21, 1] = 11
uni_series[22, 1] = 10
uni_labels[20] = 1
uni_labels[21] = 1
uni_labels[22] = 1
uni_labels[23] = 1
# only avg detected and avg + const detected
uni_series[30, 0] = 2
uni_labels[30] = 1
uni_labels[31] = 1

case_num = 2
np.savetxt(str(this / f"mixed_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mixed_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mixed_score": 1}, this / f"mixed_case_{case_num}_result.json")

# case 3: case 1 but on two dimensions
uni_series, uni_labels = reset_series_multi()
# fix to solve problem with window 2
uni_labels[0] = 1
# only std detected and std + avg detected
uni_series[10, 0] = 0.25
uni_series[11, 0] = 0.75
uni_series[12, 0] = 0.25
uni_labels[10] = 1
uni_labels[11] = 1
uni_labels[12] = 1
uni_labels[13] = 1
# avg + const detected, avg + std detected and all detected
uni_series[20, 1] = 10
uni_series[21, 1] = 11
uni_series[22, 1] = 12
uni_labels[20] = 1
uni_labels[21] = 1
uni_labels[22] = 1
uni_labels[23] = 1
# only avg detected and avg + const detected
uni_series[31, 0] = 2
uni_labels[31] = 1
uni_labels[32] = 1
# undetectable anomalies
uni_labels[50] = 1
uni_labels[51] = 1
uni_labels[52] = 1

case_num = 3
np.savetxt(str(this / f"mixed_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mixed_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mixed_score": 11/14}, this / f"mixed_case_{case_num}_result.json")
