import numpy as np

from anomalearn.analysis.dataset_simplicity import _analyse_constant_simplicity, _find_constant_score, _fast_execute_movement_simplicity, _find_best_constants, _fast_execute_mixed_score_simplicity
from anomalearn.utils import mov_avg

if __name__ == "__main__":
    method = None
    
    welcome_msg = "To diagnose the parallelism obtained in functions: " \
                  "_analyse_constant_simplicity, _find_constant_score, " \
                  "_fast_execute_movement_simplicity, _find_best_constants, " \
                  "_fast_execute_mixed_score_simplicity you  can insert as " \
                  "commands analyse, find, movement, constant, mixed " \
                  "respectively."
    print(welcome_msg, end="\n\n")
    
    while method is None or method not in ["analyse", "find", "movement", "constant", "mixed"]:
        method = input("Which function do you want to diagnose in terms of parallelism? [analyse/find/movement/constant/mixed] ")

        dummy_series, dummy_labels = np.random.rand(100, 3), np.zeros(100)
        dummy_labels[50] = 1
        
        if method == "analyse":
            _ = _analyse_constant_simplicity(dummy_series, dummy_labels, 1)
            print("_analyse_constant_simplicity is not parallel")
        elif method == "find":
            _ = _find_constant_score(dummy_series, dummy_labels)
            _find_constant_score.parallel_diagnostics(level=4)
        elif method == "constant":
            dummy_uni = np.random.rand(100)
            dummy_uni_labels = np.zeros(100)
            dummy_uni_labels[50] = 1
            asc = np.array(np.sort(dummy_uni, axis=0))
            desc = np.array(np.flip(asc, axis=0))
            _ = _find_best_constants(dummy_uni, desc, asc, dummy_uni_labels)
            _find_best_constants.parallel_diagnostics(level=4)
        elif method == "movement":
            _ = _fast_execute_movement_simplicity(dummy_series, dummy_labels, 1, [2, 3], mov_avg)
            print("_fast_execute_movement_simplicity is not parallel")
        elif method == "mixed":
            _ = _fast_execute_mixed_score_simplicity(dummy_series, dummy_labels, 1, [2, 3])
            _fast_execute_mixed_score_simplicity.parallel_diagnostics(level=4)
