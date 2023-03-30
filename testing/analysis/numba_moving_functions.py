import numpy as np

from anomalearn.utils.moving_functions import _mov_avg, _mov_std

if __name__ == "__main__":
    method = None
    
    while method is None or method not in ["avg", "std"]:
        method = input("Which function do you want to diagnose in terms of parallelism? [avg/std] ")
    
        if method == "avg":
            _ = _mov_avg(np.random.rand(100, 3), 2)
            _mov_avg.parallel_diagnostics(level=4)
        elif method == "std":
            _ = _mov_std(np.random.rand(100, 3), 2)
            _mov_std.parallel_diagnostics(level=4)
