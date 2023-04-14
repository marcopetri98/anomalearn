from pathlib import Path

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np

from anomalearn.reader.time_series import NABReader
from anomalearn.visualizer import line_plot


if __name__ == "__main__":
    reader = NABReader(Path(__file__).parent / "../../../data/anomaly_detection/nab")
    
    for ds in reader:
        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        gs = gridspec.GridSpec(2, 1)
    
        series = fig.add_subplot(gs[0, 0])
        line_plot(np.arange(ds["timestamp"].values.shape[0]),
                  ds["value"].values,
                  ax=series)
    
        targets = fig.add_subplot(gs[1, 0])
        line_plot(np.arange(ds["timestamp"].values.shape[0]),
                  ds["class"].values,
                  ax=targets)
    
        plt.show()
