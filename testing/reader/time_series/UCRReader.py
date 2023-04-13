from pathlib import Path
from matplotlib import pyplot as plt, gridspec

from anomalearn.reader.time_series import UCRReader
from anomalearn.visualizer import line_plot

if __name__ == "__main__":
    reader = UCRReader(Path(__file__).parent / "../../../data/anomaly_detection/ucr")
    
    for ds in reader:
        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        gs = gridspec.GridSpec(2, 1)
    
        series = fig.add_subplot(gs[0, 0])
        line_plot(ds["timestamp"].values,
                  ds["value"].values,
                  ax=series)
    
        targets = fig.add_subplot(gs[1, 0])
        line_plot(ds["timestamp"].values,
                  ds["class"].values,
                  ax=targets)
    
        plt.show()
