from pathlib import Path

from matplotlib import gridspec
from matplotlib import pyplot as plt

from anomalearn.reader.time_series import GHLReader
from anomalearn.visualizer import line_plot


if __name__ == "__main__":
    reader = GHLReader(Path(__file__).parent / "../../../data/anomaly_detection/ghl")
    
    for ds in reader:
        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        if "class_0" in ds.columns:
            is_testing = True
            gs = gridspec.GridSpec(6, 1)
        else:
            is_testing = False
            gs = gridspec.GridSpec(3, 1)
    
        series = fig.add_subplot(gs[0, 0])
        line_plot(ds["timestamp"].values,
                  ds["channel_0"].values,
                  ax=series)
    
        series = fig.add_subplot(gs[1, 0])
        line_plot(ds["timestamp"].values,
                  ds["channel_1"].values,
                  ax=series)
    
        series = fig.add_subplot(gs[2, 0])
        line_plot(ds["timestamp"].values,
                  ds["channel_2"].values,
                  ax=series)
    
        if is_testing:
            series = fig.add_subplot(gs[3, 0])
            line_plot(ds["timestamp"].values,
                    ds["class_0"].values,
                    ax=series)
    
            series = fig.add_subplot(gs[4, 0])
            line_plot(ds["timestamp"].values,
                    ds["class_1"].values,
                    ax=series)
    
            targets = fig.add_subplot(gs[5, 0])
            line_plot(ds["timestamp"].values,
                    ds["class_2"].values,
                    ax=targets)
    
        plt.show()
