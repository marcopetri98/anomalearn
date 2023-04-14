from pathlib import Path

from matplotlib import gridspec
from matplotlib import pyplot as plt

from anomalearn.reader.time_series import NASAReader
from anomalearn.visualizer import line_plot


if __name__ == "__main__":
    reader = NASAReader(Path(__file__).parent / "../../../data/anomaly_detection/nasa_msl_smap/labeled_anomalies.csv")
    
    for ds in reader:
        fig = plt.figure(figsize=(8, 8), tight_layout=True)
        gs = gridspec.GridSpec(2, 1)
    
        series = fig.add_subplot(gs[0, 0])
        line_plot(ds["timestamp"].values,
                  ds["channel_telemetry"].values,
                  ax=series)
    
        targets = fig.add_subplot(gs[1, 0])
        line_plot(ds["timestamp"].values,
                  ds["class"].values,
                  ax=targets)
    
        plt.show()
