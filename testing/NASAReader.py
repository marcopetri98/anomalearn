from matplotlib import pyplot as plt, gridspec

from anomalearn.reader.time_series import NASAReader
from anomalearn.visualizer import line_plot

reader = NASAReader("../data/anomaly_detection/nasa_msl_smap/labeled_anomalies.csv")

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
