from matplotlib import pyplot as plt, gridspec

from anomalearn.reader.time_series import ExathlonReader
from anomalearn.visualizer import line_plot

reader = ExathlonReader("../data/anomaly_detection/exathlon", "train")

for ds in reader:
    fig = plt.figure(figsize=(8, 8), tight_layout=True)
    gs = gridspec.GridSpec(4, 1)

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

    series = fig.add_subplot(gs[3, 0])
    line_plot(ds["timestamp"].values,
              ds["class"].values,
              ax=series)

    plt.show()
