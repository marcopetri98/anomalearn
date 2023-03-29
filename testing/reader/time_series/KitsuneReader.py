from matplotlib import pyplot as plt, gridspec

from anomalearn.reader.time_series import KitsuneReader
from anomalearn.visualizer import line_plot

reader = KitsuneReader("../data/anomaly_detection/kitsune")

for ds in reader:
    fig = plt.figure(figsize=(8, 8), tight_layout=True)
    gs = gridspec.GridSpec(6, 1)

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
              ds["channel_3"].values,
              ax=series)

    series = fig.add_subplot(gs[4, 0])
    line_plot(ds["timestamp"].values,
              ds["channel_4"].values,
              ax=series)

    targets = fig.add_subplot(gs[5, 0])
    line_plot(ds["timestamp"].values,
              ds["class"].values,
              ax=targets)

    plt.show()
