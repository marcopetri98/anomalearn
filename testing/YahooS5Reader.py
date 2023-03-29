import matplotlib.pyplot as plt
from matplotlib import gridspec

from anomalearn.reader.time_series import YahooS5Reader
from anomalearn.visualizer import line_plot

BENCHMARK = "A4"
SERIES = 1

for SERIES in range(100):
    reader = YahooS5Reader("../data/anomaly_detection/yahoo_s5/")
    ds = reader.read(SERIES, benchmark=BENCHMARK).get_dataframe()

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
