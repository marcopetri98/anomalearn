import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from anomalearn.algorithms.models.time_series.anomaly.naive import TSAConstAvgStd
from anomalearn.reader.time_series import ODINTSReader
from anomalearn.visualizer import line_plot


def train_evaluate_plot(values, targets, first_print, classifier, test_perc, verbose=True):
    print(first_print)

    last_train_point = int(values.shape[0] * test_perc)
    classifier.fit(values[0:last_train_point],
                          targets[0:last_train_point],
                   verbose=verbose)
    predictions = classifier.classify(values)
    half = int((classifier.get_w() - 1) / 2)

    print(f"The model learnt is {classifier.get_a()} * movavg(x, {classifier.get_w()}) + {classifier.get_b()} * movstd(x, {classifier.get_w()}) + {classifier.get_c()} < x")
    print(f"The f1 score of the method is: {f1_score(targets[half:-half], predictions)}")
    print(f"The f1 score of the method on test set is: {f1_score(targets[last_train_point:][half:-half], predictions[last_train_point:])}")

    fig = plt.figure(figsize=(8, 8))
    gs = plt.GridSpec(2, 1)
    ax = fig.add_subplot(gs[0, 0])
    line_plot(range(values.shape[0]),
              values,
              ax=ax)
    line_plot(range(predictions.shape[0] - half * 2),
              classifier.get_upper_series().flatten(),
              colors="green",
              ax=ax)
    line_plot(range(predictions.shape[0] - half * 2),
              classifier.get_lower_series().flatten(),
              colors="green",
              ax=ax)

    ax = fig.add_subplot(gs[1, 0])
    line_plot(range(targets.shape[0]),
              targets,
              colors="red",
              ax=ax)
    line_plot(range(predictions.shape[0] - half * 2),
              predictions[half:-half],
              colors="green",
              ax=ax)
    plt.show()


if __name__ == "__main__":
    FRIDGE = 1
    TEST_PERC = 0.5
    
    constant_function = TSAConstAvgStd(max_window=200)
    reader = ODINTSReader(f"data/anomaly_detection/private_fridge/fridge{FRIDGE}/anomalies_fridge{FRIDGE}.csv",
                          "ctime",
                          "device_consumption")
    dataset = reader.read(f"data/anomaly_detection/private_fridge/fridge{FRIDGE}/all_fridge{FRIDGE}.csv").get_dataframe()

    train_evaluate_plot(dataset["value"].values, dataset["target"].values, "Learning to detect anomalies", constant_function, TEST_PERC)
    train_evaluate_plot(np.diff(dataset["value"].values), dataset["target"].values[1:], "Learning to detect anomalies on diff(series)", constant_function, TEST_PERC)
    train_evaluate_plot(np.abs(np.diff(dataset["value"].values)), dataset["target"].values[1:], "Learning to detect anomalies abs(diff(series))", constant_function, TEST_PERC)
