import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from anomalearn.algorithms.models.time_series.anomaly.naive import TSAConstant
from anomalearn.reader.time_series import ODINTSReader
from anomalearn.visualizer import line_plot

FRIDGE = 1
TEST_PERC = 0.5


def train_evaluate_plot(values, targets, first_print, classifier):
    print(first_print)

    last_train_point = int(values.shape[0] * TEST_PERC)
    classifier.fit(values[0:last_train_point],
                          targets[0:last_train_point])
    predictions = classifier.classify(values)

    print(f"The constant learned is {classifier.get_constant()} with comparison {classifier.get_comparison()}")
    print(f"The f1 score of the method is: {f1_score(targets, predictions)}")
    print(f"The f1 score of the method on test set is: {f1_score(targets[last_train_point:], predictions[last_train_point:])}")

    fig = plt.figure(figsize=(8, 8))
    gs = plt.GridSpec(2, 1)
    ax = fig.add_subplot(gs[0, 0])
    line_plot(range(values.shape[0]),
              values,
              ax=ax)
    line_plot(range(predictions.shape[0]),
              np.array([classifier.get_constant() for _ in range(predictions.shape[0])]).flatten(),
              colors="green",
              ax=ax)

    ax = fig.add_subplot(gs[1, 0])
    line_plot(range(targets.shape[0]),
              targets,
              colors="red",
              ax=ax)
    line_plot(range(predictions.shape[0]),
              predictions,
              colors="green",
              ax=ax)
    plt.show()


if __name__ == "__main__":
    constant_function = TSAConstant(learning="supervised")
    reader = ODINTSReader(f"data/anomaly_detection/private_fridge/fridge{FRIDGE}/anomalies_fridge{FRIDGE}.csv",
                          "ctime",
                          "device_consumption")
    dataset = reader.read(f"data/anomaly_detection/private_fridge/fridge{FRIDGE}/all_fridge{FRIDGE}.csv").get_dataframe()

    train_evaluate_plot(dataset["value"].values, dataset["target"].values, "Learning to detect anomalies", constant_function)
    train_evaluate_plot(np.diff(dataset["value"].values), dataset["target"].values[1:], "Learning to detect anomalies on diff(series)", constant_function)
    train_evaluate_plot(np.abs(np.diff(dataset["value"].values)), dataset["target"].values[1:], "Learning to detect anomalies abs(diff(series))", constant_function)
