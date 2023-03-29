import time

from anomalearn.analysis import analyse_constant_simplicity
from anomalearn.reader.time_series import SMDReader

reader = SMDReader("../data/anomaly_detection/smd")

for i, series_df in enumerate(reader):
    labels = series_df["class"].values
    series = series_df.drop(labels=["timestamp", "is_training", "interpretation", "class"], axis=1).values
    
    print("Start to analyse constant simplicity")
    start = time.time()
    results = analyse_constant_simplicity(series, labels)
    end = time.time()
    print("Ended to analyse constant simplicity")
    
    print(f"Analysing constant simplicity of {i}th series of SMD used {end - start} seconds")
    print(f"The constant score is: {results['constant_score']}")
    print(f"The upper bound is: {results['upper_bound']}")
    print(f"The lower bound is: {results['lower_bound']}", end="\n\n")
    
