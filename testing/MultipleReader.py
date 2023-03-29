import os

from anomalearn.reader.time_series import TSMultipleReader

multiple_reader = TSMultipleReader()
all_paths = list()

dir_path = "data/badec/train/clean_data/1m"
dir_contents_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
dir_files = [f for f in dir_contents_paths if os.path.isfile(f)]

dataframes = multiple_reader.read_multiple(dir_files).get_all_dataframes()
for df in dataframes:
    print(df.shape)
