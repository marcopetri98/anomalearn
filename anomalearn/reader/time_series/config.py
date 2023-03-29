import configparser
import os

# get this file directory
dir_path = os.path.dirname(os.path.realpath(__file__))

# load the package configuration file
rts_config = configparser.ConfigParser()
rts_config.read(os.path.join(dir_path, "time_series_config.ini"))