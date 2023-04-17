import configparser
from pathlib import Path

rts_config = configparser.ConfigParser()
"""This variable contains all the configurations present in the configuration
file present in this package.
"""
rts_config.read(Path(__file__).parent / "time_series_config.ini")

# TODO: add a singleton object to allow the user to change configs by commands instead of browsing files and searching for the .ini
