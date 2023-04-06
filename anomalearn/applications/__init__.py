"""
A machine and deep learning collection of presented approaches.

This module contains the implementation of several methods presented and
accepted at conferences or journals. Moreover, the implementation also provides
the code to automatically download and set up the data folder for the algorithms
(when the dataset is publicly available and downloadable) and the code to run
the training exactly as it was presented in the paper to reproduce the
experiments. So, in the best case for each presented approach we will have three
components: the approach, the code to download and set up the data folder and
the code to run the training. However, there are also two other pieces of code
available in this package: a script downloading all the datasets when possible
and a script able to automatically execute all the experiments for the
implemented approaches with the option of choosing whether to execute only the
experiments for the approaches with publicly available and downloadable data or
all the approaches with publicly available data (given that the user has
downloaded and correctly set up the data folder for the dataset that are
publicly available, but not publicly downloadable). In case some approaches with
private dataset has been implemented, only the approach code will be present.
However, it will be possible to set up a dataset on which to test all the
approaches with private dataset via a simple enumeration. The default option is
not to run those experiments since they lack of reproducibility, and I want to
stress the importance of conducting experiments on public available data to
maximize the contribution to the scientific community. An experiment that cannot
be reproduced is useless by itself. The reproducibility is at the basis of the
development of scientific knowledge since it allows other scientists to improve
others work or collaborate if an error or imprecision is found.
"""

from .IExperimentLoader import IExperimentLoader
from .ExperimentLoader import ExperimentLoader

__all__ = ["IExperimentLoader", "ExperimentLoader"]
