"""
A machine and deep learning library easy to use.

This library has been developed to implement abstract concepts used in machine
and deep learning to perform anomaly detection on time series. Its primary aim
is to consolidate the basis over which this task is performed and to enhance
the reproducibility of experiments. Many algorithms have several steps in
common. Therefore, we can build a framework to easy the process of developing
new algorithms by abstracting those concepts with abstract classes, partially
implemented.
"""

from .abc import EqualityABC, ObtainableABC
