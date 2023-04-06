# anomalearn: time series anomaly detection library

## When will the code be fully released?

ASAP since it is part of a thesis work whose dissertation is planned
for 4th May 2023.

## What is it?

anomalearn is a [**Python**][python-url] package that provides modular and
extensible functionalities for developing anomaly detection methods for time
series data, reading publicly available time series anomaly detection datasets,
creating the loading of data for experiments, and dataset evaluation functions.
Additionally, anomalearn development's plans include the implementation of
several state-of-the-art and historical anomaly detection methods, and the
implementation of objects to automate the training process of methods. See
Discussion and development section for more details.

[python-url]: https://www.python.org/

## Documentation

Every functionality in anomalearn is documented. The official documentation is
hosted at [**https://marcopetri98.github.io/anomalearn/docs/stable**][documentation-url].

[documentation-url]: https://marcopetri98.github.io/anomalearn/docs/stable

## Main features

Here you find a list of the features offered by anomalearn:

- Implementation of state-of-the-art and historical anomaly detection methods
  for time series. The bare models are located in **anomalearn.algorithms.models**.
  Where bare models mean the model without the preprocessing or postprocessing
  operations.
- Implementation of data readers of commonly used publicly accessible time
  series anomaly detection datasets. Data readers are all located in the package
  **anomalearn.reader** or in **anomalearn.reader.time_series**. All data
  readers return a [**pandas**][pandas-url] [**DataFrame**][dataframe-url].
- Implementation of some data analysis functions, such as simplicity scoring
  functions, stationarity tests and time series decomposition functions. These
  functions are all located in **anomalearn.analysis**.
- Implementation of helpers for creating experiments. Currently, only the
  helper for data loading has been implemented capable of taking data readers
  and returning all or a subset of series with a default or specific split. The
  experiment helpers are all located in **anomalearn.applications**.

[dataframe-url]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html

## Installation

The source code is available at [**anomalearn github repo**][anomalearn-repo].

Currently, the library is shipped only to the
[**Python Package Index (PyPI)**][pypi-url].

```sh
# install from PyPI
pip install anomalearn --pre
```

[pypi-url]: https://pypi.org
[anomalearn-repo]: https://github.com/marcopetri98/anomalearn

## Installation from source

Firstly, download or clone the repository and place it in any location on your
computer. We will call REPO_PATH. Open the terminal and navigate to the folder:

```sh
cd REPO_PATH
```

Secondly, install the repository using pip:

```sh
pip install .
```

## Dependencies

This repository is strongly based on other existing high-quality Python packages
for machine learning and for general programming:

- [**Numpy**][numpy-url]: adds support for efficient array operations.
- [**Scipy**][scipy-url]: adds support for scientific computing.
- [**Numba**][numba-url]: adds a Just In Time compiler for functions that have
  to be efficient and leaves the package a pure Python package.
- [**Pandas**][pandas-url]: adds support for working with data structures.
- [**Scikit-learn**][sklearn-url]: adds support for model development.
- [**Scikit-optimize**][skopt-url]: adds support for searching hyper-parameters
  of models.
- [**Statsmodels**][statsmodels-url]: adds support for statistical tests and
  models.
- [**Matplotlib**][matplotlib-url]: adds supports for plotting.

[numpy-url]: https://numpy.org/
[scipy-url]: https://scipy.org/
[numba-url]: https://numba.pydata.org/
[pandas-url]: https://pandas.pydata.org/
[sklearn-url]: https://scikit-learn.org/stable/
[skopt-url]: https://scikit-optimize.github.io/stable/
[statsmodels-url]: https://www.statsmodels.org/stable/index.html
[matplotlib-url]: https://matplotlib.org/

## Getting help

For the moment, the suggested way to get help is by posting questions to
[**StackOverflow**][stackoverflow-url]. Then, until the community will grow
bigger, consider sending the URL of the questions to the author via email.

[stackoverflow-url]: https://stackoverflow.com

## Background

This work started with Marco Petri's thesis work. The work initially aimed to
develop new anomaly detection methods for time series to reach new
state-of-the-art performances. However, given the scarcity of tools specifically
aimed for time series anomaly detection, the thesis developed anomalearn and a
way to evaluate the simplicity of a dataset.

## Discussion and development

Currently, the development of the first stable version of anomalearn is ongoing.
If you want to use it, you can help us in testing the functionalities by
providing feedback on the clarity of the documentation, the naming of functions,
ease of use, and in proposing new functionalities to implement.

In the future, once the first stable version will be published, a structured and
well documented on how to contribute to the library will be written.

## Contributing to code

Firstly, download or clone the repository and place it in any location on your
computer. We will call REPO_PATH. Open the terminal and navigate to the folder:

```sh
cd REPO_PATH
```

The library uses [**poetry**][poetry-url] for managing dependencies, building,
and publishing. Therefore, it is strongly recommended to carefully read its docs
to be able to contribute and install it from source. **Be careful, the installed
version of poetry must be at least 1.4.1**.

```sh
poetry init
```

Now, poetry will recognize the project. You can install the library and its
dependencies by using the poetry lock file such that every contributor will use
the exact same versions of packages:

```sh
# this command will install the library using the lock file
poetry install
```

Now, you can add functionalities to the library. To ask for changes to be
merged, create a pull request. However, **it is strongly suggested to ask if a
feature can be implemented in anomalearn such that it does not violate any
design choice**.

[poetry-url]: https://python-poetry.org/

## Citation

Currently, neither the thesis has been published nor a paper presenting the
library to a conference or in a journal has been published. I strongly ask you
to wait till the 4th of May 2023 to get the citation (the date on which the
dissertation will happen and the thesis will be published).
