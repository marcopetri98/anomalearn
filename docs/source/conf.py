# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
from pathlib import Path

# add the path of the library to sys.path in order to use autodoc
library_path = Path(__file__).parents[2]
if str(library_path) not in sys.path:
    sys.path.append(str(library_path))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'anomalearn'
copyright = '2023, Marco Petri'
author = 'Marco Petri'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.napoleon', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary',
              'sphinx.ext.intersphinx', 'sphinx_design']

autodoc_member_order = "groupwise"
autodoc_inherit_docstrings = True
autosummary_generate = False

intersphinx_mapping = {
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "python": ("https://docs.python.org/3/", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", ("../inventories/numba.inv", None)),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "skopt": ("https://scikit-optimize.github.io/stable/", None)
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

html_css_files = ["css/grids.css"]

html_theme_options = {
    "navigation_depth": 7
}
