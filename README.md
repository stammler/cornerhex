# cornerhex

[![Documentation Status](https://readthedocs.org/projects/cornerhex/badge/?version=latest)](https://cornerhex.readthedocs.io/en/latest/?badge=latest) [![GitHub](https://img.shields.io/github/license/stammler/cornerhex) ](https://github.com/stammler/cornerhex/blob/master/LICENSE) [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/stammler/cornerhex/blob/master/.github/CODE_OF_CONDUCT.md)  
[![PyPI - Downloads](https://img.shields.io/pypi/dm/cornerhex?label=PyPI%20downloads)](https://pypistats.org/packages/cornerhex)


`cornerhex` is a package to visualize multidimensional data in matrix corner plots, for example the results of Markov Chain Monte Carlo (MCMC) methods. Instead of 2d histograms or scatter plots it uses `matplotlib.pyplot.hexbin`. `cornerhex` can be easily costumized with different color schemes.

![Cornerhex](docs/source/_static/cornerhex.jpg)

## Installation

`cornerhex` can be installed via PyPI.

`pip install cornerhex`

## Documentation

For the usage of `cornerhex` please have a look at its [documentation](https://cornerhex.rtfd.io/).

* [1. Quickstart](https://cornerhex.readthedocs.io/en/latest/1_quickstart.html)
* [2. Customizations](https://cornerhex.readthedocs.io/en/latest/2_customizations.html)
* [A. Contributing / Bug reports / Features requests](https://cornerhex.readthedocs.io/en/latest/A_contrib_bug_feature.html)

## Acknowledgements

`cornerhex` is free to use and modify. Please acknowledge this repository when using `cornerhex` in a publication.

`cornerhex` has been inspired by `corner.py` [(Foreman-Mackey, 2016)](http://dx.doi.org/10.21105/joss.00024).