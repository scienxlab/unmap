# unmap

[![Run tests](https://github.com/kwinkunks/unmap/actions/workflows/run-tests.yml/badge.svg)](https://github.com/kwinkunks/unmap/actions/workflows/run-tests.yml)
[![Build docs](https://github.com/kwinkunks/unmap/actions/workflows/build-docs.yml/badge.svg)](https://github.com/kwinkunks/unmap/actions/workflows/build-docs.yml)
[![PyPI version](https://img.shields.io/pypi/v/unmap.svg)](https://pypi.org/project/unmap//)
[![PyPI versions](https://img.shields.io/pypi/pyversions/unmap.svg)](https://pypi.org/project/unmap//)
[![PyPI license](https://img.shields.io/pypi/l/unmap.svg)](https://pypi.org/project/unmap/)


Unmap data from pseudocolor images, with knowledge of the colourmap for now, but the goal is to drop this requirement. 

in other words, this library is the opposite of `plt.imshow()`. 


## Installation

You can install this package with `pip`:

    pip install unmap

There are `dev`, `test` and `docs` options for installing dependencies for those purposes, eg `pip install unmap[dev]`.


## Documentation

Read [the documentation](https://kwinkunks.github.io/unmap), especially [the examples](https://kwinkunks.github.io/unmap/userguide/Unmap_data_from_an_image.html).


## Contributing

Take a look at [`CONTRIBUTING.md`](https://github.com/kwinkunks/unmap/blob/main/CONTRIBUTING.md).


## Testing

You can run the tests (requires `pytest` and `pytest-cov`) with

    pytest


## Building

This repo uses PEP 517-style packaging. [Read more about this](https://setuptools.pypa.io/en/latest/build_meta.html) and [about Python packaging in general](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

Building the project requires `build`, so first:

    pip install build

Then to build `unmap` locally:

    python -m build

The builds both `.tar.gz` and `.whl` files, either of which you can install with `pip`.
