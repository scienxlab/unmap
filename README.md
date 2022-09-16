# snowfake

[![Run tests](https://github.com/agilescientific/snowfake/actions/workflows/run-tests.yml/badge.svg)](https://github.com/agilescientific/snowfake/actions/workflows/run-tests.yml)
[![Build docs](https://github.com/agilescientific/snowfake/actions/workflows/build-docs.yml/badge.svg)](https://github.com/agilescientific/snowfake/actions/workflows/build-docs.yml)
[![PyPI version](https://img.shields.io/pypi/v/snowfake.svg)](https://pypi.org/project/snowfake//)
[![PyPI versions](https://img.shields.io/pypi/pyversions/snowfake.svg)](https://pypi.org/project/snowfake//)
[![PyPI license](https://img.shields.io/pypi/l/snowfake.svg)](https://pypi.org/project/snowfake/)


Make Gravner-Griffeath "snowfakes"! This code implements:

> Janko Gravner, David Griffeath (2008). Modeling snow crystal growth II: A mesoscopic lattice map with plausible dynamics. _Physica D: Nonlinear Phenomena_ **237** (3), p 385-404. [DOI: 10.1016/j.physd.2007.09.008](https://doi.org/10.1016/j.physd.2007.09.008).

![Snowfakes](https://www.dropbox.com/s/8mquyaiumdiuwwf/snowfakes.png?raw=1)


## Installation

You can install this package with `pip` (be careful not to type "snowflake"):

    pip install snowfake

Installing `scikit-image` allows you to use a different affine transformation, but I haven't figured out yet if it's better or not. 

    pip install snowfake[skimage]


## Documentation

Read [the documentation](https://code.agilescientific.com/snowfake)


## Example

You can produce a random snowfake with:

```python
import snowfake
s = snowfake.random()
```

Alternatively, this code produces the crystal in Figure 5b of the Gravner & Griffeath (2008):

```python
from snowfake import Snowfake

params =  {
    'ρ': 0.35,  # or 'rho': 0.35 if you prefer...
    'β': 1.4,
    'α': 0.001,
    'θ': 0.015,
    'κ': 0.05,
    'μ': 0.015,
    'γ': 0.01,
    'σ': 0.00005,
    'random': False,
}
s = Snowfake(size=801, **params)
```

Now you're ready to grow and plot the snowfake:

```python
s.grow()
s.plot()
```

The various physical parameter arrays are available as `s.a` (attachment flag), `s.b` (boundary mass), `s.c` (the crystal itself) and `s.d` (the vapour). The arrays exist on hexgrids; you can rectify them with, for example, `s.rectify('c')`.

The parameter `σ` (note that you can also spell out `sigma` if you prefer) can be a 1D array with one sample per epoch. This will vary the vapour density `ρ` through _time_. The parameter `ρ` can be a 2D array of shape `(size, size)`; this will vary the initial vapour density through _space_.


## Testing

You can run the tests (requires `pytest` and `pytest-cov`) with

    python run_tests.py


## Building

This repo uses PEP 517-style packaging. [Read more about this](https://setuptools.pypa.io/en/latest/build_meta.html) and [about Python packaging in general](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

Building the project requires `build`, so first:

    pip install build

Then to build `snowfake` locally:

    python -m build

The builds both `.tar.gz` and `.whl` files, either of which you can install with `pip`.


## Continuous integration

This repo has two GitHub 'workflows' or 'actions':

- Push to `main`: Run all tests on all version of Python. This is the **Run tests** workflow.
- Publish a new release: Build and upload to PyPI. This is the **Publish to PyPI** workflow. Publish using the GitHub interface, for example ([read more](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
