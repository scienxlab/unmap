# unmap

[![Run tests](https://github.com/kwinkunks/unmap/actions/workflows/run-tests.yml/badge.svg)](https://github.com/kwinkunks/unmap/actions/workflows/run-tests.yml)
[![Build docs](https://github.com/kwinkunks/unmap/actions/workflows/build-docs.yml/badge.svg)](https://github.com/kwinkunks/unmap/actions/workflows/build-docs.yml)
[![PyPI version](https://img.shields.io/pypi/v/unmap.svg)](https://pypi.org/project/unmap//)
[![PyPI versions](https://img.shields.io/pypi/pyversions/unmap.svg)](https://pypi.org/project/unmap//)
[![PyPI license](https://img.shields.io/pypi/l/unmap.svg)](https://pypi.org/project/unmap/)

Unmap data from pseudocolor images, with or without knowledge of the colourmap. This tool has 2 main components:

1. Guess the colourmap that was used for a pseudocolour visualization, in cases where it's unknown and a colourbar is not included in the image.
2. 'Unmap' a pseudocolour visualization, separating the data from the image; essentially this is the opposite of what `plt.imshow()` does. 


## Similar projects

There are some other approaches to both Task 1 (above) and Task 2:

- [`unmap`](https://github.com/jperryhouts/unmap) (I swear I didn't know about this tool when I named mine!) &mdash; does the data ripping part. The colourmap must be provided, but the tool also provides a way to interactively identify a colourbar in the image.
- [Poco et al.](https://ieeexplore.ieee.org/document/8017646) ([GitHub](https://github.com/uwdata/rev)) attempts to both find the colourbar in a visualization, then use it to perform Task 2. The visualization must contain a colourbar.
- [Yuan et al.](https://github.com/yuanlinping/deep_colormap_extraction) attempts Task 1 using deep learning. The prediction from a CNN is refined with either Laplacian eigenmapping (manifold+based dimensionality reduction, for continuous colourmaps) or DBSCAN (for categorical colourmaps).

Of these projects, only Yuan et al. ('deep colormap extraction') requires no _a priori_ knowledge of the colourmap. 


## Stack Exchange questions about this topic

- https://datascience.stackexchange.com/questions/27247/can-i-get-numeric-data-from-a-color-map
- https://stackoverflow.com/questions/71090534/how-to-extract-a-colormap-from-a-colorbar-image-and-use-it-in-a-heatmap
- https://stackoverflow.com/questions/63233529/given-a-jpg-of-2d-colorplot-colorbar-how-can-i-sample-the-image-to-extract-n
- https://stackoverflow.com/questions/3720840/how-to-reverse-a-color-map-image-to-scalar-values
- https://stackoverflow.com/questions/62267694/extract-color-table-values
- https://stackoverflow.com/questions/14445102/invert-not-reverse-a-colormap-in-matplotlib


## Installation

You can install this package with `pip`:

    pip install unmap

There are `dev`, `test` and `docs` options for installing dependencies for those purposes, eg `pip install unmap[dev]`.


## Documentation

Read [the documentation](https://kwinkunks.github.io/unmap), especially [the examples](https://kwinkunks.github.io/unmap/userguide/Unmap_data_from_an_image.html).


## Contributing

Take a look at [`CONTRIBUTING.md`](https://github.com/kwinkunks/unmap/blob/main/CONTRIBUTING.md).


## Testing

After cloning this repository and installing the dependencies required for testing, you can run the tests (requires `pytest` and `pytest-cov`) with

    pytest


## Building

This repo uses PEP 517-style packaging, with the entire build system and requirements defined in the `pyproject.toml` file. [Read more about this](https://setuptools.pypa.io/en/latest/build_meta.html) and [about Python packaging in general](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

Building the project requires `build`, so first:

    pip install build

Then to build `unmap` locally:

    python -m build

The builds both `.tar.gz` and `.whl` files, either of which you can install with `pip`.
