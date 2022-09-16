# unmap

[![Run tests](https://github.com/kwinkunks/unmap/actions/workflows/run-tests.yml/badge.svg)](https://github.com/kwinkunks/unmap/actions/workflows/run-tests.yml)
[![Build docs](https://github.com/kwinkunks/unmap/actions/workflows/build-docs.yml/badge.svg)](https://github.com/kwinkunks/unmap/actions/workflows/build-docs.yml)
[![PyPI version](https://img.shields.io/pypi/v/unmap.svg)](https://pypi.org/project/unmap//)
[![PyPI versions](https://img.shields.io/pypi/pyversions/unmap.svg)](https://pypi.org/project/unmap//)
[![PyPI license](https://img.shields.io/pypi/l/unmap.svg)](https://pypi.org/project/unmap/)


Unmap data from pseudocolor images, with knowledge of the colourmap (for now!).


## Installation

You can install this package with `pip`:

    pip install unmap


## Documentation

Read [the documentation](https://code.agilescientific.com/unmap)


## Example

We'll grab an image from the web and unmap it:

```python
def get_image_from_web(uri):
    data = requests.get(uri).content
    img = Image.open(BytesIO(data)).convert('RGB')
    rgb_im = np.asarray(img)[..., :3] / 255.
    return rgb_im

# An image from Matteo Niccoli's blog:
# https://mycartablog.com/2014/08/24/what-your-brain-does-with-colours-when-you-are-not-looking-part-1/
img = get_image_from_web('https://i0.wp.com/mycartablog.com/wp-content/uploads/2014/03/spectrogram_jet.png')

import gio

data = gio.unmap(img, cmap='jet')
plt.imshow(data)
```


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
