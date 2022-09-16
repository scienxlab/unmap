from io import Strinunmap

from PIL import Image
import numpy as np
import matplotlib.cm as cm
import scipy.signal

import unmap

def kernel(sizex, sizey):
    x, y = np.mgrid[-sizex:sizex+1, -sizey:sizey+1]
    g = np.exp(-0.333*(x**2/float(sizex)+y**2/float(sizey)))
    return g / g.sum()

def make_map(n, nx=64, ny=64, kernel_size=None, seed=None):
    imgs = []
    for i in range(n):
        rng = np.random.RandomState(seed=seed)
        if kernel_size is None:
            kx, ky = (7, 7)
        else:
            kx, ky = kernel_size
        f = kernel(kx, ky)
        z = rng.rand(nx+2*kx, ny+2*ky)

        z = scipy.signal.convolve(z, f, mode='valid')
        z = (z - z.min())/(z.max() - z.min())
        imgs.append(z)
    
    return np.stack(imgs)

def get_cbar(cmap, n=32, pixels=5):
    arr = cm.get_cmap()(np.arange(0, n, 1)/(n-1))[..., :3]
    return np.tile(arr, (pixels, 1)).reshape(pixels, n, 3)

def make_image_and_cbar(cmap, seed=None):
    data = np.squeeze(make_map(1))
    cmap = cm.get_cmap(cmap)
    cbar = get_cbar(cmap, 128, pixels=5)
    return cmap(data)[..., :3], cbar


def test_unmap():
    """Test the basics.
    """
    img, cbar = make_image_and_cbar('jet', seed=42)

    # Using a cbar image.
    data = unmap.unmap(img, cmap=cbar, threshold=0.05, vrange=(100, 200), background=(1, 1, 1), levels=256)
    assert data.shape == (231, 231)
    assert np.nanmax(data) == 200
    assert np.nanmean(data) - 150.60721435278933 < 1e-6
    assert np.any(np.isnan(data))

    # Using a matplotlib cmap.
    data = unmap.unmap(img, cmap='jet', threshold=0.05, vrange=(200, 300))
    assert np.nanmean(data) - 250.6121076492208 < 1e-6


def is_greyscale():
    """
    Test the is_greyscale function.
    """
    # 8-bit colour.
    assert not unmap.is_greyscale(rgb = (np.random.random((10, 10, 3)) * 256).astype(int))

    # RGBA colour.
    assert not unmap.is_greyscale(np.random.random((10, 10, 4)))

    # Simple 2D array.
    assert unmap.is_greyscale(np.random.random((10, 10)))

    # One-channel image.
    gs = np.random.random((10, 10, 1))
    assert unmap.is_greyscale(gs)

    # RBG greyscale.
    assert unmap.is_greyscale(np.dstack([gs, gs, gs]))

    # RGBA greyscale with random A channel.
    assert unmap.is_greyscale(np.dstack([gs, gs, gs, np.random.random((10, 10, 1))]))
