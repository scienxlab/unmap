import numpy as np
import matplotlib.cm as cm
import scipy.signal

import unmap

def kernel(sizex, sizey):
    x, y = np.mgrid[-sizex:sizex+1, -sizey:sizey+1]
    g = np.exp(-0.333*(x**2/float(sizex)+y**2/float(sizey)))
    return g / g.sum()

def make_map(nx=64, ny=64, kernel_size=None, seed=None):
    rng = np.random.RandomState(seed=seed)
    if kernel_size is None:
        kx, ky = (7, 7)
    else:
        kx, ky = kernel_size
    f = kernel(kx, ky)
    z = rng.rand(nx+2*kx, ny+2*ky)
    z = scipy.signal.convolve(z, f, mode='valid')
    z = (z - z.min())/(z.max() - z.min())
    
    return z

def get_cbar(cmap, n=32, pixels=5):
    arr = cm.get_cmap(cmap)(np.arange(0, n, 1)/(n-1))[..., :3]
    return np.tile(arr, (pixels, 1)).reshape(pixels, n, 3)

def make_image_and_cbar(cmap, seed=None):
    data = make_map()
    c = cm.get_cmap(cmap)
    return c(data)[..., :3], get_cbar(cmap, 256, pixels=5)


def test_unmap():
    """Test the basics.
    """
    img, cbar = make_image_and_cbar('jet', seed=42)

    # Add some white pixels as 'background'; will become NaNs.
    img[:3, :3, :] = 1

    # Using a cbar image.
    data = unmap.unmap(img, cmap=cbar, vrange=(100, 200))
    assert data.shape == (64, 64)
    assert np.nanmax(data) == 200
    assert np.nanmean(data) - 161.4341107536765 < 1e-6
    assert np.any(np.isnan(data))

    # Using a matplotlib cmap.
    data = unmap.unmap(img, cmap='jet', vrange=(200, 300))
    assert np.nanmean(data) - 261.4341107536765 < 1e-6


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
