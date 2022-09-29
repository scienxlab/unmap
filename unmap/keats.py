"""
keats.py

Functions for 'unweaving the rainbow': recovering colour maps from
images with no a priori knowledge.

Author: Matt Hall
Copyright: 2022, Matt Hall
Licence: Apache 2.0
"""
import numpy as np
import networkx as nx
from scipy.cluster.vq import kmeans
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from matplotlib.colors import LinearSegmentedColormap


def remove_greyish(pixels, threshold=0.05, levels=2):
    """
    Remove greyish colours from a list of colours.

    Args:
        pixels: A list or array of RGB colours.
        threshold: The maximum distance from the grey axis.
        levels: The number of grey levels to ignore.

    Returns:
        A list or array of RGB colours.
    """
    # Make a bunch of grey shades an appropriate distance apart.
    # Levels = 1 is black only.
    # Levels = 2 is black and white.
    # More levels for more levels of grey.
    if levels is None:
        levels = 1 + int(np.sqrt(3) / threshold)
    grey = np.repeat(np.linspace(0, 1, levels), 3).reshape(-1, 3)
    
    # Get the distance from every pixel to every grey shade.
    dist = np.linalg.norm(pixels.reshape(-1, 1, 3) - grey, axis=-1)
    
    # If we're half the threshold away, it's greyish.
    greyish = np.sum(dist < threshold, axis=-1).astype(bool)

    return pixels[~greyish]


def unique_colours(imarray, n=128, method='scipy', grey_levels_to_ignore=2, grey_distance_threshold=0.05):
    """
    Get unique colour dictionaries for an image (RGB) array.

    Args:
        imarray: An image array.
        n: The number of colours to try to return. Not guaranteed. If n is None,
            the number of colours returned will be the number of unique colours
            in the image. This may be very large.
        method: The method to use to find unique colours, can be 'scipy' or
            'sklearn'. Both methods use k-means clustering.
        grey_levels_to_ignore: The number of grey levels to ignore. If None,
            all grey levels will be ignored. If 0, no grey levels will be
            ignored. If 1, only black will be ignored. If 2, black and white
            will be ignored. If 3, black, white, and grey will be ignored, etc.
        grey_distance_threshold: The maximum distance from the grey axis.
    
    Returns:
        An array of unique colours.
    """
    pixels = imarray.reshape(-1, 3)
    if n is not None:
        sample = shuffle(pixels, random_state=0, n_samples=10_000)
        if method == 'scipy':  # Assumes equal-size clusters.
            uniq, _ = kmeans(sample, n)
        elif method == 'sklearn':
            clu = KMeans(n_clusters=n).fit(sample)
            uniq = clu.cluster_centers_
    else:
        uniq = np.unique(pixels, axis=0)
    uniq = np.clip(uniq, a_min=0, a_max=1)  # Reduce precision errors.
    return remove_greyish(uniq, threshold=grey_distance_threshold, levels=grey_levels_to_ignore)


def get_transition_matrix(imarray,
                          unique_colours,
                          normalize=True,
                          remove_self=True,
                          stride=2,
                         ):
    """
    Get a transition matrix, aka adjaceny matrix, from an image array.

    Args:
        imarray: An image array.
        unique_colours: A list of unique colours.
        normalize: Whether to normalize the matrix.
        remove_self: Whether to remove self-loops.
        stride: The stride to use when calculating the transition matrix. Using
            a stride of 2 oversamples the transitions, but seems to work best.
            A stride of 3 should be sufficient.
    
    Returns:
        A symmetric transition matrix.
    """
    kdtree = cKDTree(unique_colours)

    # Make the empty array.
    M = np.zeros((len(unique_colours), len(unique_colours)))

    # Visit every other pixel and increment the
    # relevant element in the transition matrix.
    offsets = (np.indices((3, 3)) - 1).reshape(-1, 2)
    rows, cols, _ = imarray.shape
    for row in range(1, rows - 1, stride):
        for col in range(1, cols - 1, stride):
            _, cval = kdtree.query(imarray[row, col])
            for (r, c) in offsets:
                _, val = kdtree.query(imarray[row+r, col+c])
                M[cval][val] += 1
    
    # Make symmetric.
    M += M.T

    # Remove self-transitions.
    if remove_self:
        np.fill_diagonal(M, 0)
    
    # Normalize by row sums.
    if normalize:
        M = (M.T / (1e-6+np.sum(M.T, axis=0))).T

    return M


def longest_shortest_path(M,
                          unique_colours,
                          edge_weight_threshold=0.05,
                          edge_distance_threshold=0.25,
                          degree_threshold=10
                          ):
    """
    Get the longest shortest path through a transition matrix.

    Args:
        M: A transition matrix.
        unique_colours: A list of unique colours.
        edge_weight_threshold: The minimum edge weight to consider. The weights
            are the values in the transition matrix.
        edge_distance_threshold: The maximum Euclidean distance between two
            colours to consider them connected by an edge.
        degree_threshold: The maximum degree of a node to consider it. Nodes
            of very high degree are connected to many other nodes, and are
            therefore difficult to place in the colourmap.

    Returns:
        A list of unique colours, in the order they appear in the colourmap.
    """
    G = nx.from_numpy_array(M)
    
    # Need dists not weights.
    edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())

    # Squared distance to penalize connections between very different colours.
    dist_dict = {(a, b): np.linalg.norm(unique_colours[a]-unique_colours[b])**2 for a, b in edges}
    nx.set_edge_attributes(G, dist_dict, 'dist2')
    
    # Remove edges by weight.
    edge_weights = nx.get_edge_attributes(G, 'weight')
    G.remove_edges_from(e for e, w in edge_weights.items() if w < edge_weight_threshold)

    # Remove edges by distance.
    edge_weights = nx.get_edge_attributes(G, 'dist2')
    G.remove_edges_from(e for e, w in edge_weights.items() if np.sqrt(w) > edge_distance_threshold)

    # Remove nodes with very high degree.
    remove = [n for n, d in dict(G.degree()).items() if d > degree_threshold]
    G.remove_nodes_from(remove)

    # Find the longest shortest path.
    paths = nx.shortest_path_length(G, weight='dist2')
    longest, s, t = 0, None, None
    for source, path_dict in paths:
        for target, path_length in path_dict.items():
            if path_length > longest:
                s, t = source, target
                longest = path_length

    path = nx.shortest_path(G,
                            weight='dist2',
                            source=s,
                            target=t)
    
    return [unique_colours[n] for n in path]


def path_to_cmap(path, n=None, name='recovered'):
    """
    Convert a list of colours to a matplotlib colormap.
    
    Args:
        path: A list of RGB colours.
        n: The number of colours to interpolate to. Defaults to twice the
            length of the path.
        name: The name of the colourmap.

    Returns:
        A matplotlib LinearSegmentedColormap.
    """
    if n is None:
        n = 2 * len(path)  # Not sure what the default should be.
    return LinearSegmentedColormap.from_list(name, path, N=n)


def guess_cmap(imarray,
               n=128,
               grey_levels_to_ignore=2,
               grey_distance_threshold=0.05,
               method='sklearn',
               normalize=True,
               remove_self=True,
               stride=2,
               reverse='auto',
               name='recovered',
               ):
    """
    Recover a colormap from an image.

    Args:
        imarray: An image array.
        n: The number of colours to try to place in the colourmap.
        grey_levels_to_ignore: The number of grey levels to ignore when
            calculating the unique colours.
        grey_distance_threshold: The maximum Euclidean distance from pure grey
            to consider a colour grey.
        method: The library to use to calculate the unique colours. Can be
            'sklearn' or 'numpy'. Unique colours are quantized using k-means.
        normalize: Whether to normalize the transition matrix.
        remove_self: Whether to remove self-loops.
        stride: The stride to use when calculating the transition matrix. Using
            a stride of 2 oversamples the transitions, but seems to work best.
        reverse: Whether to reverse the colourmap. Can be 'auto', True, or False.
            If 'auto', the colourmap will start with the end closest to dark blue.
        name: The name of the colourmap.

    Returns:
        A matplotlib LinearSegmentedColormap.
    """
    uniq = unique_colours(imarray,
                          n=n,
                          method=method,
                          grey_levels_to_ignore=grey_levels_to_ignore,
                          grey_distance_threshold=grey_distance_threshold
                          )
    M = get_transition_matrix(imarray,
                              uniq,
                              normalize=normalize,
                              remove_self=remove_self,
                              stride=stride
                              )
    path = longest_shortest_path(M, uniq)
    if reverse == 'auto':
        dark_blue = np.array([0, 0, 0.5])
        if np.linalg.norm(path[0] - dark_blue) > np.linalg.norm(path[-1] - dark_blue):
            reverse = True
        else:
            reverse = False
    if reverse:
        path = path[::-1]
    return path_to_cmap(path, n=None, name=name)
