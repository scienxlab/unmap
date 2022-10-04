"""
unweave.py

Functions for 'unweaving the rainbow': recovering colour maps from
images with no a priori knowledge. Refactored from an earlier version.

Author: Matt Hall
Copyright: 2022, Matt Hall
Licence: Apache 2.0
"""
import fsspec
import numpy as np
import networkx as nx
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix


def ordered_unique(seq):
    return list(dict.fromkeys(seq))


def convert_imarray(imarray, colors=256):
    """
    Convert an RGB image array to an index array and colourtable. The array
    will be quantized to the specified number of colours, and will be no larger
    than 512x512 pixels.

    Args:
        imarray (np.ndarray): The RGB or RGBA image array.
        colors (int): Number of colours to reduce to.

    Returns:
        imarray (np.ndarray): Array of indices into the colourtable.
        unique_colors (np.ndarray): Colourtable.
    """
    if np.min(imarray) < 0 or np.max(imarray) > 255:
        raise ValueError("Image array must be in the range [0, 255] or [0, 1].")
    elif np.max(imarray) <= 1.0:
        imarray = imarray * 255
    imp = Image.fromarray(np.uint8(imarray))
    imp = imp.quantize(colors=colors, dither=Image.NONE)
    imp.thumbnail((512, 512))
    imarray = np.asarray(imp)
    palette = np.asarray(imp.getpalette()).reshape(-1, 3)
    unique = ordered_unique(tuple(i) for i in palette[:colors]/255)
    return imarray, np.array(unique)


def construct_graph(imarray, colors=256, normed=True):
    """
    Construct an undirected value adjacency graph from an image array.

    Weights are the number of times a pair of values co-occur in the image,
    normalized per value (i.e. per node in the graph).

    Args:
        imarray (np.ndarray): Array of values.
        colors (int): Number of colours in the image.
        normed (bool): Whether to normalize the weights.

    Returns:
        G (nx.Graph): Value adjacency graph.
    """
    glcm = graycomatrix(imarray,
                        distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=colors,
                        symmetric=True
                        )

    # Add transitions over all directions.
    glcm = np.sum(np.squeeze(glcm), axis=-1)

    # Normalize.
    if normed:
        glcm = glcm / (1e-9 + np.sum(glcm, axis=-1))

    # Construct and remove self-loops.
    G = nx.from_numpy_array(glcm)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    return G


def plot_graph(G, unique_colors, layout='kamada_kawai', ax=None, figsize=(12, 8)):
    """
    Plot a graph with colours.
    
    Args:
        G (nx.Graph): Graph to plot.
        unique_colors (np.ndarray): Colourtable.
        layout (str): Layout to use.
        ax (matplotlib.axes.Axes): Axes to plot on.
        figsize (tuple): Figure size.
        
    Returns:
        ax (matplotlib.axes.Axes): Axes.
    """
    if layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G, weight='dist')
    else:
        raise ValueError("`layout` must be one of 'spring', 'spectral', or 'kamada_kawai' (default).")
    color = [unique_colors[n] for n in G]
    _, wt = zip(*nx.get_edge_attributes(G, 'weight').items())

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    nx.draw(G, pos=pos, ax=ax, node_size=30,
            node_color=color,
            edge_color=wt,
            edge_cmap=plt.cm.Greys,
            edge_vmin=-0.05,
            edge_vmax=0.25, alpha=0.75
            )
    return ax


def prune_graph(G, unique_colors, min_weight=0.025, max_dist=0.25, max_neighbours=20):
    """
    Prune a graph to remove edges with low weight and high distance.
    
    Args:
        G (nx.Graph): Graph to prune.
        unique_colors (np.ndarray): Colourtable.
        min_weight (float): Minimum weight to keep.
        max_dist (float): Maximum distance to keep.
        max_neighbours (int): Maximum number of neighbours to allow. Nodes with
            more neighbours than this will be removed.

    Returns:
        G (nx.Graph): Pruned graph.
    """
    G = G.copy()
    
    dist = lambda u, v: np.linalg.norm(unique_colors[u] - unique_colors[v])
    
    # Calculate RGB distances.
    dist_dict = {(u, v): dist(u, v) for u, v, _ in G.edges.data()}
    nx.set_edge_attributes(G, dist_dict, 'dist')

    # Calculate normalized weights.
    dist_dict = {(u, v): dist(u, v) for u, v, _ in G.edges.data()}
    nx.set_edge_attributes(G, dist_dict, 'dist')

    # Prune edges.
    remove = [(u, v) for u, v, d in G.edges.data() if d['weight'] < min_weight]
    remove += [(u, v) for u, v, d in G.edges.data() if d['dist'] > max_dist]
    G.remove_edges_from(remove)

    # Prune vertices.
    remove = [n for n, d in dict(G.degree()).items() if d > max_neighbours]
    G.remove_nodes_from(remove)

    # Return the giant component.
    Gcc = sorted(nx.connected_components(G), key=len)
    return G.subgraph(Gcc[-1])


def longest_shortest_path(G):
    """
    Find the longest shortest path in a graph. This should be the path between
    the ends of the longest chain in the graph.

    Args:
        G (nx.Graph): Graph to search.
    
    Returns:
        path (list): Longest shortest path.
    """
    
    dist = lambda u, v, d: d['dist']**2

    # Find the longest shortest path.
    paths = nx.shortest_path_length(G, weight=dist)
    longest, s, t = 0, None, None
    for source, path_dict in paths:
        for target, path_length in path_dict.items():
            if path_length > longest:
                s, t = source, target
                longest = path_length

    return nx.shortest_path(G,
                            weight=dist,
                            source=s,
                            target=t)


def path_to_cmap(path, unique_colors, colors=256, reverse='auto', equilibrate=False):
    """
    Convert a path through the graph to a colormap.

    Args:
        path (list): Path to convert.
        unique_colors (np.ndarray): Colourtable.
        colors (int): Number of colours to return. Default is 256. Use None to
            use twice the number of colours in the path.
        reverse (bool): Whether to reverse the colormap. If 'auto', the
            colormap will start with the end closest to dark blue. If False,
            the direction is essentially random.
        equilibrate (bool): Whether to equilibrate the colormap. This will
            try to ensure that the colormap's colors are as evenly spaced as
            possible.

    Returns:
        matplotlib.colors.LinearSegmentedColormap: Colormap.
    """
    cpath = np.asarray(unique_colors)[path]
    if reverse == 'auto':
        cool_dark = np.array([0, 0, 0.5])
        if np.linalg.norm(cpath[0] - cool_dark) > np.linalg.norm(cpath[-1] - cool_dark):
            reverse = True
        else:
            reverse = False
    if reverse:
        cpath = cpath[::-1]
    if colors is None:
        colors = 2 * len(cpath)  # Not sure what the default should be.
    cmap = LinearSegmentedColormap.from_list("recovered", cpath, N=colors)
    if equilibrate:
        dists = np.linalg.norm(cpath[:-1] - cpath[1:], axis=-1)
        invdist = np.cumsum(1 / dists) / (1 / dists).sum()
        cmap = LinearSegmentedColormap.from_list('recovered', cmap(invdist), N=colors)
    return cmap


def guess_cmap_from_array(array,
               source_colors=256,
               target_colors=256,
               min_weight=0.025,
               max_dist=0.25,
               max_neighbours=20,
               reverse='auto',
               equilibrate=False
               ):
    """
    Guess the colormap of an image.

    Args:
        array (np.ndarray): The RGB or RGBA image array.
        source_colors (int): Number of colours to detect in the source image.
        target_colors (int): Number of colours to return in the colormap.
        min_weight (float): Minimum weight to keep. See `prune_graph`.
        max_dist (float): Maximum distance to keep. See `prune_graph`.
        max_neighbours (int): Maximum number of neighbours to allow. See `prune_graph`.
        reverse (bool): Whether to reverse the colormap. If 'auto', the
            colormap will start with the end closest to dark blue. If False,
            the direction is essentially random.

    """
    imarray, uniq = convert_imarray(array, colors=source_colors)
    G = construct_graph(imarray, colors=source_colors)
    G0 = prune_graph(G, uniq, min_weight=min_weight, max_dist=max_dist, max_neighbours=max_neighbours)
    path = longest_shortest_path(G0)
    return path_to_cmap(path, uniq, colors=target_colors, reverse=reverse, equilibrate=equilibrate)


def guess_cmap_from_image(fname,
                          source_colors=256,
                          target_colors=256,
                          min_weight=0.025,
                          max_dist=0.25,
                          max_neighbours=20,
                          reverse='auto',
                          equilibrate=False
                          ):
    """
    Guess the colormap of an image.

    Args:
        fname (str): Filename or URL of image to guess.
        source_colors (int): Number of colours to detect in the source image.
        target_colors (int): Number of colours to return in the colormap.
        min_weight (float): Minimum weight to keep. See `prune_graph`.
        max_dist (float): Maximum distance to keep. See `prune_graph`.
        max_neighbours (int): Maximum number of neighbours to allow. See `prune_graph`.
        reverse (bool): Whether to reverse the colormap. If 'auto', the
            colormap will start with the end closest to dark blue. If False,
            the direction is essentially random.

    """
    with fsspec.open(fname) as f:
        img = Image.open(f)
        return guess_cmap_from_array(np.asarray(img),
                                     source_colors=source_colors,
                                     target_colors=target_colors,
                                     min_weight=min_weight,
                                     max_dist=max_dist,
                                     max_neighbours=max_neighbours,
                                     reverse=reverse,
                                     equilibrate=equilibrate
                                     )
