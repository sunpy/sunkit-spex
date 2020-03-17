
import numpy as np

def get_reverse_indices(x, nbins, min_range, max_range):
    """
    For a set of contiguous equal sized 1D bins, generates index of lower edge of bin in which each element of x belongs and the indices of x in each bin.

    Parameters
    ----------
    x: array-like
        Values to be binned.

    nbins: `int`
        Number of bins to divide range into.

    min_range: `float` or `int`
        Lower limit of range of bins. Default=min(x)

    max_range: `float` or `int`
        Upper limit of range of bins. Default=max(x)

    Returns
    -------
    arrays_bin_indices: `np.ndarray`
        Index of lower edge of bin into which each element of x goes. Same length as x.

    bins_array_indices: `tuple` of `np.ndarray`s
        Indices of elements of x in each bin. One set of indices for each bin.

    bin_edges: `np.ndarray`
        Edges of bins. Length is nbins+1.

    """
    bin_edges = np.linspace(min_range, max_range, nbins+1)
    arrays_bin_indices = (float(nbins)/(max_range - min_range)*(x - min_range)).astype(int)
    bins_array_indices = tuple([np.where(arrays_bin_indices == i)[0] for i in range(nbins)])
    return arrays_bin_indices, bins_array_indices, bin_edges
