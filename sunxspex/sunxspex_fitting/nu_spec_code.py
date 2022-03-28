"""
The following code is used to make the SRM/counts data in consistent units from NuSTAR spectral data.
"""

import numpy as np
from . import io

__all__ = ["flux_cts_spec", "col2arr_py", "vrmf2arr_py", "make_srm"]

def flux_cts_spec(file, bin_size):
    """ Takes a .pha file and returns plotting information.

    Parameters
    ----------
    file : Str
            String for the .pha file of the spectrum under investigation.

    Returns
    -------
    The count rate per keV (cts), and its error (cts_err).
    """

    _, counts, livetime = io._read_pha(file)

    cts = (counts / bin_size) / livetime # now in cts keV^-1 s^-1

    cts_err = (np.sqrt(counts) / bin_size) / livetime

    return cts, cts_err


def col2arr_py(data):
    """ Takes a list of parameters for each energy channel from a .rmf file and returns it in the correct format.

    From: https://lost-contact.mit.edu/afs/physics.wisc.edu/home/craigm/lib/idl/util/vcol2arr.pro

    Parameters
    ----------
    data : array/list-like object
            One parameter's array/list from the .rmf file.

    Returns
    -------
    A 2D numpy array of the correctly ordered input data.

    Example
    -------
    data = FITS_rec([(  1.6 ,   1.64,   1, [0]   , [18]  , [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                     (  1.64,   1.68,   1, [0]   , [20]  , [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                     (  1.68,   1.72,   2, [0,22], [20,1], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                     dtype=(numpy.record, [('ENERG_LO', '>f4'), ('ENERG_HI', '>f4'), ('N_GRP', '>i2'),
                                           ('F_CHAN', '>i4', (2,)), ('N_CHAN', '>i4', (2,)), ('MATRIX', '>i4', (2,))]))

    >>> col2arr_py(data['F_CHAN'])
    array([[  0.,   0.],
           [  0.,   0.],
           [  0.,  22.]])
    ## max row length of 2 so 2 columns, each row is an energy channel.
    """

    ## this is the quicker way I have chosen to do in Python (this may be revised later but is ~30x faster than way below in Python)
    max_len = np.max([len(r) for r in data]) # find max row length
    chan_array_py = np.array([[*r, *(max_len-len(r))*[0]] for r in data]) # make each row that length (padding with 0)

    return chan_array_py


def vrmf2arr_py(data=None, n_grp_list=None, f_chan_array=None, n_chan_array=None):
    """ Takes redistribution parameters for each energy channel from a .rmf file and returns it in the correct format.

    From: https://lost-contact.mit.edu/afs/physics.wisc.edu/home/craigm/lib/idl/spectral/vrmf2arr.pro

    Parameters
    ----------
    data : array/list-like object
            Redistribution matrix parameter array/list from the .rmf file. Units are counts per photon.
            Default : None

    no_of_channels : int
            Number of entries is the total number of photon channels, the entries themselves show the total number
            of count channels to which that photon channel contributes.
            Default : None

    f_chan_array : numpy.array
            The index of each sub-set channel from each energy bin from the .rmf file run through col2arr_py().
            Default : None

    n_chan_array : numpy.array
            The number of sub-set channels in each index for each energy bin from the .rmf file run through col2arr_py().
            Default : None

    Returns
    -------
    A 2D numpy array of the correctly ordered input data with dimensions of energy in the rows and channels in
    the columns.

    Example
    -------
    >>> d_rmf = 'directory/'
    >>> f_rmf = 'file.rmf'
    >>> e_lo, e_hi, ngrp, fchan, nchan, matrix = io._read_rmf(d_rmf+f_rmf)

    >>> fchan_array = nu_spec.col2arr_py(fchan)
    >>> nchan_array = nu_spec.col2arr_py(nchan)

    >>> rmf = nu_spec.vrmf2arr_py(data=matrix,
                                  n_grp_list=ngrp,
                                  f_chan_array=fchan_array,
                                  n_chan_array=nchan_array)
    >>> rmf

    array([[0.00033627, 0.0007369 , 0.00113175, ..., 0.        , 0.        , 0.        ],
           [0.00039195, 0.00079259, 0.00138341, ..., 0.        , 0.        , 0.        ],
           [0.00042811, 0.00083381, 0.00157794, ..., 0.        , 0.        , 0.        ],
                                                ...,
           [0.        , 0.        , 0.        , ..., 0.00408081, 0.00409889, 0.00403308],
           [0.        , 0.        , 0.        , ..., 0.00405333, 0.00413722, 0.00413216],
           [0.        , 0.        , 0.        , ..., 0.        , 0.        , 0.        ]])
    ## rows = photon/energy channels, columns = counts channels

    What's Going On?
    ----------------
    The RMF file has the photon-to-counts conversion information in it.
    The martix has the photon-to-count conversion value for each count channel (columns) that is involved with theach photon channel (rows).
            E.g., matrix = [ [a, b, c, d, e, f, ...] ,
                             [        ...          ] ,
                             [        ...          ] ,
                                      ...             ]
    F_chan is the starting index of contiguous counts channels that are involved with the photon channel.
            E.g., f_chan = [ [0, 5, 0, 0, 0, ...] ,
                             [       ...        ] ,
                             [       ...        ] ,
                                     ...           ]
                            For the first photon channel, there are rows of counts channels starting at index 0 and 5
    N_chan is the corresponding number of counts channels from each index in the f_chan array.
            E.g., n_chan = [ [2, 3, 0, 0, 0, ...] ,
                             [        ...        ] ,
                             [        ...        ] ,
                                      ...           ]
                            Starting at index 0 for the first photon channel we have the first 2 matrix values, then at index 5 we have the next 3.
                            The total of each row is the same as the n_grp_list and the number of entries in each row of the matrix entry.
    Putting all this together, the rmf matrix is:
            rmf_matrix = [ [a, b, 0, 0, 0, c , d , e, 0 , 0 , ...] ,   #<-- index 0 (f_chan) with 2 entries (n_chan) with photon-to-counts conversion (matrix)
                         [                 ...                   ] ,
                         [                 ...                   ] ,
                                           ...                      ]
    """
    # this was is about >6x quicker in than the IDL code written in Python

    n_grp_list = n_grp_list.astype("<i2") # change from ‘big-endian’ (">i2") to ‘little-endian’ ("<i2")

    # find the non-zero entries in Nchan, this is the number to counts channels
    #  in a row that contribute so will have a value if it is useful
    b = np.nonzero(n_chan_array)

    # now only want the useful entries from the pre-formatted Nchan and Fchan arrays
    c = f_chan_array[b]
    d = n_chan_array[b]

    # to help with indexing, this provides a running sum of the number of counts
    #  channels that a single photon channel contributes to
    e = np.cumsum(n_chan_array, axis=1)

    # these entries will give the final indices in the row on counts channels
    final_inds = e[b]

    # need to find the starting index so -1, but that means any entry that is
    #  -1 will be where a zero is needed
    starting_inds = b[1]-1

    # get the  starting indices but the ones that should be 0 are replaced with
    #  the final one in the list at the minute (-1 in starting_inds)
    start_inds = np.cumsum(n_chan_array, axis=1)[(b[0], starting_inds)]

    # where starting_inds==-1 that value should be 0, i.e. starting from the first
    #  value in the rmf matrix
    new_e = np.where(starting_inds!=-1, start_inds, 0)

    # initialise the rmf matrix
    mat_array_py = np.zeros((len(data),len(n_grp_list)))

    # now go through row by row (this is the slowest part and needs to be made faster).
    #  Here we go through each photon channel's number of discrete rows of counts channels.
    for r in range(len(c)):
        mat_array_py[b[0][r], c[r]:c[r]+d[r]] = data[b[0][r]][new_e[r]:final_inds[r]]

    return mat_array_py


def make_srm(rmf_matrix=(), arf_array=()):
    """ Takes rmf and arf and produces the spectral response matrix fro NuSTAR.

    From: https://github.com/ianan/nsigh_nov14/blob/master/make_ns_srm.pro

    Parameters
    ----------
    rmf_matrix : numpy 2D array
            Array representing the redistribution matrix.
            Default : None

    arf_array : numpy 1D array/list
            List representing the ancillary response.
            Default : None

    Returns
    -------
    An array that is the spectral response (srm).
    """

    if len(rmf_matrix) == 0 or len(arf_array) == 0:
        print('Need both RMF and ARF information to proceed.')
        return

    # srm = np.array([rmf_matrix[r, :] * arf_array[r] for r in range(len(arf_array))]) # each energy bin row in the rmf is multiplied the arf value for the same energy bin
    ## this line is >2x faster
    srm = arf_array[:, None]*rmf_matrix
    return srm
