# from dcor import distance_correlation as dc
import numpy as np
import dcor


def distance_corr(X, Y):
    """
    Computes the distance correlation between X and Y.
    Taken from pypi package dcor based on the paper:
    *Measuring and testing dependence by correlation of distances*
    by Gábor et Al (2007)
    Parameters
    ----------
    X : numpy array like object where the rows correspond to the samples
        and the columns to features.

    Y : numpy array like, of same size as X and one single output.

    Returns
    -------
    numpy array of size the number of input features of X
    which holds the distance correlation between each feature
    and Y.
    """
    n, d = X.shape
    if Y is not None:
        ny, nd = Y.shape

        assert n == ny
        assert nd == 1
    else:
        Y = X
        nd = d

    dc_stats = np.zeros((d, nd))

    for j in range(nd):
        for i in range(d):
            dc_stats[i, j] = dcor.distance_correlation(X[:, i], Y[:, j])

    return dc_stats
