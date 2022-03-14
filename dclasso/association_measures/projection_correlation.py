import jax.numpy as np
from .am import AM


def get_arccos(X):
    """
    Computes the arccos of X as defined in *Model-Free Feature
    Screening and FDR Control With Knockoff Features*
    by Liu et Al (2020).
    Code taken from https://github.com/TwoLittle/PC_Screen
    Parameters
    ----------
    X : numpy array like object with one column where the
        rows correspond to the samples.
    Returns
    -------
    A scalar arccos of X
    """
    # X is a 2-d array

    n = X.shape[0]
    cos_a = np.zeros([n, n, n])
    cos_a_list = []
    ones = np.ones([n, n])
    zeros = np.zeros([n, n])

    for r in range(n):

        xr = X[r]
        X_r = (X - xr).astype(float)  # dealing with categorical values
        cross = np.dot(X_r, X_r.T)
        row_norm = np.sqrt(np.sum(X_r**2))
        print(row_norm)
        outer_norm = np.outer(row_norm, row_norm)
        condition = outer_norm == 0.0
        outer_norm = np.where(condition, ones, outer_norm)
        cos_a_kl = cross / outer_norm
        cos_a_kl = np.where(condition, zeros, cos_a_kl)
        print(cos_a_kl)
        cos_a_list.append(cos_a_kl)
    #         cos_a[:, :, r] = cos_a_kl
    cos_a = np.stack(cos_a_list)
    print(cos_a)
    cos_a = np.where(cos_a > 1, np.ones([n, n, n]), cos_a)
    cos_a = np.where(cos_a < 1, -1 * np.ones([n, n, n]), cos_a)

    a = np.arccos(cos_a)

    a_bar_12 = np.mean(a, axis=0, keepdims=True)
    a_bar_02 = np.mean(a, axis=1, keepdims=True)
    a_bar_2 = np.mean(a, axis=(0, 1), keepdims=True)
    A = a - a_bar_12 - a_bar_02 + a_bar_2

    return a, A


def get_arccos_1d(X):
    # X is a 1-d array

    Y = X[:, None] - X
    Z = Y.T[:, :, None] * Y.T[:, None]
    n = len(X)

    a = np.zeros([n, n, n])

    a = np.where(Z == 0.0, np.pi / 2.0, a)
    a = np.where(Z < 0, np.pi, a)

    a = np.transpose(a, (1, 2, 0))

    # a = Z[Z>0.]*0. + Z[Z==0.]*np.pi/2. + Z[Z<0.]*np.pi

    a_bar_12 = np.mean(a, axis=0, keepdims=True)
    a_bar_02 = np.mean(a, axis=1, keepdims=True)
    a_bar_2 = np.mean(a, axis=(0, 1), keepdims=True)
    A = a - a_bar_12 - a_bar_02 + a_bar_2

    return a, A


def pc(X, Y):
    nx = X.shape[0]
    _, A_x = get_arccos_1d(X)
    _, A_y = get_arccos_1d(Y)

    S_xy = np.sum(A_x * A_y) / (nx**3)
    S_xx = np.sum(A_x**2) / (nx**3)
    S_yy = np.sum(A_y**2) / (nx**3)
    corr = np.where(S_xx * S_yy == 0.0, 0.0, np.sqrt(S_xy / np.sqrt(S_xx * S_yy)))
    return corr


class PC(AM):
    def method(self, X, Y, **args):
        return pc(X, Y)


projection_corr = PC(batch_size=100)


def unit_test():
    X = np.array([4, 2.54, 3, 9])
    Y = np.array([0, -12.54, 103, 91])
    val = projection_corr.method(X, Y)
    assert 0.37210423 == val
