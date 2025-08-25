import jax.numpy as np
from .am import AM


def get_arccos_1d(X):
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
