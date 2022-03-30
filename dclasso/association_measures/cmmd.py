import jax.numpy as np

from .am import AM
from .kernel_tools import precompute_kernels


class cMMD_object(AM):
    def method(
        self,
        X,
        Y,
        y_levels,
        precompute=None,
        kernel="gaussian",
        normalised=False,
        sigma=None,
        **args
    ):

        n = X.shape[0]
        cmmd = 0

        Kx = precompute_kernels(X, kernel=kernel, sigma=sigma, center_kernel=normalised)

        for y in y_levels:
            cmmd += np.where(Y == y, x=Kx, y=0).sum() / sum(Y == y)

        cmmd = cmmd / n - 1 / n**2 * Kx.sum()

        return cmmd


cMMD = cMMD_object()
