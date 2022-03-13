import jax.numpy as np

from .am import AM
from .kernel_tools import precompute_kernels


class cMMD_object(AM):
    def method(
        self,
        X,
        Y,
        precompute=None,
        kernel="gaussian",
        normalised=False,
        sigma=None,
        **args
    ):

        n = X.shape[0]
        m = Y.shape[0]

        if precompute is None:
            Kx = precompute_kernels(
                X, kernel=kernel, sigma=sigma, center_kernel=normalised
            )
            Ky = precompute_kernels(
                Y, kernel=kernel, sigma=sigma, center_kernel=normalised
            )
        else:
            Kx = precompute[0]
            Ky = precompute[1]

        Kxy = precompute_kernels(
            X, Y, kernel=kernel, sigma=sigma, center_kernel=normalised
        )

        mmd2 = 1 / n**2 * Kx.sum() + 1 / m**2 * Ky.sum() - 2 / (n * m) * Kxy.sum()
        mmd = np.sqrt(mmd2)

        return mmd


cMMD = cMMD_object()
