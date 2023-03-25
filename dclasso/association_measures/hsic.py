from .am import AM
import jax.numpy as np

# from jax import vmap
from .kernel_tools import precompute_kernels


class HSIC_object(AM):
    def method(
        self,
        X,
        Y,
        precompute=None,
        kernel="gaussian",
        normalised=True,
        sigma=None,
        **args
    ):
        # we could save some computation by saving Kx and Ky, because we could
        # compute them once instead of d*d.
        if precompute is None:
            Kx = precompute_kernels(X, kernel=kernel, sigma=sigma)
            Ky = precompute_kernels(Y, kernel=kernel, sigma=sigma)
        else:
            Kx = precompute[0]
            Ky = precompute[1]
        hsic = np.trace(np.matmul(Kx, Ky))

        if normalised:
            hsic_xx = np.trace(np.matmul(Kx, Kx))
            hsic_yy = np.trace(np.matmul(Ky, Ky))
            norm = (hsic_xx * hsic_yy) ** 0.5
            hsic = hsic / norm
        return hsic


HSIC = HSIC_object(batch_size=500)
