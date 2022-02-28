from .am import AM
import jax.numpy as np

# from jax import vmap
from .kernel_tools import center, get_kernel_function



def precompute_kernels(X, kernel="gaussian", sigma=None):
    kernel, kernel_params = get_kernel_function(kernel, nfeats=sigma)
    Kx = center(kernel(X, **kernel_params))
    return Kx

class HSIC_object(AM):
    def method(
        self, X, Y, precompute=None, kernel="gaussian", normalised=False, sigma=None, **args
    ):

        # we could save some computation by saving Kx and Ky, because we could compute them
        # once instead of d*d.
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

HSIC = HSIC_object()
