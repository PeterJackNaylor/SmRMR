import functools
from typing import Callable, Dict
import jax
import jax.numpy as np

from .dist import sqeuclidean_distance


# @functools.partial(jax.jit, static_argnums=(0))
@functools.lru_cache(maxsize=None)
def gram(
    func: Callable,
    params: Dict,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Computes the gram matrix.

    Given a function `Callable` and some `params`, we can
    use the `jax.vmap` function to calculate the gram matrix
    as the function applied to each of the points.

    Parameters
    ----------
    func : Callable
        a callable function (kernel or distance)
    params : Dict
        the parameters needed for the kernel
    x : jax.numpy.ndarray
        input dataset (n_samples, n_features)
    y : jax.numpy.ndarray
        other input dataset (n_samples, n_features)

    Returns
    -------
    mat : np.ndarray
        the gram matrix.

    Examples
    --------

    >>> gram(kernel_rbf, {"gamma": 1.0}, X, Y)
    """
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(params, x1, y1))(y))(x)


def covariance_matrix(
    func: Callable,
    params: Dict[str, float],
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Computes the covariance matrix.

    Given a function `Callable` and some `params`, we can
    use the `jax.vmap` function to calculate the gram matrix
    as the function applied to each of the points.

    Parameters
    ----------
    kernel_func : Callable
        a callable function (kernel or distance)
    params : Dict
        the parameters needed for the kernel
    x : jax.numpy.ndarray
        input dataset (n_samples, n_features)
    y : jax.numpy.ndarray
        other input dataset (n_samples, n_features)

    Returns
    -------
    mat : jax.ndarray
        the gram matrix.

    Notes
    -----

        There is little difference between this function
        and `gram`

    See Also
    --------
    jax.kernels.gram

    Examples
    --------

    >>> covariance_matrix(kernel_rbf, {"gamma": 1.0}, X, Y)
    """
    mapx1 = jax.vmap(lambda x, y: func(params, x, y), in_axes=(0, None), out_axes=0)
    mapx2 = jax.vmap(lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)
    return mapx2(x, y)


def linear_kernel(params: Dict[str, float], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear kernel

    Parameters
    ----------
    params : None
        kept for compatibility
    x : jax.numpy.ndarray
        the inputs
    y : jax.numpy.ndarray
        the inputs

    Returns
    -------
    kernel_mat : jax.numpy.ndarray
        the kernel matrix (n_samples, n_samples)

    """
    return np.sum(x * y)


def rbf_kernel(params: Dict[str, float], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Radial Basis Function (RBF) Kernel.

    The most popular kernel in all of kernel methods.

    Parameters
    ----------
    params : Dict
        the parameters needed for the kernel
    x : jax.numpy.ndarray
        input dataset (n_samples, n_features)
    y : jax.numpy.ndarray
        other input dataset (n_samples, n_features)

    Returns
    -------
    kernel_mat : jax.numpy.ndarray
        the kernel matrix (n_samples, n_samples)

    References
    ----------
    .. [1] David Duvenaud, *Kernel Cookbook*
    """
    return np.exp(-params["gamma"] * sqeuclidean_distance(x, y))


# ARD Kernel
def ard_kernel(params: Dict[str, float], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Automatic Relevance Determination (ARD) Kernel.

    This is an RBF kernel with a variable length scale. It
    *should* be the most popular kernel of all of the kernel
    methods.

    Parameters
    ----------
    params : Dict
        the parameters needed for the kernel
    x : jax.numpy.ndarray
        input dataset (n_samples, n_features)
    y : jax.numpy.ndarray
        other input dataset (n_samples, n_features)

    Returns
    -------
    kernel_mat : jax.numpy.ndarray
        the kernel matrix (n_samples, n_samples)

    References
    ----------
    .. [1] David Duvenaud, *Kernel Cookbook*
    """
    # divide by the length scale
    x = x / params["length_scale"]
    y = y / params["length_scale"]

    # return the ard kernel
    return params["var_f"] * np.exp(-sqeuclidean_distance(x, y))


# Rational Quadratic Kernel
def rq_kernel(params: Dict[str, float], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rational Quadratic Function (RQF) Kernel.

    A generalization of the RBF kernel function. It is
    equivalent to adding many RBF kernels together.

    Parameters
    ----------
    params : Dict
        the parameters needed for the kernel
    x : jax.numpy.ndarray
        input dataset (n_samples, n_features)
    y : jax.numpy.ndarray
        other input dataset (n_samples, n_features)

    Notes
    -----

    This kernel is equivalent to the RBF kernel as
    alpha goes to 0

    References
    ----------
    .. [1] David Duvenaud, *Kernel Cookbook*
    """
    # divide by the length scale
    x = x / params["length_scale"]
    y = y / params["length_scale"]

    # return the ard kernel
    return params["var_f"] * np.exp(1 + sqeuclidean_distance(x, y)) ** (
        -params["scale_mixture"]
    )
