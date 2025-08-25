import jax.numpy as np
from jax.random import PRNGKey, uniform
import pytest

from smrmr.association_measures.kernel_tools import get_kernel_function

key = PRNGKey(42)
X = uniform(key, shape=(10, 50))
Y = uniform(key, shape=(10,))

x = np.array([1, 1, 1])
y = np.array([-1, -1, -1])


def test_sigmoid():
    kernel_sigmoid, _ = get_kernel_function("sigmoid")
    assert np.trace(kernel_sigmoid(10 * x)) == 3
    assert np.all(kernel_sigmoid(10 * x, alpha=-10000) == 0)
    assert np.trace(kernel_sigmoid(10 * x, 10 * y)) == pytest.approx(0)
    assert np.trace(kernel_sigmoid(10 * x, 10 * y, alpha=10000)) == 3


kernel_and_bound = {
    "gaussian": (0, 1),
    "laplacian": (0, 1),
    "sigmoid": (1 / (1 + np.exp(-1)), 1),  # min is 1 / ( 1 + exp(-alpha) )
    "tanh": (-1, 1),
    "inverse-M": (0, 1),  # max is 1 / alpha ** beta
    "linear": (0, np.inf),
    "distance": (0, np.inf),
}


def test_kernelfunction():
    for name, (min_bound, max_bound) in kernel_and_bound.items():
        kernel, kernel_params = get_kernel_function(name)

        Ky = kernel(Y, **kernel_params)
        assert (Ky >= min_bound).all()
        assert (Ky <= max_bound).all()
        for p in range(X.shape[1]):
            Xp = X[:, p]
            Kx = kernel(Xp, **kernel_params)
            Kxy = kernel(Xp, Y, **kernel_params)
            assert (Kx >= min_bound).all()
            assert (Kx <= max_bound).all()
            assert (Kxy >= min_bound).all()
            assert (Kxy <= max_bound).all()
