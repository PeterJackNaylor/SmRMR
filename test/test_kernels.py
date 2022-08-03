import jax.numpy as np
from jax.random import PRNGKey, uniform
import pytest

from dclasso.association_measures.projection_correlation import pc
from dclasso.association_measures.kernel_tools import kernel_sigmoid
from dclasso.association_measures.distance_correlation import distance_corr

key = PRNGKey(42)
X = uniform(key, shape=(10, 50))
Y = uniform(key, shape=(10, 1))
# Y_c = np.random.randint(0, 2, size=10)

x_r = np.array([35, 23, 47, 17, 10, 43, 9, 6, 28]).reshape(9, 1)
y_r = np.array([30, 33, 45, 23, 8, 49, 12, 4, 31]).reshape(9, 1)
# y_c = np.array([0, 0, 0, 1, 1, 0, 1, 1, 0]).reshape(9, 1)

# reference implementation from

x = np.array([1, 1, 1])
y = np.array([-1, -1, -1])


def test_sigmoid():

    assert np.trace(kernel_sigmoid(10 * x)) == 3
    assert np.all(kernel_sigmoid(10 * x, coeff0=-10000) == 0)
    assert np.trace(kernel_sigmoid(10 * x, 10 * y)) == pytest.approx(0)
    assert np.trace(kernel_sigmoid(10 * x, 10 * y, coeff0=10000)) == 3


def test_projection_correlation():
    # sanity check
    stats = pc(X, Y)
    assert len(stats) == X.shape[1]

    pc_ = pc(x, y)
    ans = 0.65911018
    np.testing.assert_almost_equal(pc_, ans)


def test_distance_correlation():
    # sanity check
    stats = distance_corr.method(X, Y)
    assert len(stats) == X.shape[1]

    # checked with https://rdrr.io/cran/Rfast/
    dc_ = distance_corr.method(x, y)
    ans = 0.9513721
    np.testing.assert_almost_equal(dc_, ans)


# def test_hsic():
#     hsic = HSIC(X, Y, kernel="gaussian", sigma=50) / X.shape[0] ** 2

#     ref = np.zeros((50, 1))
#     for i in range(50):
#         ref[i] = ref_hsic(X[:, i].reshape(10, 1), Y, kernel="gaussian", sigma=50)

#     assert len(hsic) == X.shape[1]
#     assert all(hsic >= 0)

#     np.testing.assert_almost_equal(ref, hsic)

#     # I got 0.09954543 with a R method..
#     hsic_ = HSIC(x, y, sigma=1) / x.shape[0] ** 2
#     ans = 0.09528812
#     np.testing.assert_almost_equal(ans, hsic_)

#     x_2 = np.random.rand(10, 1)
#     assert np.all(HSIC(x_2, x_2**2) > hsic)


# def test_kernelfunction():
#     d = 50
#     kernel, kernel_params = get_kernel_function("gaussian", nfeats=d)
#     Ky = kernel(Y[:, 0].reshape(10, 1), **kernel_params)
#     assert (Ky >= 0).all()
#     assert (Ky <= 1).all()
#     for i in range(d):
#         Xi = X[:, i].reshape(10, 1)
#         Kx = kernel(Xi, **kernel_params)
#         Kxy = kernel(Xi, Y[:, 0].reshape(10, 1), **kernel_params)
#         assert (Kx >= 0).all()
#         assert (Kx <= 1).all()
#         assert (Kxy >= 0).all()
#         assert (Kxy <= 1).all()
