import jax.numpy as np
import pytest

from dclasso.association_measures.kernel_tools import kernel_sigmoid

x = np.array([1, 1, 1])
y = np.array([-1, -1, -1])


def test_sigmoid():

    assert np.trace(kernel_sigmoid(10 * x)) == 3
    assert np.all(kernel_sigmoid(10 * x, coeff0=-10000) == 0)
    assert np.trace(kernel_sigmoid(10 * x, 10 * y)) == pytest.approx(0)
    assert np.trace(kernel_sigmoid(10 * x, 10 * y, coeff0=10000)) == 3
