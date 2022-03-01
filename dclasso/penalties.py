import jax.numpy as np


def none(theta):
    return 0


def lasso(theta, lamb=0.5):
    return (lamb * np.absolute(theta)).sum()


def scad(theta, lamb=0.5, a=3.7):
    assert lamb < a * lamb
    abs_theta = np.absolute(theta)
    pen = np.where(
        abs_theta < lamb,
        lamb * abs_theta,
        np.where(
            abs_theta < a * lamb,
            (2 * a * lamb * abs_theta - abs_theta**2 - lamb**2) / (2 * a - 2),
            0.5 * (a + 1) * lamb**2,
        ),
    )
    return pen.sum()


def mcp(theta, lamb=0.5, b=3):
    abs_theta = np.absolute(theta)
    pen = lamb * np.where(
        abs_theta < lamb * b, abs_theta - abs_theta**2 / (2 * b * lamb), 0.5 * b
    )
    return pen.sum()


penalty_dic = {"none": none, "l1": lasso, "scad": scad, "mcp": mcp}


def scad_test():
    """Scad test
    Manually checked
    """
    beta = np.array([-0.3, 0.5, 1, -1.5, 2, 2.5])
    result = np.array([0.15, 0.25, 0.4537037, 0.56481487, 0.5875, 0.5875])
    assert scad(beta) == result
