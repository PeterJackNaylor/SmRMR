import jax.numpy as np
from functools import partial


def none(theta, **kwargs):
    return 0


def lasso(theta, lamb=0.5, **kwargs):
    return (lamb * np.absolute(theta)).sum()


def scad(theta, lamb=0.5, a=3.7, **kwargs):
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


def scad_derivative(theta, lamb=0.5, a=3.7, **kwargs):
    abs_theta = np.absolute(theta)
    pen = np.where(
        abs_theta <= lamb,
        lamb,
        np.where(
            abs_theta < a * lamb,
            (a * lamb - abs_theta) / (a - 1),
            0,
        ),
    )
    return pen


def mcp(theta, lamb=0.5, b=3.5, **kwargs):
    abs_theta = np.absolute(theta)
    pen = lamb * np.where(
        abs_theta < lamb * b, abs_theta - abs_theta**2 / (2 * b * lamb), 0.5 * b
    )
    return pen.sum()


def mcp_derivative(theta, lamb=0.5, b=3.5, **kwargs):
    abs_theta = np.absolute(theta)
    pen = np.where(abs_theta <= lamb * b, (lamb - abs_theta / b) * np.sign(theta), 0)
    return pen


def pic_penalty(kwargs):
    name = kwargs["name"]
    match name:
        case "None":
            f = none
        case "l1":
            f = lasso
        case "scad":
            f = scad
        case "mcp":
            f = mcp
        case _:
            error_msg = f"Unkown penalty: given {name}"
            raise ValueError(error_msg)
    f = partial(f, **kwargs)
    return f


def pic_derivative(kwargs):
    name = kwargs["name"]
    match name:
        case "scad":
            f = scad_derivative
        case "mcp":
            f = mcp_derivative
        case _:
            error_msg = f"Unkown derivative penalty: given {name}"
            raise ValueError(error_msg)
    f = partial(f, **kwargs)
    return f


def pic_penalty_cvx(kwargs):
    name = kwargs["name"]
    match name:
        case "None":
            f = none
        case "l1":
            f = lasso
        case "scad":
            f = scad
        case "mcp":
            f = mcp
        case _:
            error_msg = f"Unkown penalty: given {name}"
            raise ValueError(error_msg)
    f = partial(f, **kwargs)
    return f


def scad_test():
    """Scad test
    Manually checked
    """
    beta = np.array([-0.3, 0.5, 1, -1.5, 2, 2.5])
    result = np.array([0.15, 0.25, 0.4537037, 0.56481487, 0.5875, 0.5875])
    assert scad(beta) == result
