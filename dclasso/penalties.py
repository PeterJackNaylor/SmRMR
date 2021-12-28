import jax.numpy as np


def none(theta):
    return 0


def lasso(theta, lamb=0.5):
    return (lamb * np.absolute(theta)).sum()


def scad(theta, a=3.7):
    pass


def mcp(theta, b=3):
    pass


penalty_dic = {"none": none, "l1": lasso, "scad": scad, "MCP": mcp}
