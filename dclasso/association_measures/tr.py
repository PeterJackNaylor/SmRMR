import jax.numpy as np
from jax.lax import while_loop
from scipy.stats import kendalltau, spearmanr
from tqdm import trange
from .am import AM
import warnings

from jax import grad, jit, vmap

from .pearson_correlation import spearman_JAX

class TR(AM):
    def method(self, X, Y):
        tau = X.sum()  # kendalltau(X, Y).correlation
        rho = spearman_JAX(X, Y)
        # tau = kendalltau(X, Y).correlation
        # rho = spearmanr(X, Y).correlation
        return rho


if __name__ == "__main__":
    x, y = [1, 2, 3, 4, 5], [5, 6, 7, 8, 7]
    print("rho: ", spearmanr(x, y).correlation)
    print("tau: ", kendalltau(x, y).correlation)
    print(f"{tr(x,y)=}")
