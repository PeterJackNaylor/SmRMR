from abc import abstractmethod

from jax import vmap
from jax import jit
import jax.numpy as np
from tqdm import trange

import numpy as onp


class AM:
    def __init__(self, batch_size=1000):
        self.batch_size = batch_size

    def __call__(self, X, Y=None, max_batch=None, **args):
        batch_size = max_batch if max_batch else self.batch_size
        n, d = X.shape
        y_1d = True if Y is not None else False
        if y_1d:
            ny, nd = Y.shape
            assert n == ny
            assert nd == 1
        else:
            Y = X
            nd = d

        y_levels = onp.unique(Y)

        if not y_1d:
            indices = np.triu_indices(d, k=0, m=nd)
        else:
            indices = np.triu_indices(d, k=-d, m=nd)
        n_indices = indices[0].size
        if "precompute" in args.keys():
            Kx = args["precompute"][0]
            if y_1d:
                Ky = args["precompute"][1]
            else:
                Ky = Kx

            # we only want to give individual sliced arrays and not the full one..
            del args["precompute"]

            @jit
            def func_with_indices(el):
                i, j = el
                return self.method(
                    X[:, i],
                    Y[:, j],
                    precompute=(Kx[i], Ky[j]),
                    y_levels=y_levels,
                    **args
                )

        else:

            @jit
            def func_with_indices(el):
                i, j = el
                return self.method(X[:, i], Y[:, j], y_levels=y_levels, **args)

        batch_mode = determine_batch_mode(batch_size, n_indices)

        result = compute(func_with_indices, indices, batch_mode, batch_size, n_indices)

        # lax map seems to be so slow...
        # result = jax.lax.map(func_with_indices, indices)
        if not y_1d:
            result_r = np.zeros((d, nd))
            result_r = result_r.at[indices].set(result)

            i_lower = np.tril_indices(d, -1, m=nd)
            result_r = result_r.at[i_lower].set(result_r.T[i_lower])
            result = result_r

        return result

    @abstractmethod
    def method(self, X, Y, **args):
        return NotImplemented


def determine_batch_mode(batch_size, n):
    if not batch_size:
        batch_mode = False
    else:
        if n <= batch_size:
            batch_mode = False
        else:
            batch_mode = True
    return batch_mode


def compute(func, ind, batch_mode, bs, n_ind):

    if not batch_mode:
        result = vmap(func)(ind)
    else:
        ix, iy = ind

        def helper(i):
            return vmap(func)((ix[i : (i + bs)], iy[i : (i + bs)]))

        result = np.concatenate([helper(i) for i in trange(0, n_ind, bs)])
    return result
