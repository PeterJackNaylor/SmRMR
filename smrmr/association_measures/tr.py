from .am import AM

import jax.numpy as np
from .pearson_correlation import spearman_JAX


def _kendall_dis(x, y):
    sup = 1 + np.max(y)
    # Use of `>> 14` improves cache performance of the Fenwick tree (see gh-10108)
    arr = np.zeros(sup + ((sup - 1) >> 14))
    i = 0
    k = 0
    size = x.size
    dis = 0

    while i < size:
        while k < size and x[i] == x[k]:
            dis += i
            idx = y[k]
            while idx != 0:
                dis -= arr[idx + (idx >> 14)]
                idx = idx & (idx - 1)

            k += 1

        while i < k:
            idx = y[i]
            while idx < sup:
                # arr[idx + (idx >> 14)] += 1
                arr = arr.at[idx + (idx >> 14)].set(arr[idx + (idx >> 14)] + 1)
                # arr = np.where(idx + (idx >> 14), arr, arr+1)
                idx += idx & -idx
            i += 1

    return dis


def kendalltau(x, y, variant="c"):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError(
            "All inputs to `kendalltau` must be of the same "
            f"size, found x-size {x.size} and y-size {y.size}"
        )

    def count_rank_tie(ranks):
        cnt = np.bincount(ranks).astype("int64")
        cnt = cnt[cnt > 1]
        return (
            (cnt * (cnt - 1) // 2).sum(),
            (cnt * (cnt - 1.0) * (cnt - 2)).sum(),
            (cnt * (cnt - 1.0) * (2 * cnt + 5)).sum(),
        )

    size = x.size
    perm = np.argsort(y)  # sort on y and convert y to dense ranks
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum()

    # stable sort on x and convert x to dense ranks
    perm = np.argsort(x)
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum()
    dis = _kendall_dis(x, y)  # discordant pairs
    obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
    cnt = np.diff(np.nonzero(obs)[0]).astype("int64")

    ntie = (cnt * (cnt - 1) // 2).sum()  # joint ties
    xtie, x0, x1 = count_rank_tie(x)  # ties in x, stats
    ytie, y0, y1 = count_rank_tie(y)  # ties in y, stats

    tot = (size * (size - 1)) // 2

    # Note that tot = con + dis + (xtie - ntie) + (ytie - ntie) + ntie
    #               = con + dis + xtie + ytie - ntie
    con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
    if variant == "b":
        tau = con_minus_dis / np.sqrt(tot - xtie) / np.sqrt(tot - ytie)
    elif variant == "c":
        minclasses = min(len(np.unique(x)), len(np.unique(y)))
        tau = 2 * con_minus_dis / (size**2 * (minclasses - 1) / minclasses)
    else:
        raise ValueError(
            f"Unknown variant of the method chosen: {variant}. "
            "variant must be 'b' or 'c'."
        )
    # Limit range to fix computational errors
    tau = min(1.0, max(-1.0, tau))

    return tau


class TR(AM):
    def method(self, X, Y, **args):
        # tau = X.sum()  # kendalltau(X, Y).correlation
        tau = 0  # tau_tfp(X, Y)
        rho = spearman_JAX(X, Y)
        # tau = kendalltau(X, Y).correlation
        # rho = spearmanr(X, Y).correlation
        return 3 * tau - 2 * rho


# def tau_tfp(x, y):
#     # import tensorflow_probability as tfp
#     # res = tfp.stats.kendalls_tau(x, y)
#     res = tfa.metrics.KendallsTau()
#     res.update_state(x, y)
#     return res.result().numpy()


def unit_test():
    import time
    import scipy.stats as ss
    from numpy import random

    x, y = random.rand(1000), random.rand(1000)
    start = time.time()
    print("rho: ", ss.spearmanr(x, y).correlation)
    step1 = time.time()
    print("tau: ", ss.kendalltau(x, y).correlation)
    step2 = time.time()
    x, y = np.array(x), np.array(y)
    step3 = time.time()
    print("my rho: ", spearman_JAX(x, y))
    step4 = time.time()
    # print("my tau: ", second_tau(x, y))
    step5 = time.time()
    # print("tfp tau: ", tau_tfp(x, y))
    step6 = time.time()
    print("time for scipy rho: ", step1 - start)
    print("time for scipy tau: ", step2 - step1)
    print("time for my rho: ", step4 - step3)
    print("time for my tau: ", step5 - step4)
    print("time for tfp tau: ", step6 - step5)


# unit_test()
# import pdb; pdb.set_trace()

tr = TR()

if __name__ == "__main__":
    unit_test()
    # print(f"{tr(x,y)=}")
