# import itertools
import numpy as np
import jax
import jax.numpy as jnp

# import optax
from jax import jit, vmap
from scipy.sparse.linalg import eigsh

# from tqdm import trange
from .association_measures.hsic import precompute_kernels
from .association_measures.distance_correlation import pdist_p, fill_diagonal

# from optax._src.base import GradientTransformation


def argmin_lst(lst):
    return lst.index(min(lst))


def knock_off_check_parameters(n, p, n1, d):
    """
    Checks a variety of things with respect to n, p, n1 and d.
    Returns wether to perform screening.
    """
    screening, stop = False, False
    N1 = int(n1 * n)
    if n in [1, 2, 3]:
        # not possible because we want d < n / 2
        msg = "Fit is not possible, data too small and \
can't satisfy condition d < n_2 / 2"
        stop = True

    if p < n / 2:
        # we skip if we don't need to further reduce the number of features
        # in order to create exact knockoff features.
        screening = False
        msg = "No screening, setting d=p"
        d = p
    else:
        N2 = n - N1
        if d is None:
            # we set d with respect to N2, as the knockoff features are created then
            d = N2 / 2
            d = d if d < N2 / 2 else d - 1
        # need to check
        elif d >= N2 / 2:
            # d not set correctly so we set it to the highest plausible value
            d = N2 / 2
            d = d if d < N2 / 2 else d - 1
            if d <= 0:
                msg = "Fit is not possible, data too small and \
can't satisfy condition d < n_2 / 2"
                stop = True
            else:
                msg = "d badly set, reseting"
        else:
            msg = "d correctly set"
        if not stop:
            screening = True
            msg = "Splitting the data"
            # split data

    return stop, screening, msg, int(d)


def generate_random_sets(n, n1, key):
    indices = jnp.arange(n)
    indices = jax.random.permutation(key, indices)
    set_one = indices[: int(n1 * n)]
    set_two = indices[int(n1 * n) :]
    return set_one, set_two


def orthonormalize_qr(X):
    q, _ = jnp.linalg.qr(X, mode="reduced")
    return q


def shift_until_PSD(M, tol):
    """Add the identity until a p x p matrix M has eigenvalues of at least tol"""
    p = M.shape[0]
    try:
        mineig = float(eigsh(np.array(M), k=1, which="SA")[0].squeeze())
    except:
        print("PSD transformation failling")
        mineig = float(eigsh(np.array(M) + tol, k=1, which="SA")[0].squeeze())
    if mineig < tol:
        M = M + (tol - mineig) * np.eye(p)
    return M


def scale_until_PSD_and_cho(S, A_1, num_iter=10):
    """
    Perform a binary search to find the largest ``gamma`` such that the minimum
    eigenvalue of ``2*Sigma - gamma*S`` is at least ``tol``.

    Returns
    -------
    gamma * S : np.ndarray
        See description.
    gamma : float
        See description
    """

    # Raise value error if S is not PSD
    # Binary search to find minimum gamma
    lower_bound = 0  # max feasible gamma
    upper_bound = 2  # min infeasible gamma
    for j in range(num_iter):
        gamma = (lower_bound + upper_bound) / 2
        A = jnp.linalg.cholesky(2 * S - gamma * A_1)
        if jnp.isnan(A).any():
            upper_bound = gamma
        else:
            good_gamma = gamma
            lower_bound = gamma
            if good_gamma == 1:
                break
    # Scale S properly, be a bit more conservative
    V = 2 * S - good_gamma * A_1
    A = jnp.linalg.cholesky(V).T
    return A


def get_equi_features(X, key, eps=1e-5):
    """
    Builds the knockoff variables with the equicorrelated procedure.
    Code taken from https://github.com/TwoLittle/PC_Screen
    """
    # X is 2-d array
    n, p = X.shape
    scale = jnp.sqrt(jnp.sum(X**2, axis=0))
    Xstd = X / scale
    sigma = Xstd.T.dot(Xstd)
    lambd_min = float(eigsh(np.array(sigma), k=1, which="SA")[0].squeeze())

    sigma_inv = jnp.linalg.inv(sigma * jnp.power(scale, 2))
    sj = min([1.0, 2.0 * lambd_min])
    if sj <= 0:
        sj = eps

    sj = jnp.array(sj)
    mat_s = jnp.diag(np.repeat(sj, p))

    # added line
    mat_s = mat_s * jnp.power(scale, 2)

    # added line
    A_1 = jnp.dot(mat_s, jnp.dot(sigma_inv, mat_s))
    C = scale_until_PSD_and_cho(mat_s, A_1)

    XX = jnp.hstack([X, jnp.zeros((n, p))])

    XXo = orthonormalize_qr(XX)
    U = XXo[:, p : (2 * p)]

    Xnew = jnp.dot(X, jnp.eye(p) - jnp.dot(sigma_inv, mat_s)) + jnp.dot(U, C)
    return Xnew


def precompute_kernels_match(measure_stat, X, y, kernel, ms_kwargs, normalise_input):
    if measure_stat in ["HSIC", "DC"] and normalise_input:
        X = X / jnp.linalg.norm(X, ord=2, axis=0)

    match measure_stat:
        case "HSIC" | "cMMD":
            ms_kwargs["precompute"] = compute_kernels_for_am(X, y, kernel, **ms_kwargs)
        case "DC":
            ms_kwargs["precompute"] = compute_distance_for_am(X, y, **ms_kwargs)
    return ms_kwargs


def alpha_threshold(
    alpha,
    wjs,
    indices,
    hard_alpha=True,
    alpha_increase=0.05,
    return_alpha=False,
    verbose=True,
    conservative=True,
):
    """
    Computes the selected features with respect to alpha.
    Parameters
    ----------
    alpha : threshold value to use for post inference selection.
    Returns
    -------
    A 3 element tuple where:
        1 - indices corresponding to indexes of the chosen features in
        the original array.
        2 - the threshold value.
        3 - the number of selected features.
    """
    init_alpha = alpha
    n_out = []
    while not n_out and alpha < 1.0:
        alpha_indices_, t_alpha_ = threshold_alpha(
            wjs, indices, alpha, conservative, verbose
        )
        n_features_out_ = len(alpha_indices_)

        n_out = list(alpha_indices_)
        if not n_out:
            alpha += alpha_increase
        if hard_alpha:
            break

    if verbose:
        if len(alpha_indices_):
            print("selected features: ", alpha_indices_)
            print(f"init alpha: {init_alpha}; last alpha: {alpha}")
        else:
            print("No features were selected, returning empty set.")
    if return_alpha:
        return alpha_indices_, t_alpha_, n_features_out_, alpha
    return alpha_indices_, t_alpha_, n_features_out_


# def apply_at(fn, pos_lst, iterable):
#     """
#     Apply function at given indices in the iterable.

#     Args:
#         fn (function): Function to apply
#         pos_lst list: list-like array containing the indices
#         iterable list: array on which to apply the transformation

#     Returns:
#         list: with elements in pos_lst modified by fn.
#     """
#     pos_lst = set(pos_lst)
#     return [fn(x) if i in pos_lst else x for (i, x) in enumerate(iterable)]


# def pos_proj(array):
#     """Positive projection of the array
#     by simple positive clipping.

#     Args:
#         array (numpy array like): input aray

#     Returns:
#         Same type as input: clipped array
#     """
#     return array.clip(0)


# def minimize_loss(
#     step_function, opt_state, beta, max_epoch, eps_stop, patience=5, verbose=False
# ):
#     # error_tmp = []
#     prev = jnp.inf
#     i = 0
#     # Minimizing loss function
#     range_epoch = trange(max_epoch) if verbose else range(max_epoch)
#     for _ in range_epoch:
#         value, beta, opt_state = step_function(beta, opt_state)
#         # error_tmp.append(float(value))
#         if abs(value - prev) < eps_stop / patience:
#             i += 1
#             if i == patience:
#                 break
#         else:
#             i = 0
#         prev = value
#     return beta, value


# def get_optimizer(opt, opt_kwargs):
#     scheduler = optax.exponential_decay(**opt_kwargs)

#     match opt:
#         case "SGD":
#             optimizer = optax.chain(
#                 optax.identity(),
#                 optax.scale_by_schedule(scheduler),
#                 optax.scale(-1.0),
#                 optax.keep_params_nonnegative(),
#             )

#         case "adam":
#             # Combining gradient transforms using `optax.chain`.
#             optimizer = optax.chain(
#                 optax.scale_by_adam(),  # Use the updates from adam.
#                 optax.scale_by_schedule(
#                     scheduler
#                 ),  # Use the learning rate from the scheduler.
#                 # Scale updates by -1 since optax.apply_updates is additive and we
#                 # want to descend on the loss.
#                 optax.scale(-1.0),
#                 optax.keep_params_nonnegative(),
#             )
#         case opt if opt.__class__ == GradientTransformation:
#             optimizer = opt
#         case _:
#             error_msg = f"Unkown optimizer: {opt}, should be a known string or of \
#             GradientTransformation class of the optax python package"
#             raise ValueError(error_msg)

#     return optimizer


def compute_kernels_for_am(X, y, kernel, **kwargs):
    _, p = X.shape
    # mostly for HSIC and cMMD
    indicesX = jnp.arange(p)
    indicesY = jnp.arange(1)

    def jit_precompute_kernels(x):
        return precompute_kernels(x, kernel=kernel, **kwargs)

    @jit
    def precompX(k):
        return jit_precompute_kernels(X[:, k])

    @jit
    def precompY(k):
        return jit_precompute_kernels(y)

    Kx = vmap(precompX)(indicesX)
    Ky = vmap(precompY)(indicesY)
    return Kx, Ky


def compute_distance_for_am(X, y, **kwargs):
    # mostly for DC
    _, p = X.shape
    indicesX = jnp.arange(p)
    indicesY = jnp.arange(1)
    p = kwargs["order_x"] if "order_x" in kwargs.keys() else 2
    q = kwargs["order_x"] if "order_y" in kwargs.keys() else 2

    def pdist_A(x, y, p, unbiased=False):
        n = X.shape[0]
        A = pdist_p(x, y, p)
        if unbiased:
            A_1 = jnp.tile(A.sum(axis=1), (n, 1)) / (n - 2)
            A_0 = jnp.tile(A.sum(axis=0), (1, n)).reshape(n, n, order="F") / (n - 2)
            A = A - A_0 - A_1 + A.sum() / ((n - 1) * (n - 2))
            A = fill_diagonal(A, 0)
        else:
            A_1 = jnp.tile(A.mean(axis=1), (n, 1))
            A_0 = jnp.tile(A.mean(axis=0), (1, n)).reshape(n, n, order="F")
            mean = A.mean()
            A = A - A_0 - A_1 + mean
        return A

    def jit_precompute_dist_x(x):
        return pdist_A(x, x, p=p)

    def jit_precompute_dist_y(y):
        return pdist_A(y, y, p=q)

    @jit
    def precompX(k):
        return jit_precompute_dist_x(X[:, k])

    @jit
    def precompY(k):
        return jit_precompute_dist_y(y)

    Kx = vmap(precompX)(indicesX)
    Ky = vmap(precompY)(indicesY)
    return Kx, Ky


def threshold_alpha(Ws, w_indice, alpha, conservative=True, verbose=True):
    """
    Computes the set defined by equation 3.8
    Parameters
    ----------
    Ws : list like object corresponding to the estimator W_j
        for each feature.
    w_indice : numpy array like object corresponding to the indices
    of the W_j in the original dataset.
    alpha : float between 0 and 1. Sets the FDR rate.
    Returns
    -------
    indices : numpy array like corresponding to the selected features.
    t_alpha : float, which is the threshold used to select the set of active
    features.
    """

    ts = jnp.sort(abs(Ws))

    add = 1 if conservative else 0

    def fraction_3_6(t):
        num = (Ws <= -abs(t)).sum() + add
        den = max((Ws >= abs(t)).sum(), 1)
        return num / den

    fraction_3_6_v = np.vectorize(fraction_3_6)
    fdp = fraction_3_6_v(ts)

    t_alpha = jnp.where(fdp <= alpha)
    if t_alpha[0].size == 0:
        # no one selected..
        if verbose:
            print("alpha_min set to jnp.inf, no one selected...")
        t_alpha_min = jnp.inf
    else:
        t_alpha_min = min(ts[t_alpha])
    indices = w_indice[Ws >= t_alpha_min]
    ########################################
    # For debugging purposes
    # def numerator(t):
    #    num = (Ws <= -abs(t)).sum() + 1
    #    return num

    # def denominator(t):
    #    den = max((Ws >= abs(t)).sum(), 1)
    #    return den

    # nume_v = npy.vectorize(numerator)
    # denomi_3_6_v = npy.vectorize(denominator)
    # nn = nume_v(ts)
    # dd = denomi_3_6_v(ts)
    #######################################

    return indices, t_alpha_min


# def gene_generators(param_grid, kernel_ms=[]):
#     measure_stat = param_grid["ms"]
#     kernel = param_grid["kernel"]

#     def ms_kernel_gen():
#         for ms in measure_stat:
#             if ms in kernel_ms:
#                 for k in kernel:
#                     yield (ms, k)
#             else:
#                 yield (ms, None)

#     penalty = param_grid["penalty"]
#     optimizer = param_grid["optimizer"]
#     learning_rate = param_grid["learning_rate"]
#     lambda_ = param_grid["lambda"]

#     def hp_gen():
#         dic = {}
#         dic["penalty_kwargs"] = {}
#         dic["opt_kwargs"] = {"transition_steps": 100, "decay_rate": 0.99}
#         iters = itertools.product(penalty, optimizer, learning_rate)
#         for pen, opt, lr in iters:
#             dic["penalty_kwargs"]["name"] = pen
#             dic["opt_kwargs"]["init_value"] = lr
#             dic["optimizer"] = opt
#             if pen in ["None", "l1"]:
#                 if lr != learning_rate[0]:
#                     continue
#             for la in lambda_:
#                 dic["penalty_kwargs"]["lamb"] = la
#                 if pen == "None" and dic["penalty_kwargs"]["lamb"] == lambda_[0]:
#                     dic["penalty_kwargs"]["lamb"] = 0
#                     yield dic
#                 elif pen != "None" and dic["penalty_kwargs"]["lamb"] != 0:
#                     yield dic

#     return ms_kernel_gen, hp_gen


def selected_top_k(beta, d):
    return jnp.argsort(beta)[-d:][::-1]
