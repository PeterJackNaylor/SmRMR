import numpy as np
import jax
import jax.numpy as jnp
from scipy.sparse.linalg import eigsh
from tqdm import trange


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


def orthonormalize(X):
    """
    Orthonomalizes X.
    Code taken from https://github.com/TwoLittle/PC_Screen
    """
    # X is a 2-d array
    # output: Gram-Schmidt orthogonalization of X

    n, p = X.shape
    Y = jnp.zeros([n, p])
    Y = Y.at[:, 0].set(X[:, 0] / jnp.sqrt(jnp.sum(X[:, 0] ** 2)))

    for j in trange(1, p):

        Yj = Y[:, 0:j]
        xj = X[:, j]
        w = jnp.dot(xj, Yj)
        xj_p = jnp.sum(w * Yj, axis=1)
        yj = xj - xj_p
        yj = yj / jnp.sqrt(jnp.sum(yj**2))

        Y = Y.at[:, j].set(yj)

    return Y


def orthonormalize_qr(X):
    q, _ = jnp.linalg.qr(X, mode="reduced")
    return q


def orthonormalize_q(X):
    n, p = X.shape
    Xpadded = jnp.ones([p, p])
    Xpadded = Xpadded.at[0:n].set(X)

    Q = jnp.linalg.qr(Xpadded, mode="complete")[0][0:n]
    return Q


def shift_until_PSD(M, tol):
    """Add the identity until a p x p matrix M has eigenvalues of at least tol"""
    p = M.shape[0]
    mineig = float(eigsh(np.array(M), k=1, which="SA")[0].squeeze())
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

    # sigma_inv = jnp.linalg.inv(sigma)
    sigma_inv = jnp.linalg.inv(sigma * jnp.power(scale, 2))
    sj = min([1.0, 2.0 * lambd_min])
    if sj <= 0:
        sj = eps
    # why does this line make A invertible...
    # sj = jnp.array(sj - 0.00001)
    sj = jnp.array(sj)
    mat_s = jnp.diag(np.repeat(sj, p))

    # added line
    mat_s = mat_s * jnp.power(scale, 2)

    # added line
    A_1 = np.dot(mat_s, np.dot(sigma_inv, mat_s))
    C = scale_until_PSD_and_cho(mat_s, A_1)

    # mat_s =  jnp.array(mat_s - eps)

    # A = 2 * mat_s - sj * sj * sigma_inv

    # C = jnp.linalg.cholesky(A).T

    # Xn = jax.random.normal(key, (n, p))

    # XX = jnp.hstack([Xstd, Xn])
    XX = jnp.hstack([X, jnp.zeros((n, p))])

    # XXo = orthonormalize(XX)
    XXo = orthonormalize_qr(XX)
    U = XXo[:, p : (2 * p)]

    # Xnew = jnp.dot(Xstd, jnp.eye(p) - sigma_inv * sj) + jnp.dot(U, C)
    Xnew = jnp.dot(X, jnp.eye(p) - np.dot(sigma_inv, mat_s)) + jnp.dot(U, C)
    return Xnew
