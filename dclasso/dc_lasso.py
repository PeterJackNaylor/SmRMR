import numpy.typing as npt
from tqdm import trange
import numpy as npy
import jax.numpy as np
from jax.lax import top_k
from jax import jit, vmap, value_and_grad, random

import optax
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y

from . import association_measures as am
from .association_measures.hsic import precompute_kernels
from .association_measures.distance_correlation import pdist_p
from .association_measures.kernel_tools import check_vector
from .penalties import penalty_dic
from .utils import (
    knock_off_check_parameters,
    get_equi_features,
    generate_random_sets,
)

from optax._src.base import GradientTransformation

available_am = ["PC", "DC", "TR", "HSIC", "cMMD", "pearson_correlation"]
kernel_am = ["HSIC", "cMMD"]
available_kernels = ["distance", "gaussian", "linear"]


class DCLasso(BaseEstimator, TransformerMixin):
    """
    The DClasso object is a transformer from the sklearn
    base object.
    :param alpha: float between 0 and 1. Sets the FDR rate.
    :param measure_stat: string, sets the association measure

    The model parameters once fitted will be alpha_indices.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        measure_stat: str = "PC",
        kernel: str = "linear",
        penalty: str = "None",
        optimizer: str = "SGD",
        normalise_input: bool = True,
        ms_kwargs: int = None,
        opt_kwargs: int = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        assert measure_stat in available_am, "measure_stat incorrect"
        assert kernel in available_kernels, "kernel incorrect"
        self.measure_stat = measure_stat
        self.kernel = kernel
        self.normalise_input = normalise_input
        self.penalty = penalty
        self.optimizer = optimizer
        self.learning_rate = 0.001
        self.lambda_ = 0.15 * 0.2
        self.ms_kwargs = ms_kwargs if ms_kwargs else {}
        self.opt_kwargs = opt_kwargs if opt_kwargs else {}

    def _compute_assoc(self, x, y=None, **kwargs):

        args = {}
        if self.measure_stat in kernel_am:
            args["kernel"] = self.kernel

        if self.normalise_input:
            x = x / np.linalg.norm(x, ord=2, axis=0)

        assoc_func = self._get_assoc_func()
        return assoc_func(x, y, **args, **kwargs)

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        n1: float,
        d: int = None,
        seed: int = 42,
        max_epoch: int = 151,
        eps_stop: float = 1e-8,
        mode: str = "competitive",
        init="from_convex_solve",
    ):
        self.precomputed_elements = False
        key = random.PRNGKey(seed)

        # we have to prescreen and split
        X, y = check_X_y(X, y)
        X = np.asarray(X)
        n, p = X.shape

        (ny,) = y.shape
        assert n == ny
        y = check_vector(y)

        self.n_features_in_ = p

        stop, screening, msg, d = knock_off_check_parameters(n, p, n1, d)

        if stop:
            # raise ValueError(msg)
            print(msg)
            self.alpha_indices_ = []
            self.n_features_out_ = 0
            return self

        if screening:
            print("Starting screening")

            s1, s2 = generate_random_sets(n, n1, key)
            X1, y1 = X[s1, :], y[s1]
            X2, y2 = X[s2, :], y[s2]

            screened_indices = self.screen(X1, y1, d)
            X2 = X2[:, screened_indices]

        else:
            print("No screening")
            X2, y2 = X, y
            screened_indices = np.arange(p)

        # Compute knock-off variables
        Xhat = get_equi_features(X2, key)

        if mode == "competitive":
            Xs = [np.concatenate([X2, Xhat], axis=1)]
        else:
            Xs = [X2, Xhat]

        betas = []
        for x in Xs:
            loss_fn = self.compute_loss_fn(x, y2)
            beta = self.initialisation(x.shape[1], key, init)
            step_function, opt_state = self.set_optimizer(loss_fn, beta)

            error_tmp = []
            prev = np.inf
            for _ in trange(max_epoch):
                value, beta, opt_state = step_function(beta, opt_state)
                error_tmp.append(float(value))
                if abs(value - prev) < eps_stop:
                    break
                else:
                    prev = value

            betas.append(beta)

        if mode == "competitive":
            self.beta_ = betas[0]
        else:
            self.beta_ = np.concatenate([betas[0], betas[1]], axis=0)

        self.wjs_ = self.beta_[:d] - self.beta_[d:]
        alpha_thres = alpha_threshold(self.alpha, self.wjs_, screened_indices)

        self.alpha_indices_ = alpha_thres[0]
        self.t_alpha_ = alpha_thres[1]
        self.n_features_out_ = alpha_thres[2]

        print("what do we keep...?")
        return self

    def fit_transform(self, X, y, **fit_params):
        """Fits and transforms an input dataset X and y.

        Parameters
        ----------
        X : numpy array like object where the rows correspond to the samples
            and the columns to features.

        y : numpy array like, which can be multi-dimensional.

        Returns
        -------
        The new version of X corresponding to the selected features.
        """

        return self.fit(X, y, **fit_params).transform(X, y)

    def _get_assoc_func(self):
        """Returns the correct association measure
        given the attribute in __init__.
        """
        match self.measure_stat:
            case "PC":
                f = am.projection_corr
            case "TR":
                f = am.tr
            case "HSIC":
                f = am.HSIC
            case "cMMD":
                f = am.cMMD
            case "DC":
                f = am.distance_corr
            case "pearson_correlation":
                f = am.pearson_correlation
            case _:
                error_msg = f"associative measure undefined {self.measure_stat}"
                raise ValueError(error_msg)
        return f

    def penalty_func(self, beta):
        return self.lambda_ * penalty_dic[self.penalty](beta)

    def _more_tags(self):
        return {"stateless": True}

    def get_optimizer(self):
        opt = self.optimizer
        scheduler = optax.exponential_decay(
            init_value=self.learning_rate, transition_steps=200, decay_rate=0.99
        )

        match opt:
            case "SGD":
                optimizer = optax.chain(
                    optax.identity(),
                    optax.scale_by_schedule(scheduler),
                    optax.scale(-1.0),
                    optax.keep_params_nonnegative(),
                )

            case "adam":
                # Combining gradient transforms using `optax.chain`.
                optimizer = optax.chain(
                    optax.scale_by_adam(),  # Use the updates from adam.
                    optax.scale_by_schedule(
                        scheduler
                    ),  # Use the learning rate from the scheduler.
                    # Scale updates by -1 since optax.apply_updates is additive and we
                    # want to descend on the loss.
                    optax.scale(-1.0),
                    optax.keep_params_nonnegative(),
                )
            case opt if opt.__class__ == GradientTransformation:
                optimizer = opt
            case _:
                error_msg = f"Unkown optimizer: {opt}, should be a known string or of \
                GradientTransformation class of the optax python package"
                raise ValueError(error_msg)

        return optimizer

    def initialisation(self, p, key, init):
        match init:
            case "random":
                beta = random.uniform(
                    key, shape=(p,), dtype="float32", minval=0.0, maxval=2.0
                )
            case "from_convex_solve":
                # without penalty
                beta = np.matmul(np.linalg.inv(self.Dxx), self.Dxy)
                beta = pos_proj(beta)
        return beta

    def set_optimizer(self, loss_fn, beta):
        optimizer = self.get_optimizer()
        opt_state = optimizer.init(beta)

        def step(params, opt_state):
            value, grads = value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            return value, params, opt_state

        return step, opt_state

    def screen(self, X, y, d):

        Dxy = self._compute_assoc(X, y, **self.ms_kwargs)
        screened_indices = top_k(Dxy, d)[1]

        return screened_indices

    def compute_loss_fn(self, X, y):
        match self.measure_stat:
            case "HSIC" | "cMMD":
                self.ms_kwargs["precompute"] = compute_kernels_for_am(
                    X, y, self.kernel, **self.ms_kwargs
                )
            case "DC":
                self.ms_kwargs["precompute"] = compute_distance_for_am(
                    X, y, **self.ms_kwargs
                )
        # made as self so that they can re-used for the
        # initialisation
        self.Dxy = self._compute_assoc(X, y, **self.ms_kwargs)
        self.Dxx = self._compute_assoc(X, **self.ms_kwargs)

        def loss(b):
            xy_term = -(b * self.Dxy).sum()
            xx_term = 0.5 * (b * (self.Dxx * b).T).sum()
            return xy_term + xx_term + self.penalty_func(b)

        return loss


def alpha_threshold(alpha, wjs, indices):
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
    alpha_indices_, t_alpha_ = threshold_alpha(wjs, indices, alpha)
    if len(alpha_indices_):
        print("selected features: ", alpha_indices_)
    else:
        print("No features were selected, returning empty set.")
    n_features_out_ = len(alpha_indices_)
    return alpha_indices_, t_alpha_, n_features_out_


def apply_at(fn, pos_lst, iterable):
    """
    Apply function at given indices in the iterable.

    Args:
        fn (function): Function to apply
        pos_lst list: list-like array containing the indices
        iterable list: array on which to apply the transformation

    Returns:
        list: with elements in pos_lst modified by fn.
    """
    pos_lst = set(pos_lst)
    return [fn(x) if i in pos_lst else x for (i, x) in enumerate(iterable)]


def pos_proj(array):
    """Positive projection of the array
    by simple positive clipping.

    Args:
        array (numpy array like): input aray

    Returns:
        Same type as input: clipped array
    """
    return array.clip(0)


def compute_kernels_for_am(X, y, kernel, **kwargs):
    _, p = X.shape
    # mostly for HSIC and cMMD
    indicesX = np.arange(p)
    indicesY = np.arange(1)

    def jit_precompute_kernels(x):
        return precompute_kernels(x, kernel=kernel, **kwargs)

    @jit
    def precompX(k):
        return jit_precompute_kernels(X[:, k])

    @jit
    def precompY(k):
        return jit_precompute_kernels(y[:, k])

    Kx = vmap(precompX)(indicesX)
    Ky = vmap(precompY)(indicesY)
    return Kx, Ky


def compute_distance_for_am(X, y, **kwargs):
    # mostly for DC
    _, p = X.shape
    # mostly for HSIC and cMMD
    indicesX = np.arange(p)
    indicesY = np.arange(1)
    p = kwargs["order_x"] if "order_x" in kwargs.keys() else 2
    q = kwargs["order_x"] if "order_y" in kwargs.keys() else 2

    def jit_precompute_dist_x(x):
        return pdist_p(x, x, p=p)

    def jit_precompute_dist_y(x):
        return pdist_p(x, x, p=q)

    @jit
    def precompX(k):
        return jit_precompute_dist_x(X[:, k])

    @jit
    def precompY(k):
        return jit_precompute_dist_y(y[:, k])

    Kx = vmap(precompX)(indicesX)
    Ky = vmap(precompY)(indicesY)
    return Kx, Ky


def threshold_alpha(Ws, w_indice, alpha):
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
    ts = np.sort(abs(Ws))

    def fraction_3_6(t):
        num = (Ws <= -abs(t)).sum() + 1
        den = max((Ws >= abs(t)).sum(), 1)
        return num / den

    fraction_3_6_v = npy.vectorize(fraction_3_6)
    fdp = fraction_3_6_v(ts)

    t_alpha = np.where(fdp <= alpha)
    if t_alpha[0].size == 0:
        # no one selected..
        print("alpha_min set to np.inf, no one selected...")
        t_alpha_min = np.inf
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

    return indices, t_alpha
