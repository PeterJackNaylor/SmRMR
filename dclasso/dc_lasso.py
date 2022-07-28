from functools import partial
from collections.abc import Callable
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
from .penalties import pic_penalty
from .utils import (
    knock_off_check_parameters,
    get_equi_features,
    generate_random_sets,
)

from optax._src.base import GradientTransformation

available_am = ["PC", "DC", "TR", "HSIC", "cMMD", "pearson_correlation"]
kernel_am = ["HSIC", "cMMD"]
available_kernels = [
    "distance",
    "gaussian",
    "laplacian",
    "linear",
    "sigmoid",
    "tanh",
    "inverse-M",
]


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
        ms_kwargs: dict = {},
        normalise_input: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        assert measure_stat in available_am, "measure_stat incorrect"
        if measure_stat in kernel_am:
            assert kernel in available_kernels, "kernel incorrect"
        self.measure_stat = measure_stat
        self.kernel = kernel
        self.ms_kwargs = ms_kwargs
        self.normalise_input = normalise_input  # default value
        self.verbose = verbose

    def _compute_assoc(self, x, y=None, **kwargs):

        args = {}
        if self.measure_stat in kernel_am:
            args["kernel"] = self.kernel

        if self.normalise_input:
            x = x / np.linalg.norm(x, ord=2, axis=0)
        args["verbose"] = self.verbose
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
        optimizer: str = "SGD",
        penalty_kwargs: dict = {"name": "None", "lamb": 0.5},
        opt_kwargs: dict = {
            "init_value": 0.001,
            "transition_steps": 100,
            "decay_rate": 0.99,
        },
    ):

        key = random.PRNGKey(seed)

        X, y = check_X_y(X, y)
        X = np.asarray(X)
        n, p = X.shape

        (ny,) = y.shape
        assert n == ny
        y = np.asarray(y)

        self.n_features_in_ = p

        # we have to prescreen and split if needed
        X2, y2, self.screen_indices_, d = self.screen_split(X, y, n, p, n1, d, key)

        # Compute knock-off variables
        Xhat = get_equi_features(X2, key)

        if mode == "competitive":
            Xs = [np.concatenate([X2, Xhat], axis=1)]
        else:
            Xs = [X2, Xhat]

        # Iterate on X if competitive
        betas = []
        for x in Xs:
            # Penalty/Loss setting
            loss_fn = self.compute_loss_fn(x, y2, penalty_kwargs)
            step_function, opt_state, beta = self.setup_optimisation(
                loss_fn,
                optimizer,
                x.shape[1],
                key,
                init,
                self.Dxx,
                self.Dxy,
                opt_kwargs,
            )
            beta, value = minimize_loss(
                step_function,
                opt_state,
                beta,
                max_epoch,
                eps_stop,
                verbose=self.verbose,
            )
            betas.append(beta)

        if mode == "competitive":
            self.beta_ = betas[0]
        else:
            self.beta_ = np.concatenate([betas[0], betas[1]], axis=0)

        self.wjs_ = self.beta_[:d] - self.beta_[d:]
        alpha_thres = alpha_threshold(
            self.alpha, self.wjs_, self.screen_indices_, verbose=self.verbose
        )

        self.alpha_indices_ = alpha_thres[0]
        self.t_alpha_ = alpha_thres[1]
        self.n_features_out_ = alpha_thres[2]
        self.final_loss_ = value

        return self

    def cv_fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        X_val: npt.ArrayLike,
        y_val: npt.ArrayLike,
        param_grid: dict,
        n1: float,
        d: int = None,
        seed: int = 42,
        max_epoch: int = 151,
        eps_stop: float = 1e-8,
        mode: str = "competitive",
        init="from_convex_solve",
        evaluate_function: Callable[
            [npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike], float
        ] = None,
        refit: bool = True,
    ):
        if mode != "competitive":
            raise NotImplementedError("Only competitive mode is implemented")
        key = random.PRNGKey(seed)
        X, y = check_X_y(X, y)
        X = np.asarray(X)
        X_val, y_val = check_X_y(X_val, y_val)
        X_val = np.asarray(X_val)
        n, p = X.shape

        (ny,) = y.shape
        assert n == ny
        assert X_val.shape[0] == y_val.shape[0]
        assert p == X_val.shape[1]
        y = np.asarray(y)

        self.n_features_in_ = p

        generator_ms_kernel, hyperparameter_generator = gene_generators(
            param_grid, self.measure_stat, self.kernel
        )

        best_score = np.inf
        best_features = []
        best_wj = []
        dict_scores = {}
        for ms, kernel in generator_ms_kernel:
            self.measure_stat = ms
            self.kernel = kernel
            X2, y2, screen_indices, d = self.screen_split(X, y, n, p, n1, d, key)
            Xhat = get_equi_features(X2, key)
            self.ms_kwargs = precompute_kernels_match(ms, X, y, kernel, self.ms_kwargs)
            Xs = np.concatenate([X2, Xhat], axis=1)
            Dxy = self._compute_assoc(Xs, y2, **self.ms_kwargs)
            Dxx = self._compute_assoc(Xs, **self.ms_kwargs)

            for hyper in hyperparameter_generator:
                if self.verbose:
                    print(hyper)
                optimizer = hyper["optimizer"]
                penalty_kwargs = hyper["penalty_kwargs"]
                opt_kwargs = hyper["opt_kwargs"]
                loss_fn = partial(
                    loss, Dxy=Dxy, Dxx=Dxx, penalty_func=pic_penalty(penalty_kwargs)
                )
                step_function, opt_state, beta = self.setup_optimisation(
                    loss_fn, optimizer, Xs.shape[1], key, init, Dxx, Dxy, opt_kwargs
                )
                beta, _ = minimize_loss(
                    step_function,
                    opt_state,
                    beta,
                    max_epoch,
                    eps_stop,
                    verbose=self.verbose,
                )
                wj = beta[:d] - beta[d:]
                selected_features, threshold, nout = alpha_threshold(
                    self.alpha, wj, screen_indices, verbose=self.verbose
                )
                name = (
                    f"ms={ms}_kernel={kernel}_optimizer={optimizer}"
                    + f"_penalty={penalty_kwargs['name']}_lamb="
                    + f"{penalty_kwargs['lamb']}_lr={opt_kwargs['init_value']}"
                )
                if nout:
                    score = evaluate_function(
                        X2[:, selected_features], y2, X_val[:, selected_features], y_val
                    )
                    dict_scores[name] = (score, selected_features)
                    if score < best_score:
                        best_score = score
                        best_features = selected_features
                        best_wj = wj[wj >= threshold]
                else:
                    dict_scores[name] = (np.inf, [])
        if refit:
            if nout:
                self.alpha_indices_ = np.array(best_features)
            else:
                print("Warning: no features selected")
        return best_score, best_features, best_wj, dict_scores

    def transform(self, X: npt.ArrayLike) -> npt.ArrayLike:
        if hasattr(self, "alpha_indices_"):
            return X[:, self.alpha_indices_]
        else:
            print("Warning: not fitted, doing nothing..")
            return X

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

        return self.fit(X, y, **fit_params).transform(X)

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

    def _more_tags(self):
        return {"stateless": True}

    def setup_optimisation(self, loss_fn, opt, p, key, init, Dxx, Dxy, opt_kwargs):
        """
        Sets up the loss function for the optimisation.
        """
        beta = self.initialisation(p, key, init, Dxx, Dxy)
        step_function, opt_state = self.set_optimizer(opt, loss_fn, beta, opt_kwargs)
        return step_function, opt_state, beta

    def initialisation(self, p, key, init, Dxx, Dxy):
        match init:
            case "random":
                beta = random.uniform(
                    key, shape=(p,), dtype="float32", minval=0.0, maxval=2.0
                )
            case "from_convex_solve":
                # without penalty
                beta = np.matmul(np.linalg.inv(Dxx), Dxy)
                beta = pos_proj(beta)
        return beta

    def set_optimizer(self, opt, loss_fn, beta, opt_kwargs):
        optimizer = get_optimizer(opt, opt_kwargs)
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

    def screen_split(self, X, y, n, p, n1, d, key):
        """
        If needed, screen the features of X and y and split X, y.
        """

        stop, screening, msg, d = knock_off_check_parameters(n, p, n1, d)

        if stop:
            raise ValueError(msg)

        if screening:
            if self.verbose:
                print("Starting screening")

            s1, s2 = generate_random_sets(n, n1, key)
            X1, y1 = X[s1, :], y[s1]
            X2, y2 = X[s2, :], y[s2]

            screened_indices = self.screen(X1, y1, d)
            X2 = X2[:, screened_indices]

        else:
            if self.verbose:
                print("No screening")
            X2, y2 = X, y
            screened_indices = np.arange(p)

        return X2, y2, screened_indices, d

    def compute_loss_fn(self, X, y, penalty_kwargs):
        self.ms_kwargs = precompute_kernels_match(
            self.measure_stat, X, y, self.kernel, self.ms_kwargs
        )

        # made as self so that they can re-used for the
        # initialisation
        self.Dxy = self._compute_assoc(X, y, **self.ms_kwargs)
        self.Dxx = self._compute_assoc(X, **self.ms_kwargs)

        loss_fn = partial(
            loss, Dxy=self.Dxy, Dxx=self.Dxx, penalty_func=pic_penalty(penalty_kwargs)
        )

        return loss_fn


def precompute_kernels_match(measure_stat, X, y, kernel, ms_kwargs):
    match measure_stat:
        case "HSIC" | "cMMD":
            ms_kwargs["precompute"] = compute_kernels_for_am(X, y, kernel, **ms_kwargs)
        case "DC":
            ms_kwargs["precompute"] = compute_distance_for_am(X, y, **ms_kwargs)
    return ms_kwargs


def alpha_threshold(alpha, wjs, indices, verbose=True):
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
    alpha_indices_, t_alpha_ = threshold_alpha(wjs, indices, alpha, verbose)
    if verbose:
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


def loss(b, Dxy, Dxx, penalty_func):
    xy_term = -(b * Dxy).sum()
    xx_term = 0.5 * (b * (Dxx * b).T).sum()
    return xy_term + xx_term + penalty_func(b)


def minimize_loss(step_function, opt_state, beta, max_epoch, eps_stop, verbose):
    # error_tmp = []
    prev = np.inf
    # Minimizing loss function
    range_epoch = trange(max_epoch) if verbose else range(max_epoch)
    for _ in range_epoch:
        value, beta, opt_state = step_function(beta, opt_state)
        # error_tmp.append(float(value))
        if abs(value - prev) < eps_stop:
            break
        else:
            prev = value
    return beta, value


def get_optimizer(opt, opt_kwargs):
    scheduler = optax.exponential_decay(**opt_kwargs)

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
        return jit_precompute_kernels(y)

    Kx = vmap(precompX)(indicesX)
    Ky = vmap(precompY)(indicesY)
    return Kx, Ky


def compute_distance_for_am(X, y, **kwargs):
    # mostly for DC
    _, p = X.shape
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
        return jit_precompute_dist_y(y)

    Kx = vmap(precompX)(indicesX)
    Ky = vmap(precompY)(indicesY)
    return Kx, Ky


def threshold_alpha(Ws, w_indice, alpha, verbose=True):
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
        if verbose:
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

    return indices, t_alpha_min


def gene_generators(param_grid, measure_stat, kernel):
    def ms_kernel_gen(param_grid, measure_stat, kernel):
        if "measure_stat" in param_grid:
            measure_stat = param_grid["measure_stat"]
        else:
            measure_stat = [measure_stat]
        if "kernel" in param_grid:
            kernel = param_grid["kernel"]
        else:
            kernel = [kernel]
        for ms in measure_stat:
            if ms in kernel_am:
                for k in kernel:
                    yield (ms, k)
            else:
                yield (ms, kernel[0])

    def hp_gen(param_grid):
        if "lambda" not in param_grid:
            lambda_ = [0.5]
        else:
            lambda_ = param_grid["lambda"]

        if "penalty" not in param_grid:
            penalty = ["l1"]
        else:
            penalty = param_grid["penalty"]
            if not isinstance(penalty, list):
                penalty = [penalty]

        if "learning_rate" not in param_grid:
            learning_rate = [0.001]
        else:
            learning_rate = param_grid["learning_rate"]

        if "optimizer" not in param_grid:
            optimizer = ["adam"]
        else:
            optimizer = param_grid["optimizer"]
            if not isinstance(optimizer, list):
                optimizer = [optimizer]

        dic = {}
        dic["penalty_kwargs"] = {}
        dic["opt_kwargs"] = {"transition_steps": 100, "decay_rate": 0.99}
        for pen in penalty:
            dic["penalty_kwargs"]["name"] = pen
            for lr in learning_rate:
                dic["opt_kwargs"]["init_value"] = lr
                for opt in optimizer:
                    dic["optimizer"] = opt
                    for lam in lambda_:
                        dic["penalty_kwargs"]["lamb"] = lam
                        if (
                            pen == "None"
                            and dic["penalty_kwargs"]["lamb"] == lambda_[0]
                        ):
                            dic["penalty_kwargs"]["lamb"] = 0
                            yield dic
                        elif pen != "None" and dic["penalty_kwargs"]["lamb"] != 0:
                            yield dic

    return ms_kernel_gen(param_grid, measure_stat, kernel), hp_gen(param_grid)
