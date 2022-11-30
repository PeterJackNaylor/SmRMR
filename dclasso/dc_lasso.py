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
from .association_measures.distance_correlation import pdist_p, fill_diagonal
from .penalties import pic_penalty
from .utils import (
    knock_off_check_parameters,
    get_equi_features,
    generate_random_sets,
)

from optax._src.base import GradientTransformation

# General comments
# - Private methods should be named __method_name

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
        alpha: float = 0.2,
        measure_stat: str = "HSIC",
        kernel: str = "gaussian",
        ms_kwargs: dict = {},
        normalise_input: bool = True,
        hard_alpha: bool = True,
        alpha_increase: float = 0.05,
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
        self.hard_alpha = hard_alpha
        self.alpha_increase = alpha_increase
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
        max_epoch: int = 301,
        eps_stop: float = 1e-8,
        init="from_convex_solve",
        data_recycling: bool = True,
        optimizer: str = "adam",
        penalty_kwargs: dict = {"name": "l1", "lamb": 0.5},
        opt_kwargs: dict = {
            "init_value": 0.001,
            "transition_steps": 100,
            "decay_rate": 0.99,
        },
        conservative: bool = True,
    ):
        self.verbose = True
        key = random.PRNGKey(seed)

        X, y = check_X_y(X, y)
        X = np.asarray(X)
        n, p = X.shape

        (ny,) = y.shape
        assert n == ny
        y = np.asarray(y)

        self.n_features_in_ = p

        # we have to prescreen and split if needed
        X2, y2, X1, y1, self.screen_indices_, d = self.screen_split(
            X,
            y,
            n,
            p,
            n1,
            d,
            penalty_kwargs["name"],
            key,
        )

        # Compute knock-off variables
        Xhat = get_equi_features(X2, key)

        # TODO is there any downside to always do data recycling?
        if data_recycling:
            # TODO why are we not computing the knockoffs for X1?
            X1_tild = np.concatenate([X1, X1], axis=1)
            X2_tild = np.concatenate([X2, Xhat], axis=1)
            Xs = np.concatenate([X1_tild, X2_tild], axis=0)
            ys = np.concatenate([y1, y2], axis=0)
        else:
            Xs = np.concatenate([X2, Xhat], axis=1)
            ys = y2

        self.beta_, value = self.minimize_loss_function(
            Xs,
            ys,
            penalty_kwargs,
            optimizer,
            init,
            opt_kwargs,
            max_epoch,
            eps_stop,
            key,
        )

        self.wjs_ = self.beta_[:d] - self.beta_[d:]
        alpha_thres = alpha_threshold(
            self.alpha,
            self.wjs_,
            self.screen_indices_,
            hard_alpha=self.hard_alpha,
            alpha_increase=self.alpha_increase,
            conservative=conservative,
            verbose=self.verbose,
        )

        self.alpha_indices_ = alpha_thres[0]
        self.t_alpha_ = alpha_thres[1]
        self.n_features_out_ = alpha_thres[2]
        self.final_loss_ = value

        return self

    def minimize_loss_function(
        self, x, y, pen_kwargs, optimizer, init, opt_kwargs, max_epoch, eps_stop, key
    ):
        if self.verbose:
            print("Starting to compute the loss")
        # Penalty/Loss setting
        loss_fn = self.compute_loss_fn(x, y, pen_kwargs)
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
        if self.verbose:
            print("Starting to optimise the loss")
        beta, value = minimize_loss(
            step_function,
            opt_state,
            beta,
            max_epoch,
            eps_stop,
            verbose=self.verbose,
        )
        if self.verbose:
            print("The loss has been optimised")
        return beta, value

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
        max_epoch: int = 301,
        eps_stop: float = 1e-8,
        init: str = "from_convex_solve",
        data_recycling: bool = False,
        minimize_val_loss: bool = False,
        evaluate_function: Callable[
            [npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike], float
        ] = None,
        refit: bool = True,
        conservative: bool = True,
    ):

        key = random.PRNGKey(seed)
        X, y = check_X_y(X, y)
        X = np.asarray(X)
        y = np.asarray(y)
        X_val, y_val = check_X_y(X_val, y_val)
        y_val = np.asarray(y_val)
        X_val = np.asarray(X_val)
        n, p = X.shape
        (ny,) = y.shape
        assert n == ny
        assert X_val.shape[0] == y_val.shape[0]
        assert p == X_val.shape[1]

        self.n_features_in_ = p

        generator_ms_kernel, hyperparameter_generator = gene_generators(
            param_grid, self.measure_stat, self.kernel
        )

        dict_scores = {}

        for ms, kernel in generator_ms_kernel:
            self.measure_stat = ms
            self.kernel = kernel
            # we have to prescreen and split if needed
            if len(param_grid["penalty"]) == 1:
                pen = param_grid["penalty"][0]
            else:
                pen = "None"
            X2, y2, X1, y1, screen_indices, d = self.screen_split(
                X,
                y,
                n,
                p,
                n1,
                d,
                pen,
                key,
            )
            Xhat = get_equi_features(X2, key)
            if data_recycling:
                X1_tild = np.concatenate([X1, X1], axis=1)
                X2_tild = np.concatenate([X2, Xhat], axis=1)
                Xs = np.concatenate([X1_tild, X2_tild], axis=0)
                ys = np.concatenate([y1, y2], axis=0)
            else:
                Xs = np.concatenate([X2, Xhat], axis=1)
                ys = y2

            self.ms_kwargs = precompute_kernels_match(
                ms, Xs, ys, kernel, self.ms_kwargs, self.normalise_input
            )

            Dxy = self._compute_assoc(Xs, ys, **self.ms_kwargs)
            Dxx = self._compute_assoc(Xs, **self.ms_kwargs)

            if "precompute" in self.ms_kwargs:
                del self.ms_kwargs["precompute"]

            if minimize_val_loss:

                Dxy_val = self._compute_assoc(
                    X_val[:, screen_indices], y_val, **self.ms_kwargs
                )
                Dxx_val = self._compute_assoc(
                    X_val[:, screen_indices], **self.ms_kwargs
                )

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

                selected_features, threshold, nout, adapted_alpha = alpha_threshold(
                    self.alpha,
                    wj,
                    screen_indices,
                    hard_alpha=self.hard_alpha,
                    alpha_increase=self.alpha_increase,
                    return_alpha=True,
                    conservative=conservative,
                    verbose=self.verbose,
                )
                name = (
                    f"ms={ms}_kernel={kernel}_optimizer={optimizer}"
                    + f"_penalty={penalty_kwargs['name']}_lamb="
                    + f"{penalty_kwargs['lamb']}_lr={opt_kwargs['init_value']}"
                )
                if nout:
                    if minimize_val_loss:
                        loss_val = partial(
                            loss,
                            Dxy=Dxy_val,
                            Dxx=Dxx_val,
                            penalty_func=pic_penalty({"name": "None"}),
                        )
                        score = float(loss_val(beta[:d]))
                    elif evaluate_function:
                        score = evaluate_function(
                            X2[:, selected_features],
                            y2,
                            X_val[:, selected_features],
                            y_val,
                        )
                    else:
                        raise NotImplementedError("No valid way of cross-validating")
                    dict_scores[name] = (
                        score,
                        np.array(selected_features),
                        adapted_alpha,
                        threshold,
                        wj.copy(),
                    )
                else:
                    dict_scores[name] = (np.inf, [], 1.0, 0.0, [])

        # Selected best set of features, best score and closest to alpha
        best_score = np.inf
        best_features = []
        best_wj = []
        for key, item in dict_scores.items():
            score, selected_features, aalpha, threshold, wj = item
            if len(selected_features):
                if score < best_score:
                    best_score = score
                    best_features = selected_features.tolist()
                    best_wj = wj[wj >= float(threshold)]

        if refit:
            if best_features:
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
        if self.verbose:
            print(f"Starting initialisation: {init}")
        match init:
            case "random":
                beta = random.uniform(
                    key, shape=(p,), dtype="float32", minval=0.0, maxval=2.0
                )
            case "from_convex_solve":
                # without penalty
                Dxx_minus_1 = np.linalg.inv(Dxx)
                eps = 1e-6
                while np.isnan(Dxx_minus_1).any():
                    Dxx_minus_1 = np.linalg.inv(Dxx + eps * np.eye(Dxx.shape[0]))
                    eps *= 10
                beta = np.matmul(Dxx_minus_1, Dxy)
                beta = pos_proj(beta)
        if self.verbose:
            print(f"Initialisation {init} done")
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

    def marginal_screen(self, X, y, d):

        Dxy = self._compute_assoc(X, y, **self.ms_kwargs)
        screened_indices = top_k(Dxy, d)[1]

        return screened_indices

    def feature_feature_screen(self, X, y, d, penalty, key):

        Dxy = self._compute_assoc(X, y, **self.ms_kwargs)
        Dxx = self._compute_assoc(X, **self.ms_kwargs)

        # Default parameters
        max_epoch = 200
        eps_stop = 1e-8
        init = "from_convex_solve"
        penalty_kwargs = {"name": f"{penalty}", "lamb": 0.5}
        optimizer = "adam"
        opt_kwargs = {
            "init_value": 0.001,
            "transition_steps": 100,
            "decay_rate": 0.99,
        }
        loss_fn = partial(
            loss, Dxy=Dxy, Dxx=Dxx, penalty_func=pic_penalty(penalty_kwargs)
        )
        step_function, opt_state, beta = self.setup_optimisation(
            loss_fn, optimizer, X.shape[1], key, init, Dxx, Dxy, opt_kwargs
        )
        beta, _ = minimize_loss(
            step_function,
            opt_state,
            beta,
            max_epoch,
            eps_stop,
            verbose=self.verbose,
        )
        screened_indices = selected_top_k(beta, d)
        return screened_indices

    def feature_feature_screen_nonzeros(self, X, y, d, penalty, key):

        Dxy = self._compute_assoc(X, y, **self.ms_kwargs)
        Dxx = self._compute_assoc(X, **self.ms_kwargs)

        # Default parameters
        max_epoch = 150
        eps_stop = 1e-8
        init = "from_convex_solve"
        penalty_kwargs = {"name": f"{penalty}", "lamb": 0.5}
        optimizer = "adam"
        opt_kwargs = {
            "init_value": 0.001,
            "transition_steps": 100,
            "decay_rate": 0.99,
        }
        loss_fn = partial(
            loss, Dxy=Dxy, Dxx=Dxx, penalty_func=pic_penalty(penalty_kwargs)
        )
        step_function, opt_state, beta = self.setup_optimisation(
            loss_fn, optimizer, X.shape[1], key, init, Dxx, Dxy, opt_kwargs
        )
        beta, _ = minimize_loss(
            step_function,
            opt_state,
            beta,
            max_epoch,
            eps_stop,
            verbose=self.verbose,
        )

        nonzeros = np.where(beta != 0)[0]
        screened_indices = beta.argsort()[-nonzeros.shape[0] :][::-1]
        return screened_indices

    def screen_split(self, X, y, n, p, n1, d, penalty, key):
        """
        If needed, screen the features of X and y and split X, y.
        """

        stop, screening, msg, d = knock_off_check_parameters(n, p, n1, d)

        if stop:
            raise ValueError(msg)

        if screening:
            # 1 - pre-screening by only checking the marginals
            if self.verbose:
                print("Starting screening")

            # randomly split the samples
            s1, s2 = generate_random_sets(n, n1, key)
            X1, y1 = X[s1, :], y[s1]
            X2, y2 = X[s2, :], y[s2]

            # if too many features, select the top K = 4 * d
            # TODO rename d=? and p=n_features?
            screened_indices = np.arange(p)
            if 4 * d < p:
                screened_indices = self.marginal_screen(X1, y1, 4 * d)
                X1 = X1[:, screened_indices]
                X2 = X2[:, screened_indices]

            # TODO what does this do?
            screened_indices_2 = self.feature_feature_screen(X2, y2, d, penalty, key)
            X1 = X1[:, screened_indices_2]
            X2 = X2[:, screened_indices_2]
            screened_indices = screened_indices[screened_indices_2]

        else:
            if self.verbose:
                print("Only non-zero screening")
            # screened_indices = np.arange(p)
            # TODO same as above
            screened_indices = self.feature_feature_screen_nonzeros(
                X, y, d, penalty, key
            )
            d = screened_indices.shape[0]

            X = X[:, screened_indices]

            X1, y1 = np.zeros(shape=(0, d)), np.zeros(shape=(0,))
            X2, y2 = X, y

        return X2, y2, X1, y1, screened_indices, d

    def compute_loss_fn(self, X, y, penalty_kwargs):

        self.ms_kwargs = precompute_kernels_match(
            self.measure_stat, X, y, self.kernel, self.ms_kwargs, self.normalise_input
        )
        # made as self so that they can re-used for the
        # initialisation
        self.Dxy = self._compute_assoc(X, y, **self.ms_kwargs)
        self.Dxx = self._compute_assoc(X, **self.ms_kwargs)

        if "precompute" in self.ms_kwargs:
            del self.ms_kwargs["precompute"]

        loss_fn = partial(
            loss, Dxy=self.Dxy, Dxx=self.Dxx, penalty_func=pic_penalty(penalty_kwargs)
        )

        return loss_fn


def precompute_kernels_match(measure_stat, X, y, kernel, ms_kwargs, normalise_input):
    if measure_stat in ["HSIC", "DC"] and normalise_input:
        X = X / np.linalg.norm(X, ord=2, axis=0)

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


def minimize_loss(
    step_function, opt_state, beta, max_epoch, eps_stop, patience=5, verbose=False
):
    # error_tmp = []
    prev = np.inf
    i = 0
    # Minimizing loss function
    range_epoch = trange(max_epoch) if verbose else range(max_epoch)
    for _ in range_epoch:
        value, beta, opt_state = step_function(beta, opt_state)
        # error_tmp.append(float(value))
        if abs(value - prev) < eps_stop / patience:
            i += 1
            if i == patience:
                break
        else:
            i = 0
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

    def pdist_A(x, y, p, unbiased=False):
        n = X.shape[0]
        A = pdist_p(x, y, p)
        if unbiased:
            A_1 = np.tile(A.sum(axis=1), (n, 1)) / (n - 2)
            A_0 = np.tile(A.sum(axis=0), (1, n)).reshape(n, n, order="F") / (n - 2)
            A = A - A_0 - A_1 + A.sum() / ((n - 1) * (n - 2))
            A = fill_diagonal(A, 0)
        else:
            A_1 = np.tile(A.mean(axis=1), (n, 1))
            A_0 = np.tile(A.mean(axis=0), (1, n)).reshape(n, n, order="F")
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

    ts = np.sort(abs(Ws))

    add = 1 if conservative else 0

    def fraction_3_6(t):
        num = (Ws <= -abs(t)).sum() + add
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


def selected_top_k(beta, d):
    return np.argsort(beta)[-d:][::-1]
