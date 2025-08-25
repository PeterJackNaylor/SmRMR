from functools import partial

# from collections.abc import Callable
import numpy.typing as npt
from numpy import asarray
import jax.numpy as np
from jax.lax import top_k
from jax import random  # value_and_grad,

import optuna
import cvxpy as cp

# from .penalties_cvx import mcp_cvx, huber

# import optax
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y

from . import association_measures as am

from .penalties import pic_penalty, pic_derivative
from .utils import (
    knock_off_check_parameters,
    get_equi_features,
    generate_random_sets,
    # argmin_lst,
    # pos_proj,
    # get_optimizer,
    # minimize_loss,
    selected_top_k,
    precompute_kernels_match,
    alpha_threshold,
    shift_until_PSD,
    # gene_generators,
)


# General comments
# - Private methods should be named __method_name

available_ms = ["PC", "DC", "TR", "HSIC", "cMMD", "pearson_correlation"]
kernel_ms = ["HSIC", "cMMD"]
available_kernels = [
    "distance",
    "gaussian",
    "laplacian",
    "linear",
    "sigmoid",
    "tanh",
    "inverse-M",
]


class smrmr(BaseEstimator, TransformerMixin):
    """
    The smrmr object is a transformer from the sklearn
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
        if measure_stat is not None:
            assert measure_stat in available_ms, "measure_stat incorrect"
        if kernel is not None:
            if measure_stat in kernel_ms:
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
        if self.measure_stat in kernel_ms:
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
        # max_epoch: int = 301,
        # eps_stop: float = 1e-8,
        data_recycling: bool = True,
        # optimizer: str = "adam",
        penalty_kwargs: dict = {"name": "l1", "lamb": 0.5},
        # opt_kwargs: dict = {
        #     "init_value": 0.001,
        #     "transition_steps": 100,
        #     "decay_rate": 0.99,
        # },
        conservative: bool = True,
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

        if data_recycling:
            X1_tild = np.concatenate([X1, X1], axis=1)
            X2_tild = np.concatenate([X2, Xhat], axis=1)
            Xs = np.concatenate([X1_tild, X2_tild], axis=0)
            ys = np.concatenate([y1, y2], axis=0)
        else:
            Xs = np.concatenate([X2, Xhat], axis=1)
            ys = y2

        self.compute_matrix(Xs, ys)
        self.beta_, value = self.minimize_loss_function(
            penalty_kwargs,
            # optimizer,
            # opt_kwargs,
            # max_epoch,
            # eps_stop,
            # key,
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
        self,
        pen_kwargs
        # , optimizer, opt_kwargs, max_epoch, eps_stop
    ):
        if self.verbose:
            print("Starting to compute the loss")
        # Penalty/Loss setting

        beta, value = self.cvx_solve(pen_kwargs)

        # The optimal value for x is stored in `x.value`.
        if pen_kwargs["name"] in ["scad", "mcp"]:
            if self.verbose:
                print("Starting to optimise with scad//mcp")
            beta, value = self.cvx_solve(pen_kwargs, init_beta=beta)
        if self.verbose:
            print("The loss has been optimised")
        return beta, value

    def cvx_solve(self, pen_kwargs, init_beta=None):
        p = self.Dxx.shape[1]
        theta = cp.Variable(p)
        constraints = [theta >= 0]
        if init_beta is None:
            regulariser = "l1"
        else:
            regulariser = pen_kwargs["name"]
        match regulariser:
            case "None":
                reg = 0
            case "l1":
                reg = pen_kwargs["lamb"] * cp.norm(theta, 1)
            case "scad" | "mcp":
                lamb_theta = pic_derivative(pen_kwargs)(init_beta)
                reg = cp.sum(lamb_theta @ cp.abs(theta))
        self.Dxx = shift_until_PSD(self.Dxx, 0.0001)
        objective = cp.Minimize(
            -self.Dxy.T @ theta + 0.5 * cp.quad_form(theta, self.Dxx) + reg
        )
        problem = cp.Problem(objective, constraints)
        result = problem.solve()
        beta = theta.value
        value = result

        # set to 0 values that are too small with respect to the rest
        if beta is None:
            beta = 0
            value = np.inf
        else:
            threshold = float(np.abs(beta).max()) / 1e5
            beta[beta < threshold] = 0

        return beta, value

    def cv_fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        X_val: npt.ArrayLike,
        y_val: npt.ArrayLike,
        penalty: str,
        lambda_range: list,
        n1: float,
        d: int = None,
        n_trials: int = 50,
        seed: int = 42,
        data_recycling: bool = False,
        conservative: bool = True,
        **pen_kwargs,
    ):
        self.verbose = True
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
        X2, y2, X1, y1, screen_indices, d = self.screen_split(
            X, y, n, p, n1, d, penalty, key, **pen_kwargs
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

        self.compute_matrix(Xs, ys)

        Dxy_val = self._compute_assoc(X_val[:, screen_indices], y_val, **self.ms_kwargs)
        Dxx_val = self._compute_assoc(X_val[:, screen_indices], **self.ms_kwargs)
        val_penalty = {"name": "None"}
        loss_val = partial(
            loss, Dxy=Dxy_val, Dxx=Dxx_val, penalty_func=pic_penalty(val_penalty)
        )
        if penalty != "None":

            def objective(trial):
                lambda_ = trial.suggest_float(
                    "lambda", float(lambda_range[0]), float(lambda_range[1]), log=True
                )
                penalty_kwargs = {"name": penalty, "lamb": lambda_}
                penalty_kwargs.update(pen_kwargs)
                beta, train_loss = self.minimize_loss_function(
                    penalty_kwargs,
                )
                validation_loss = float(loss_val(beta[:d]))
                return validation_loss

            study = optuna.create_study(
                direction="minimize", sampler=optuna.samplers.TPESampler()
            )
            study.optimize(objective, n_trials=n_trials)
            self.best_lambda = study.best_params["lambda"]
            val_loss = study.best_value
        else:
            self.best_lambda = 0

        penalty_kwargs = {"name": penalty, "lamb": self.best_lambda}
        beta, train_loss = self.minimize_loss_function(
            penalty_kwargs,
        )
        if penalty == "None":
            val_loss = float(loss_val(beta[:d]))

        self.beta_ = beta
        self.screen_indices_ = screen_indices
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
        self.feature_score = self.wjs_[asarray(self.wjs_ >= self.t_alpha_)]

        if not self.n_features_out_:
            print("No features selected, taking best feat")
            idx = np.argmax(self.wjs_)
            self.alpha_indices_ = self.screen_indices_[idx : idx + 1]
            self.feature_score = self.wjs_[idx : idx + 1]
            self.n_features_out_ = 1

        return train_loss, val_loss

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

    def marginal_screen(self, X, y, d):
        Dxy = self._compute_assoc(X, y, **self.ms_kwargs)
        screened_indices = top_k(Dxy, d)[1]

        return screened_indices

    def feature_feature_screen(self, X, y, d, penalty, key, **pen_kwargs):

        self.compute_matrix(X, y)
        penalty_kwargs = {"name": f"{penalty}", "lamb": 5.0}
        penalty_kwargs.update(**pen_kwargs)
        beta = np.zeros(X.shape[1])
        iteration = 0
        while int(np.count_nonzero(beta)) < d and iteration < 5:
            iteration += 1
            penalty_kwargs["lamb"] /= 10
            beta, _ = self.cvx_solve(penalty_kwargs)
            if penalty_kwargs["name"] in ["scad", "mcp"]:
                beta, _ = self.cvx_solve(penalty_kwargs, init_beta=beta)

        screened_indices = selected_top_k(beta, d)
        return screened_indices

    def screen_split(self, X, y, n, p, n1, d, penalty, key, **pen_kwargs):
        """
        If needed, screen the features of X and y and split X, y.
        """

        stop, screening, msg, d = knock_off_check_parameters(n, p, n1, d)

        if stop:
            raise ValueError(msg)

        screened_indices = np.arange(p)
        if screening:
            # 1 - pre-screening by only checking the marginals
            if self.verbose:
                print("Starting screening")

            # randomly split the samples
            s1, s2 = generate_random_sets(n, n1, key)
            X1, y1 = X[s1, :], y[s1]
            X2, y2 = X[s2, :], y[s2]

            # if too many features, select the top K = 4 * d
            # First marginal screening
            # TODO rename d=? and p=n_features?
            if 4 * d < p:
                screened_indices = self.marginal_screen(X1, y1, 4 * d)
                X1 = X1[:, screened_indices]
                X2 = X2[:, screened_indices]

            # TODO what does this do?
            # Second screening in order to reduce even further the number of features
            # this time, it takes into account feature feature relations
            screened_indices_2 = self.feature_feature_screen(
                X2, y2, d, penalty, key, **pen_kwargs
            )
            X1 = X1[:, screened_indices_2]
            X2 = X2[:, screened_indices_2]
            screened_indices = screened_indices[screened_indices_2]

        else:
            d = screened_indices.shape[0]

            X1, y1 = np.zeros(shape=(0, d)), np.zeros(shape=(0,))
            X2, y2 = X, y

        return X2, y2, X1, y1, screened_indices, d

    def compute_matrix(self, X, y):

        self.ms_kwargs = precompute_kernels_match(
            self.measure_stat, X, y, self.kernel, self.ms_kwargs, self.normalise_input
        )
        # made as self so that they can re-used for the
        # initialisation
        self.Dxy = self._compute_assoc(X, y, **self.ms_kwargs)
        self.Dxx = self._compute_assoc(X, **self.ms_kwargs)

        if "precompute" in self.ms_kwargs:
            del self.ms_kwargs["precompute"]

    def compute_loss_fn(self, penalty_kwargs):
        loss_fn = partial(
            loss, Dxy=self.Dxy, Dxx=self.Dxx, penalty_func=pic_penalty(penalty_kwargs)
        )

        return loss_fn


def loss(b, Dxy, Dxx, penalty_func):
    xy_term = -(b * Dxy).sum()
    xx_term = 0.5 * (b * (Dxx * b).T).sum()
    return xy_term + xx_term + penalty_func(b)
