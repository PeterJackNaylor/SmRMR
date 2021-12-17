import numpy.typing as npt
from collections import namedtuple
from tqdm import trange

import jax.numpy as np
from jax import jit, vmap, value_and_grad, random
from jax.example_libraries import optimizers

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y

from . import association_measures as am
from .association_measures.hsic import precompute_kernels
from .association_measures.kernel_tools import check_vector
from .penalties import penalty_dic


OptimizerState = namedtuple(
    "OptimizerState", ["packed_state", "tree_def", "subtree_defs"]
)

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
        self.lambda_ = 1.0
        self.ms_kwargs = ms_kwargs if ms_kwargs else {}
        self.opt_kwargs = opt_kwargs if opt_kwargs else {}

    def get_assoc(self, x, y=None, **kwargs):

        args = {}
        if self.measure_stat in kernel_am:
            args["kernel"] = self.kernel

        if self.normalise_input:
            x = x / np.linalg.norm(x, ord=2, axis=0)

        assoc_func = self.get_association_measure()
        return assoc_func(x, y, **args, **kwargs)

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        seed: int = 42,
    ):

        if seed:
            key = random.PRNGKey(seed)
        print("maybe here is a good place to add knockoff generation..")
        X, y = check_X_y(X, y)
        X = np.asarray(X)
        n, p = X.shape

        (ny,) = y.shape
        assert n == ny
        y = check_vector(y)

        loss_fn = self.compute_loss_fn(X, y)
        step_function, opt_state, get_beta = self.set_optimizer(loss_fn, p, key)

        for epoch in trange(150):
            value, opt_state = step_function(epoch, opt_state)

        self.beta_ = get_beta(opt_state)

        import pdb

        pdb.set_trace()
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

    def get_association_measure(self):
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
        match opt:
            case  "SGD":
                init, update, params = optimizers.sgd(
                    self.learning_rate, **self.opt_kwargs
                )
            case "adam":
                init, update, params = optimizers.adam(
                    self.learning_rate, **self.opt_kwargs
                )
            case opt if opt.__class__ == OptimizerState:
                init, update, params = opt
            case _:
                error_msg = f"Unkown optimizer: {opt}, should be known string or of OptimizerState class"
                raise ValueError(error_msg)

        return init, update, params

    def set_optimizer(self, loss_fn, p, key):
        opt_init, opt_update, get_params = self.get_optimizer()
        beta = random.uniform(key, shape=(p,), dtype="float32", minval=0.0, maxval=5.0)
        opt_state = opt_init(beta)

        def step(step, opt_state):
            value, grads = value_and_grad(loss_fn)(get_params(opt_state))
            opt_state = opt_update(step, grads, opt_state)
            states_flat, tree, subtrees = opt_state
            # we might have to be careful of the ordering with the positivity constraint
            new_states_flat = (apply_at(pos_proj, [0], states_flat[0]),)
            opt_state = OptimizerState(new_states_flat, tree, subtrees)
            return value, opt_state

        return step, opt_state, get_params

    def compute_loss_fn(self, X, y):
        D = self.get_assoc

        if self.measure_stat in kernel_am:
            self.ms_kwargs["precompute"] = compute_kernels_for_am(
                X, y, self.kernel, **self.ms_kwargs
            )
        # D = jit(D)
        Dxy = D(X, y, **self.ms_kwargs)
        # import pdb; pdb.set_trace()
        Dxx = D(X)#, **self.ms_kwargs)
        def loss(b):
            xy_term = (b * Dxy).sum()
            xx_term = (b * (Dxx * b).T).sum()
            return xy_term + xx_term / 2 + self.penalty_func(b)

        return loss


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
