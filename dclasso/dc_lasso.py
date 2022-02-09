import numpy.typing as npt
from collections import namedtuple
from tqdm import trange
import numpy as npy
import jax.numpy as np
from jax.lax import top_k
from jax import jit, vmap, value_and_grad, random, grad
from jax.example_libraries import optimizers

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y

from . import association_measures as am
from .association_measures.hsic import precompute_kernels
from .association_measures.kernel_tools import check_vector
from .penalties import penalty_dic
from .utils import generate_random_sets, knock_off_check_parameters, get_equi_features, generate_random_sets


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
        self.learning_rate = 0.01
        self.lambda_ = 0.15 * .2
        self.ms_kwargs = ms_kwargs if ms_kwargs else {}
        self.opt_kwargs = opt_kwargs if opt_kwargs else {}

    def D(self, x, y=None, **kwargs):

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
        n1: float,
        d: int = None,
        seed: int = 42,
        max_epoch: int = 1500,
        eps_stop:float = 1e-8,
    ):
        self.precomputed_elements = False
        if seed:
            key = random.PRNGKey(seed)


        ## we have to prescreen and split
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

            screened_indices = self.screen(
                X1, y1, d
            )
            X2 = X2[:, screened_indices]

        else:
            print("No screening")
            X2, y2 = X, y
            screened_indices = np.arange(p)

        # Compute knock-off variables
        Xhat = get_equi_features(X2, key)
        X2 = np.concatenate([X2, Xhat], axis=1)

        loss_fn = self.compute_loss_fn(X2, y2)
        step_function, opt_state, get_beta = self.set_optimizer(loss_fn, 2*d, key)
        
        error_tmp = []
        beta_max = []
        beta_min = []
        prev = np.inf
        for epoch in trange(max_epoch):
            value, opt_state = step_function(epoch, opt_state)
            error_tmp.append(float(value))
            beta_max.append(float(get_beta(opt_state).max()))
            beta_min.append(float(get_beta(opt_state).min()))
            if abs(value - prev) < eps_stop:
                break
            else:
                prev = value

        self.beta_ = get_beta(opt_state)

        self.wjs_ =   self.beta_[:d] - self.beta_[d:]
        alpha_thres = alpha_threshold(self.alpha, self.wjs_, screened_indices)

        self.alpha_indices_ = alpha_thres[0]
        self.t_alpha_ = alpha_thres[1]
        self.n_features_out_ = alpha_thres[2]
        
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
                error_msg = f"Unkown optimizer: {opt}, should be a known string or of OptimizerState class"
                raise ValueError(error_msg)

        return init, update, params

    def set_optimizer(self, loss_fn, p, key):
        opt_init, opt_update, get_params = self.get_optimizer()
        beta = random.uniform(key, shape=(p,), dtype="float32", minval=0.0, maxval=5.0)
        opt_state = opt_init(beta)

        def step(step, opt_state):
            # value, grads = value_and_grad(loss_fn)(get_params(opt_state))
            grads = grad(loss_fn)(get_params(opt_state))
            opt_state = opt_update(step, grads, opt_state)
            states_flat, tree, subtrees = opt_state
            # we might have to be careful of the ordering with the positivity constraint
            new_states_flat = (apply_at(pos_proj, [0], states_flat[0]),)
            opt_state = OptimizerState(new_states_flat, tree, subtrees)
            value = loss_fn(get_params(opt_state))
            return value, opt_state

        return step, opt_state, get_params

    def screen(self, X, y, d):

        Dxy = self.D(X, y, **self.ms_kwargs)
        screened_indices = top_k(Dxy, d)[1]
        return screened_indices

    def compute_loss_fn(self, X, y):
        if self.measure_stat in kernel_am:
            self.ms_kwargs["precompute"] = compute_kernels_for_am(
                X, y, self.kernel, **self.ms_kwargs
            )

        Dxy = self.D(X, y, **self.ms_kwargs)
        Dxx = self.D(X, **self.ms_kwargs)

        def loss(b):
            xy_term = - (b * Dxy).sum()
            xx_term = 0.5 * (b * (Dxx * b).T).sum()
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
    alpha_indices_, t_alpha_ = threshold_alpha(
        wjs, indices, alpha
    )
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
    indices = w_indice[np.where(Ws >= t_alpha_min)[0]]
    import pdb; pdb.set_trace()
    return indices, t_alpha