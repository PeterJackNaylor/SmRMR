import jax.numpy as np


def precompute_kernels(X, Y=None, kernel="gaussian", sigma=None, center_kernel=True):
    kernel, kernel_params = get_kernel_function(kernel, nfeats=sigma)
    if center_kernel:
        Kx = center(kernel(X, Y, **kernel_params))
    else:
        Kx = kernel(X, Y, **kernel_params)
    return Kx


def get_kernel_function(name, nfeats=1):
    """
    Get the correct kernel function given the name.
    For the gaussian kernel nfeats designates sigma.
    Parameters
    ----------
    name: string, could be gaussian, linear or distance.

    nfeats: None or scalar, if None, it is set as the median
        of the distance matrix of the input, otherwise it is
        set as sqrt(nfeats)

    Returns
    -------
    A tuple, where the first element is the kernel function
    and the second it's hyper parameter dictionnary.
    """
    match name:
        case "gaussian":
            kernel = kernel_gaussian
            if nfeats is not None:
                kernel_params = {"sigma": np.sqrt(nfeats)}
            else:
                kernel_params = {"sigma": None}
        case "laplace":
            kernel = kernel_laplace
            if nfeats is not None:
                kernel_params = {"sigma": np.sqrt(nfeats)}
            else:
                kernel_params = {"sigma": None}
        case "tanh":
            kernel = kernel_tanh
        case "inverse-M":
            kernel = kernel_inverse_M

        case "linear":
            kernel = kernel_linear
            kernel_params = {}
        case "distance":
            kernel = kernel_alpha
            kernel_params = {"alpha": 1.0}
        case "sigmoid":
            kernel = kernel_sigmoid
            kernel_params = {}
        case _:
            raise ValueError("No valid kernel.")

    return kernel, kernel_params


def compute_distance_matrix(x1, x2=None):
    """
    Computes the l2 distance matrix between x1 and x2.
    If x2 isn't given, it will set x2 as x1 and compute
    the inner distance matrix of x1.
    Parameters
    ----------
    x1: numpy array like object with one feature where the rows
        designate samples.

    x2: None or like x1.

    Returns
    -------
    The distance matrix of x1 and x2.
    """
    x1 = check_vector(x1)
    x1_2 = np.power(x1, 2)

    x2 = x1 if x2 is None else check_vector(x2)
    x2_2 = np.power(x2, 2)

    dist_2 = x2_2 + x1_2.T - 2 * np.dot(x2, x1.T)
    return dist_2


def compute_l1_distance_matrix(x1, x2=None):
    """
    Computes the l1 distance matrix between x1 and x2.
    If x2 isn't given, it will set x2 as x1 and compute
    the inner distance matrix of x1.
    Parameters
    ----------
    x1: numpy array like object with one feature where the rows
        designate samples.

    x2: None or like x1.

    Returns
    -------
    The distance matrix of x1 and x2.
    """
    x1 = check_vector(x1)

    x2 = x1 if x2 is None else check_vector(x2)

    dist_1 = np.absolute(x2 - x1.T)
    return dist_1


def kernel_gaussian(x1, x2=None, sigma=None):
    """
    Computes the distance matrix with the gaussian kernel.
    If x2 isn't given, it will set x2 as x1 and compute
    the inner distance matrix of x1.
    If sigma is not given, it will be set according to the
    median of the distance matrix.
    Parameters
    ----------
    x1: numpy array like object with one feature where the rows
        designate samples.

    x2: None or like x1.

    sigma: None or float, hyper parameter for the gaussian kernel.
        If set to None, it takes sigma as the median of distance matrix.

    Returns
    -------
    The gaussian kernel of the distance matrix of x1 and x2.
    """
    dist_2 = compute_distance_matrix(x1, x2)
    if sigma is None:
        sigma = 0.5 * np.sqrt(np.var(dist_2))
    K = np.exp(-0.5 * dist_2 / sigma)

    return K


def kernel_laplace(x1, x2=None, sigma=None):
    """
    Computes the distance matrix with the laplacian kernel.
    If x2 isn't given, it will set x2 as x1 and compute
    the inner distance matrix of x1.
    If sigma is not given, it will be set according to the
    median of the distance matrix.
    Parameters
    ----------
    x1: numpy array like object with one feature where the rows
        designate samples.

    x2: None or like x1.

    sigma: None or float, hyper parameter for the gaussian kernel.
        If set to None, it takes sigma as the median of distance matrix.

    Returns
    -------
    The gaussian kernel of the distance matrix of x1 and x2.
    """
    dist_1 = compute_l1_distance_matrix(x1, x2)
    if sigma is None:
        sigma = np.sqrt(np.var(dist_1))
    K = np.exp(dist_1 / sigma)

    return K


def kernel_linear(x1, x2=None):
    """
    Computes the distance matrix with the linear kernel.
    If x2 isn't given, it will set x2 as x1 and compute
    the inner distance matrix of x1.
    Parameters
    ----------
    x1: numpy array like object with one feature where the rows
        designate samples.

    x2: None or like x1.

    Returns
    -------
    The linear kernel of the distance matrix of x1 and x2.
    """
    x1 = check_vector(x1)
    x2 = x1 if x2 is None else check_vector(x2)

    result = np.dot(x2, x1.T)
    return result


def kernel_tanh(x1, x2=None, d=1.0, alpha=1.0):
    """
    Computes the distance matrix with the tanh kernel.
    If x2 isn't given, it will set x2 as x1 and compute
    the inner distance matrix of x1.
    Parameters
    ----------
    x1: numpy array like object with one feature where the rows
        designate samples.

    x2: None or like x1.

    Returns
    -------
    The tanh kernel of the distance matrix of x1 and x2.
    """

    result = np.tanh(kernel_linear(x1, x2) / d + alpha)
    return result


def kernel_inverse_M(x1, x2=None, alpha=1.0, beta=1.0):
    """
    Computes the distance matrix with the inverse-M kernel.
    If x2 isn't given, it will set x2 as x1 and compute
    the inner distance matrix of x1.
    Parameters
    ----------
    x1: numpy array like object with one feature where the rows
        designate samples.

    x2: None or like x1.

    Returns
    -------
    The inverse-M kernel of the distance matrix of x1 and x2.
    """

    x1 = check_vector(x1)
    x2 = x1 if x2 is None else check_vector(x2)

    dist_2 = compute_distance_matrix(x1, x2)

    result = (alpha + dist_2) ** (-beta)
    return result


def kernel_alpha(x1, x2=None, alpha=None):
    """
    Computes the distance matrix with the distance alpha kernel.
    If x2 isn't given, it will set x2 as x1 and compute
    the inner distance matrix of x1.
    If alpha isn't set, it's default value is 1.
    Parameters
    ----------
    x1: numpy array like object with one feature where the rows
        designate samples.

    x2: None or like x1.

    sigma: None or float, hyper parameter for the distance alpha kernel.
        If set to None, it takes the value of 1.

    Returns
    -------
    The distance alpha kernel of the distance matrix of x1 and x2.
    """
    x1 = check_vector(x1)
    x1_alpha = np.power(np.abs(x1), alpha)

    x2 = x1 if x2 is None else check_vector(x2)
    x2_alpha = np.power(np.abs(x2), alpha)

    result = 0.5 * (x1_alpha + x2_alpha.T - np.power(np.abs(x2.T - x1), alpha))
    return result


def kernel_sigmoid(x1, x2=None, gamma=None, coeff0=1):

    x1 = check_vector(x1)
    x2 = x1 if x2 is None else check_vector(x2)
    gamma = 1.0 / len(x1) if gamma is None else gamma

    x = gamma * np.dot(x1, x2.T) + coeff0

    return 1 / (1 + np.exp(-x))


def check_vector(x):
    """
    Checks wether the numpy array x needs to be expended
    to contain a second dimension.
    Parameters
    ----------
    x:  numpy array like object with one feature where the rows
        designate samples.
    Returns
    -------
    The same vector x with an extra dimension if it didn't originally
    have one.
    """
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=1)

    return x


def center(K):
    """
    Matrix centering
    Parameters
    ----------
    x:  square numpy array like object
    Returns
    -------
    The centered matrix.
    """
    n, d = K.shape

    assert n == d

    H = np.eye(n) - 1 / n * np.ones((n, n))
    KH = np.matmul(K, H)

    return KH
