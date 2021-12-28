import jax.numpy as np


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
