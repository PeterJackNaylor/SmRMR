import jax.numpy as np
from jax import vmap

# from dclasso.jaxkern.kernels import ard_kernel, linear_kernel, rbf_kernel, rq_kernel


from .jaxkern.dependence import hsic
# , mmd
from .jaxkern import kernels as k
from .jaxkern import sigma as s


available_am = ["PC", "DC", "TR", "HSIC", "cMMD", "pearson_correlation"]
kernel_am = ["HSIC", "cMMD"]
available_kernels = ["distance", "gaussian", "linear"]


class association_measure():
    def method(self, x, y):
        pass

    def __call__(self, x, y, **args):
        """[summary]

        Args:
            x ([type]): [description]
            y ([type]): [description]

        Returns:
            [type]: [description]
        """
        n, d = x.shape
        ny, nd = y.shape

        indice_h, indice_v = np.triu_indices(d, k=0, m=d)

        if nd == 1:
            # We only want the first line
            indice_h = indice_h[:d]
            indice_v = indice_v[:d]

        def func_with_indices(i, j):
            return self.method(x[:, i], y[:, j], **args)

        result = vmap(func_with_indices)(indice_h, indice_v)

        if nd != 1:
            # Reshaping to become square
            result_r = np.zeros((d, nd))
            # fill top triangle
            result_r = result_r.at[(indice_h, indice_v)].set(result)
            # fill bottom triangle
            i_lower = np.tril_indices(d, -1, m=nd)
            result_r = result_r.at[i_lower].set(result_r.T[i_lower])
            result = result_r

        return result


def get_kernel(name):
    match name:
        case "linear":
            return k.linear_kernel
        case "rbf":
            return k.rbf_kernel
        case "ard":
            return k.ard_kernel
        case "rq":
            return k.rq_kernel
        case "distance":
            return k.distance_kernel


class HSIC(association_measure):
    def method(self, x, y, kernel="linear", gamma="adaptative"):
        if isinstance(gamma, str):
            match gamma:
                case "adaptative":
                    gamma_x = s.sigma_to_gamma(s.k_estimate_sigma_median(x))
                    gamma_y = s.sigma_to_gamma(s.k_estimate_sigma_median(y))
                case _:
                    print("Unkown sigma str, set to 1.")
                    gamma = 1.0
        kernel_f = get_kernel(kernel)
        params_x = {'gamma': gamma_x}
        params_y = {'gamma': gamma_y}
        value = hsic(x, y, kernel_f, params_x, params_y, bias=False)
        return value


hsic_d = HSIC()


class dcor(association_measure):
    def method(self, x, y):
        pass


dcor_d = dcor()


def pick_association_measure(name):
    """
    This function returns an association measure
    based on the given name. One could give a custom
    association name but this would need a call function.

    Args:
        name (Str or object): if the string is in the list
        `available_am` it will return the corresponding
        association measure but if it is a

    Returns:
        association_measure object:
    """
    match name:
        case "HSIC":
            return hsic_d
        case "dcor":
            return dcor_d
        case "ard":
            return k.ard_kernel
        case "rq":
            return k.rq_kernel
        case "distance":
            return k.distance_kernel
        case _:
            print(f"Association name {name} not defined")
