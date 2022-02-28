from .am import AM
import jax.numpy as np

def spearman_JAX(X, Y):
    rs = np.corrcoef(X, Y)
    if rs.shape == (2, 2):
        return rs[1, 0]
    else:
        return rs
        
class pearson_correlation_object(AM):
    def method(self, X, Y):
        rho = np.absolute(spearman_JAX(X, Y))
        return rho

pearson_correlation = pearson_correlation_object()