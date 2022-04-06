#!/usr/bin/env python
"""
Input variables:
    - DATA_NPZ: path to a .npz file containing three elements: an X matrix, a y vector,
    and a featnames vector (optional)
    - CAUSAL_NPZ: path to a .npz file containing true covariates
    - AM: string, association measure to use
    - KERNEL: string, kernel to be used
Output files:
    - performance.tsv: path to a single-line tsv with the TSV results.
"""

import numpy as np
from sklearn.utils.validation import check_X_y

from dclasso import DCLasso
from dclasso.association_measures.kernel_tools import check_vector
import utils as u


def find_minimum_size_model(cf, order_idx):
    for i in range(len(cf), len(order_idx)):
        result = all(elem in order_idx[:i] for elem in cf)
        if result:
            break
    return i


am_kernels = ["HSIC", "cMMD"]

# Read data
############################
X, y, featnames = u.read_data("${DATA_NPZ}")
X, y = check_X_y(X, y)
X = np.asarray(X)
n, p = X.shape

(ny,) = y.shape
assert n == ny
y = check_vector(y)

p = X.shape[1]
causal_feats = np.load("${CAUSAL_NPZ}")
causal_feats = causal_feats["featnames"][causal_feats["selected"]]


# Hyper-parameters
############################
am = "${AM}"
kernel = "${KERNEL}" if am in am_kernels else ""

dl = DCLasso(alpha="", measure_stat=am, kernel=kernel, penalty="", optimizer="")

# Start screening
############################
indices_in_order = dl.screen(X, y, p)
n_min = find_minimum_size_model(causal_feats, indices_in_order)

# Save results
############################
simu, _ = "${PARAMS}".split("(")
u.save_analysis_tsv(
    run=simu,
    n=n,
    p=p,
    AM=am,
    kernel=kernel,
    metric=["minimal_model_size"],
    value=[n_min],
)
