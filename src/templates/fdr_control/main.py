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
from scipy.sparse.linalg import eigsh

try:
    from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence
except ImportError:
    from scipy.sparse.linalg import ArpackNoConvergence
from dclasso import DCLasso, pic_penalty
from dclasso.dc_lasso import alpha_threshold

import utils as u


def fdr(causal_features, selected_features):

    if len(selected_features) == 0:
        print("No feature selection process happened")
        return -1
    else:
        n_selected = len(selected_features)
        intersection = list(set(selected_features) & set(causal_features))
        number_of_correct_positives = len(intersection)
        fdr = (n_selected - number_of_correct_positives) / n_selected
        return fdr


am_kernels = ["HSIC", "cMMD"]

# Read data
############################
X, y, featnames, _ = u.read_data("${DATA_NPZ}")
X, y = check_X_y(X, y)
X = np.asarray(X)
n, p = X.shape

(ny,) = y.shape
assert n == ny

p = X.shape[1]
causal_feats = np.load("${CAUSAL_NPZ}")
causal_feats = list(causal_feats["featnames"][causal_feats["selected"]])


# Hyper-parameters
############################
am = "${AM}"
kernel = "${KERNEL}" if am in am_kernels else ""
optimizer = "${OPTIMIZER}"
penalty_kwargs = {"name": "${PENALTY}", "lamb": float("${LAMBDA}")}

# File parameters
alpha_list = np.arange(0.1, 1, 0.1)
fdr_alpha = []
selected_variables = []


# Process

dl = DCLasso(alpha=0.1, measure_stat=am, kernel=kernel)

dl.fit(X, y, n1=0.5, max_epoch=300, penalty_kwargs=penalty_kwargs, optimizer=optimizer)
selected = list(np.array(dl.alpha_indices_))
loss_value = float(dl.final_loss_)
fdr_ = fdr(causal_feats, selected)
fdr_alpha.append(fdr_)
selected_variables.append(selected)

for alpha in alpha_list[1:]:

    selected_features, _, _ = alpha_threshold(alpha, dl.wjs_, dl.screen_indices_)
    selected_features = list(np.array(selected_features))
    fdr_alpha.append(fdr(causal_feats, selected_features))
    selected_variables.append(selected_features)

# Some theoritical results
try:
    alpha2 = float(eigsh(np.array(dl.Dxx), k=1, which="SA")[0].squeeze())
except ArpackNoConvergence:
    alpha2 = 0
Cst = 1.5
R = float(Cst / penalty_kwargs["lamb"] * pic_penalty(penalty_kwargs)(dl.beta_))
N1 = np.abs(dl.beta_).sum()
# Save results
############################
simu, _ = "${PARAMS}".split("(")
u.save_analysis_tsv(
    run=simu,
    n=n,
    p=p,
    AM=am,
    kernel=kernel,
    penalty=penalty_kwargs["name"],
    optimizer=optimizer,
    metric="fdr",
    lamb=penalty_kwargs["lamb"],
    alpha=alpha_list,
    value=fdr_alpha,
    selected=selected_variables,
    loss=loss_value,
    alpha2=alpha2,
    R=R,
    norm_1=N1,
)
