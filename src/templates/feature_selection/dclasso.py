#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing three elements: an X matrix, a y vector,
    and a featnames vector (optional)
  - PARAMS_FILE: path to a json file with the hyperparameters
    - num_feat
    - weight_decay
    - penalisation
    - optimizer
    - learning rate
    - covars
Output files:
  - selected.npz: contains the featnames of the selected features, their scores and the
    hyperparameters selected by cross-validation
  - selected.tsv: like selected.npz, but in tsv format.
"""

from dclasso import DCLasso

import utils as u

ms_kernels = ["HSIC", "cMMD"]

# Read data
############################
X, y, featnames = u.read_data("${TRAIN_NPZ}")
param_grid = u.read_parameters("${PARAMS_FILE}", "dclasso")
print(param_grid)

measure_stats = param_grid['measure_stat']
alphas = param_grid['alpha']
optimizers = param_grid['optimizer']
kernels = param_grid['kernel']
penalties = param_grid['penalty']
n1 = param_grid['n1'][0]
# Run algorithm
############################

for ms in measure_stats:
  for alpha in alphas:
    for opt in optimizers:
      for penalty in penalties:
        kernel_loop = kernels if ms in ms_kernels else [kernels[0]]
        for kernel in kernel_loop:
          dl = DCLasso(
            alpha = alpha,
            measure_stat = ms,
            kernel = kernel,
            penalty = penalty,
            optimizer = opt
          )
          dl.fit(X, y, n1, d=None)
