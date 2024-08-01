#!/usr/bin/env python
"""
Input variables:
    - PARAMS: model parameters
    - Y_TEST: path to numpy array with validation Y vector.
    - Y_PRED: path to numpy array with prediction vector.
Output files:
    - prediction_stats: path to a single-line tsv with the TSV results.
"""

import numpy as np
from sklearn.metrics import confusion_matrix

import utils as u

causal = np.load("${CAUSAL_NPZ}")["selected"]
selected = np.load("${SCORES_NPZ}")["selected"]

score = np.nan
if len(causal):
    tn, fp, fn, tp = confusion_matrix(causal, selected, labels=[0, 1]).ravel()
    if tp + fn != 0:
        tpr = tp / (tp + fn)
    else:
        tpr = "NA"

    if fp + tn != 0:
        fpr = fp / (fp + tn)
    else:
        fpr = 0

    if tp + fp != 0:
        fdr = fp / (tp + fp)
    else:
        fdr = 0
    n_selected = selected.sum()
else:
    tpr = fpr = fdr = n_selected = "NA"

u.save_analysis_tsv(
    run="${PARAMS}",
    metric=["tpr_causal", "fpr_causal", "fdr_causal", "n_selected"],
    value=[tpr, fpr, fdr, n_selected],
)
