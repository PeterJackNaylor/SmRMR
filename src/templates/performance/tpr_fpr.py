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

_, y, _, _ = u.read_data("${TEST_NPZ}", "")
y_pred = np.load("${PRED_NPZ}")["preds"]

score = np.nan
if len(y_pred):
    labels = list(np.unique(y).astype(int))
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=labels).ravel()
    acc = (tn + tp) / (tn + fp + fn + tp)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    f1 = 2 * tp / (2 * tp + fp + fn)

else:
    tpr = fpr = "NA"

u.save_analysis_tsv(
    run="${PARAMS}", metric=["acc", "tpr", "fpr", "f1"], value=[acc, tpr, fpr, f1]
)
