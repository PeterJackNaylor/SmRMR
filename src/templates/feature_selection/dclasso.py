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

from base.sklearn import SklearnModel
from dclasso import DCLasso


class DCLassoModel(SklearnModel):
    def __init__(self) -> None:
        dl = DCLasso()
        super().__init__(dl, "feature_selection", "dc_lasso")

    def score_features(self):
        return self.wjs_

    def select_features(self):
        return self.alpha_indices_


if __name__ == "__main__":
    model = DCLassoModel()
    model.train("${TRAIN_NPZ}", "", "${PARAMS_FILE}")
