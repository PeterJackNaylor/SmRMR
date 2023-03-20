from abc import abstractmethod
import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit

import utils as u


class SklearnModel:
    def __init__(self, model, model_type, config_name, name):
        u.set_random_state()

        self.model = model
        self.type = model_type
        self.config_name = config_name
        self.name = name
        if self.type == "classification":
            self.scoring = "accuracy"
        else:
            self.scoring = "neg_mean_squared_error"

    def train(self, train_npz, scores_npz, params_file):

        X, y, featnames = u.read_data(train_npz, scores_npz)
        featnames = np.squeeze(featnames)
        X = X[:, featnames]
        param_grid = u.read_parameters(params_file, self.config_name, self.name)

        self.clf = GridSearchCV(self.model, param_grid, scoring=self.scoring)
        self.clf.fit(X, y)

        self.best_hyperparams = {k: self.clf.best_params_[k] for k in param_grid.keys()}

        scores = self.score_features()
        scores = u.sanitize_vector(scores)
        selected = self.select_features(scores)

        u.save_scores_npz(featnames, selected, scores, self.best_hyperparams)
        u.save_scores_tsv(featnames, selected, scores, self.best_hyperparams)

    def train_validate(self, train_npz, val_npz, scores_npz, params_file):

        X, y, featnames, selected = u.read_data(train_npz, scores_npz)
        self.model_input_features = selected
        # featnames = np.squeeze(featnames)
        if len(featnames.shape) >= 2:
            featnames = featnames[0]
            # if featnames.shape[1] > 1:
            #     featnames = np.squeeze(featnames)
            # else:
            #     featnames =
        X_val, y_val, _, _ = u.read_data(val_npz, scores_npz)
        X = X[:, selected]
        X_val = X_val[:, selected]
        param_grid = u.read_parameters(params_file, self.config_name, self.name)
        X_join = np.concatenate((X, X_val), axis=0)
        y_join = np.concatenate((y, y_val), axis=0)
        split_index = np.concatenate(
            (np.zeros(len(y), dtype=int) - 1, np.zeros(len(y_val), dtype=int)), axis=0
        )
        pds = PredefinedSplit(test_fold=split_index)

        self.clf = GridSearchCV(self.model, param_grid, cv=pds, scoring=self.scoring)
        self.clf.fit(X_join, y_join)

        self.best_hyperparams = {k: self.clf.best_params_[k] for k in param_grid.keys()}

        scores = self.score_features()
        scores = u.sanitize_vector(scores)
        self_selected = self.select_features(scores)

        featnames = featnames[self_selected]
        selected_feats = np.zeros(shape=(selected.shape[0],), dtype=bool)

        if list(featnames):
            selected_feats[np.array(featnames)] = True
        self.select_features = selected_feats

        u.update_save_scores_npz(
            featnames,
            selected_feats,
            scores,
            self.best_hyperparams,
            scores_npz,
            "new_scores.npz",
        )
        # u.save_scores_npz(featnames, selected_feats, scores, self.best_hyperparams)
        # u.save_scores_tsv(featnames, selected_feats, scores, self.best_hyperparams)

    def predict_proba(self, test_npz, scores_npz):

        X_test, _, _, _ = u.read_data(test_npz, scores_npz)

        y_proba = self.clf.predict_proba(X_test[:, self.model_input_features])
        u.save_proba_npz(y_proba, self.best_hyperparams)

    def predict(self, test_npz, scores_npz):

        X_test, _, _, _ = u.read_data(test_npz, scores_npz)

        y_pred = self.clf.predict(X_test[:, self.model_input_features])
        u.save_preds_npz(y_pred, self.best_hyperparams)

    @abstractmethod
    def score_features(self):
        raise NotImplementedError

    @abstractmethod
    def select_features(self, scores):
        raise NotImplementedError
