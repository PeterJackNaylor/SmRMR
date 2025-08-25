from sklearn.utils.estimator_checks import check_estimator

from smrmr.smrmr_class import smrmr


def test_sklearn():

    check_estimator(smrmr())
