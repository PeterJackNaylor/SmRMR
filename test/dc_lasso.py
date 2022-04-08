from sklearn.utils.estimator_checks import check_estimator

from dclasso.dc_lasso import DCLasso


def test_sklearn():

    check_estimator(DCLasso())
