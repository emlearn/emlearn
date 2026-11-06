import numpy
import numpy.testing
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
import emlearn

random = 42

def test_gbc_predict_matches_sklearn():
    X, y = datasets.make_classification(n_samples=100, random_state=random)
    clf = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=random)
    clf.fit(X, y)
    cmodel = emlearn.convert(clf)
    numpy.testing.assert_equal(cmodel.predict(X), clf.predict(X))

def test_gbc_predict_proba_sums_to_one():
    X, y = datasets.make_classification(n_samples=100, random_state=random)
    clf = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=random)
    clf.fit(X, y)
    cmodel = emlearn.convert(clf)
    proba = cmodel.predict_proba(X)
    # probabilities must sum to 1 for each sample
    numpy.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert proba.shape == (len(X), 2)  # binary: 2 Klassen


def test_gbc_save_generates_c_code():
    X, y = datasets.make_classification(n_samples=100, random_state=random)
    clf = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=random)
    clf.fit(X, y)
    cmodel = emlearn.convert(clf)
    code = cmodel.save(name='testmodel')
    assert isinstance(code, str)
    assert 'testmodel_predict_proba' in code
    assert 'testmodel_tree_' in code

def test_gbc_multiclass_matches_sklearn():

    for n_classes in (3, 5):
        X, y = make_classification(
            n_samples=200,
            n_features=6,
            n_informative=5,
            n_redundant=0,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=42,
        )

        model = GradientBoostingClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42,
        )
        model.fit(X, y)

        converted = emlearn.convert(model, method='inline')

        expected = model.predict(X)
        actual = converted.predict(X)

        numpy.testing.assert_array_equal(actual, expected)

        expected_proba = model.predict_proba(X)
        actual_proba = converted.predict_proba(X)

        assert actual_proba.shape == expected_proba.shape
        numpy.testing.assert_allclose(
            actual_proba,
            expected_proba,
            rtol=1e-4,
            atol=1e-5,
        )
