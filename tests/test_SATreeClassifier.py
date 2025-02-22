import numpy as np
import pytest
from satree.SATreeClassifier import SATreeClassifier


@pytest.fixture
def simple_tree_model():
    # A very simple tree: if feature[0] <= 10, go left and return label 0; otherwise return label 1.
    return [
        {'type': 'branching', 'feature': '0', 'threshold': 10, 'children': [1, 2]},
        {'type': 'leaf', 'label': 0},
        {'type': 'leaf', 'label': 1}
    ]


@pytest.fixture
def classifier(simple_tree_model):
    return SATreeClassifier(simple_tree_model)


def test_predict(classifier):
    # For two samples, the first with feature 5 (<=10) should yield 0 and the second with feature 15 (>10) yield 1.
    data = np.array([[5, 100], [15, 200]])
    predictions = classifier.predict(data)
    np.testing.assert_array_equal(predictions, np.array([0, 1]))


def test_score(classifier):
    data = np.array([[5, 100], [15, 200]])
    y_true = np.array([0, 1])
    score = classifier.score(data, y_true)
    assert score == 1.0


def test_classification_report(classifier):
    data = np.array([[5, 100], [15, 200]])
    y_true = np.array([0, 1])
    report = classifier.get_classification_report(data, y_true)
    assert "precision" in report


def test_confusion_matrix(classifier):
    data = np.array([[5, 100], [15, 200]])
    y_true = np.array([0, 1])
    cm = classifier.get_confusion_matrix(data, y_true)
    assert cm.shape == (2, 2)
