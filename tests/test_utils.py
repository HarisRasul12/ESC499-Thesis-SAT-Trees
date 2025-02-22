import numpy as np
import pytest
from satree.utils import k_fold_tester
from satree.SATreeCraft import SATreeCraft


# Monkey-patch the SATreeCraft.solve method to avoid heavy solving.
@pytest.fixture(autouse=True)
def patch_satreecraft(monkeypatch):
    def dummy_solve(self):
        # Dummy model that makes a simple decision:
        # if feature[0] <= 10 then label 0 else label 1.
        self.model = [
            {'type': 'branching', 'feature': '0', 'threshold': 10, 'children': [1, 2]},
            {'type': 'leaf', 'label': 0},
            {'type': 'leaf', 'label': 1}
        ]

    monkeypatch.setattr(SATreeCraft, "solve", dummy_solve)


def test_k_fold_tester():
    features = np.array(['0', '1'])
    labels = np.array([0, 1])
    dataset = np.array([[5, 100], [15, 200], [5, 150], [15, 250]])
    true_labels = np.array([0, 1, 0, 1])
    accuracies, mean_score = k_fold_tester(k=2, depth=1, dataset=dataset,
                                           true_labels_for_points=true_labels,
                                           labels=labels, features=features)
    # Our dummy model correctly predicts: if feature[0] <= 10 → 0; else → 1.
    # In our dataset the correct predictions are achieved.
    assert mean_score == 1.0
