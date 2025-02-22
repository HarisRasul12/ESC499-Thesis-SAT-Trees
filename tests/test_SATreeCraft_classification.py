import numpy as np
import pytest
from satree.SATreeCraft import SATreeCraft


@pytest.fixture
def dummy_classification_solution(monkeypatch):
    # Replace the min-height method with a dummy that returns a fixed tree model.
    def dummy_find_min_depth_tree_problem(*_args, **_kwargs):
        tree = [
            {'type': 'branching', 'feature': '0', 'threshold': 10, 'children': [1, 2]},
            {'type': 'leaf', 'label': 0},
            {'type': 'leaf', 'label': 1}
        ]
        literals = {'a_0_0': 1}
        depth = 1
        solution = [1]  # dummy solution
        cnf = None
        return tree, literals, depth, solution, cnf

    monkeypatch.setattr(
        "satree.SATreeCraft.SATreeCraft.find_min_depth_tree_problem",
        dummy_find_min_depth_tree_problem
    )
    return dummy_find_min_depth_tree_problem


@pytest.fixture
def dummy_satreecraft(dummy_classification_solution):
    # Use a very simple dataset.
    dataset = np.array([[5, 100], [15, 200]])
    features = np.array(['0', '1'])
    labels = np.array([0, 1])
    true_labels = np.array([0, 1])
    craft = SATreeCraft(dataset=dataset, features=features, labels=labels, true_labels_for_points=true_labels)
    craft.solve()  # Our dummy method is called here.
    return craft


def test_satreecraft_classification(dummy_satreecraft):
    model = dummy_satreecraft.model
    # Ensure the model is a list containing at least one branching node and at least one leaf.
    branching_nodes = [node for node in model if node['type'] == 'branching']
    leaf_nodes = [node for node in model if node['type'] == 'leaf']
    assert len(branching_nodes) == 1
    assert len(leaf_nodes) == 2
