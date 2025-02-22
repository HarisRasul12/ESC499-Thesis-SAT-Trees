import subprocess
import numpy as np
import pytest
from satree.loandra_support.loandra import run_loandra_and_parse_results, transform_tree_from_loandra


def dummy_subprocess_run(args, stdout, stderr, text):
    print(args, stdout, stderr, text)

    # Simulate Loandra output:
    # One cost line and one model line.
    class DummyCompletedProcess:
        def __init__(self):
            self.stdout = "o 0\nv 101010\ns OPTIMUM FOUND\n"
            self.stderr = ""

    return DummyCompletedProcess()


@pytest.fixture(autouse=True)
def patch_subprocess(monkeypatch):
    monkeypatch.setattr(subprocess, "run", dummy_subprocess_run)


def test_run_loandra_and_parse_results():
    model, cost = run_loandra_and_parse_results("dummy_path", "dummy_execution.cnf")
    # For "101010", the dummy code produces:
    expected_model = [1, -2, 3, -4, 5, -6]
    assert model == expected_model
    assert cost == 0


def test_transform_tree_from_loandra():
    dummy_model = [1, -2, 3, -4]
    dummy_literals = {'g_0_0': 1, 'g_0_1': 2, 'a_0_0': 3}
    dummy_leaf_indices = [0]
    dummy_tree_structure = [{'type': 'leaf', 'label': None}]
    dummy_labels = [0, 1]
    dummy_features = np.array(['0'])
    transform_tree_from_loandra(dummy_model, dummy_literals, dummy_leaf_indices, dummy_tree_structure, dummy_labels,
                                dummy_features)
    # Our dummy model has literal 1 (which is the value for 'g_0_0') so label 0 should be set.
    assert dummy_tree_structure[0]['label'] == 0
