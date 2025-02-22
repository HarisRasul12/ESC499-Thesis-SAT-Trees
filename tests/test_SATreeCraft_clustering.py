import numpy as np
import pytest
from satree.SATreeCraft import SATreeCraft


@pytest.fixture
def dummy_clustering_solution(monkeypatch):
    def dummy_solve_clustering_problem_max_diameter(*_args, **_kwargs):
        # Return dummy clustering results.
        cluster_assignments = {0: [0, 1], 1: [2, 3]}
        cluster_diameters = {0: 1.0, 1: 2.0}
        literals = {'dummy': 1}
        solution = [1, -2, 3]
        return cluster_assignments, cluster_diameters, literals, solution

    monkeypatch.setattr(
        "satree.SATreeCraft.SATreeCraft.solve_clustering_problem_max_diameter",
        dummy_solve_clustering_problem_max_diameter
    )
    return dummy_solve_clustering_problem_max_diameter


@pytest.fixture
def dummy_satreecraft_clustering(dummy_clustering_solution):
    dataset = np.array([[1, 1], [1, 2], [7, 7], [7, 8]])
    features = np.array(['0', '1'])
    # Here we specify a clustering problem via k_clusters and fixed_depth.
    craft = SATreeCraft(dataset=dataset, features=features, k_clusters=2, fixed_depth=2)
    craft.solve()  # This uses our dummy clustering solver.
    return craft


def test_satreecraft_clustering(dummy_satreecraft_clustering):
    assert hasattr(dummy_satreecraft_clustering, "cluster_assignments")
    assert hasattr(dummy_satreecraft_clustering, "cluster_diameters")
    assignments = dummy_satreecraft_clustering.cluster_assignments
    diameters = dummy_satreecraft_clustering.cluster_diameters
    assert isinstance(assignments, dict)
    assert isinstance(diameters, dict)
