from typing import List, Tuple, Dict

import numpy as np
from scipy.spatial.distance import euclidean
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

from satree.clustering.literals import create_literal_matrices_modular


def solve_wcnf_clustering(wcnf: WCNF) -> List[int]:
    """
    Solve the weighted CNF clustering problem using a Partial MaxSAT solver.

    Args:
        wcnf: The weighted CNF object containing the clustering clauses.

    Returns:
        A list of literal indices representing the SAT model, or an empty list if no solution was found.
    """
    solver = RC2(wcnf)
    solution = solver.compute()
    return solution if solution is not None else []


def assign_clusters_and_diameters(x_i_c_matrix: np.ndarray,
                                  dataset: np.ndarray,
                                  k_clusters: int) -> Tuple[Dict[int, List[int]], Dict[int, float]]:
    """
    Assign clusters to data points based on the cluster assignment matrix and compute the maximum diameter for each cluster.

    Args:
        x_i_c_matrix: A matrix representing cluster assignments for data points.
        dataset: The original dataset with data points.
        k_clusters: The total number of clusters.

    Returns:
        A tuple containing:
          - A dictionary mapping cluster IDs to lists of data point indices.
          - A dictionary mapping cluster IDs to the maximum diameter (largest pairwise distance) within that cluster.
    """
    # Assign clusters based on unique patterns in the x_i_c_matrix
    unique_patterns = np.unique(x_i_c_matrix, axis=0)
    pattern_to_cluster = {tuple(pattern): cluster_id for cluster_id, pattern in enumerate(unique_patterns)}

    cluster_assignments = {cluster_id: [] for cluster_id in range(k_clusters)}
    for data_point_index, pattern in enumerate(x_i_c_matrix):
        cluster_id = pattern_to_cluster[tuple(pattern)]
        cluster_assignments[cluster_id].append(data_point_index)

    # Calculate the maximum diameter for each cluster
    cluster_diameters = {}
    for cluster_id, data_points in cluster_assignments.items():
        max_diameter = 0
        # Calculate all pairwise distances within the cluster
        for i in range(len(data_points)):
            for j in range(i + 1, len(data_points)):
                dist = euclidean(dataset[data_points[i]], dataset[data_points[j]])
                max_diameter = max(max_diameter, dist)
        cluster_diameters[cluster_id] = max_diameter

    return cluster_assignments, cluster_diameters


def process_clustering_solution(wcnf: WCNF,
                                literals: Dict[str, int],
                                dataset: np.ndarray,
                                features: np.ndarray,
                                k_clusters: int,
                                branch_nodes: List[int],
                                leaf_nodes: List[int],
                                distance_classes: List[np.ndarray]) -> Tuple[
    Dict[int, List[int]], Dict[int, float], List[int]]:
    """
    Solve the clustering SAT problem and process the solution to obtain cluster assignments and diameters.

    Args:
        wcnf: The weighted CNF object containing clustering clauses.
        literals: A dictionary mapping literal names to variable indices.
        dataset: The dataset containing data points.
        features: An array of feature identifiers.
        k_clusters: The total number of clusters.
        branch_nodes: Indices of branching nodes.
        leaf_nodes: Indices of leaf nodes.
        distance_classes: List of arrays for each distance class.

    Returns:
        A tuple containing:
          - A dictionary mapping cluster IDs to lists of data point indices.
          - A dictionary mapping cluster IDs to the maximum diameter of each cluster.
          - The SAT solver's solution as a list of literal indices.
    """
    solution = solve_wcnf_clustering(wcnf)

    a_matrix, s_matrix, z_matrix, g_matrix, x_i_c_matrix, bw_m_vector, bw_p_vector = create_literal_matrices_modular(
        literals=literals,
        solution=solution,
        dataset_size=len(dataset),
        k_clusters=k_clusters,
        branch_nodes=branch_nodes,
        leaf_nodes=leaf_nodes,
        num_features=len(features),
        distance_classes=distance_classes,
        bicriteria=True
    )

    cluster_assignments, cluster_diameters = assign_clusters_and_diameters(
        x_i_c_matrix, dataset, k_clusters
    )

    return cluster_assignments, cluster_diameters, solution
