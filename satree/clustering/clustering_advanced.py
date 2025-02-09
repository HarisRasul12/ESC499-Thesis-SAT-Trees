"""
=========== Module Description ===========

Module for solving clustering problems using a complete binary tree and SAT solvers. This module contains functions
to solve clustering problems using a complete binary tree and SAT solvers.
"""

from typing import Any, List, Tuple, Dict
from itertools import combinations
from collections import OrderedDict

import numpy as np
from scipy.spatial.distance import euclidean
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

from satree.clustering.clustering_clauses import construct_clustering_clauses, add_clustering_encodings


def create_distance_classes(dataset: np.ndarray,
                            epsilon: float = 0) -> Tuple[
    OrderedDict[str, List[Tuple[Tuple[int, int], float]]], OrderedDict[str, List[Tuple[int, int]]], List[np.ndarray]]:
    """
    Create non-overlapping distance classes for all unique pairs of data points.

    A distance class groups pairs of points whose distances are less than epsilon apart.
    Each class is labeled starting from D1, D2, ..., Dm, where m is the number of classes.

    Parameters:
    - dataset (array_like): The dataset containing n-dimensional data points.
    - epsilon (float, optional): The maximum distance difference to consider two distances similar.

    Returns:
    - OrderedDict: An ordered dictionary where keys are class labels (D1, D2, ...) and
      values are lists of point index pairs belonging to each distance class.
    """

    def euclidean_distance(p1, p2):
        return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

    distances = {}
    for (idx1, point1), (idx2, point2) in combinations(enumerate(dataset), 2):
        dist = euclidean_distance(point1, point2)
        distances[(idx1, idx2)] = dist

    sorted_distances = sorted(distances.items(), key=lambda item: item[1])

    distance_classes_with_dist = OrderedDict()
    distance_classes_simplified = OrderedDict()
    current_class_label = 1
    for (pair, dist) in sorted_distances:
        placed = False
        for d_class, pairs in distance_classes_with_dist.items():
            class_dist = next(iter(pairs))[1]  # Get the reference distance for this class
            if abs(class_dist - dist) <= epsilon:
                distance_classes_with_dist[d_class].append((pair, dist))
                distance_classes_simplified[d_class].append(pair)
                placed = True
                break
        if not placed:
            distance_classes_with_dist[f'D{current_class_label}'] = [(pair, dist)]
            distance_classes_simplified[f'D{current_class_label}'] = [pair]
            current_class_label += 1

    distance_pairs_array = [np.array(pairs) for pairs in distance_classes_simplified.values()]
    distance_classes = distance_pairs_array

    return distance_classes_with_dist, distance_classes_simplified, distance_classes


def build_clauses_cluster_tree_md(literals: Dict[str, int],
                                  dataset: np.ndarray,
                                  branch_nodes: List[int],
                                  leaf_nodes: List[int],
                                  num_features: int,
                                  k_clusters: int,
                                  cl_pairs: List[Tuple[int, int]],
                                  ml_pairs: List[Tuple[int, int]],
                                  distance_classes: List[np.ndarray]) -> WCNF:
    """
    Constructs the clauses for the SAT solver based on the decision tree encoding. Now includes MAX SOLVER PROBLEM FOR FIXED HEIGHT 

    Args:
        literals (dict): A dictionary mapping literals to variable indices.
        dataset (list): The dataset, a list of tuples representing data points.
        branch_nodes (list): Indices of branching nodes.
        leaf_nodes (list): Indices of leaf nodes.
        num_features (int): Number of features in the dataset.
        k_clusters: number of clusters, will need to turn this into a list for operations on each clause
        cl_pairs (list): Cannot-link pairs.
        ml_pairs (list): Must-link pairs.
        distance_classes (list): list pairs in ecah distace classes  

    Returns:
        wcnf: A wcnf object containing all the clauses, with hard clauses for the tree structure/clustering and soft clauses for maximization
    """
    ##################################################  BASE TREE ENCODINGS ################################################

    # Now the problem has become Partial MaxSAT - we will assign weights to the soft clauses Eq. (13). Eq(1-10,12) HARD clauses
    wcnf = construct_clustering_clauses(literals, dataset, branch_nodes, leaf_nodes, num_features)
    wcnf = add_clustering_encodings(wcnf, literals, dataset, leaf_nodes, k_clusters, cl_pairs, ml_pairs,
                                    distance_classes)

    # Clause 32: Ensures that if bw^-_w is true, then the points in distance class w 
    # cannot be clustered with the points in distance class w-1 if bw^-_(w-1) is false.
    for w in range(1, len(distance_classes)):  # Starting from 1 since we're checking w against w-1
        # print('clause 32: ', [-literals[f'bw_m_{w}'], literals[f'bw_m_{w-1}']])
        wcnf.append([-literals[f'bw_m_{w}'], literals[f'bw_m_{w - 1}']])

    # Clause 37: For each distance class w, we add a soft clause for the corresponding b^-_w literal
    # to encourage points within that class to be clustered separately
    # Max diameter solve problem
    for w in range(len(distance_classes)):
        # print("clause 37: ", [-literals[f'bw_m_{w}']])
        wcnf.append([-literals[f'bw_m_{w}']], weight=1)

    return wcnf


def solve_wcnf_clustering(wcnf: WCNF) -> List[int]:
    """
    Solve the weighted CNF problem and return the model if found.
    """
    solver = RC2(wcnf)
    solution = solver.compute()
    return solution if solution is not None else []


def assign_clusters_and_diameters(x_i_c_matrix: np.ndarray,
                                  dataset: np.ndarray,
                                  k_clusters: int) -> Tuple[Dict[int, List[int]], Dict[int, float]]:
    """
    Assigns data points to clusters based on the unique patterns in the x_i_c_matrix
    and calculates the maximum diameter for each cluster.

    Parameters:
    - x_i_c_matrix (np.ndarray): The matrix containing cluster assignments of data points.
    - dataset (np.ndarray): The original dataset with the data points.
    - k_clusters (int): The number of clusters.

    Returns:
    - cluster_assignments (dict): A dictionary with keys as cluster IDs and values as lists of data points in each cluster.
    - cluster_diameters (dict): A dictionary with keys as cluster IDs and values as the maximum diameter of each cluster.
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
