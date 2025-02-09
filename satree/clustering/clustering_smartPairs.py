"""
=========== Module Description ===========

This module contains functions to solve the clustering problem using a minimum height tree with a maximum diameter and
minimum split criteria.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import euclidean
from pysat.formula import WCNF

from satree.treemodder.builder import build_complete_tree

from satree.clustering.core import create_literals_cluster_tree
from satree.clustering.clustering_advanced import create_distance_classes
from satree.clustering.clustering_clauses import add_clustering_encodings, add_distance_class_clauses, \
    construct_clustering_clauses
from satree.clustering.clustering_minsplit import process_clustering_solution


def build_clauses_cluster_tree_md_ms_smart_pair(literals: Dict[str, int],
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
        distance_classes (list): list pairs in each distance classes  

    Returns:
        wcnf: A wcnf object containing all the clauses, with hard clauses for the tree structure/clustering and soft clauses for maximization bicriteria
    """
    ##################################################  BASE TREE ENCODINGS ################################################

    wcnf = construct_clustering_clauses(literals, dataset, branch_nodes, leaf_nodes, num_features)

    ################################################## CLUSTERING PROBLEM ENCODINGS ################################################

    # Smart Pairs initialization
    E_plus = set()
    E_minus = set()

    # Process must-link pairs
    ML_pairs_sorted = sorted(ml_pairs, key=lambda pair: euclidean(dataset[pair[0]], dataset[pair[1]]))
    for (i, i_prime) in ML_pairs_sorted:
        if (i, i_prime) not in E_plus:
            E_plus.add((i, i_prime))
            # Add must-link clauses
            for c in range(k_clusters - 1):
                wcnf.append([-literals[f'x_{i}_{c}'], literals[f'x_{i_prime}_{c}']])
                wcnf.append([literals[f'x_{i}_{c}'], -literals[f'x_{i_prime}_{c}']])

    # Process cannot-link pairs
    CL_pairs_sorted = sorted(cl_pairs, key=lambda pair: -euclidean(dataset[pair[0]], dataset[pair[1]]))
    for (i, i_prime) in CL_pairs_sorted:
        if (i, i_prime) not in E_minus:
            E_minus.add((i, i_prime))
            # Add cannot-link clauses
            for c in range(k_clusters - 1):
                wcnf.append([-literals[f'x_{i}_{c}'], -literals[f'x_{i_prime}_{c}']])

    # Process distance classes
    for w, pairs in enumerate(distance_classes):
        for (i, i_prime) in pairs:
            if (i, i_prime) in E_plus:
                continue
            if (i, i_prime) in E_minus:
                wcnf.append([literals[f'bw_m_{w}']])
            else:
                # Add conditional clauses for co-clustering
                wcnf.append([literals[f'bw_m_{w}'], literals[f'x_{i}_0'], literals[f'x_{i_prime}_0']])
                wcnf.append([literals[f'bw_m_{w}'], -literals[f'x_{i}_{k_clusters - 2}'],
                             -literals[f'x_{i_prime}_{k_clusters - 2}']])
                c = 0  # Ensure c is defined even if the loop doesn't execute
                for c in range(k_clusters - 2):
                    wcnf.append([literals[f'bw_m_{w}'], -literals[f'x_{i}_{c}'], -literals[f'x_{i_prime}_{c}'],
                                 literals[f'x_{i}_{c + 1}'], literals[f'x_{i_prime}_{c + 1}']])
                wcnf.append([-literals[f'bw_p_{w}'], -literals[f'x_{i}_{c}'], literals[f'x_{i_prime}_{c}']])
                wcnf.append([-literals[f'bw_p_{w}'], literals[f'x_{i}_{c}'], -literals[f'x_{i_prime}_{c}']])

    wcnf = add_clustering_encodings(wcnf, literals, dataset, leaf_nodes, k_clusters, cl_pairs, ml_pairs,
                                    distance_classes)
    wcnf = add_distance_class_clauses(wcnf, literals, k_clusters, distance_classes)

    return wcnf


def min_split_clustering_problem_smart_pair(dataset: np.ndarray,
                                            features: np.ndarray,
                                            k_clusters: int,
                                            depth: int,
                                            epsilon: float = 0,
                                            cl_pairs: np.ndarray = np.array([]),
                                            ml_pairs: np.ndarray = np.array([])) -> Tuple[
    Dict[int, List[int]], Dict[int, float]]:
    """
    Solves a clustering minimum split BICRITERIA problem by constructing a complete binary tree of a specified depth,
    creating literals for a SAT solver, building clauses for the SAT problem, and then solving
    the weighted CNF problem to determine the cluster assignments and the maximum diameter
    of each cluster. USING THE SMART PAIR ALGORITM

    Args:
        dataset (np.ndarray): The dataset containing n-dimensional data points.
        features (np.ndarray): Array of feature names or indices.
        k_clusters (int): The desired number of clusters to form.
        depth (int): The depth of the complete binary tree for clustering.
        epsilon (float, optional): The maximum distance difference to consider two distances as similar, defaults to 0.
        cl_pairs (np.ndarray, optional): An array of data point pairs that cannot be in the same cluster (cannot-link constraints).
        ml_pairs (np.ndarray, optional): An array of data point pairs that must be in the same cluster (must-link constraints).

    Returns:
    - cluster_assignments (dict): A dictionary with keys as cluster IDs and values as lists of data points in each cluster.
    - cluster_diameters (dict): A dictionary with keys as cluster IDs and values as the maximum diameter of each cluster.

    The function performs the following steps:
    - Creates non-overlapping distance classes for all unique pairs of data points in the dataset.
    - Constructs a complete binary tree for the given depth and assigns branching and leaf nodes.
    - Generates literals required for the SAT solver based on the tree structure and dataset.
    - Builds clauses for the SAT solver based on the decision tree encoding.
    - Solves the weighted CNF problem to find a solution for the clustering.
    - Assigns data points to clusters based on the solution and calculates the maximum diameter for each cluster.
    """
    dataset_size = len(dataset)
    num_features = len(features)
    dist1, dist2, distance_classes = create_distance_classes(dataset, epsilon)
    tree_structure, TB, TL = build_complete_tree(depth)
    literals = create_literals_cluster_tree(TB, TL, features, k_clusters, dataset_size, distance_classes, True)
    wcnf = build_clauses_cluster_tree_md_ms_smart_pair(literals, dataset, TB, TL, num_features, k_clusters,
                                                       cl_pairs, ml_pairs, distance_classes)

    return process_clustering_solution(wcnf, literals, dataset, features, k_clusters, TB, TL, distance_classes)[:2]
