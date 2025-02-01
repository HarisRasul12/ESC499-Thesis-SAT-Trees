"""
=========== Module Description ===========

This module contains functions to solve the clustering problem using a minimum height tree with a maximum diameter and
minimum split criteria.
"""

import numpy as np
from scipy.spatial.distance import euclidean

from satree.clustering.core import create_literals_cluster_tree
from satree.treemodder.builder import build_complete_tree
from satree.clustering.clustering_advanced import create_distance_classes
from clustering_clauses import add_clustering_encodings, add_distance_class_clauses, construct_clustering_clauses
from clustering_minsplit import process_clustering_solution


def build_clauses_cluster_tree_MD_MS_Smart_Pair(literals, X, TB, TL, num_features, k_clusters, CL_pairs, ML_pairs, distance_classes):
    """
    Constructs the clauses for the SAT solver based on the decision tree encoding. Now includes MAX SOLVER PROBLEM FOR FIXED HEIGHT 

    Args:
        literals (dict): A dictionary mapping literals to variable indices.
        X (list): The dataset, a list of tuples representing data points.
        TB (list): Indices of branching nodes.
        TL (list): Indices of leaf nodes.
        num_features (int): Number of features in the dataset.
        k_clusters: number of clusters, will need to turn this into a list for operations on each clause
        distance_classes (list): list pairs in each distance classes  

    Returns:
        wcnf: A wcnf object containing all the clauses, with hard clauses for the tree structure/clustering and soft clauses for maximization bicriteria
    """
    ##################################################  BASE TREE ENCODINGS ################################################

    wcnf = construct_clustering_clauses(literals, X, TB, TL, num_features, k_clusters, CL_pairs, ML_pairs, distance_classes)

    ################################################## CLUSTERING PROBLEM ENCODINGS ################################################
    
    # Smart Pairs initialization
    E_plus = set()
    E_minus = set()
    
    # Process must-link pairs
    ML_pairs_sorted = sorted(ML_pairs, key=lambda pair: euclidean(X[pair[0]], X[pair[1]]))
    for (i, i_prime) in ML_pairs_sorted:
        if (i, i_prime) not in E_plus:
            E_plus.add((i, i_prime))
            # Add must-link clauses
            for c in range(k_clusters - 1):
                wcnf.append([-literals[f'x_{i}_{c}'], literals[f'x_{i_prime}_{c}']])
                wcnf.append([literals[f'x_{i}_{c}'], -literals[f'x_{i_prime}_{c}']])

    # Process cannot-link pairs
    CL_pairs_sorted = sorted(CL_pairs, key=lambda pair: -euclidean(X[pair[0]], X[pair[1]]))
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
                wcnf.append([literals[f'bw_m_{w}'], -literals[f'x_{i}_{k_clusters - 2}'], -literals[f'x_{i_prime}_{k_clusters - 2}']])
                for c in range(k_clusters - 2):
                    wcnf.append([literals[f'bw_m_{w}'], -literals[f'x_{i}_{c}'], -literals[f'x_{i_prime}_{c}'], literals[f'x_{i}_{c+1}'], literals[f'x_{i_prime}_{c+1}']])
                wcnf.append([-literals[f'bw_p_{w}'], -literals[f'x_{i}_{c}'], literals[f'x_{i_prime}_{c}']])
                wcnf.append([-literals[f'bw_p_{w}'], literals[f'x_{i}_{c}'], -literals[f'x_{i_prime}_{c}']])


    wcnf = add_clustering_encodings(wcnf, literals, X, TL, k_clusters, CL_pairs, ML_pairs, distance_classes)
    wcnf = add_distance_class_clauses(wcnf, literals, k_clusters, distance_classes)

    return wcnf


def min_split_clustering_problem_SmartPair(dataset,features,k_clusters, depth, epsilon = 0, CL_pairs = np.array([]), ML_pairs = np.array([])):
    """
    Solves a clustering minimum split BICRITERIA problem by constructing a complete binary tree of a specified depth,
    creating literals for a SAT solver, building clauses for the SAT problem, and then solving
    the weighted CNF problem to determine the cluster assignments and the maximum diameter
    of each cluster. USING THE SMART PAIR ALGORITM

    Parameters:
    - dataset (np.ndarray): The dataset containing n-dimensional data points.
    - features (np.ndarray): Array of feature names or indices.
    - k_clusters (int): The desired number of clusters to form.
    - depth (int): The depth of the complete binary tree for clustering.
    - epsilon (float, optional): The maximum distance difference to consider two distances as similar, defaults to 0.
    - CL_pairs (np.ndarray, optional): An array of data point pairs that cannot be in the same cluster (cannot-link constraints).
    - ML_pairs (np.ndarray, optional): An array of data point pairs that must be in the same cluster (must-link constraints).

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
    wcnf = build_clauses_cluster_tree_MD_MS_Smart_Pair(literals, dataset, TB, TL, num_features, k_clusters,
                                  CL_pairs, ML_pairs, distance_classes)

    return process_clustering_solution(wcnf, literals, dataset, features, k_clusters, TB, TL, distance_classes)[:2]



# if __name__ == "__main__":
# #     # Define the test dataset parameters
#     # Data points
#     F = np.array(['0', '1'])
#     dataset = np.array([[1, 1], [1, 2], [7, 7], [7, 8], [15,5],[15,6]])  # Dataset X
#     dataset_size = len(dataset)
#     epsilon = 1 
#     k_clusters = 3
#     depth = 3

#     # CL_pairs = np.array([])
#     ML_pairs = np.array([])
#     CL_pairs = np.array([[2,3]])
#     # # ML_pairs = np.array([[4,5],[0,1],[2,3]])

#     cluster_assignments, cluster_diameters = min_split_clustering_problem_SmartPair(dataset=dataset,
#                                                                 features=F,
#                                                                 k_clusters=k_clusters,
#                                                                 depth = depth,
#                                                                 epsilon= epsilon,
#                                                                 CL_pairs=CL_pairs,
#                                                                 ML_pairs= ML_pairs)

#     print(cluster_assignments)
#     print(cluster_diameters)

#     # Plot the clusters
#     plot_and_save_clusters(dataset, cluster_assignments, k_clusters)


#     # dist1, dist2, distance_classes = create_distance_classes(dataset, epsilon)
    
#     # print('distance classes created: ')
#     # print(distance_classes)
#     # # print(dist1)
#     # print("\nNumber of distance classes: ")
#     # print(len(distance_classes))

#     # k_clusters = 2
#     # depth = 2
#     # tree_structure, TB, TL = build_complete_tree_clustering(depth)
#     # # print(tree_structure)
    
#     # literals = create_literals_cluster_tree(TB, TL, F, k_clusters, dataset_size,distance_classes, False)
#     # print("\nliterals map: ")
#     # for key, value in literals.items():
#     #     print(f'{key}: {value}')

#     # num_features = len(F)
#     # CL_pairs = np.array([])
#     # ML_pairs = np.array([])
#     # # CL_pairs = np.array([[2,3]])
#     # # ML_pairs = np.array([[4,5],[0,1],[2,3]])
#     # X = dataset

#     # wcnf = build_clauses_cluster_tree_MD(literals, X, TB, TL, num_features, k_clusters,
#     #                               CL_pairs, ML_pairs, distance_classes)
#     # # print(wcnf)
#     # solution = solve_wcnf_clustering(wcnf)
#     # print('\nthe solution: ')
#     # print(solution)

#     # # Call the function with the appropriate parameters
#     # print('\nsolution breakdown:\n')
#     # a_matrix, s_matrix, z_matrix, g_matrix, x_i_c_matrix, bw_m_vector = create_literal_matrices(
#     #     literals=literals,
#     #     solution=solution,
#     #     dataset_size=len(dataset),
#     #     k_clusters=k_clusters,
#     #     TB=TB,
#     #     TL=TL,
#     #     num_features=len(F),
#     #     distance_classes= distance_classes
#     # )

#     # # Call the function with the example dataset and number of clusters
#     # cluster_assignments_example, cluster_diameters_example = assign_clusters_and_diameters(
#     #     x_i_c_matrix, dataset, k_clusters
#     # )

#     # print(cluster_assignments)
#     # print(cluster_diameters)

#     # # Plot the clusters
#     # plot_and_save_clusters(dataset, cluster_assignments, k_clusters)