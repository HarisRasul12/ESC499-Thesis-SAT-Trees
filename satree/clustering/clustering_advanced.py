"""
=========== Module Description ===========

Module for solving clustering problems using a complete binary tree and SAT solvers. This module contains functions
to solve clustering problems using a complete binary tree and SAT solvers.
"""

from itertools import combinations
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from pysat.examples.rc2 import RC2

from satree.clustering.core import create_literals_cluster_tree, create_literal_matrices_modular
from satree.treemodder.builder import build_complete_tree
from clustering_clauses import construct_clustering_clauses, add_clustering_encodings


def create_distance_classes(dataset, epsilon=0):
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
    def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))
    
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
    

def build_clauses_cluster_tree_MD(literals, X, TB, TL, num_features, k_clusters,
                                  CL_pairs, ML_pairs, distance_classes):
    """
    Constructs the clauses for the SAT solver based on the decision tree encoding. Now includes MAX SOLVER PROBLEM FOR FIXED HEIGHT 

    Args:
        literals (dict): A dictionary mapping literals to variable indices.
        X (list): The dataset, a list of tuples representing data points.
        TB (list): Indices of branching nodes.
        TL (list): Indices of leaf nodes.
        num_features (int): Number of features in the dataset.
        k_clusters: number of clusters, will need to turn this into a list for operations on each clause
        distance_classes (list): list pairs in ecah distace classes  

    Returns:
        wcnf: A wcnf object containing all the clauses, with hard clauses for the tree structure/clustering and soft clauses for maximization
    """
    ##################################################  BASE TREE ENCODINGS ################################################

    # Now the problem has become Partial MaxSAT - we will assign weights to the soft clauses Eq. (13). Eq(1-10,12) HARD clauses
    wcnf = construct_clustering_clauses(literals, X, TB, TL, num_features, k_clusters, CL_pairs, ML_pairs, distance_classes)
    wcnf = add_clustering_encodings(wcnf, literals, X, TL, k_clusters, CL_pairs, ML_pairs, distance_classes)

    # Clause 32: Ensures that if bw^-_w is true, then the points in distance class w 
    # cannot be clustered with the points in distance class w-1 if bw^-_(w-1) is false.
    for w in range(1, len(distance_classes)):  # Starting from 1 since we're checking w against w-1
        # print('clause 32: ', [-literals[f'bw_m_{w}'], literals[f'bw_m_{w-1}']])
        wcnf.append([-literals[f'bw_m_{w}'], literals[f'bw_m_{w-1}']])
    
    # Clause 37: For each distance class w, we add a soft clause for the corresponding b^-_w literal
    # to encourage points within that class to be clustered separately
    # Max diameter solve problem
    for w in range(len(distance_classes)):
        # print("clause 37: ", [-literals[f'bw_m_{w}']])
        wcnf.append([-literals[f'bw_m_{w}']], weight=1)

    return wcnf


def solve_wcnf_clustering(wcnf):
    """
    Solve the weighted CNF problem and return the model if found.
    """
    solver = RC2(wcnf)
    solution = solver.compute()
    return solution if solution is not None else []


def assign_clusters_and_diameters(x_i_c_matrix, dataset, k_clusters):
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

# Save the plot to the specified directory with the given filename format
def plot_and_save_clusters(dataset, cluster_assignments, k_clusters):
    """
    Plots the dataset points before and after clustering if the dataset has 1 or 2 features.
    Creates a side-by-side plot showing the dataset before clustering and after with cluster IDs.
    Saves the plot to the specified directory with a filename based on the number of clusters.
    Does not display the plot in the output.

    Parameters:
    - dataset (np.ndarray): The original dataset with data points.
    - cluster_assignments (dict): A dictionary with cluster IDs and lists of data points in each cluster.
    - k_clusters (int): The number of clusters.

    Returns:
    - full_path (str): The path to the saved plot image.
    """
    # Define the directory and filename
    directory = 'images/cluster_trees/'
    filename = f'cluster_tree_with_cluster_size{k_clusters}.png'
    full_path = directory + filename

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    
    if dataset.shape[1] == 2:  # If 2D dataset
        axes[0].scatter(dataset[:, 0], dataset[:, 1], c='gray', label='Data Points')
        axes[0].set_title('Before Clustering')
        axes[1].scatter(dataset[:, 0], dataset[:, 1], c='gray', label='Data Points')
        axes[1].set_title('After Clustering')
    elif dataset.shape[1] == 1:  # If 1D dataset
        axes[0].scatter(dataset[:, 0], np.zeros_like(dataset[:, 0]), c='gray', label='Data Points')
        axes[0].set_title('Before Clustering')
        axes[1].scatter(dataset[:, 0], np.zeros_like(dataset[:, 0]), c='gray', label='Data Points')
        axes[1].set_title('After Clustering')
    else:
        return 'can only plot 2d or 1d datasets'

    # Assign colors to clusters
    colors = plt.cm.tab10(np.linspace(0, 1, k_clusters))
    for cluster_id, data_points in cluster_assignments.items():
        if dataset.shape[1] == 2:
            axes[1].scatter(dataset[data_points, 0], dataset[data_points, 1], 
                            color=colors[cluster_id], label=f'Cluster {cluster_id}')
        elif dataset.shape[1] == 1:
            axes[1].scatter(dataset[data_points, 0], np.zeros_like(dataset[data_points, 0]), 
                            color=colors[cluster_id], label=f'Cluster {cluster_id}')

    # Add legend to the second plot
    axes[1].legend()

    # Save the figure
    fig.savefig(full_path)
    plt.show()
    plt.close(fig)  # Close the figure to prevent it from displaying in the output

    return full_path

def clustering_problem(dataset,features,k_clusters, depth, epsilon = 0, CL_pairs = np.array([]), ML_pairs = np.array([])):
    """
    Solves a clustering problem by constructing a complete binary tree of a specified depth,
    creating literals for a SAT solver, building clauses for the SAT problem, and then solving
    the weighted CNF problem to determine the cluster assignments and the maximum diameter
    of each cluster.

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
    literals = create_literals_cluster_tree(TB, TL, features, k_clusters, dataset_size, distance_classes, False)
    wcnf = build_clauses_cluster_tree_MD(literals, dataset, TB, TL, num_features, k_clusters,
                                  CL_pairs, ML_pairs, distance_classes)
    
    solution = solve_wcnf_clustering(wcnf)

    a_matrix, s_matrix, z_matrix, g_matrix, x_i_c_matrix, bw_m_vector = create_literal_matrices_modular(
        literals=literals,
        solution=solution,
        dataset_size=len(dataset),
        k_clusters=k_clusters,
        TB=TB,
        TL=TL,
        num_features=len(features),
        distance_classes= distance_classes,
        bicriteria=False
    )

    cluster_assignments, cluster_diameters = assign_clusters_and_diameters(
        x_i_c_matrix, dataset, k_clusters
    )

    return cluster_assignments, cluster_diameters



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

#     cluster_assignments, cluster_diameters = clustering_problem(dataset=dataset,
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