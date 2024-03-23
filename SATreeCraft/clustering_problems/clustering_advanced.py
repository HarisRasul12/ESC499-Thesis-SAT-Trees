# Created by: Haris Rasul
# Date: March 21 2024
# Python script to build the complete tree and create literals
# for a given depth and dataset. Will attempt to maximize the number of correct labels given to training dataset 
# adding soft clauses for maximizing corrcet solution for a given depth and hard clauses 

from pysat.formula import CNF
from pysat.solvers import Solver
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from graphviz import Digraph
import numpy as np
from classification_problems.min_height_tree_module import *
import math
from itertools import combinations
from collections import defaultdict, OrderedDict
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt


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
    

def build_complete_tree_clustering(depth):
    """
    Construct a complete binary tree of a specified depth with feature and threshold values for branching nodes.

    Parameters:
    - depth (int): The depth of the tree, with the root node at depth 0.

    Returns:
    - tree_structure (list): A list where each element represents a node in the tree.
    - TB (list): The indices of the branching nodes within the tree list.
    - TL (list): The indices of the leaf nodes within the tree list.
    """
    num_nodes = (2 ** (depth + 1)) - 1
    tree_structure = [None] * num_nodes
    TB, TL = [], []

    for node in range(num_nodes):
        if node < ((2 ** depth) - 1):
            TB.append(node)
            # Include feature and threshold keys for branching nodes
            tree_structure[node] = {
                'type': 'branching', 
                'children': [2 * node + 1, 2 * node + 2], 
                'feature': None, 
                'threshold': None
            }
        else:
            TL.append(node)
            tree_structure[node] = {'type': 'leaf', 'label': None}
    
    return tree_structure, TB, TL

# Define the function to create literals based on the tree structure
def create_literals_cluster_tree(TB, TL, F, k_clusters, dataset_size,distance_classes):
    """
    Create the literals for the SAT solver based on the tree structure and dataset size.

    This function creates four types of literals:
    - 'a' literals for feature splits at branching nodes,
    - 's' literals for data points directed to left or right,
    - 'z' literals for data points that end up at a leaf node,
    - 'g' literals for assigning class labels to leaf nodes.
    - 'x' The cluster assigned to point 𝑖 is or comes after 𝑐lass label c
    - 'bw_p' The pairs in class 𝑤 should be clustered together
    - 'bw_m' (The negation of) whether the pairs in distance class 𝑤 should be clustered separately

    Parameters:
    - TB (list): Indices of branching nodes in the tree.
    - TL (list): Indices of leaf nodes in the tree.
    - F (list): The array of features
    - k_clusters (int) - based on clusters , so we need to turn this into arrat: eg C= k_clusters, C =2, C-> [0,1], turn into list of cluster ids
    - dataset_size (int): The number of data points in the dataset.
    - data_classes 


    Returns:
    - literals (dict): A dictionary where keys are literal names and values are their corresponding indices for the SAT solver.
    """
    C = list(range(k_clusters))  # List of cluster IDs
    literals = {}
    current_index = 1

    # Create 'a' literals for feature splits at branching nodes
    for t in TB:
        for j in F:
            literals[f'a_{t}_{j}'] = current_index
            current_index += 1

    # Create 's' literals for data points directed left or right at branching nodes
    for i in range(dataset_size):
        for t in TB:
            literals[f's_{i}_{t}'] = current_index
            current_index += 1

    # Create 'z' literals for data points ending up at leaf nodes
    for i in range(dataset_size):
        for t in TL:
            literals[f'z_{i}_{t}'] = current_index
            current_index += 1

    # Create 'g' literals for clusters at leaf nodes
    for t in TL:
        for c in C:
            literals[f'g_{t}_{c}'] = current_index
            current_index += 1

    # Create 'x' The cluster assigned to point 𝑖 is or comes after 𝑐
    for i in range(dataset_size):
        for c in C:
            literals[f'x_{i}_{c}'] = current_index
            current_index += 1
    
    # Create 'bw_m' literals (points in class w should NOT be clustered together)
    for w, pairs in enumerate(distance_classes):  # data_classes is a list of numpy arrays
        literals[f'bw_m_{w}'] = current_index
        current_index += 1

    # Create 'bw_p' literals (points in class w should be clustered together)
    # for w, pairs in enumerate(distance_classes):
    #     literals[f'bw_p_{w}'] = current_index
    #     current_index += 1

    return literals
    

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
    wcnf = WCNF()
    
    # Clause (7) and (8): Feature selection at branching nodes
    for t in TB:
        # At least one feature is chosen (Clause 7)
        clause = [literals[f'a_{t}_{j}'] for j in range(num_features)]
        wcnf.append(clause)
        
        # No two features are chosen (Clause 8)
        for j in range(num_features):
            for jp in range(j + 1, num_features):
                clause = [-literals[f'a_{t}_{j}'], -literals[f'a_{t}_{jp}']]
                wcnf.append(clause)

    # Clause (9) and (10): Data point direction based on feature values
    for j in range(num_features):
        Oj = compute_ordering(X, j)
        for (i, ip) in Oj:
            if X[i][j] < X[ip][j]:  # Different feature values (Clause 9)
                for t in TB:
                    wcnf.append([-literals[f'a_{t}_{j}'], literals[f's_{i}_{t}'], -literals[f's_{ip}_{t}']])
            if X[i][j] == X[ip][j]:  # Equal feature values (Clause 10)
                for t in TB:
                    wcnf.append([-literals[f'a_{t}_{j}'], literals[f's_{i}_{t}'], -literals[f's_{ip}_{t}']])
                    wcnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{i}_{t}'], literals[f's_{ip}_{t}']])

    # Clause (11 and 12): Path valididty form right traversla and left traversal 
    for t in TL:
        left_ancestors = get_ancestors(t, 'left')
        right_ancestors = get_ancestors(t, 'right')
        for i in range(len(X)):
            # Data point i ends at leaf node t (Clause 11 and 12) - assumption made!!!
            if left_ancestors:
                wcnf.append([-literals[f'z_{i}_{t}']] + [literals[f's_{i}_{a}'] for a in left_ancestors])
            if right_ancestors:
                wcnf.append([-literals[f'z_{i}_{t}']] + [-literals[f's_{i}_{a}'] for a in right_ancestors])

    # Clause (13): Each data point that does not end up in leaf node t has at least one deviation from the path
    for i in range(len(X)):
        for t in TL:
            deviations = []
            left_ancestors = get_ancestors(t, 'left')  # Get left ancestors using TB indices
            right_ancestors = get_ancestors(t, 'right')  # Get right ancestors using TB indices
            # Only append deviations if there are ancestors on the corresponding side
            if left_ancestors:
                deviations.extend([-literals[f's_{i}_{ancestor}'] for ancestor in left_ancestors])
            if right_ancestors:
                deviations.extend([literals[f's_{i}_{ancestor}'] for ancestor in right_ancestors])
            # Only append the clause if there are any deviations
            if deviations:
                wcnf.append([literals[f'z_{i}_{t}']] + deviations)    

    # Clause (14) and (15): Redundant constraints to prune the search space
    # These clauses are optimizations
    for t in TB:
        # Find the data point with the lowest and highest feature value for each feature
        for j in range(num_features):
            sorted_by_feature = sorted(range(len(X)), key=lambda k: X[k][j])
            lowest_value_index = sorted_by_feature[0]
            highest_value_index = sorted_by_feature[-1]

            # Clause (14): The data point with the lowest feature value is directed left
            wcnf.append([-literals[f'a_{t}_{j}'], literals[f's_{lowest_value_index}_{t}']])

            # Clause (15): The data point with the highest feature value is directed right
            wcnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{highest_value_index}_{t}']])

    ################################################## CLUSTERING PROBLEM ENDCODINGS ################################################
    
    # Add remaining clauses without CONSIDEIRNG THE MS PORBLEM - ONLY MD; recall they have ids 1,...k, in application ist say 0,...,k-1 for any k
    
    # Clause 16: Unary encoding of cluster labels in each leaf
    for t in TL:  # For each leaf node
        for c in range(k_clusters - 2):  # Remember, k_clusters already 1 less than total count
            clause = [literals[f'g_{t}_{c}'], -literals[f'g_{t}_{c+1}']]
            # print("clause 16: ", clause)
            wcnf.append(clause)
    
    # Clause 17: Data points ending at leaf node t are assigned to cluster c if g_t,c is true
    for t in TL:  # For each leaf node
        for i in range(len(X)):  # For each data point
            for c in range(k_clusters - 1):  # Cluster indexes 0 to k-2, as the last index is not needed here
                clause = [-literals[f'z_{i}_{t}'], -literals[f'g_{t}_{c}'], literals[f'x_{i}_{c}']]
                # print("clause 17: ", clause)
                wcnf.append(clause)

    # Clause 18: Data points ending at leaf node t are NOT assigned to cluster c if g_t,c is false
    for t in TL:  # For each leaf node
        for i in range(len(X)):  # For each data point
            for c in range(k_clusters - 1):  # Cluster indexes 0 to k-2, as the last index is not needed here
                clause = [-literals[f'z_{i}_{t}'], literals[f'g_{t}_{c}'], -literals[f'x_{i}_{c}']]
                # print("clause 18: ", clause)
                wcnf.append(clause)
    
    # Clause 19: Ensure no cluster is empty by ensuring there's at least one data point in each cluster
    for c in range(k_clusters - 1):  # Iterate over all clusters except the last one
        # Add a clause that states that there's at least one data point in cluster c
        # print("clause 19: ", [literals[f'x_{c}_{c}']])
        wcnf.append([literals[f'x_{c}_{c}']])
    
    # Clause 20: If xi is not in cluster c, then there must be some xi' in cluster c-1, for all c < i
    for i in range(1,len(X)):  # We start from 1 to ensure there's at least one i' < i
        for c in range(1, k_clusters - 1):  # We can safely start from 1 since we need c to be at least 1 to have a c-1
            clause = [-literals[f'x_{i}_{c}']]
            for i_prime in range(i):  # For each i' less than i
                clause.append(literals[f'x_{i_prime}_{c-1}'])
                # print("clause 20: ", clause)
            wcnf.append(clause)
    
    # Clause 21: Ensure all clusters are non-empty by requiring at least one point is assigned to each cluster
    # In this case, we focus on the second-to-last cluster k-2.
    clauseTW = [literals[f'x_{i}_{k_clusters - 2}'] for i in range(len(X))]
    # print("clause 21: ", clauseTW)
    wcnf.append(clauseTW)

    # Clause 22: Ensure that pairs in CL are not clustered in the first cluster (0-indexed)
    for i, i_prime in CL_pairs:  # For each pair (i, i') in the cannot-link set
        # print("clause 22: ", [literals[f'x_{i}_0'], literals[f'x_{i_prime}_0']])
        wcnf.append([literals[f'x_{i}_0'], literals[f'x_{i_prime}_0']])

    # Clause 23: Ensure that pairs in CL are not clustered in the last cluster (k-2 in 0-indexed system)
    for i, i_prime in CL_pairs:  # For each pair (i, i') in the cannot-link set
        # print("clause 23: ", [-literals[f'x_{i}_{k_clusters - 2}'], -literals[f'x_{i_prime}_{k_clusters - 2}']])
        wcnf.append([-literals[f'x_{i}_{k_clusters - 2}'], -literals[f'x_{i_prime}_{k_clusters - 2}']])

    # Clause 24: Unconditional separating clauses for cannot-link pairs, applied to clusters from 0 to k-3
    for (i, i_prime) in CL_pairs:  # For each cannot-link pair
        for c in range(k_clusters - 2):  # Up to k-3, because we're considering c and c+1
            # print("clause 24: ", [
            #     -literals[f'x_{i}_{c}'], 
            #     -literals[f'x_{i_prime}_{c}'], 
            #     literals[f'x_{i}_{c+1}'], 
            #     literals[f'x_{i_prime}_{c+1}']
            # ])

            wcnf.append([
                -literals[f'x_{i}_{c}'], 
                -literals[f'x_{i_prime}_{c}'], 
                literals[f'x_{i}_{c+1}'], 
                literals[f'x_{i_prime}_{c+1}']
            ])

    # Clause 25 and 26: Ensure that pairs in ML are clustered together for each cluster
    for i, i_prime in ML_pairs:  # For each must-link pair
        for c in range(k_clusters - 1):  # Iterate over all clusters except the last one
            # print("clause 25: ", [-literals[f'x_{i}_{c}'], literals[f'x_{i_prime}_{c}']])
            wcnf.append([-literals[f'x_{i}_{c}'], literals[f'x_{i_prime}_{c}']]) # clause 25
            
            # print("clause 26: ", [literals[f'x_{i}_{c}'], -literals[f'x_{i_prime}_{c}']])
            wcnf.append([literals[f'x_{i}_{c}'], -literals[f'x_{i_prime}_{c}']])
    
    # Clause 27: Conditional separating clauses using distance classes and bw_m literals
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            # Adding a clause that says if bw_m_w is false, then i and i_prime must not be in the first cluster together
            # print('clause 27: ',[literals[f'bw_m_{w}'], literals[f'x_{i}_0'], literals[f'x_{i_prime}_0']])
            wcnf.append([literals[f'bw_m_{w}'], literals[f'x_{i}_0'], literals[f'x_{i_prime}_0']])
    
    # Clause 28: Ensure that if bw^-_w is true, then the pair (i, i') from Dw cannot be in the second to last cluster k-2
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            # If bw^-_w is true, then i and i' must not be in cluster k-2 together
            # print('clause 28: ', [literals[f'bw_m_{w}'], -literals[f'x_{i}_{k_clusters - 2}'], -literals[f'x_{i_prime}_{k_clusters - 2}']])
            wcnf.append([literals[f'bw_m_{w}'], -literals[f'x_{i}_{k_clusters - 2}'], -literals[f'x_{i_prime}_{k_clusters - 2}']])
    
    # Clause 29: Conditional co-separation for non-adjacent clusters
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            # Apply the constraint for all clusters except the last (since we start from zero)
            for c in range(k_clusters - 2):
                # If bw^-_w is true, then i and i_prime must not be in cluster c together and must be in c+1 or any cluster after
                # print('clause 29: ', [literals[f'bw_m_{w}'], -literals[f'x_{i}_{c}'], 
                #              -literals[f'x_{i_prime}_{c}'], literals[f'x_{i}_{c+1}'], literals[f'x_{i_prime}_{c+1}']])
                
                wcnf.append([literals[f'bw_m_{w}'], -literals[f'x_{i}_{c}'], 
                             -literals[f'x_{i_prime}_{c}'], literals[f'x_{i}_{c+1}'], literals[f'x_{i_prime}_{c+1}']])
        
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


def create_literal_matrices(literals, solution, dataset_size, k_clusters, TB, TL, num_features, distance_classes):
    # Initialize matrices with zeros
    a_matrix = np.zeros((len(TB), num_features), dtype=int)
    s_matrix = np.zeros((dataset_size, len(TB)), dtype=int)
    z_matrix = np.zeros((dataset_size, len(TL)), dtype=int)
    g_matrix = np.zeros((len(TL), k_clusters), dtype=int)
    x_i_c_matrix = np.zeros((dataset_size, k_clusters), dtype=int)  # Added x_i_c matrix back
    bw_m_vector = np.zeros(len(distance_classes), dtype=int)  # Corrected length based on the number of distance classes
    
    # Helper to update the matrix based on the literal and its presence in the solution
    def update_matrix(matrix, i, j, literal_index):
        if literal_index in solution:
            matrix[i, j] = 1
        elif -literal_index in solution:
            matrix[i, j] = 0

    # Iterate over all literals to fill in the matrices
    for literal, index in literals.items():
        parts = literal.split('_')
        if literal.startswith('a_'):
            t = TB.index(int(parts[1]))
            j = int(parts[2])
            update_matrix(a_matrix, t, j, index)
        elif literal.startswith('s_'):
            i = int(parts[1])
            t = TB.index(int(parts[2]))
            update_matrix(s_matrix, i, t, index)
        elif literal.startswith('z_'):
            i = int(parts[1])
            t = TL.index(int(parts[2]))
            update_matrix(z_matrix, i, t, index)
        elif literal.startswith('g_'):
            t = TL.index(int(parts[1]))
            c = int(parts[2])
            update_matrix(g_matrix, t, c, index)
        elif literal.startswith('x_'):
            i = int(parts[1])
            c = int(parts[2])
            update_matrix(x_i_c_matrix, i, c, index)

    for literal, index in literals.items():
        if literal.startswith('bw_m_'):
            w = int(literal.split('_')[2])  # Extract the class index directly from the literal
            # If the index is positive in the solution, it's true, else false.
            bw_m_vector[w] = 1 if index in solution else 0

    # Print matrices
    # print("a_matrix (Feature selection at branching nodes):\n", a_matrix)
    # print("s_matrix (Data point direction at branching nodes):\n", s_matrix)
    # print("z_matrix (Data point end at leaf nodes):\n", z_matrix)
    # print("g_matrix (Cluster assignment to leaf nodes):\n", g_matrix)
    # print("x_i_c_matrix (Cluster assigned to data points):\n", x_i_c_matrix)
    # print("bw_m_vector (Separate clustering for distance classes):\n", bw_m_vector)

    # Return the matrices
    return a_matrix, s_matrix, z_matrix, g_matrix, x_i_c_matrix, bw_m_vector


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
    plt.close(fig)  # Close the figure to prevent it from displaying in the output

    return full_path

def clustering_problem(dataset,features,k_clusters, depth, epsilon = 0, CL_pairs = np.array([]), ML_pairs = np.array([])):
    
    dataset_size = len(dataset)
    num_features = len(features)
    dist1, dist2, distance_classes = create_distance_classes(dataset, epsilon)
    tree_structure, TB, TL = build_complete_tree_clustering(depth)
    literals = create_literals_cluster_tree(TB, TL, features, k_clusters, dataset_size,distance_classes)
    wcnf = build_clauses_cluster_tree_MD(literals, dataset, TB, TL, num_features, k_clusters,
                                  CL_pairs, ML_pairs, distance_classes)
    
    solution = solve_wcnf_clustering(wcnf)

    a_matrix, s_matrix, z_matrix, g_matrix, x_i_c_matrix, bw_m_vector = create_literal_matrices(
        literals=literals,
        solution=solution,
        dataset_size=len(dataset),
        k_clusters=k_clusters,
        TB=TB,
        TL=TL,
        num_features=len(features),
        distance_classes= distance_classes
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
    
#     # literals = create_literals_cluster_tree(TB, TL, F, k_clusters, dataset_size,distance_classes)
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