"""
=========== Module Description ===========

Module to build the complete minimum depth tree and create literals. This module includes a modified decoded threshold
compared to the original paper and should be tested on test accuracy later.
"""

import numpy as np
from graphviz import Digraph
from pysat.formula import CNF
from pysat.solvers import Solver

from satree.treemodder.builder import build_complete_tree, create_literals
from satree.classification.classification_clauses import add_redundant_constraints, construct_maxsat_clauses


def get_ancestors(node_index, side):
    """
    Find all the ancestors of a given node in the tree on the specified side (left or right).

    Parameters:
    - tree_structure (list): The complete binary tree structure.
    - node_index (int): The index of the leaf node for which to find ancestors.
    - side (str): Side of the ancestors to find ('left' or 'right').

    Returns:
    - ancestors (list): A list of indices of the ancestors on the specified side.
    """
    ancestors = []
    current_index = node_index
    while True:
        parent_index = (current_index - 1) // 2
        if parent_index < 0:
            break
        # Check if current node is a left or right child
        if (side == 'left' and current_index % 2 == 1) or (side == 'right' and current_index % 2 == 0):
            ancestors.append(parent_index)
        current_index = parent_index
    return ancestors

# Helper function to sort data points by feature and create O_j
def compute_ordering(X, feature_index):
    sorted_indices = sorted(range(len(X)), key=lambda i: X[i][feature_index])
    return [(sorted_indices[i], sorted_indices[i + 1]) for i in range(len(sorted_indices) - 1)]

def build_clauses(literals, X, TB, TL, num_features, labels,true_labels):
    """
    Constructs the clauses for the SAT solver based on the decision tree encoding.

    Args:
        literals (dict): A dictionary mapping literals to variable indices.
        X (list): The dataset, a list of tuples representing data points.
        TB (list): Indices of branching nodes.
        TL (list): Indices of leaf nodes.
        num_features (int): Number of features in the dataset.
        labels (list): Possible class labels for the data points.

    Returns:
        CNF: A CNF object containing all the clauses.
    """
    cnf = CNF()

    cnf = construct_maxsat_clauses(cnf, literals, X, TB, TL, num_features, labels)

    cnf = add_redundant_constraints(cnf, literals, X, TB, num_features)

    # Clause (11): Correct class labels for leaf nodes
    for t in TL:
        for i, xi in enumerate(X):
            label = true_labels[i]
            cnf.append([-literals[f'z_{i}_{t}'], literals[f'g_{t}_{label}']])
    
    return cnf

def set_branch_node_features(model, literals, tree_structure,features,datasetX):
    """
    Set the chosen feature and threshold for each branching node in the tree structure
    based on the given SAT model.

    Args:
    - model (list): The model returned by the SAT solver.
    - literals (dict): A dictionary mapping literals to variable indices.
    - tree_structure (list): The complete binary tree structure.
    - dataset (list): The dataset, a list of tuples representing data points.
    - features (list): List of features in the dataset.
    """
    # For each branching node, determine the chosen feature and threshold
    for node_index in range(len(tree_structure)):
        #print(node_index)
        node = tree_structure[node_index]
        if node['type'] == 'branching':
            # Find which feature is used for splitting at the current node
            chosen_feature = None
            for feature in features:
                if literals[f'a_{node_index}_{feature}'] in model:
                    chosen_feature = feature
                    break
            
            # If a feature is chosen, set the feature and find the threshold
            if chosen_feature is not None:
                # Set the chosen feature and computed threshold in the tree structure
                node['feature'] = chosen_feature


def solve_cnf(cnf, literals, TL, tree_structure, labels,features,datasetX):
    """
    Attempts to solve the given CNF using a SAT solver.

    If a solution is found, it updates the tree structure with the correct labels for leaf nodes.

    Args:
    - cnf (CNF): The CNF object containing all clauses for the SAT solver.
    - literals (dict): A dictionary mapping literals to variable indices.
    - TL (list): Indices of leaf nodes in the tree.
    - tree_structure (list): The complete binary tree structure.
    - labels (list): The list of class labels for the dataset.

    Returns:
    - solution (list or str): The solution to the SAT problem if it exists, otherwise "No solution exists".
    """
    solver = Solver()
    solver.append_formula(cnf)
    if solver.solve():
        model = solver.get_model()
        # Update the tree structure with the correct labels for leaf nodes
        for t in TL:
            for label in labels:
                if literals[f'g_{t}_{label}'] in model:
                    tree_structure[t]['label'] = label
                    break
         # Set details for branching nodes
        set_branch_node_features(model, literals, tree_structure,features,datasetX)
        return model
    else:
        #print("no solution!")
        return "No solution exists"

#adjusted Logic to compute threshold on the entire dataset at each feature node branch 
def add_thresholds(tree_structure, literals, model_solution, dataset):
    """
    Compute the threshold for each branching node in the tree structure
    based on the entire dataset.

    Args:
    - tree_structure (list): The binary tree structure containing nodes.
    - literals (dict): The mapping of literals to variable indices.
    - model_solution (list): The model solution from the SAT solver.
    - dataset (np.array): The dataset containing all the data points.

    Returns:
    - tree_structure (list): The updated tree structure with thresholds set for branching nodes.
    """
    def get_literal_value(literal):
        # Helper function to get the value of a literal from the model solution.
        return literals[literal] if literals[literal] in model_solution else -literals[literal]

    def set_thresholds(node_index, dataset):
        node = tree_structure[node_index]
        if node['type'] == 'branching':
            feature_index = int(node['feature'])

            # Instead of using the dataset_indices, we will compute the threshold based on the entire dataset.
            feature_values = dataset[:, feature_index]
            sorted_indices = np.argsort(feature_values)

            # Initialize the threshold
            threshold = None

            # Find the first instance where the direction changes and set the threshold.
            for i in range(1, len(sorted_indices)):
                left_index = sorted_indices[i - 1]
                right_index = sorted_indices[i]
                if get_literal_value(f's_{left_index}_{node_index}') > 0 and get_literal_value(f's_{right_index}_{node_index}') < 0:
                    threshold = (feature_values[left_index] + feature_values[right_index]) / 2
                    break

            # If no change in direction is found, threshold remains None.
            node['threshold'] = threshold
            
            # Continue for children nodes
            left_child_index, right_child_index = node['children'][0], node['children'][1]
            if left_child_index < len(tree_structure): # Check index is within bounds
                set_thresholds(left_child_index, dataset)
            if right_child_index < len(tree_structure): # Check index is within bounds
                set_thresholds(right_child_index, dataset)

    # Apply the threshold setting function starting from the root node
    set_thresholds(0, dataset)

    return tree_structure

# Create a matrix for each type of variable
def create_solution_matrix(literals, solution, var_type):
    # Find the maximum index for this var_type
    max_index = max(int(key.split('_')[1]) for key, value in literals.items() if key.startswith(var_type)) + 1
    max_sub_index = max(int(key.split('_')[2]) for key, value in literals.items() if key.startswith(var_type)) + 1
    
    # Initialize the matrix with zeros
    matrix = [[0 for _ in range(max_sub_index)] for _ in range(max_index)]
    
    # Fill in the matrix with 1 where the literals are true according to the solution
    for key, value in literals.items():
        if key.startswith(var_type):
            index, sub_index = map(int, key.split('_')[1:])
            matrix[index][sub_index] = 1 if value in solution else 0

    return matrix

# visualization code
def add_nodes(dot, tree, node_index=0):
    node = tree[node_index]
    if node['type'] == 'branching':
        dot.node(str(node_index), label=f"BranchNode:\n{node_index}\nFeature:{node['feature']}\nThreshold:{node['threshold']}")
        for child_index in node['children']:
            add_nodes(dot, tree, child_index)
            dot.edge(str(node_index), str(child_index))
    elif node['type'] == 'leaf':
        dot.node(str(node_index), label=f"LeafNode:\n{node_index}\nLabel: {node['label']}")

#visualization code
def visualize_tree(tree_structure):
    dot = Digraph()
    add_nodes(dot, tree_structure)
    return dot


def find_min_depth_tree(features, labels, true_labels_for_points, dataset):
    depth = 1  # Start with a depth of 1
    solution = "No solution exists"
    tree_with_thresholds = None
    tree = None
    literals = None

    while solution == "No solution exists":
        tree, TB, TL = build_complete_tree(depth)
        literals = create_literals(TB, TL, features, labels, len(dataset), False)[0]
        cnf = build_clauses(literals, dataset, TB, TL, len(features), labels, true_labels_for_points)
        solution = solve_cnf(cnf, literals, TL, tree, labels, features, dataset)
        
        if solution != "No solution exists":
            tree_with_thresholds = add_thresholds(tree, literals, solution, dataset)
            dot = visualize_tree(tree_with_thresholds)
            dot.render(f'images/min_height/binary_decision_tree_min_depth_{depth}', format='png', cleanup=True)
        else:
            print('no solution at depth', depth)
            depth += 1  # Increase the depth and try again
    
    return tree_with_thresholds, literals, depth, solution

