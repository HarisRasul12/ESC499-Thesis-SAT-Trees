"""
=========== Module Description ===========

Base module to help solve SAT problems with categorical and numerical features.
"""

from typing import List, Dict, Any
from pysat.formula import CNF

from satree.classification.classification_core import compute_numerical_threshold
from satree.classification.min_height_tree_module import solve_cnf, visualize_tree
from satree.treemodder.builder import build_complete_tree, create_literals
from satree.classification.classification_clauses import add_clauses_for_features_and_paths, add_feature_selection_clauses_for_branching_nodes


def build_clauses_categorical(literals, dataset, branch_nodes, leaf_nodes, num_features, features_categorical, features_numerical, labels, true_labels):
    """
    Constructs the clauses for the SAT solver based on the decision tree encoding.

    Args:
        literals (dict): A dictionary mapping literals to variable indices.
        dataset (list): The dataset, a list of tuples representing data points.
        branch_nodes (list): Indices of branching nodes.
        leaf_nodes (list): Indices of leaf nodes.
        num_features (int): Number of features in the dataset.
        features_categorical (list): List of categorical features.
        features_numerical (list): List of numerical features.
        labels (list): Possible class labels for the data points.
        true_labels (list): The true class labels for the data points.

    Returns:
        CNF: A CNF object containing all the clauses.
    """
    cnf = CNF()
    cnf = add_feature_selection_clauses_for_branching_nodes(cnf, literals, branch_nodes, num_features)
    cnf = add_clauses_for_features_and_paths(cnf, literals, dataset, branch_nodes, leaf_nodes, num_features, features_categorical, features_numerical, labels)

    # Clause (25): Correct class labels for leaf nodes
    for t in leaf_nodes:
        for i, xi in enumerate(dataset):
            label = true_labels[i]
            cnf.append([-literals[f'z_{i}_{t}'], literals[f'g_{t}_{label}']])
    
    return cnf


def add_thresholds_categorical(tree_structure: List[Dict[str, Any]], literals, model_solution, dataset, features_categorical):
    """
    Adds thresholds to each branching node in the tree structure based on the entire dataset.

    For categorical features, the threshold is the sorted list of unique values that went left.
    For numerical features, the threshold is computed as the average of two adjacent data point values
    where the data point direction changes.

    Args:
        tree_structure (list): The complete tree structure (list of nodes).
        literals (dict): A dictionary mapping literal names to variable indices.
        model_solution (list): The SAT solver's model solution.
        dataset (array): The dataset containing data points.
        features_categorical (list): List of categorical features.

    Returns:
        list: The updated tree structure with thresholds added for branching nodes.
    """

    def get_literal_value(literal):
        return literals[literal] if literals[literal] in model_solution else -literals[literal]

    def set_thresholds_categorical(node_index, data):
        node = tree_structure[node_index]
        if node['type'] == 'branching':
            feature_index = int(node['feature'])
            is_categorical = str(feature_index) in features_categorical

            if is_categorical:
                # For categorical features, list the unique values that went left.
                categories_that_went_left = set()
                for i, data_point in enumerate(data):
                    if get_literal_value(f's_{i}_{node_index}') > 0:
                        categories_that_went_left.add(data_point[feature_index])
                node['threshold'] = sorted(list(categories_that_went_left))
            else:
                # For numerical features, use the helper function.
                feature_values = data[:, feature_index].astype(float)
                node['threshold'] = compute_numerical_threshold(feature_values, node_index, get_literal_value)

            # Continue for children nodes.
            left_child_index, right_child_index = node['children'][0], node['children'][1]
            if left_child_index < len(tree_structure):
                set_thresholds_categorical(left_child_index, data)
            if right_child_index < len(tree_structure):
                set_thresholds_categorical(right_child_index, data)

    set_thresholds_categorical(0, dataset)
    return tree_structure


def find_min_depth_tree_categorical(features, features_categorical, features_numerical, labels, true_labels_for_points, dataset):
    """
    Finds a minimum-depth decision tree for a categorical classification problem using SAT solving.

    This function incrementally increases the depth of a complete binary tree until a SAT solver solution is found.
    For each depth, it:
      1. Builds a complete tree.
      2. Creates SAT literals.
      3. Constructs CNF clauses using a categorical encoding.
      4. Attempts to solve the CNF using a SAT solver.
      5. If a solution is found, it adds thresholds to the tree nodes and visualizes the tree.
      6. Otherwise, it increases the depth and tries again.

    Args:
        features (list): List of feature names or indices used for splitting.
        features_categorical (list): List of indices or identifiers for categorical features.
        features_numerical (list): List of indices or identifiers for numerical features.
        labels (list): Possible class labels for the data points.
        true_labels_for_points (list): The true class labels for each data point.
        dataset (array or list): The dataset containing data points (each data point is a tuple or array).

    Returns:
        tuple: A tuple containing:
            - tree_with_thresholds: The decision tree with thresholds added (if a solution is found).
            - literals (dict): A dictionary mapping literal names to their indices.
            - depth (int): The depth of the found tree.
            - solution (list or str): The SAT solver's model solution, or "No solution exists" if unsolvable.
    """

    depth = 1  # Start with a depth of 1
    solution = "No solution exists"
    tree_with_thresholds = None
    literals = None

    while solution == "No solution exists":
        tree, TB, TL = build_complete_tree(depth)
        literals = create_literals(TB, TL, features, labels, len(dataset), False)[0]
        cnf = build_clauses_categorical(literals, dataset, TB, TL, len(features), features_categorical, features_numerical, labels, true_labels_for_points)
        solution = solve_cnf(cnf, literals, TL, tree, labels, features)
        
        if solution != "No solution exists":
            tree_with_thresholds = add_thresholds_categorical(tree, literals, solution, dataset, features_categorical)
            dot = visualize_tree(tree_with_thresholds)
            dot.render(f'images/min_height/binary_decision_tree_min_depth_with_categorical_features_depth_{depth}', format='png', cleanup=True)
        else:
            print("No solution at depth: ", depth)
            depth += 1  # Increase the depth and try again
    
    return tree_with_thresholds, literals, depth, solution
