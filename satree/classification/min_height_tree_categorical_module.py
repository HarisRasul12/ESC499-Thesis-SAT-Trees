"""
=========== Module Description ===========

Base module to help solve SAT problems with categorical and numerical features.
"""

import numpy as np
from pysat.formula import CNF

from satree.classification.min_height_tree_module import solve_cnf, visualize_tree
from satree.treemodder.builder import build_complete_tree, create_literals
from satree.classification.classification_clauses import add_clauses_for_features_and_paths, add_feature_selection_clauses_for_branching_nodes


# Caluses builder for dataeest with cateorgoialc features
def build_clauses_categorical(literals, X, TB, TL, num_features, features_categorical, features_numerical, labels,true_labels):
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

    cnf = add_feature_selection_clauses_for_branching_nodes(cnf, literals, TB, num_features)

    cnf = add_clauses_for_features_and_paths(cnf, literals, X, TB, TL, num_features, features_categorical, features_numerical, labels)

    # Clause (25): Correct class labels for leaf nodes
    for t in TL:
        for i, xi in enumerate(X):
            label = true_labels[i]
            cnf.append([-literals[f'z_{i}_{t}'], literals[f'g_{t}_{label}']])
    
    return cnf

#adjusted Logic to compute threshold on the entire dataset at each feature node branch and categorical feature 
def add_thresholds_categorical(tree_structure, literals, model_solution, dataset, features_categorical):
    def get_literal_value(literal):
        return literals[literal] if literals[literal] in model_solution else -literals[literal]

    def set_thresholds_categorical(node_index, dataset):
        node = tree_structure[node_index]
        if node['type'] == 'branching':
            feature_index = int(node['feature'])
            is_categorical = str(feature_index) in features_categorical
            
            if is_categorical:
                # For categorical features, list the unique values that went left
                categories_that_went_left = set()
                for i, data_point in enumerate(dataset):
                    if get_literal_value(f's_{i}_{node_index}') > 0:
                        categories_that_went_left.add(data_point[feature_index])
                node['threshold'] = sorted(list(categories_that_went_left))
            else:
                # For numerical features, use the existing logic to find the threshold
                feature_values = dataset[:, feature_index].astype(float)
                sorted_indices = np.argsort(feature_values)
                threshold = None
                for i in range(1, len(sorted_indices)):
                    left_index = sorted_indices[i - 1]
                    right_index = sorted_indices[i]
                    if get_literal_value(f's_{left_index}_{node_index}') > 0 and get_literal_value(f's_{right_index}_{node_index}') < 0:
                        threshold = (feature_values[left_index] + feature_values[right_index]) / 2
                        break
                node['threshold'] = threshold
            
            # Continue for children nodes
            left_child_index, right_child_index = node['children'][0], node['children'][1]
            if left_child_index < len(tree_structure):
                set_thresholds_categorical(left_child_index, dataset)
            if right_child_index < len(tree_structure):
                set_thresholds_categorical(right_child_index, dataset)

    # Apply the threshold setting function starting from the root node
    set_thresholds_categorical(0, dataset)
    return tree_structure

def find_min_depth_tree_categorical(features, features_categorical, features_numerical, labels, true_labels_for_points, dataset):
    depth = 1  # Start with a depth of 1
    solution = "No solution exists"
    tree_with_thresholds = None
    tree = None
    literals = None

    while solution == "No solution exists":
        tree, TB, TL = build_complete_tree(depth)
        literals = create_literals(TB, TL, features, labels, len(dataset), False)[0]
        cnf = build_clauses_categorical(literals, dataset, TB, TL, len(features), features_categorical, features_numerical, labels, true_labels_for_points)
        solution = solve_cnf(cnf, literals, TL, tree, labels, features, dataset)
        
        if solution != "No solution exists":
            tree_with_thresholds = add_thresholds_categorical(tree, literals, solution, dataset, features_categorical)
            dot = visualize_tree(tree_with_thresholds)
            dot.render(f'images/min_height/binary_decision_tree_min_depth_with_categorical_features_depth_{depth}', format='png', cleanup=True)
        else:
            print("No solution at depth: ", depth)
            depth += 1  # Increase the depth and try again
    
    return tree_with_thresholds, literals, depth, solution
