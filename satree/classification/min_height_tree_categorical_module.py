"""
=========== Module Description ===========

This module implements a SAT-based approach for constructing minimum-depth decision trees
tailored for categorical classification problems. The module encodes the structure and
constraints of a decision tree as a CNF formula by introducing SAT literals that represent:
  • Feature selection at branching nodes (denoted by 'a' literals),
  • Data point routing decisions at branching nodes (denoted by 's' literals),
  • Data point-to-leaf assignments (denoted by 'z' literals), and
  • Label assignments at the leaves (denoted by 'g' literals).

Key mathematical and algorithmic components include:
  - Building CNF clauses that capture the decision tree constraints, including both hard clauses
    (which enforce the tree’s structure, valid splits, and unique label assignments) and soft clauses
    (which bias the solver toward maximizing correct classifications).
  - Computing an ordering of data point indices for each feature: for categorical features, indices
    are grouped by unique categories; for numerical features, indices are sorted by their numeric values.
  - Determining decision thresholds for branching nodes. For categorical features, the threshold is
    defined as the sorted list of unique category values that are directed left, while for numerical
    features the threshold is computed as the average of two adjacent values where the routing decision
    (as indicated by the SAT literals) changes.
  - Iteratively increasing the tree depth until a satisfiable solution is found by the SAT solver.
    Once a solution is obtained, the module augments the tree structure with the computed thresholds and
    visualizes the decision tree using Graphviz.

In summary, this module provides a mathematically rigorous framework for encoding and solving
decision tree classification problems via SAT, ensuring that the resulting tree structure optimally
satisfies both the structural constraints and the classification objectives for datasets with
categorical (and numerical) features.
"""

from typing import List, Dict, Any, Tuple, Union

import numpy as np
from pysat.formula import CNF

from satree.classification.classification_core import compute_numerical_threshold
from satree.classification.min_height_tree_module import solve_cnf, visualize_tree
from satree.treemodder.builder import build_complete_tree, create_literals
from satree.classification.classification_clauses import add_clauses_for_features_and_paths, \
    add_feature_selection_clauses_for_branching_nodes


def build_clauses_categorical(literals: Dict[str, int],
                              dataset: np.ndarray,
                              branch_nodes: List[int],
                              leaf_nodes: List[int],
                              num_features: int,
                              features_categorical: List[str],
                              features_numerical: List[str],
                              labels: List[Any],
                              true_labels: List[Any]) -> CNF:
    """
    Constructs the clauses for the SAT solver based on the decision tree encoding.

    Args:
        literals: A dictionary mapping literals to variable indices.
        dataset: The dataset, a list of tuples representing data points.
        branch_nodes: Indices of branching nodes.
        leaf_nodes: Indices of leaf nodes.
        num_features: Number of features in the dataset.
        features_categorical: List of categorical features.
        features_numerical: List of numerical features.
        labels: Possible class labels for the data points.
        true_labels: The true class labels for the data points.

    Returns:
        A CNF object containing all the clauses.
    """
    cnf = CNF()
    cnf = add_feature_selection_clauses_for_branching_nodes(cnf, literals, branch_nodes, num_features)
    cnf = add_clauses_for_features_and_paths(cnf, literals, dataset, branch_nodes, leaf_nodes, num_features,
                                             features_categorical, features_numerical, labels)

    # Clause (25): Correct class labels for leaf nodes
    for t in leaf_nodes:
        for i, xi in enumerate(dataset):
            label = true_labels[i]
            cnf.append([-literals[f'z_{i}_{t}'], literals[f'g_{t}_{label}']])

    return cnf


def add_thresholds_categorical(tree_structure: List[Dict[str, Any]],
                               literals: Dict[str, int],
                               model_solution: List[int],
                               dataset: np.ndarray,
                               features_categorical: List[str]) -> List[Dict[str, Any]]:
    """
    Adds thresholds to each branching node in the tree structure based on the entire dataset.

    For categorical features, the threshold is the sorted list of unique values that went left.
    For numerical features, the threshold is computed as the average of two adjacent data point values
    where the data point direction changes.

    Args:
        tree_structure: The complete tree structure (list of nodes).
        literals: A dictionary mapping literal names to variable indices.
        model_solution: The SAT solver's model solution.
        dataset: The dataset containing data points.
        features_categorical: List of categorical features.

    Returns:
        The updated tree structure with thresholds added for branching nodes.
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


def find_min_depth_tree_categorical(features: List[str],
                                    features_categorical: List[str],
                                    features_numerical: List[str],
                                    labels: List[Any],
                                    true_labels_for_points: List[Any],
                                    dataset: np.ndarray) -> Tuple[
    List[Dict[str, Any]], Dict[str, int], int, Union[List[int], str]]:
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
        features: List of feature names or indices used for splitting.
        features_categorical: List of indices or identifiers for categorical features.
        features_numerical: List of indices or identifiers for numerical features.
        labels: Possible class labels for the data points.
        true_labels_for_points: The true class labels for each data point.
        dataset: The dataset containing data points (each data point is a tuple or array).

    Returns:
        A tuple containing:
            - tree_with_thresholds: The decision tree with thresholds added (if a solution is found).
            - literals: A dictionary mapping literal names to their indices.
            - depth: The depth of the found tree.
            - solution: The SAT solver's model solution, or "No solution exists" if unsolvable.
    """

    depth = 1  # Start with a depth of 1
    solution = "No solution exists"
    tree_with_thresholds = None
    literals = None

    while solution == "No solution exists":
        tree, TB, TL = build_complete_tree(depth)
        literals = create_literals(TB, TL, features, labels, len(dataset), False)[0]
        cnf = build_clauses_categorical(literals, dataset, TB, TL, len(features), features_categorical,
                                        features_numerical, labels, true_labels_for_points)
        solution = solve_cnf(cnf, literals, TL, tree, labels, features)

        if solution != "No solution exists":
            tree_with_thresholds = add_thresholds_categorical(tree, literals, solution, dataset, features_categorical)
            dot = visualize_tree(tree_with_thresholds)
            dot.render(f'images/min_height/binary_decision_tree_min_depth_with_categorical_features_depth_{depth}',
                       format='png', cleanup=True)
        else:
            print("No solution at depth: ", depth)
            depth += 1  # Increase the depth and try again

    return tree_with_thresholds, literals, depth, solution
