"""
=========== Module Description ===========

This module implements a SAT-based approach for constructing and visualizing decision trees
of minimal depth. By incrementally increasing the tree depth until a satisfiable solution
is found, the module encodes constraints and objectives as a CNF formula, leverages a SAT
solver to identify valid assignments, and assigns computed thresholds to branching nodes.

Key mathematical and algorithmic aspects include:

1. **CNF Construction**:
   - A maxSAT-like encoding is employed, wherein clauses represent:
     • Feature selection (a_{t}_{j}),
     • Data point routing decisions (s_{i}_{t}),
     • Data point-to-leaf assignments (z_{i}_{t}),
     • Label assignments at leaf nodes (g_{t}_{label}).
   - Redundant constraints prune the search space by guiding extreme data-point routing (lowest or highest values).
   - Once the constraints are constructed, they are passed to a SAT solver.

2. **SAT Solving and Model Extraction**:
   - If a solution is found, the module updates the tree structure:
     • Assigning features to branching nodes,
     • Computing thresholds for each branching node (using numerical splits),
     • Assigning labels to leaves.

3. **Visualization**:
   - A Graphviz Digraph is created to illustrate the resultant decision tree, showing branching nodes,
     threshold values, and leaf labels.

By exploring increasingly deep trees, this module ensures a minimal-depth tree that satisfies
the encoding’s constraints, effectively combining structural validity with classification needs
in a rigorous mathematical framework.
"""

from typing import List, Dict, Any, Tuple, Union

import numpy as np
from graphviz import Digraph
from pysat.formula import CNF
from pysat.solvers import Solver

from satree.classification.common_ops import compute_numerical_threshold
from satree.treemodder.builder import build_complete_tree, create_literals
from satree.classification.sat_clauses import construct_maxsat_clauses
from satree.common_sat_clauses import add_redundant_constraints


def build_clauses(literals: Dict[str, int],
                  dataset: np.ndarray,
                  branch_nodes: List[int],
                  leaf_nodes: List[int],
                  num_features: int,
                  labels: List[Any],
                  true_labels: List[Any]) -> CNF:
    """
    Constructs a CNF encoding for the min-depth optimal decision tree problem.

    The formulation includes clauses for feature selection, ensuring valid data routing through the tree, and
    enforcing that each leaf node’s assigned label matches the training data. This encoding mirrors the rules
    set out in Sections 3.1 and 3.2 of the paper.

    Args:
        literals: A dictionary mapping literals to variable indices.
        dataset: The dataset, a list of tuples representing data points.
        branch_nodes: Indices of branching nodes.
        leaf_nodes: Indices of leaf nodes.
        num_features: Number of features in the dataset.
        labels: Possible class labels for the data points.
        true_labels: The true class labels for the data points.

    Returns:
        A CNF object containing all the clauses.
    """
    cnf = CNF()

    cnf = construct_maxsat_clauses(cnf, literals, dataset, branch_nodes, leaf_nodes, num_features, labels)

    cnf = add_redundant_constraints(cnf, literals, dataset, branch_nodes, num_features)

    # Clause (11): Correct class labels for leaf nodes
    for t in leaf_nodes:
        for i, xi in enumerate(dataset):
            label = true_labels[i]
            cnf.append([-literals[f'z_{i}_{t}'], literals[f'g_{t}_{label}']])

    return cnf


def set_branch_node_features(model: List[int],
                             literals: Dict[str, int],
                             tree_structure: List[Dict[str, Any]],
                             features: List[str]) -> None:
    """
    Decodes the SAT solution to determine the selected feature at each branching node.

    By inspecting which feature selection literal is set to true in the SAT model, the function updates the tree
    structure accordingly. This decoding directly implements the mapping from SAT variables to decision tree splits
    as described in the paper.

    Args:
        model: The model returned by the SAT solver.
        literals: A dictionary mapping literals to variable indices.
        tree_structure: The complete binary tree structure.
        features: List of features in the dataset.
    """

    # For each branching node, determine the chosen feature and threshold
    for node_index in range(len(tree_structure)):
        # print(node_index)
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


def solve_cnf(cnf: CNF,
              literals: Dict[str, int],
              leaf_nodes: List[int],
              tree_structure: List[Dict[str, Any]],
              labels: List[Any],
              features: List[str]) -> Union[List[int], str]:
    """
    Attempts to solve the CNF encoding of the decision tree using a SAT solver.

    If a satisfying assignment is found, the function decodes the model to update leaf node labels and branching
    decisions, effectively realizing the SAT decoding step in the mathematical framework.

    Args:
        cnf: The CNF object containing all clauses for the SAT solver.
        literals: A dictionary mapping literals to variable indices.
        leaf_nodes: Indices of leaf nodes in the tree.
        tree_structure: The complete binary tree structure.
        labels: The list of class labels for the dataset.
        features: The list of feature names in the dataset.

    Returns:
        The solution to the SAT problem if it exists, otherwise "No solution exists".
    """
    solver = Solver()
    solver.append_formula(cnf)
    if solver.solve():
        model = solver.get_model()
        # Update the tree structure with the correct labels for leaf nodes
        for t in leaf_nodes:
            for label in labels:
                if literals[f'g_{t}_{label}'] in model:
                    tree_structure[t]['label'] = label
                    break
        # Set details for branching nodes
        set_branch_node_features(model, literals, tree_structure, features)
        return model
    else:
        # print("no solution!")
        return "No solution exists"


def add_thresholds(tree_structure: List[Dict[str, Any]],
                   literals: Dict[str, int],
                   model_solution: List[int],
                   dataset: np.ndarray) -> List[Dict[str, Any]]:
    """
    Computes and assigns numerical thresholds to branching nodes based on the SAT model solution.

    For each branching node, the threshold is determined by locating the transition in literal values over the
    sorted feature order and averaging the corresponding feature values. This mirrors the threshold decoding
    procedure outlined in the paper.

    Args:
        tree_structure: The binary tree structure containing nodes.
        literals: The mapping of literals to variable indices.
        model_solution: The model solution from the SAT solver.
        dataset: The dataset containing all the data points.

    Returns:
        The updated tree structure with thresholds set for branching nodes.
    """

    def get_literal_value(literal):
        return literals[literal] if literals[literal] in model_solution else -literals[literal]

    def set_thresholds(node_index, data):
        node = tree_structure[node_index]
        if node['type'] == 'branching':
            feature_index = int(node['feature'])
            feature_values = data[:, feature_index]
            # Use the helper function to compute the threshold.
            node['threshold'] = compute_numerical_threshold(feature_values, node_index, get_literal_value)

            # Continue for children nodes.
            left_child_index, right_child_index = node['children'][0], node['children'][1]
            if left_child_index < len(tree_structure):
                set_thresholds(left_child_index, data)
            if right_child_index < len(tree_structure):
                set_thresholds(right_child_index, data)

    set_thresholds(0, dataset)
    return tree_structure


def add_nodes(dot: Digraph,
              tree: List[Dict[str, Any]],
              node_index: int = 0) -> None:
    """
    Recursively adds nodes and edges to a Graphviz Digraph object based on the given tree structure.

    This function traverses the tree (represented as a list of nodes, where each node is a dictionary)
    starting from the specified node index, adding each node and its connecting edges to the provided
    Digraph object for visualization.

    Args:
        dot: A Graphviz Digraph object used for visualizing the tree.
        tree: The complete tree structure (list of nodes).
        node_index: The index of the current node to process (defaults to 0 for the root).
    """
    node = tree[node_index]
    if node['type'] == 'branching':
        dot.node(str(node_index),
                 label=f"BranchNode:\n{node_index}\nFeature:{node['feature']}\nThreshold:{node['threshold']}")
        for child_index in node['children']:
            add_nodes(dot, tree, child_index)
            dot.edge(str(node_index), str(child_index))
    elif node['type'] == 'leaf':
        dot.node(str(node_index), label=f"LeafNode:\n{node_index}\nLabel: {node['label']}")


def visualize_tree(tree_structure: List[Dict[str, Any]]) -> Digraph:
    """
    Creates a Graphviz Digraph visualization of the decision tree structure.

    By leveraging a recursive node addition process, this function produces a diagram that clearly displays branching
    nodes (with their features and thresholds) and leaf nodes (with assigned class labels), thus bridging the gap
    between the SAT encoding and an interpretable decision model.

    Args:
        tree_structure: The complete tree structure (list of nodes) to be visualized.

    Returns:
        A Graphviz Digraph object representing the tree.
    """
    dot = Digraph()
    add_nodes(dot, tree_structure)
    return dot


def find_min_depth_tree(features: List[str],
                        labels: List[Any],
                        true_labels_for_points: List[Any],
                        dataset: np.ndarray) -> Tuple[List[Dict[str, Any]], Dict[str, int], int, Union[List[int], str]]:
    """
    Finds the minimum depth decision tree for a classification problem using SAT-based encoding.

    The function incrementally increases the depth of a complete binary tree until a satisfiable solution is found.
    For each depth, it:
      - Builds a complete tree.
      - Generates SAT literals.
      - Constructs the corresponding CNF clauses.
      - Attempts to solve the CNF with a SAT solver.
      - If a solution is found, it computes thresholds for the branching nodes and visualizes the tree.
      - If no solution is found, it increases the depth and tries again.

    Args:
        features: List of feature identifiers used for splitting in the decision tree.
        labels: List of possible class labels.
        true_labels_for_points: The true class labels for each data point.
        dataset: The dataset containing data points (each data point is represented as a tuple or array).

    Returns:
        A tuple containing:
            - tree_with_thresholds: The final decision tree structure with computed thresholds.
            - literals: A dictionary mapping SAT literal names to their variable indices.
            - depth: The depth of the found decision tree.
            - solution: The SAT solver's solution if found, or "No solution exists" if unsolvable.
    """
    depth = 1  # Start with a depth of 1
    solution = "No solution exists"
    tree_with_thresholds = None
    literals = None

    while solution == "No solution exists":
        tree, TB, TL = build_complete_tree(depth)
        literals = create_literals(TB, TL, features, labels, len(dataset), False)[0]
        cnf = build_clauses(literals, dataset, TB, TL, len(features), labels, true_labels_for_points)
        solution = solve_cnf(cnf, literals, TL, tree, labels, features)

        if solution != "No solution exists":
            tree_with_thresholds = add_thresholds(tree, literals, solution, dataset)
            dot = visualize_tree(tree_with_thresholds)
            dot.render(f'images/min_height/binary_decision_tree_min_depth_{depth}', format='png', cleanup=True)
        else:
            print('no solution at depth', depth)
            depth += 1  # Increase the depth and try again

    return tree_with_thresholds, literals, depth, solution
