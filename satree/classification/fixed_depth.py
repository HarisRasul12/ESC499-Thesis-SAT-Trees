"""
=========== Module Description ===========

This module implements a SAT-based framework for constructing fixed-depth decision trees
for classification problems. It operates by first building a complete binary tree of a
specified depth, then encoding the tree structure as a weighted CNF (WCNF) formula. In
this encoding, SAT literals are introduced to represent:

  • Feature selection at branching nodes (denoted by 'a' literals),
  • Data point routing decisions at branching nodes (denoted by 's' literals),
  • Data point-to-leaf assignments (denoted by 'z' literals), and
  • Label assignments at leaf nodes (denoted by 'g' literals).

The module combines several types of constraints:
  - **Hard constraints** enforce the correct tree structure, valid feature splits, and unique
    label assignments.
  - **Soft constraints** (with unit weights) are added to favor solutions that maximize the number
    of correctly classified data points.
  - **Redundant constraints** are included to prune the search space by enforcing extreme routing
    (i.e., data points with the lowest feature value are directed left and those with the highest
    value are directed right).

After constructing the WCNF, the module employs a Partial MaxSAT solver (RC2 from the PySAT library)
to compute an optimal (or near-optimal) solution. The resulting SAT model is then used to update the
tree structure—assigning features and computed thresholds to branching nodes and correct labels to
leaf nodes. Finally, the decision tree is visualized using Graphviz, providing a clear picture of the
branching decisions and threshold values.

In summary, this module provides a mathematically rigorous method for encoding and solving decision
tree classification problems via SAT, effectively handling both numerical and categorical features.
"""

from typing import List, Dict, Any, Tuple, Union

import numpy as np
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

from satree.treemodder.builder import build_complete_tree, create_literals

from satree.common_sat_clauses import add_redundant_constraints
from satree.classification.min_depth import set_branch_node_features, add_thresholds, visualize_tree
from satree.classification.sat_clauses import construct_maxsat_clauses, add_classification_clauses


def build_clauses_fixed_tree(literals: Dict[str, int],
                             dataset: np.ndarray,
                             branch_nodes: List[int],
                             leaf_nodes: List[int],
                             num_features: int,
                             labels: List[Any],
                             true_labels: List[Any]) -> WCNF:
    """
    Constructs the clauses for the SAT solver based on the decision tree encoding. Now includes MAX SOLVER PROBLEM FOR FIXED HEIGHT 

    Args:
        literals: A dictionary mapping literals to variable indices.
        dataset: The dataset, a list of tuples representing data points.
        branch_nodes: Indices of branching nodes.
        leaf_nodes: Indices of leaf nodes.
        num_features: Number of features in the dataset.
        labels: Possible class labels for the data points.
        true_labels: True labels for each data point in the dataset.

    Returns:
        A WCNF object containing all the clauses, with hard clauses for the tree structure and soft clauses for maximizing correctly classified points
    """

    wcnf = WCNF()

    # Now the problem has become Partial MaxSAT - we will assign weights to the soft clauses Eq. (13). Eq(1-10,12) HARD clauses
    wcnf = construct_maxsat_clauses(wcnf, literals, dataset, branch_nodes, leaf_nodes, num_features, labels)

    # Redundant constraints to prune the search space
    wcnf = add_redundant_constraints(wcnf, literals, dataset, branch_nodes, num_features)

    # Add the classification clauses to the CNF
    wcnf = add_classification_clauses(wcnf, literals, dataset, leaf_nodes, true_labels)

    return wcnf


def solve_wcnf(wcnf: WCNF,
               literals: Dict[str, int],
               leaf_nodes: List[int],
               tree_structure: List[Dict[str, Any]],
               labels: List[Any],
               features: np.ndarray) -> Union[Tuple[List[int], Union[int, float]], str]:
    """
    Attempts to solve the given CNF using a SAT solver.

    If a solution is found, it updates the tree structure with the correct labels for leaf nodes.

    Args:
        wcnf: The CNF object containing all clauses for the SAT solver.
        literals: A dictionary mapping literals to variable indices.
        leaf_nodes: Indices of leaf nodes in the tree.
        tree_structure: The complete binary tree structure.
        labels: The list of class labels for the dataset.
        features: The list of feature names in the dataset.

    Returns:
        The solution to the MaxSAT problem if it exists, otherwise "No solution exists".
    """
    with RC2(wcnf) as m:
        model = m.compute()
        cost = m.cost

        # print(model)
    if model:
        # Update the tree structure with the correct labels for leaf nodes
        for t in leaf_nodes:
            for label in labels:
                if literals[f'g_{t}_{label}'] in model:
                    tree_structure[t]['label'] = label
                    break
        # Set details for branching nodes
        set_branch_node_features(model, literals, tree_structure, features)
        return model, cost
    else:
        return "No solution exists"


def find_fixed_depth_tree(features: np.ndarray,
                          labels: List[Any],
                          true_labels_for_points: List[Any],
                          dataset: np.ndarray,
                          depth: int) -> str | Tuple[
    List[Dict[str, Any]], Dict[str, int], int, Union[List[int], str], Union[int, float]]:
    """
    Finds a fixed-depth decision tree for a classification problem using SAT-based encoding.

    This function builds a complete binary decision tree of the specified depth, generates SAT literals,
    and constructs CNF clauses using a fixed-tree encoding. It then attempts to solve the CNF with a SAT solver.
    If a solution is found, thresholds are added to the tree's branching nodes and the tree is visualized.
    If no solution is found, the function prints an error message and returns "No solution".

    Args:
        features: List of feature identifiers used for splitting in the decision tree.
        labels: List of possible class labels for the data points.
        true_labels_for_points: The true labels for each data point in the dataset.
        dataset: The dataset, where each element represents a data point.
        depth: The fixed depth at which to construct the decision tree.

    Returns:
        A tuple containing:
            - tree_with_thresholds: The decision tree with thresholds assigned to branching nodes.
            - literals: A dictionary mapping literal names to their corresponding variable indices.
            - depth: The fixed depth of the decision tree.
            - solution: The SAT solver's solution if one is found, or "No solution" otherwise.
            - cost: The cost associated with the SAT solution.
    """
    tree, TB, TL = build_complete_tree(depth)
    literals = create_literals(TB, TL, features, labels, len(dataset), True)[0]
    wcnf = build_clauses_fixed_tree(literals, dataset, TB, TL, len(features), labels, true_labels_for_points)
    solution, cost = solve_wcnf(wcnf, literals, TL, tree, labels, features)

    if solution != "No solution exists":
        tree_with_thresholds = add_thresholds(tree, literals, solution, dataset)
        dot = visualize_tree(tree_with_thresholds)
        dot.render(f'images/fixed_height/binary_decision_tree_fixed_depth_{depth}', format='png', cleanup=True)
    else:
        print('could not find solution')
        return 'No solution'

    return tree_with_thresholds, literals, depth, solution, cost
