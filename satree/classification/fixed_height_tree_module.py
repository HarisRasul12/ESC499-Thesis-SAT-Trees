"""
=========== Module Description ===========

Module to build the complete tree and create literals for a given depth and dataset. This module attempts to maximize
the number of correct labels given to the training dataset by adding soft clauses for maximizing correct solutions and
hard clauses for the constraints.
"""

from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

from satree.classification.min_height_tree_module import set_branch_node_features, add_thresholds, visualize_tree
from satree.treemodder.builder import build_complete_tree, create_literals
from satree.classification.classification_clauses import construct_maxsat_clauses, add_classification_clauses, add_redundant_constraints


def build_clauses_fixed_tree(literals, dataset, branch_nodes, leaf_nodes, num_features, labels, true_labels):
    """
    Constructs the clauses for the SAT solver based on the decision tree encoding. Now includes MAX SOLVER PROBLEM FOR FIXED HEIGHT 

    Args:
        literals (dict): A dictionary mapping literals to variable indices.
        dataset (list): The dataset, a list of tuples representing data points.
        branch_nodes (list): Indices of branching nodes.
        leaf_nodes (list): Indices of leaf nodes.
        num_features (int): Number of features in the dataset.
        labels (list): Possible class labels for the data points.
        true_labels (list): True labels for each data point in the dataset.

    Returns:
        WCNF: A WCNF object containing all the clauses, with hard clauses for the tree structure and soft clauses for maximizing correctly classified points
    """

    wcnf = WCNF()

    # Now the problem has become Partial MaxSAT - we will assign weights to the soft clauses Eq. (13). Eq(1-10,12) HARD clauses
    wcnf = construct_maxsat_clauses(wcnf, literals, dataset, branch_nodes, leaf_nodes, num_features, labels)

    # Redundant constraints to prune the search space
    wcnf = add_redundant_constraints(wcnf, literals, dataset, branch_nodes, num_features)

    # Add the classification clauses to the CNF
    wcnf = add_classification_clauses(wcnf, literals, dataset, leaf_nodes, true_labels)

    return wcnf

def solve_wcnf(wcnf, literals, leaf_nodes, tree_structure, labels, features):
    """
    Attempts to solve the given CNF using a SAT solver.

    If a solution is found, it updates the tree structure with the correct labels for leaf nodes.

    Args:
        wcnf (CNF): The CNF object containing all clauses for the SAT solver.
        literals (dict): A dictionary mapping literals to variable indices.
        leaf_nodes (list): Indices of leaf nodes in the tree.
        tree_structure (list): The complete binary tree structure.
        labels (list): The list of class labels for the dataset.
        features (list): The list of feature names in the dataset.

    Returns:
    - solution (list or str): The solution to the MaxSAT problem if it exists, otherwise "No solution exists".
    """
    with RC2(wcnf) as m:
        model = m.compute()
        cost = m.cost 
    
    #print(model)
    if model:
        # Update the tree structure with the correct labels for leaf nodes
        for t in leaf_nodes:
            for label in labels:
                if literals[f'g_{t}_{label}'] in model:
                    tree_structure[t]['label'] = label
                    break
         # Set details for branching nodes
        set_branch_node_features(model, literals, tree_structure,features)
        return model, cost
    else:
        return "No solution exists"


def find_fixed_depth_tree(features, labels, true_labels_for_points, dataset, depth):
    """
    Finds a fixed-depth decision tree for a classification problem using SAT-based encoding.

    This function builds a complete binary decision tree of the specified depth, generates SAT literals,
    and constructs CNF clauses using a fixed-tree encoding. It then attempts to solve the CNF with a SAT solver.
    If a solution is found, thresholds are added to the tree's branching nodes and the tree is visualized.
    If no solution is found, the function prints an error message and returns "No solution".

    Args:
        features (list): List of feature identifiers used for splitting in the decision tree.
        labels (list): List of possible class labels for the data points.
        true_labels_for_points (list): The true labels for each data point in the dataset.
        dataset (array-like): The dataset, where each element represents a data point.
        depth (int): The fixed depth at which to construct the decision tree.

    Returns:
        tuple: A tuple containing:
            - tree_with_thresholds: The decision tree with thresholds assigned to branching nodes.
            - literals (dict): A dictionary mapping literal names to their corresponding variable indices.
            - depth (int): The fixed depth of the decision tree.
            - solution (list or str): The SAT solver's solution if one is found, or "No solution" otherwise.
            - cost (int or float): The cost associated with the SAT solution.
    """
    tree, TB, TL = build_complete_tree(depth)
    literals = create_literals(TB, TL, features, labels, len(dataset), True)[0]
    wcnf = build_clauses_fixed_tree(literals, dataset, TB, TL, len(features), labels, true_labels_for_points)
    solution,cost = solve_wcnf(wcnf, literals, TL, tree, labels, features)

    if solution != "No solution exists":
        tree_with_thresholds = add_thresholds(tree, literals, solution, dataset)
        dot = visualize_tree(tree_with_thresholds)
        dot.render(f'images/fixed_height/binary_decision_tree_fixed_depth_{depth}', format='png', cleanup=True)
    else:
        print('could not find solution')
        return 'No solution'

    return tree_with_thresholds, literals, depth, solution, cost
