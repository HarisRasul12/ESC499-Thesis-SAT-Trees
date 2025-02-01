"""
=========== Module Description ===========

Module to build the complete tree and create literals for a given depth and dataset. This module attempts to maximize
the number of correct labels given to the training dataset by adding soft clauses for maximizing correct solutions and
hard clauses for the constraints.
"""

from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

from min_height_tree_module import get_ancestors, compute_ordering, set_branch_node_features, add_thresholds, visualize_tree
from satree.treemodder.builder import build_complete_tree, create_literals
from classification_clauses import construct_maxsat_clauses, add_classification_clauses, add_redundant_constraints


def build_clauses_fixed_tree(literals, X, TB, TL, num_features, labels,true_labels):
    """
    Constructs the clauses for the SAT solver based on the decision tree encoding. Now includes MAX SOLVER PROBLEM FOR FIXED HEIGHT 

    Args:
        literals (dict): A dictionary mapping literals to variable indices.
        X (list): The dataset, a list of tuples representing data points.
        TB (list): Indices of branching nodes.
        TL (list): Indices of leaf nodes.
        num_features (int): Number of features in the dataset.
        labels (list): Possible class labels for the data points.

    Returns:
        WCNF: A WCNF object containing all the clauses, with hard clauses for the tree structure and soft clauses for maximizing correctly classified points
    """

    # Now the problem has become Partial MaxSAT - we will assign weights to the soft clauses Eq. (13). Eq(1-10,12) HARD clauses
    wcnf = construct_maxsat_clauses(literals, X, TB, TL, num_features, labels)

    # Redundant constraints to prune the search space
    wcnf = add_redundant_constraints(wcnf, literals, X, TB, num_features)

    # Add the classification clauses to the CNF
    wcnf = add_classification_clauses(wcnf, literals, X, TL, true_labels)

    return wcnf

def solve_wcnf(wcnf, literals, TL, tree_structure, labels,features,datasetX):
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
    - solution (list or str): The solution to the MaxSAT problem if it exists, otherwise "No solution exists".
    """
    with RC2(wcnf) as m:
        model = m.compute()
        cost = m.cost 
    
    #print(model)
    if model:
        # Update the tree structure with the correct labels for leaf nodes
        for t in TL:
            for label in labels:
                if literals[f'g_{t}_{label}'] in model:
                    tree_structure[t]['label'] = label
                    break
         # Set details for branching nodes
        set_branch_node_features(model, literals, tree_structure,features,datasetX)
        return model, cost
    else:
        return "No solution exists"


def find_fixed_depth_tree(features, labels, true_labels_for_points, dataset,depth):
    solution = "No solution exists"
    tree_with_thresholds = None
    tree = None
    literals = None
    cost = None

    tree, TB, TL = build_complete_tree(depth)
    literals = create_literals(TB, TL, features, labels, len(dataset), True)[0]
    wcnf = build_clauses_fixed_tree(literals, dataset, TB, TL, len(features), labels, true_labels_for_points)
    solution,cost = solve_wcnf(wcnf, literals, TL, tree, labels, features, dataset)

    if solution != "No solution exists":
        tree_with_thresholds = add_thresholds(tree, literals, solution, dataset)
        dot = visualize_tree(tree_with_thresholds)
        dot.render(f'images/fixed_height/binary_decision_tree_fixed_depth_{depth}', format='png', cleanup=True)
    else:
        print('could not find solution')
        return 'No solution'

    return tree_with_thresholds, literals, depth, solution, cost
