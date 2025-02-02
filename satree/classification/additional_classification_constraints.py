"""
=========== Module Description ===========

Module to add additional user constraints such as pairwise and cardinality constraints (minimum support and
minimum margin) for a given depth and dataset. This module attempts to maximize the number of correct labels given
to the training dataset by adding soft clauses for maximizing correct solutions and hard clauses for the constraints.
"""

from pysat.formula import WCNF
from pysat.card import CardEnc, IDPool, EncType

from satree.classification.classification_clauses import construct_maxsat_clauses, add_classification_clauses


def min_support(wcnf, literals, X, TL, min_support):
    """
    Add minimum support constraints to a WCNF object for decision tree leaf nodes.

    This function takes a WCNF object representing a set of constraints for a decision tree,
    a mapping of literals to their indices, the dataset, leaf node indices, and a minimum
    support threshold. It encodes the constraint that at least `min_support` number of data
    points must be present at each leaf node of the decision tree.

    Parameters:
    - wcnf (WCNF): The weighted CNF object to which the constraints will be added.
    - literals (dict): A dictionary mapping each literal to its unique integer identifier.
    - X (list): The dataset containing data points.
    - TL (list): The list of indices corresponding to the leaf nodes of the decision tree.
    - min_support (int): The minimum number of data points required at each leaf node.

    Returns:
    - WCNF: The updated WCNF object with the minimum support constraints included.

    Each leaf node t in TL will have a minimum support constraint ensuring that
    at least `min_support` of the literals associated with it (z literals) must be True.
    Auxiliary variables and clauses for the encoding are managed by an IDPool instance
    to maintain uniqueness of variable identifiers.
    """

    # Initialize the variable pool with the highest index plus one to avoid conflicts
    max_var_index = max(literals.values()) + 1
    vpool = IDPool(start_from=max_var_index)

    # Add the minimum support constraints for each leaf node
    for t in TL:
        # Collect all 'z' literals for the current leaf node
        z_literals = [literals[f'z_{i}_{t}'] for i in range(len(X))]

        # Encode the constraint that at least 'min_support' of these literals must be True
        min_support_clauses = CardEnc.atleast(lits=z_literals, bound=min_support, vpool=vpool, encoding=EncType.seqcounter)

        # Add the clauses for the minimum support constraint to the WCNF
        for clause in min_support_clauses.clauses:
            wcnf.append(clause)

        # Update the variable pool for the next available variable index
        max_var_index = vpool.id()
        vpool = IDPool(start_from=max_var_index)

    return wcnf

def build_clauses_fixed_tree_min_margin_constraint_add(literals, X, TB, TL, num_features, labels,true_labels, min_margin):
    """
    Constructs the clauses for the SAT solver based on the decision tree encoding with MINIUM SPLT/MARGIN
    Only works for numeircal problems fixed height problem 

    Args:
        literals (dict): A dictionary mapping literals to variable indices.
        X (list): The dataset, a list of tuples representing data points.
        TB (list): Indices of branching nodes.
        TL (list): Indices of leaf nodes.
        num_features (int): Number of features in the dataset.
        labels (list): Possible class labels for the data points.
        min_margin (int) : minumim margin constraint added 

    Returns:
        WCNF: A WCNF object containing all the clauses, with hard clauses for the tree structure and soft clauses for maximizing correctly classified points
    """
    wcnf = WCNF()
    # Now the problem has become Partial MaxSAT - we will assign weights to the soft clauses Eq. (13). Eq(1-10,12) HARD clauses
    wcnf = construct_maxsat_clauses(wcnf, literals, X, TB, TL, num_features, labels)

    # Clause (9) and (10): Redundant constraints to prune the search space
    # These clauses are optimizations
    for t in TB:
        for j in range(num_features):
            # Get the sorted indices of the data points by feature j
            sorted_by_feature = sorted(range(len(X)), key=lambda k: X[k][j])

            # Clause (9): Data point with the M-th smallest feature value directed left
            if 0 < min_margin <= len(X):
                # We subtract 1 because Python indexing is zero-based
                mth_smallest_index = sorted_by_feature[min_margin - 1]
                wcnf.append([-literals[f'a_{t}_{j}'], literals[f's_{mth_smallest_index}_{t}']])

            # Clause (10): Data point with the M-th largest feature value directed right
            if 0 < min_margin <= len(X):
                # No need to subtract 1 when using negative indexing in Python
                mth_largest_index = sorted_by_feature[-min_margin]
                wcnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{mth_largest_index}_{t}']])

    # Add the classification clauses to the CNF
    wcnf = add_classification_clauses(wcnf, literals, X, TL, true_labels)

    return wcnf


def add_oblivious_tree_constraints(cnf, features, depth, literals, dataset=None, tree_structure='Oblivious'):
    """
    Add constraints to the CNF for an oblivious tree where all nodes at the same level
    must select the same feature for splitting.

    Parameters:
    - cnf (CNF or WCNF): The current CNF formula to which we will add the constraints.
    - TB (list): Indices of branching nodes in the tree.
    - features (list): List of features in the dataset.
    - depth (int): The depth of the tree.
    - literals (dict): A dictionary mapping literals to their unique integer identifiers.
    - dataset (list): The dataset containing data points.
    - tree_structure (str): The type of tree structure to consider (Oblivious or Oblivious2).

    Returns:
    - cnf (CNF or WCNF): The CNF formula with the added constraints.
    """

    def level_nodes(level, max_depth):
        """Return the node indices at a given level."""
        start = (2 ** level) - 1
        end = min((2 ** (level + 1)) - 1, (2 ** max_depth) - 1)
        return list(range(start, end))


    for d in range(depth):  # Exclude the last level which has the leaf nodes
        nodes_at_level = level_nodes(d, depth)
        for feature in features:
            for i in range(len(nodes_at_level)):
                for j in range(i + 1, len(nodes_at_level)):
                    t1 = nodes_at_level[i]
                    t2 = nodes_at_level[j]
                    # Add clauses to enforce the same feature is chosen by both nodes
                    cnf.append([-literals[f'a_{t1}_{feature}'], literals[f'a_{t2}_{feature}']])
                    cnf.append([literals[f'a_{t1}_{feature}'], -literals[f'a_{t2}_{feature}']])

                    if tree_structure == 'Oblivious2':
                        # Add clauses to enforce the same threshold is chosen by both nodes
                        for k in range(len(dataset)):
                            cnf.append([-literals[f's_{k}_{t1}'], literals[f's_{k}_{t2}']])
                            cnf.append([literals[f's_{k}_{t1}'], -literals[f's_{k}_{t2}']])

    return cnf
