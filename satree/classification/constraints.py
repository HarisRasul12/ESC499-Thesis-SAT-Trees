"""
=========== Module Description ===========

This module augments the SAT encoding of decision trees with additional user-defined constraints
to improve classification performance. Specifically, it introduces pairwise and cardinality constraints
that enforce both a minimum support requirement at the leaf nodes and a minimum margin (or split)
constraint at the branching nodes.

The minimum support constraints ensure that each leaf node has at least a specified number of data points,
while the minimum margin constraints help maintain a defined separation between data points at branching
nodes. These constraints are encoded as hard clauses (which must be satisfied) and soft clauses (which
are weighted to favor solutions with more correct label assignments) in a Partial MaxSAT framework.

By integrating these constraints, the module aims to maximize the number of correct labels assigned to the
training data while preserving the structural integrity of the decision tree. This approach is particularly
useful when dealing with both categorical and numerical features in a dataset.
"""

from typing import List, Dict, Any, Optional, Union

import numpy as np
from pysat.formula import WCNF
from pysat.card import CardEnc, IDPool, EncType

from satree.classification.sat_clauses import construct_maxsat_clauses, add_classification_clauses


def min_support(wcnf: WCNF,
                literals: Dict[str, int],
                dataset: np.ndarray,
                leaf_nodes: List[int],
                min_sup: int) -> WCNF:
    """
    Add minimum support constraints to a WCNF object for decision tree leaf nodes.

    This function takes a WCNF object representing a set of constraints for a decision tree,
    a mapping of literals to their indices, the dataset, leaf node indices, and a minimum
    support threshold. It encodes the constraint that at least `min_support` number of data
    points must be present at each leaf node of the decision tree.

    Args:
        wcnf: The weighted CNF object to which the constraints will be added.
        literals: A dictionary mapping each literal to its unique integer identifier.
        dataset: The dataset containing data points.
        leaf_nodes: The list of indices corresponding to the leaf nodes of the decision tree.
        min_sup: The minimum number of data points required at each leaf node.

    Returns:
        The updated wcnf object with the minimum support constraints included.

    Each leaf node t in TL will have a minimum support constraint ensuring that
    at least `min_support` of the literals associated with it (z literals) must be True.
    Auxiliary variables and clauses for the encoding are managed by an IDPool instance
    to maintain uniqueness of variable identifiers.
    """

    # Initialize the variable pool with the highest index plus one to avoid conflicts
    max_var_index = max(literals.values()) + 1
    vpool = IDPool(start_from=max_var_index)

    # Add the minimum support constraints for each leaf node
    for t in leaf_nodes:
        # Collect all 'z' literals for the current leaf node
        z_literals = [literals[f'z_{i}_{t}'] for i in range(len(dataset))]

        # Encode the constraint that at least 'min_support' of these literals must be True
        min_support_clauses = CardEnc.atleast(lits=z_literals, bound=min_sup, vpool=vpool, encoding=EncType.seqcounter)

        # Add the clauses for the minimum support constraint to the WCNF
        for clause in min_support_clauses.clauses:
            wcnf.append(clause)

        # Update the variable pool for the next available variable index
        max_var_index = vpool.id()
        vpool = IDPool(start_from=max_var_index)

    return wcnf


def build_clauses_fixed_tree_min_margin_constraint_add(literals: Dict[str, int],
                                                       dataset: np.ndarray,
                                                       branch_nodes: List[int],
                                                       leaf_nodes: List[int],
                                                       num_features: int,
                                                       labels: List[Any],
                                                       true_labels: List[Any],
                                                       min_margin: int) -> WCNF:
    """
    Constructs the clauses for the SAT solver based on the decision tree encoding with MINIUM SPLIT/MARGIN
    Only works for numerical problems fixed height problem

    Args:
        literals: A dictionary mapping literals to variable indices.
        dataset: The dataset, a list of tuples representing data points.
        branch_nodes: Indices of branching nodes.
        leaf_nodes: Indices of leaf nodes.
        num_features: Number of features in the dataset.
        labels: Possible class labels for the data points.
        true_labels: True class labels for the data points.
        min_margin: minimum margin constraint added

    Returns:
        A WCNF object containing all the clauses, with hard clauses for the tree structure and soft clauses for maximizing correctly classified points
    """
    wcnf = WCNF()
    # Now the problem has become Partial MaxSAT - we will assign weights to the soft clauses Eq. (13). Eq(1-10,12) HARD clauses
    wcnf = construct_maxsat_clauses(wcnf, literals, dataset, branch_nodes, leaf_nodes, num_features, labels)

    # Clause (9) and (10): Redundant constraints to prune the search space
    # These clauses are optimizations
    for t in branch_nodes:
        for j in range(num_features):
            # Get the sorted indices of the data points by feature j
            sorted_by_feature = sorted(range(len(dataset)), key=lambda k: float(dataset[k][j]))

            # Clause (9): Data point with the M-th smallest feature value directed left
            if 0 < min_margin <= len(dataset):
                # We subtract 1 because Python indexing is zero-based
                mth_smallest_index = sorted_by_feature[min_margin - 1]
                wcnf.append([-literals[f'a_{t}_{j}'], literals[f's_{mth_smallest_index}_{t}']])

            # Clause (10): Data point with the M-th largest feature value directed right
            if 0 < min_margin <= len(dataset):
                # No need to subtract 1 when using negative indexing in Python
                mth_largest_index = sorted_by_feature[-min_margin]
                wcnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{mth_largest_index}_{t}']])

    # Add the classification clauses to the CNF
    wcnf = add_classification_clauses(wcnf, literals, dataset, leaf_nodes, true_labels)

    return wcnf


def add_oblivious_tree_constraints(cnf: Union[WCNF, Any],  # Use CNF or WCNF as appropriate
                                   features: np.ndarray,
                                   depth: int,
                                   literals: Dict[str, int],
                                   dataset: Optional[np.ndarray] = None,
                                   tree_structure: str = 'Oblivious') -> Union[WCNF, Any]:
    """
    Add constraints to the CNF for an oblivious tree where all nodes at the same level
    must select the same feature for splitting.

    Parameters:
        cnf: The current CNF formula to which we will add the constraints.
        features: List of features in the dataset.
        depth: The depth of the tree.
        literals: A dictionary mapping literals to their unique integer identifiers.
        dataset: The dataset containing data points.
        tree_structure: The type of tree structure to consider (Oblivious or Oblivious2).

    Returns:
        The CNF formula with the added constraints.
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
