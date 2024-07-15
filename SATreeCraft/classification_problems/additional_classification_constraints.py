# Created by: Haris Rasul
# Date: Macrh 4th  2024
# Python script to add on additional user constraints of their choice form pairwise and cardinality constraints such as minimum support and miniumum margin 
# for a given depth and dataset. Will attempt to maximize the number of correct labels given to training dataset 
# adding soft clauses for maximizing corrcet solution for a given depth and hard clauses 

from pysat.formula import CNF
from pysat.solvers import Solver
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from graphviz import Digraph
import numpy as np
from classification_problems.fixed_height_tree_module import *
from classification_problems.fixed_height_tree_categorical_module import *

from pysat.card import CardEnc, IDPool, EncType

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
    # Now the problem has become Partial MaxSAT - we will assign weights to the soft clauses Eq. (13). Eq(1-10,12) HARD clauses 
    wcnf = WCNF()
    
    # Clause (1) and (2): Feature selection at branching nodes
    for t in TB:
        # At least one feature is chosen (Clause 2)
        clause = [literals[f'a_{t}_{j}'] for j in range(num_features)]
        wcnf.append(clause)
        
        # No two features are chosen (Clause 1)
        for j in range(num_features):
            for jp in range(j + 1, num_features):
                clause = [-literals[f'a_{t}_{j}'], -literals[f'a_{t}_{jp}']]
                wcnf.append(clause)

    # Clause (3) and (4): Data point direction based on feature values
    for j in range(num_features):
        Oj = compute_ordering(X, j)
        for (i, ip) in Oj:
            if X[i][j] < X[ip][j]:  # Different feature values (Clause 3)
                for t in TB:
                    wcnf.append([-literals[f'a_{t}_{j}'], literals[f's_{i}_{t}'], -literals[f's_{ip}_{t}']])
            if X[i][j] == X[ip][j]:  # Equal feature values (Clause 4)
                for t in TB:
                    wcnf.append([-literals[f'a_{t}_{j}'], literals[f's_{i}_{t}'], -literals[f's_{ip}_{t}']])
                    wcnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{i}_{t}'], literals[f's_{ip}_{t}']])

    # Clause (5 and 6): Path valididty form right traversla and left traversal 
    for t in TL:
        left_ancestors = get_ancestors(t, 'left')
        right_ancestors = get_ancestors(t, 'right')
        for i in range(len(X)):
            # Data point i ends at leaf node t (Clause 5 and 6) - assumption made!!!
            if left_ancestors:
                wcnf.append([-literals[f'z_{i}_{t}']] + [literals[f's_{i}_{a}'] for a in left_ancestors])
            if right_ancestors:
                wcnf.append([-literals[f'z_{i}_{t}']] + [-literals[f's_{i}_{a}'] for a in right_ancestors])

    # Clause (7): Each data point that does not end up in leaf node t has at least one deviation from the path
    for xi in range(len(X)):
        for t in TL:
            deviations = []
            left_ancestors = get_ancestors(t, 'left')  # Get left ancestors using TB indices
            right_ancestors = get_ancestors(t, 'right')  # Get right ancestors using TB indices
            # Only append deviations if there are ancestors on the corresponding side
            if left_ancestors:
                deviations.extend([-literals[f's_{xi}_{ancestor}'] for ancestor in left_ancestors])
            if right_ancestors:
                deviations.extend([literals[f's_{xi}_{ancestor}'] for ancestor in right_ancestors])
            # Only append the clause if there are any deviations
            if deviations:
                wcnf.append([literals[f'z_{xi}_{t}']] + deviations)    

    # Clause (8): Each leaf node is assigned at most one label
    for t in TL:
        for c in range(len(labels)):
            for cp in range(c + 1, len(labels)):
                wcnf.append([-literals[f'g_{t}_{labels[c]}'], -literals[f'g_{t}_{labels[cp]}']])

    # Clause (9) and (10): Redundant constraints to prune the search space
    # These clauses are optimizations
    for t in TB:
        for j in range(num_features):
            # Get the sorted indices of the data points by feature j
            sorted_by_feature = sorted(range(len(X)), key=lambda k: X[k][j])

            # Clause (9): Data point with the M-th smallest feature value directed left
            if min_margin > 0 and min_margin <= len(X):
                # We subtract 1 because Python indexing is zero-based
                mth_smallest_index = sorted_by_feature[min_margin - 1]
                wcnf.append([-literals[f'a_{t}_{j}'], literals[f's_{mth_smallest_index}_{t}']])

            # Clause (10): Data point with the M-th largest feature value directed right
            if min_margin > 0 and min_margin <= len(X):
                # No need to subtract 1 when using negative indexing in Python
                mth_largest_index = sorted_by_feature[-min_margin]
                wcnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{mth_largest_index}_{t}']])

    # New Hard Clause (12) for ensuring pi is true only when xi ends up in a leaf node with the correct label, REMOVED (CLAUSE 11)
    for i, xi in enumerate(X):
        for t in TL:
            label = true_labels[i]
            # This adds the clause (¬pi ∨ ¬zi,t ∨ gt,γ(xi))
            wcnf.append([-literals[f'p_{i}'], -literals[f'z_{i}_{t}'], literals[f'g_{t}_{label}']])
            
    # Add the soft clauses (13) for each data point being correctly classified
    for i in range(len(X)):
        wcnf.append([literals[f'p_{i}']], weight=1)

    return wcnf

def add_oblivious_tree_constraints(cnf, features, depth, literals):
    """
    Add constraints to the CNF for an oblivious tree where all nodes at the same level
    must select the same feature for splitting.

    Parameters:
    - cnf (CNF or WCNF): The current CNF formula to which we will add the constraints.
    - TB (list): Indices of branching nodes in the tree.
    - features (list): List of features in the dataset.
    - depth (int): The depth of the tree.
    
    Returns:
    - cnf (CNF or WCNF): The CNF formula with the added constraints.
    """
    # print("we got here hello!!!")
    # print("CNF: ", cnf)
    # print("Features", features)
    # print("Depth", depth)
    # print("Literals", literals)
    def level_nodes(level, max_depth):
        """Return the node indices at a given level."""
        start = (2 ** level) - 1
        end = min((2 ** (level + 1)) - 1, (2 ** max_depth) - 1)
        return list(range(start, end))

    for d in range(depth):  # Exclude the last level which has the leaf nodes
        nodes_at_level = level_nodes(d, depth)
        # print(nodes_at_level)
        for feature in features:
            for i in range(len(nodes_at_level)):
                for j in range(i+1, len(nodes_at_level)):
                    t1 = nodes_at_level[i]
                    t2 = nodes_at_level[j]
                    # print(t1)
                    # print(t2)
                    # Add clauses to enforce the same feature is chosen by both nodes
                    cnf.append([-literals[f'a_{t1}_{feature}'], literals[f'a_{t2}_{feature}']])
                    cnf.append([literals[f'a_{t1}_{feature}'], -literals[f'a_{t2}_{feature}']])
    return cnf

def add_oblivious_tree_constraints2(cnf, features, depth, literals, dataset):
    """
    Add constraints to the CNF for an oblivious tree where all nodes at the same level
    must select the same feature and the same threshold for splitting.

    Parameters:
    - cnf (CNF or WCNF): The current CNF formula to which we will add the constraints.
    - features (list): List of features in the dataset.
    - depth (int): The depth of the tree.
    - literals (dict): Dictionary of literals used in the CNF.
    - dataset (list): The dataset being used.

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
                    
                    # Add clauses to enforce the same threshold is chosen by both nodes
                    for k in range(len(dataset)):
                        cnf.append([-literals[f's_{k}_{t1}'], literals[f's_{k}_{t2}']])
                        cnf.append([literals[f's_{k}_{t1}'], -literals[f's_{k}_{t2}']])
                        
    return cnf