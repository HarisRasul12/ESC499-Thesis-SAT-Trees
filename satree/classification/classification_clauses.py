from pysat.formula import WCNF
from min_height_tree_module import get_ancestors, compute_ordering


def construct_maxsat_clauses(literals, X, TB, TL, num_features, labels):
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
        WCNF: A WCNF object containing all the clauses, with hard clauses for the tree structure and soft clauses for maximizing correctly classified points.
    """
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

    # Clause (5 and 6): Path validity from right traversal and left traversal
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

    return wcnf


def add_classification_clauses(wcnf, literals, X, TL, true_labels):
    """
    Adds classification clauses to the WCNF object.

    This function adds hard clauses to ensure that a data point ends up in a leaf node with the correct label,
    and soft clauses to maximize the number of correctly classified data points.

    Args:
        wcnf (WCNF): The WCNF object to which the clauses will be added.
        literals (dict): A dictionary mapping literals to variable indices.
        X (list): The dataset, a list of tuples representing data points.
        TL (list): Indices of leaf nodes.
        true_labels (list): The true labels for the data points.

    Returns:
        WCNF: The updated WCNF object with the added classification clauses.
    """
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


def add_redundant_constraints(cnf, literals, X, TB, num_features):
    """
    Adds redundant constraints to prune the search space.

    This function adds clauses to ensure that the data point with the lowest feature value is directed left
    and the data point with the highest feature value is directed right for each feature at each branching node.

    Args:
        cnf (CNF): The CNF object to which the clauses will be added.
        literals (dict): A dictionary mapping literals to variable indices.
        X (list): The dataset, a list of tuples representing data points.
        TB (list): Indices of branching nodes.
        num_features (int): Number of features in the dataset.

    Returns:
        CNF: The updated CNF object with the added redundant constraints.
    """
    # Clause (9) and (10): Redundant constraints to prune the search space
    # These clauses are optimizations
    for t in TB:
        # Find the data point with the lowest and highest feature value for each feature
        for j in range(num_features):
            sorted_by_feature = sorted(range(len(X)), key=lambda k: X[k][j])
            lowest_value_index = sorted_by_feature[0]
            highest_value_index = sorted_by_feature[-1]

            # Clause (9): The data point with the lowest feature value is directed left
            cnf.append([-literals[f'a_{t}_{j}'], literals[f's_{lowest_value_index}_{t}']])

            # Clause (10): The data point with the highest feature value is directed right
            cnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{highest_value_index}_{t}']])

    return cnf