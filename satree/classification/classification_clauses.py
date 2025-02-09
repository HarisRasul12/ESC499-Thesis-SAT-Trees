from satree.classification.classification_core import compute_ordering_with_categorical, get_ancestors, compute_ordering

def append_direction_clauses(cnf, literals, t, j, i_index, ip_index, append_both=False):
    """
    Appends direction clauses for a given branching node and feature.

    This function appends:
      - Clause 1: [-a_{t}_{j}, s_{i_index}_{t}, -s_{ip_index}_{t}]
      - Clause 2: [-a_{t}_{j}, -s_{i_index}_{t}, s_{ip_index}_{t}]
        (this second clause is appended only if append_both is True)

    Args:
        cnf (list): The CNF (list of clauses) to which the new clauses will be appended.
        literals (dict): A dictionary mapping literal names (as strings) to their variable indices.
        t (int): The index of the current branching node.
        j (int): The feature index.
        i_index (int): The index of the left data point.
        ip_index (int): The index of the right data point.
        append_both (bool): If True, both clauses will be appended; if False, only the first clause is appended.
    """
    cnf.append([-literals[f'a_{t}_{j}'], literals[f's_{i_index}_{t}'], -literals[f's_{ip_index}_{t}']])
    if append_both:
        cnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{i_index}_{t}'], literals[f's_{ip_index}_{t}']])


def add_data_point_clauses(cnf, literals, dataset, branch_nodes, leaf_nodes, num_features):
    """
    Adds data point direction and path validity clauses to the CNF object.

    This function adds clauses to ensure proper data point direction based on feature values,
    path validity from right and left traversal, and deviations for data points not ending in leaf nodes.

    Args:
        cnf (CNF): The CNF object to which the clauses will be added.
        literals (dict): A dictionary mapping literals to variable indices.
        dataset (list): The dataset, a list of tuples representing data points.
        branch_nodes (list): Indices of branching nodes.
        leaf_nodes (list): Indices of leaf nodes.
        num_features (int): Number of features in the dataset.

    Returns:
        CNF: The updated CNF object with the added data point clauses.
    """
    # Clause (3) and (4): Data point direction based on feature values
    for j in range(num_features):
        Oj = compute_ordering(dataset, j)
        for (i, ip) in Oj:
            if dataset[i][j] < dataset[ip][j]:  # Different feature values (Clause 3)
                for t in branch_nodes:
                    append_direction_clauses(cnf, literals, t, j, i, ip, append_both=False)
            if dataset[i][j] == dataset[ip][j]:  # Equal feature values (Clause 4)
                for t in branch_nodes:
                    append_direction_clauses(cnf, literals, t, j, i, ip, append_both=True)

    cnf = add_path_validity_and_deviation_clauses(cnf, literals, dataset, leaf_nodes)

    return cnf


def construct_feature_selection_clauses(wcnf, literals, dataset, branch_nodes, leaf_nodes, num_features):
    """
    Constructs the feature selection clauses for the SAT solver based on the decision tree encoding.

    Args:
        wcnf (WCNF): The WCNF object to which the clauses will be added.
        literals (dict): A dictionary mapping literals to variable indices.
        dataset (list): The dataset, a list of tuples representing data points.
        branch_nodes (list): Indices of branching nodes.
        leaf_nodes (list): Indices of leaf nodes.
        num_features (int): Number of features in the dataset.

    Returns:
        WCNF: A WCNF object containing all the feature selection clauses.
    """
    wcnf = add_feature_selection_clauses_for_branching_nodes(wcnf, literals, branch_nodes, num_features)
    wcnf = add_data_point_clauses(wcnf, literals, dataset, branch_nodes, leaf_nodes, num_features)

    return wcnf


def construct_maxsat_clauses(wcnf, literals, dataset, branch_nodes, leaf_nodes, num_features, labels):
    """
    Constructs the clauses for the SAT solver based on the decision tree encoding.

    Args:
        wcnf (WCNF): The WCNF object to which the clauses will be added.
        literals (dict): A dictionary mapping literals to variable indices.
        dataset (list): The dataset, a list of tuples representing data points.
        branch_nodes (list): Indices of branching nodes.
        leaf_nodes (list): Indices of leaf nodes.
        num_features (int): Number of features in the dataset.
        labels (list): Possible class labels for the data points.

    Returns:
        WCNF: A WCNF object containing all the clauses, with hard clauses for the tree structure and soft clauses for maximizing correctly classified points.
    """

    wcnf = construct_feature_selection_clauses(wcnf, literals, dataset, branch_nodes, leaf_nodes, num_features)

    # Clause (8): Each leaf node is assigned at most one label
    for t in leaf_nodes:
        for c in range(len(labels)):
            for cp in range(c + 1, len(labels)):
                wcnf.append([-literals[f'g_{t}_{labels[c]}'], -literals[f'g_{t}_{labels[cp]}']])

    return wcnf


def add_classification_clauses(wcnf, literals, dataset, leaf_nodes, true_labels):
    """
    Adds classification clauses to the WCNF object.

    This function adds hard clauses to ensure that a data point ends up in a leaf node with the correct label,
    and soft clauses to maximize the number of correctly classified data points.

    Args:
        wcnf (WCNF): The WCNF object to which the clauses will be added.
        literals (dict): A dictionary mapping literals to variable indices.
        dataset (list): The dataset, a list of tuples representing data points.
        leaf_nodes (list): Indices of leaf nodes.
        true_labels (list): The true labels for the data points.

    Returns:
        WCNF: The updated WCNF object with the added classification clauses.
    """
    # New Hard Clause (12) for ensuring pi is true only when xi ends up in a leaf node with the correct label, REMOVED (CLAUSE 11)
    for i, xi in enumerate(dataset):
        for t in leaf_nodes:
            label = true_labels[i]
            # This adds the clause (¬pi ∨ ¬zi,t ∨ gt,γ(xi))
            wcnf.append([-literals[f'p_{i}'], -literals[f'z_{i}_{t}'], literals[f'g_{t}_{label}']])

    # Add the soft clauses (13) for each data point being correctly classified
    for i in range(len(dataset)):
        wcnf.append([literals[f'p_{i}']], weight=1)

    return wcnf


def add_redundant_constraints(cnf, literals, dataset, branch_nodes, num_features):
    """
    Adds redundant constraints to prune the search space.

    This function adds clauses to ensure that the data point with the lowest feature value is directed left
    and the data point with the highest feature value is directed right for each feature at each branching node.

    Args:
        cnf (CNF): The CNF object to which the clauses will be added.
        literals (dict): A dictionary mapping literals to variable indices.
        dataset (list): The dataset, a list of tuples representing data points.
        branch_nodes (list): Indices of branching nodes.
        num_features (int): Number of features in the dataset.

    Returns:
        CNF: The updated CNF object with the added redundant constraints.
    """
    # Clause (9) and (10): Redundant constraints to prune the search space
    # These clauses are optimizations
    for t in branch_nodes:
        # Find the data point with the lowest and highest feature value for each feature
        for j in range(num_features):
            sorted_by_feature = sorted(range(len(dataset)), key=lambda k: dataset[k][j])
            lowest_value_index = sorted_by_feature[0]
            highest_value_index = sorted_by_feature[-1]

            # Clause (9): The data point with the lowest feature value is directed left
            cnf.append([-literals[f'a_{t}_{j}'], literals[f's_{lowest_value_index}_{t}']])

            # Clause (10): The data point with the highest feature value is directed right
            cnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{highest_value_index}_{t}']])

    return cnf


def add_clauses_for_features_and_paths(cnf, literals, dataset, branch_nodes, leaf_nodes, num_features, features_categorical, features_numerical, labels):
    """
    Adds clauses for feature selection, path validity, and label assignment to the CNF object.

    This function adds clauses to ensure proper feature selection, path validity from right and left traversal,
    deviations for data points not ending in leaf nodes, and label assignment for leaf nodes.

    Args:
        cnf (CNF): The CNF object to which the clauses will be added.
        literals (dict): A dictionary mapping literals to variable indices.
        dataset (list): The dataset, a list of tuples representing data points.
        branch_nodes (list): Indices of branching nodes.
        leaf_nodes (list): Indices of leaf nodes.
        num_features (int): Number of features in the dataset.
        features_categorical (list): List of categorical feature indices.
        features_numerical (list): List of numerical feature indices.
        labels (list): Possible class labels for the data points.

    Returns:
        CNF: The updated CNF object with the added clauses.
    """
    # Clauses (16), (17), and (18)
    for j in range(num_features):
        ordering = compute_ordering_with_categorical(dataset, j, features_categorical)
        for t in branch_nodes:
            for i in range(len(ordering) - 1):
                i_index, ip_index = ordering[i], ordering[i + 1]
                if str(j) in features_categorical:
                    # Clause (18) and (17) for categorical features
                    if dataset[i_index, j] == dataset[ip_index, j]:
                        cnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{i_index}_{t}'], literals[f's_{ip_index}_{t}']])
                        cnf.append([-literals[f'a_{t}_{j}'], literals[f's_{i_index}_{t}'], -literals[f's_{ip_index}_{t}']])
                else:
                    # Clause (16) and (17) for numerical features
                    if float(dataset[i_index, j]) < float(dataset[ip_index, j]):
                        append_direction_clauses(cnf, literals, t, j, i_index, ip_index, append_both=False)
                    if float(dataset[i_index, j]) == float(dataset[ip_index, j]):
                        append_direction_clauses(cnf, literals, t, j, i_index, ip_index, append_both=True)

    cnf = add_path_validity_and_deviation_clauses(cnf, literals, dataset, leaf_nodes)

    # Clause (22): Each leaf node is assigned at most one label
    for t in leaf_nodes:
        for c in range(len(labels)):
            for cp in range(c + 1, len(labels)):
                cnf.append([-literals[f'g_{t}_{labels[c]}'], -literals[f'g_{t}_{labels[cp]}']])

    # Clause (23) and (24)
    for t in branch_nodes:
        for j in range(num_features):
            ordering = compute_ordering_with_categorical(dataset, j, features_categorical)
            if str(j) in features_categorical or str(j) in features_numerical:
                cnf.append([-literals[f'a_{t}_{j}'], literals[f's_{ordering[0]}_{t}']])
            if str(j) in features_numerical:
                cnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{ordering[-1]}_{t}']])

    return cnf


def add_feature_selection_clauses_for_branching_nodes(cnf, literals, branch_nodes, num_features):
    """
    Adds feature selection clauses to the CNF object.

    This function adds clauses to ensure that at least one feature is chosen at each branching node
    and no two features are chosen simultaneously.

    Args:
        cnf (CNF): The CNF object to which the clauses will be added.
        literals (dict): A dictionary mapping literals to variable indices.
        branch_nodes (list): Indices of branching nodes.
        num_features (int): Number of features in the dataset.

    Returns:
        CNF: The updated CNF object with the added feature selection clauses.
    """
    # Clause (14) and (15): Feature selection at branching nodes
    for t in branch_nodes:
        # At least one feature is chosen (Clause 15)
        clause = [literals[f'a_{t}_{j}'] for j in range(num_features)]
        cnf.append(clause)

        # No two features are chosen (Clause 14)
        for j in range(num_features):
            for jp in range(j + 1, num_features):
                clause = [-literals[f'a_{t}_{j}'], -literals[f'a_{t}_{jp}']]
                cnf.append(clause)

    return cnf


def add_path_validity_and_deviation_clauses(cnf, literals, dataset, leaf_nodes):
    """
    Adds path validity and deviation clauses to the CNF object.

    This function adds clauses to ensure path validity from right and left traversal,
    and deviations for data points not ending in leaf nodes.

    Args:
        cnf (CNF): The CNF object to which the clauses will be added.
        literals (dict): A dictionary mapping literals to variable indices.
        dataset (list): The dataset, a list of tuples representing data points.
        leaf_nodes (list): Indices of leaf nodes.

    Returns:
        CNF: The updated CNF object with the added path validity and deviation clauses.
    """
    # Clause (5 and 6): Path validity from right traversal and left traversal
    for t in leaf_nodes:
        left_ancestors = get_ancestors(t, 'left')
        right_ancestors = get_ancestors(t, 'right')
        for i in range(len(dataset)):
            if left_ancestors:
                cnf.append([-literals[f'z_{i}_{t}']] + [literals[f's_{i}_{a}'] for a in left_ancestors])
            if right_ancestors:
                cnf.append([-literals[f'z_{i}_{t}']] + [-literals[f's_{i}_{a}'] for a in right_ancestors])

    # Clause (7): Each data point that does not end up in leaf node t has at least one deviation from the path
    for xi in range(len(dataset)):
        for t in leaf_nodes:
            deviations = []
            left_ancestors = get_ancestors(t, 'left')
            right_ancestors = get_ancestors(t, 'right')
            if left_ancestors:
                deviations.extend([-literals[f's_{xi}_{ancestor}'] for ancestor in left_ancestors])
            if right_ancestors:
                deviations.extend([literals[f's_{xi}_{ancestor}'] for ancestor in right_ancestors])
            if deviations:
                cnf.append([literals[f'z_{xi}_{t}']] + deviations)

    return cnf