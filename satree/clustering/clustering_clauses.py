from pysat.formula import WCNF
from satree.classification.min_height_tree_module import compute_ordering, get_ancestors


def construct_clustering_clauses(literals, X, TB, TL, num_features, k_clusters, CL_pairs, ML_pairs, distance_classes):
    """
    Constructs the clauses for the SAT solver based on the decision tree encoding for clustering.

    Args:
        literals (dict): A dictionary mapping literals to variable indices.
        X (list): The dataset, a list of tuples representing data points.
        TB (list): Indices of branching nodes.
        TL (list): Indices of leaf nodes.
        num_features (int): Number of features in the dataset.
        k_clusters (int): Number of clusters.
        CL_pairs (list): Cannot-link pairs.
        ML_pairs (list): Must-link pairs.
        distance_classes (list): List of pairs in each distance class.

    Returns:
        WCNF: A WCNF object containing all the clauses, with hard clauses for the tree structure and soft clauses for maximizing correctly classified points.
    """
    wcnf = WCNF()

    # Clause (7) and (8): Feature selection at branching nodes
    for t in TB:
        # At least one feature is chosen (Clause 7)
        clause = [literals[f'a_{t}_{j}'] for j in range(num_features)]
        wcnf.append(clause)

        # No two features are chosen (Clause 8)
        for j in range(num_features):
            for jp in range(j + 1, num_features):
                clause = [-literals[f'a_{t}_{j}'], -literals[f'a_{t}_{jp}']]
                wcnf.append(clause)

    # Clause (9) and (10): Data point direction based on feature values
    for j in range(num_features):
        Oj = compute_ordering(X, j)
        for (i, ip) in Oj:
            if X[i][j] < X[ip][j]:  # Different feature values (Clause 9)
                for t in TB:
                    wcnf.append([-literals[f'a_{t}_{j}'], literals[f's_{i}_{t}'], -literals[f's_{ip}_{t}']])
            if X[i][j] == X[ip][j]:  # Equal feature values (Clause 10)
                for t in TB:
                    wcnf.append([-literals[f'a_{t}_{j}'], literals[f's_{i}_{t}'], -literals[f's_{ip}_{t}']])
                    wcnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{i}_{t}'], literals[f's_{ip}_{t}']])

    # Clause (11 and 12): Path validity from right traversal and left traversal
    for t in TL:
        left_ancestors = get_ancestors(t, 'left')
        right_ancestors = get_ancestors(t, 'right')
        for i in range(len(X)):
            # Data point i ends at leaf node t (Clause 11 and 12) - assumption made!!!
            if left_ancestors:
                wcnf.append([-literals[f'z_{i}_{t}']] + [literals[f's_{i}_{a}'] for a in left_ancestors])
            if right_ancestors:
                wcnf.append([-literals[f'z_{i}_{t}']] + [-literals[f's_{i}_{a}'] for a in right_ancestors])

    # Clause (13): Each data point that does not end up in leaf node t has at least one deviation from the path
    for i in range(len(X)):
        for t in TL:
            deviations = []
            left_ancestors = get_ancestors(t, 'left')  # Get left ancestors using TB indices
            right_ancestors = get_ancestors(t, 'right')  # Get right ancestors using TB indices
            # Only append deviations if there are ancestors on the corresponding side
            if left_ancestors:
                deviations.extend([-literals[f's_{i}_{ancestor}'] for ancestor in left_ancestors])
            if right_ancestors:
                deviations.extend([literals[f's_{i}_{ancestor}'] for ancestor in right_ancestors])
            # Only append the clause if there are any deviations
            if deviations:
                wcnf.append([literals[f'z_{i}_{t}']] + deviations)

    # Clause (14) and (15): Redundant constraints to prune the search space
    for t in TB:
        # Find the data point with the lowest and highest feature value for each feature
        for j in range(num_features):
            sorted_by_feature = sorted(range(len(X)), key=lambda k: X[k][j])
            lowest_value_index = sorted_by_feature[0]
            highest_value_index = sorted_by_feature[-1]

            # Clause (14): The data point with the lowest feature value is directed left
            wcnf.append([-literals[f'a_{t}_{j}'], literals[f's_{lowest_value_index}_{t}']])

            # Clause (15): The data point with the highest feature value is directed right
            wcnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{highest_value_index}_{t}']])

    return wcnf


def add_clustering_encodings(wcnf, literals, X, TL, k_clusters, CL_pairs, ML_pairs, distance_classes):
    """
    Adds clustering clauses to the WCNF object.

    This function adds various clauses to ensure proper clustering, including unary encoding of cluster labels,
    assignment of data points to clusters, and constraints for must-link and cannot-link pairs.

    Args:
        wcnf (WCNF): The WCNF object to which the clauses will be added.
        literals (dict): A dictionary mapping literals to variable indices.
        X (list): The dataset, a list of tuples representing data points.
        TL (list): Indices of leaf nodes.
        k_clusters (int): Number of clusters.
        CL_pairs (list): Cannot-link pairs.
        ML_pairs (list): Must-link pairs.
        distance_classes (list): List of pairs in each distance class.

    Returns:
        WCNF: The updated WCNF object with the added clustering clauses.
    """
    # Clause 16: Unary encoding of cluster labels in each leaf
    for t in TL:
        for c in range(k_clusters - 2):
            clause = [literals[f'g_{t}_{c}'], -literals[f'g_{t}_{c+1}']]
            wcnf.append(clause)

    # Clause 17: Data points ending at leaf node t are assigned to cluster c if g_t,c is true
    for t in TL:
        for i in range(len(X)):
            for c in range(k_clusters - 1):
                clause = [-literals[f'z_{i}_{t}'], -literals[f'g_{t}_{c}'], literals[f'x_{i}_{c}']]
                wcnf.append(clause)

    # Clause 18: Data points ending at leaf node t are NOT assigned to cluster c if g_t,c is false
    for t in TL:
        for i in range(len(X)):
            for c in range(k_clusters - 1):
                clause = [-literals[f'z_{i}_{t}'], literals[f'g_{t}_{c}'], -literals[f'x_{i}_{c}']]
                wcnf.append(clause)

    # Clause 19: Ensure no cluster is empty by ensuring there's at least one data point in each cluster
    for c in range(k_clusters - 1):
        wcnf.append([-literals[f'x_{c}_{c}']])

    # Clause 20: If xi is not in cluster c, then there must be some xi' in cluster c-1, for all c < i
    for i in range(1, len(X)):
        for c in range(1, k_clusters - 1):
            clause = [-literals[f'x_{i}_{c}']]
            for i_prime in range(i):
                clause.append(literals[f'x_{i_prime}_{c-1}'])
            wcnf.append(clause)

    # Clause 21: Ensure all clusters are non-empty by requiring at least one point is assigned to each cluster
    clauseTW = [literals[f'x_{i}_{k_clusters - 2}'] for i in range(len(X))]
    wcnf.append(clauseTW)

    # Clause 22: Ensure that pairs in CL are not clustered in the first cluster (0-indexed)
    for i, i_prime in CL_pairs:
        wcnf.append([literals[f'x_{i}_0'], literals[f'x_{i_prime}_0']])

    # Clause 23: Ensure that pairs in CL are not clustered in the last cluster (k-2 in 0-indexed system)
    for i, i_prime in CL_pairs:
        wcnf.append([-literals[f'x_{i}_{k_clusters - 2}'], -literals[f'x_{i_prime}_{k_clusters - 2}']])

    # Clause 24: Unconditional separating clauses for cannot-link pairs, applied to clusters from 0 to k-3
    for (i, i_prime) in CL_pairs:
        for c in range(k_clusters - 2):
            wcnf.append([
                -literals[f'x_{i}_{c}'],
                -literals[f'x_{i_prime}_{c}'],
                literals[f'x_{i}_{c+1}'],
                literals[f'x_{i_prime}_{c+1}']
            ])

    # Clause 25 and 26: Ensure that pairs in ML are clustered together for each cluster
    for i, i_prime in ML_pairs:
        for c in range(k_clusters - 1):
            wcnf.append([-literals[f'x_{i}_{c}'], literals[f'x_{i_prime}_{c}']])  # clause 25
            wcnf.append([literals[f'x_{i}_{c}'], -literals[f'x_{i_prime}_{c}']])  # clause 26

    # Clause 27: Conditional separating clauses using distance classes and bw_m literals
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            wcnf.append([literals[f'bw_m_{w}'], literals[f'x_{i}_0'], literals[f'x_{i_prime}_0']])

    # Clause 28: Ensure that if bw^-_w is true, then the pair (i, i') from Dw cannot be in the second to last cluster k-2
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            wcnf.append([literals[f'bw_m_{w}'], -literals[f'x_{i}_{k_clusters - 2}'], -literals[f'x_{i_prime}_{k_clusters - 2}']])

    # Clause 29: Conditional co-separation for non-adjacent clusters
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            for c in range(k_clusters - 2):
                wcnf.append([literals[f'bw_m_{w}'], -literals[f'x_{i}_{c}'],
                             -literals[f'x_{i_prime}_{c}'], literals[f'x_{i}_{c+1}'], literals[f'x_{i_prime}_{c+1}']])

    return wcnf


def add_distance_class_clauses(wcnf, literals, k_clusters, distance_classes):
    """
    Adds distance class clauses to the WCNF object.

    This function adds various clauses to ensure proper clustering based on distance classes,
    including constraints for must-link and cannot-link pairs within distance classes.

    Args:
        wcnf (WCNF): The WCNF object to which the clauses will be added.
        literals (dict): A dictionary mapping literals to variable indices.
        k_clusters (int): Number of clusters.
        distance_classes (list): List of pairs in each distance class.

    Returns:
        WCNF: The updated WCNF object with the added distance class clauses.
    """
    # Clause 30: If b^+_w is true, then pairs (i, i') in distance class w must be in the same cluster
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            for c in range(k_clusters - 1):
                wcnf.append([-literals[f'bw_p_{w}'], -literals[f'x_{i}_{c}'], literals[f'x_{i_prime}_{c}']])

    # Clause 31: If b^+_w is true, then points (i, i') in distance class w must be in the same cluster c
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            for c in range(k_clusters - 1):
                wcnf.append([-literals[f'bw_p_{w}'], literals[f'x_{i}_{c}'], -literals[f'x_{i_prime}_{c}']])

    # Clause 32: Ensures that if bw^-_w is true, then the points in distance class w
    # cannot be clustered with the points in distance class w-1 if bw^-_(w-1) is false.
    for w in range(1, len(distance_classes)):
        wcnf.append([-literals[f'bw_m_{w}'], literals[f'bw_m_{w - 1}']])

    # Clause 33: If bw^+_w is true, then distance class w must be clustered together with distance class w-1
    for w in range(1, len(distance_classes)):
        wcnf.append([-literals[f'bw_p_{w}'], literals[f'bw_p_{w - 1}']])

    # Clause 34: If bw^+_w is true, then distance class w cannot be clustered separately within itself
    for w in range(len(distance_classes)):
        wcnf.append([-literals[f'bw_p_{w}'], literals[f'bw_m_{w}']])

    # Clause 37: For each distance class w, we add a soft clause for the corresponding b^-_w literal
    # to encourage points within that class to be clustered separately
    for w in range(len(distance_classes)):
        wcnf.append([-literals[f'bw_m_{w}']], weight=1)

    # Clause 38: For each distance class w, we add a soft clause for the corresponding b^+_w literal
    # to encourage points within that class to be clustered together
    for w in range(len(distance_classes)):
        wcnf.append([literals[f'bw_p_{w}']], weight=1)

    return wcnf