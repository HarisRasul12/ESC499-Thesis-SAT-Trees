from satree.treemodder.builder import create_literals


def create_literals_cluster_tree(TB, TL, F, k_clusters, dataset_size, distance_classes, bicriteria=False):
    """
    Create the literals for the SAT solver based on the tree structure and dataset size.

    This function creates four types of literals:
    - 'a' literals for feature splits at branching nodes,
    - 's' literals for data points directed to left or right,
    - 'z' literals for data points that end up at a leaf node,
    - 'g' literals for assigning class labels to leaf nodes.
    - 'x' The cluster assigned to point 𝑖 is or comes after 𝑐lass label c
    - 'bw_p' The pairs in class 𝑤 should be clustered together
    - 'bw_m' (The negation of) whether the pairs in distance class 𝑤 should be clustered separately

    Parameters:
    - TB (list): Indices of branching nodes in the tree.
    - TL (list): Indices of leaf nodes in the tree.
    - F (list): The array of features
    - k_clusters (int) - based on clusters , so we need to turn this into arrat: eg C= k_clusters, C =2, C-> [0,1], turn into list of cluster ids
    - dataset_size (int): The number of data points in the dataset.
    - data_classes


    Returns:
    - literals (dict): A dictionary where keys are literal names and values are their corresponding indices for the SAT solver.
    """
    C = list(range(k_clusters))  # List of cluster IDs
    literals, current_index = create_literals(TB, TL, F, C, dataset_size, fixed_tree=False)

    # Create 'x' The cluster assigned to point 𝑖 is or comes after 𝑐
    for i in range(dataset_size):
        for c in C:
            literals[f'x_{i}_{c}'] = current_index
            current_index += 1

    # Create 'bw_m' literals (points in class w should NOT be clustered together)
    for w, pairs in enumerate(distance_classes):  # data_classes is a list of numpy arrays
        literals[f'bw_m_{w}'] = current_index
        current_index += 1

    if bicriteria:
        # Create 'bw_p' literals (points in class w should be clustered together)
        for w, pairs in enumerate(distance_classes):
            literals[f'bw_p_{w}'] = current_index
            current_index += 1

    return literals


