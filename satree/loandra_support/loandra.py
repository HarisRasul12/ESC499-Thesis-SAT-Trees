"""
=========== Module Description ===========

This module provides functionality to interface with the Loandra MaxSAT solver for SAT-based decision tree
problems. In our SAT formulation, the decision tree model is encoded as a CNF (Conjunctive Normal Form)
that captures both the hard structural constraints (such as valid tree splits and leaf assignments) and the
soft optimization objectives (e.g., minimizing tree cost or margin violations). The primary functions in this
module are:

  • run_loandra_and_parse_results: Executes the Loandra solver on a specified CNF file and parses the solver’s
    output to extract the optimal cost and the corresponding SAT model.

  • transform_tree_from_loandra: Interprets the raw SAT model produced by Loandra to update the decision tree
    structure with the appropriate labels and branch node features, thereby bridging the gap between the abstract
    SAT solution and the interpretable decision tree used in classification.

This module thus connects the mathematical underpinnings of the SAT decision tree formulation with practical solver
execution and postprocessing, ensuring that the encoded optimization objectives are faithfully translated into a
usable tree model.

References:
    See https://github.com/jezberg/loandra for details on the Loandra solver.
"""

import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from satree.classification.min_depth import set_branch_node_features


def run_loandra_and_parse_results(loandra_path: str, execution_path: str) -> Tuple[List[int], Optional[int]]:
    """
    Executes the Loandra MaxSAT solver on the provided CNF formulation and parses the solver output to extract
    the minimum cost and the corresponding SAT model, which encodes the decision tree solution.

    This function bridges the mathematical formulation of the SAT-based decision tree model with its practical
    resolution. The CNF file (located at `execution_path`) encodes constraints derived from the decision tree
    structure and classification objectives. Loandra is invoked to minimize the cost associated with these constraints.
    The output is then parsed to yield:
      - The minimum cost, representing the minimal penalty (or optimality measure) achieved by the solution.
      - The model, represented as a list of integers, where a positive integer indicates that the corresponding literal
        is True, and a negative integer (derived from a '0' in the solver's output) indicates that it is False.

    Args:
        loandra_path (str): Filesystem path to the directory containing the Loandra executable.
        execution_path (str): Filesystem path to the CNF file representing the SAT formulation of the decision tree problem.

    Returns:
        Tuple[List[int], Optional[int]]:
            - model: A list of integers representing the truth assignments of the SAT variables.
            - min_cost: The minimum cost (objective value) as determined by Loandra, or None if no cost was extracted.

    Note:
        The function uses absolute paths and captures the stdout of the solver. It is essential that the CNF file is
        properly formatted and that Loandra is correctly installed in the specified directory.
    """
    # Construct the full path to the Loandra executable
    loandra_executable = os.path.join(loandra_path, './loandra')

    # Construct the absolute path to the CNF file
    full_execution_path = os.path.abspath(execution_path)

    # Run the Loandra command with absolute paths
    result = subprocess.run(
        [loandra_executable, full_execution_path, "-print-model"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Process the output from Loandra
    output = result.stdout.splitlines()
    min_cost: Optional[int] = None
    model: List[int] = []

    # Extract minimum cost from the last 'o' line (e.g., "o 123") before "s OPTIMUM FOUND"
    o_lines = [line for line in output if line.startswith('o ')]
    if o_lines:
        try:
            min_cost = int(o_lines[-1].split()[1])
        except (IndexError, ValueError):
            min_cost = None

    # Extract model line (e.g., starting with "v") and convert to required format.
    model_line = next((line for line in output if line.startswith('v ')), None)
    if model_line:
        model_numbers = model_line[2:].strip()  # Remove the "v " prefix
        # Convert the string of 0s and 1s into a list of integers:
        # If a digit is '0', interpret it as false (mapped to a negative literal);
        # if it is '1', interpret it as true (mapped to a positive literal).
        model = [-i - 1 if num == '0' else i + 1
                 for i, num in enumerate(model_numbers) if num != ' ']

    return model, min_cost


def transform_tree_from_loandra(model: List[int],
                                literals: Dict[str, int],
                                leaf_indices: List[int],
                                tree_structure: List[Dict[str, Any]],
                                labels: List[Any],
                                features: np.ndarray) -> Union[List[int], str]:
    """
    Transforms the raw SAT model produced by Loandra into a complete decision tree structure by updating the tree's
    leaf and branch nodes. This transformation is crucial for mapping the abstract SAT solution to an interpretable
    decision tree model that can be used for classification.

    The function operates in two main phases:
      1. For each leaf node (indexed in `leaf_indices`), it examines the corresponding 'g' literals in `literals` to
         determine the correct class label from `labels`. The tree structure is then updated with this label.
      2. The branch nodes are configured by invoking `set_branch_node_features`, which updates the tree with the proper
         feature splits in accordance with the underlying mathematical formulation of the SAT decision tree.

    Args:
        model (List[int]): The SAT model as a list of integers, where positive values indicate True literals.
        literals (Dict[str, int]): A mapping from SAT literal names (e.g., 'g_{t}_{label}') to their variable indices.
        leaf_indices (List[int]): A list of indices corresponding to leaf nodes in the decision tree.
        tree_structure (List[Dict[str, Any]]): The complete decision tree represented as a list of node dictionaries,
            where each node contains its properties (e.g., type, threshold, label).
        labels (List[Any]): The list of class labels for the dataset, used to assign labels to leaf nodes.
        features (List[Any]): The list of feature identifiers used to determine splitting criteria at branch nodes.

    Returns:
        Union[List[int], str]:
            - If a valid model is provided, returns the transformed SAT model (i.e., the original model after updating the tree).
            - If the model is empty or invalid, returns the string "No solution exists".

    Note:
        This function assumes that the literals have been generated consistently with the decision tree encoding and that
        the model produced by Loandra correctly reflects a valid assignment for these literals.
    """
    if model:
        # Update each leaf node with the corresponding label
        for t in leaf_indices:
            for label in labels:
                # The literal for the leaf 'g' variable is expected to be of the form "g_{t}_{label}"
                if literals.get(f'g_{t}_{label}') in model:
                    tree_structure[t]['label'] = label
                    break
        # Configure branch nodes using the provided SAT model and literals
        set_branch_node_features(model, literals, tree_structure, features)
        return model
    else:
        return "No solution exists"
