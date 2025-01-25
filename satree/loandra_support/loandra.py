# Created by: Haris Rasul
# Date: March 16 2024
# Python script module for computing fixed height trees max accuacy given fixed depth  with categorical features 

import os
import subprocess
from classification_problems.min_height_tree_module import set_branch_node_features

def run_loandra_and_parse_results(loandra_path, execution_path):
    """
    Executes the Loandra MaxSAT solver on a given CNF file and parses the results.

    This function assumes that Loandra is installed and available at the given path.
    It runs Loandra with the provided execution path to the CNF file, extracts the minimum
    cost from Loandra's output, and interprets the model solution.

    Parameters:
    - loandra_path: A string representing the filesystem path to the directory containing the Loandra executable.
    - execution_path: A string representing the filesystem path to the CNF file that Loandra will solve.

    Returns:
    - min_cost: The minimum cost of the solution as reported by Loandra.
    - model: A list of integers representing the model solution. Positive numbers correspond to literals
             that are true, and negative numbers correspond to literals that are false. The index in the
             list represents the literal number starting from 1.

    The function changes the current working directory to the Loandra path, executes the solver, and then
    may change back to the original directory. The stdout of the solver is captured and parsed to extract
    the cost and the model. The model is returned as a list of integers, where each '0' from Loandra's
    output is represented as a negative number (literal is false), and each '1' is represented as a
    positive number (literal is true).

    Please see https://github.com/jezberg/loandra/tree/master for LOANDRA solver details and installations 
    """

    # Construct the full path to the loandra executable
    loandra_executable = os.path.join(loandra_path, './loandra')

    # Construct the full path to the CNF file
    full_execution_path = os.path.abspath(execution_path)

    # Run the Loandra command with absolute paths
    result = subprocess.run([loandra_executable, full_execution_path, "-print-model"], 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Process result
    output = result.stdout.splitlines()
    min_cost = None
    model = []

    # Extract minimum cost from the last 'o' line before "s OPTIMUM FOUND"
    o_lines = [line for line in output if line.startswith('o ')]
    if o_lines:
        min_cost = int(o_lines[-1].split()[1])  # Get the last o line's cost

    # Extract model and convert to the required format
    model_line = next((line for line in output if line.startswith('v ')), None)
    if model_line:
        model_numbers = model_line[2:].strip()  # Remove the 'v ' and trailing whitespaces
        # Convert the string of 0s and 1s into an array of positive and negative integers
        model = [-i-1 if num == '0' else i+1 for i, num in enumerate(model_numbers) if num != ' ']

    return model, min_cost


def transform_tree_from_loandra(model, literals, TL, tree_structure, labels,features,datasetX):
    """
    Attempts to complete tree from solutions 

    Args:
    - cnf (CNF): The CNF object containing all clauses for the SAT solver.
    - literals (dict): A dictionary mapping literals to variable indices.
    - TL (list): Indices of leaf nodes in the tree.
    - tree_structure (list): The complete binary tree structure.
    - labels (list): The list of class labels for the dataset.

    Returns:
    - model from prior
    """

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
        return model
    else:
        return "No solution exists"