"""
=========== Module Description ===========

SATreeCraft is a comprehensive Python library that implements SAT-based decision tree learning and clustering
techniques using advanced mathematical formulations. The library transforms both classification and clustering
tasks into SAT (CNF) formulations, rigorously capturing hard constraints such as valid tree splits, proper leaf
assignments, and overall tree consistency, while also integrating soft optimization objectives. These include
minimizing tree height for perfect training accuracy and maximizing classification performance within a fixed depth.
For clustering, SATreeCraft encodes constraints that balance inter-cluster separation with intra-cluster cohesion,
yielding robust and meaningful cluster assignments.

Key features include:
  • **Versatile Classification Methods:**
      - *Minimum Height Tree*: Identifies the smallest decision tree that achieves 100% training accuracy.
      - *Fixed Depth Maximum Accuracy*: Optimizes classification performance under a user-specified tree depth.
  • **Effective Clustering Approaches:**
      - Constructs SAT formulations that integrate structural and distance-based constraints to produce high-quality clusters.
  • **Solver Integration and Postprocessing:**
      - Generates SAT literals and CNF formulas directly from raw data and problem specifications.
      - Seamlessly interfaces with both standard SAT solvers and specialized MaxSAT solvers (e.g., Loandra) to obtain optimal solutions.
      - Decodes SAT solutions into interpretable decision trees or clustering assignments, complete with visualization and export capabilities.

Designed for researchers and practitioners alike, SATreeCraft offers a flexible and intuitive framework to tackle
complex decision tree and clustering problems using state-of-the-art SAT-based methodologies.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from pysat.formula import WCNF, CNF

from satree.treemodder.builder import build_complete_tree, create_literals

from satree.classification.min_depth import build_clauses, add_thresholds, solve_cnf, visualize_tree
from satree.classification.fixed_depth import build_clauses_fixed_tree, solve_wcnf
from satree.classification.min_depth_categorical import build_clauses_categorical, add_thresholds_categorical
from satree.classification.fixed_depth_categorical import build_clauses_categorical_fixed
from satree.classification.constraints import add_oblivious_tree_constraints, min_support, \
    build_clauses_fixed_tree_min_margin_constraint_add

from satree.clustering.solver import solve_wcnf_clustering, \
    assign_clusters_and_diameters, process_clustering_solution
from satree.clustering.distance_classes import create_distance_classes
from satree.clustering.models import build_clauses_cluster_tree_md_ms, build_clauses_cluster_tree_md_ms_smart_pair, \
    build_clauses_cluster_tree_md
from satree.clustering.literals import create_literals_cluster_tree, create_literal_matrices_modular

from satree.loandra_support.loandra import run_loandra_and_parse_results, transform_tree_from_loandra


class SATreeCraft:
    """
    SATreeCraft is a Python Library designed to solve classification problems
    using SAT-based decision trees. It supports datasets with categorical and/or
    numerical features and can optimize for minimum tree height or maximum accuracy
    given a fixed depth. It also supports clustering tree objectives such as maximizing minimum split and minimizing maximum diameter

    Attributes:
        dataset (array): The dataset to be used for tree construction.
        features (array): The list of feature names in the dataset.
        labels (array): The list of labels in the dataset.
        true_labels_for_points (array): The list of true labels for data points.
        features_numerical (array, optional): List of indices for numerical features.
        features_categorical (array, optional): List of indices for categorical features.
        is_classification (bool): Flag to indicate if the problem is classification. Default is True. Will support Clustering in future
        classification_objective (str): The objective of the classification ('min_height' or 'max_accuracy').
        fixed_depth (int, optional): The depth of the tree if 'max_accuracy' is the classification objective.
        tree_structure (str): The type of tree structure to build ('Complete' or 'Oblivious').

    Methods:
        solve: Determines the appropriate solving strategy based on the problem domain and objectives.
        export_cnf: Exports the final CNF formula to a DIMACS format file.
    """

    def __init__(self,
                 dataset: np.ndarray,
                 features: np.ndarray,
                 labels: Optional[List[Any]] = None,
                 true_labels_for_points: Optional[List] = None,
                 features_numerical: Optional[List[str]] = None,
                 features_categorical: Optional[List[str]] = None,
                 is_classification: bool = True,
                 classification_objective: str = 'min_height',
                 fixed_depth: Optional[int] = None,
                 tree_structure: str = 'Complete',
                 min_support_level: int = 0,
                 min_margin: int = 1,
                 k_clusters: Optional[int] = None,
                 clustering_objective: str = 'max_diameter',
                 is_clustering: bool = False,
                 epsilon: float = 0,
                 cl_pairs: np.ndarray = np.array([]),
                 ml_pairs: np.ndarray = np.array([]),
                 smart_pairs: bool = False) -> None:

        """
        Initializes the SATreeCraft instance with the provided dataset and configuration.

        Args:
            dataset: The input dataset as a NumPy array.
            features: Array of feature identifiers.
            labels: (Optional) Array of class labels.
            true_labels_for_points: (Optional) Array of true labels for each data point.
            features_numerical: (Optional) Numerical feature indices.
            features_categorical: (Optional) Categorical feature indices.
            is_classification: Flag indicating if the problem is classification.
            classification_objective: Classification objective ('min_height' or 'max_accuracy').
            fixed_depth: (Optional) Fixed tree depth for max accuracy problems.
            tree_structure: Type of tree structure ('Complete' or 'Oblivious').
            min_support_level: Minimum support level for additional constraints.
            min_margin: Minimum margin constraint for numerical problems.
            k_clusters: (Optional) Number of clusters for clustering problems.
            clustering_objective: Clustering objective (e.g., 'max_diameter').
            is_clustering: Flag indicating if the problem is clustering.
            epsilon: Parameter for distance class computation.
            cl_pairs: Must-link pairs as a NumPy array.
            ml_pairs: Cannot-link pairs as a NumPy array.
            smart_pairs: Flag to indicate whether smart pair encoding is used.

        Returns:
            None
        """

        self.dataset = dataset
        self.features = features

        # Classification tools
        self.labels = labels
        self.true_labels_for_points = true_labels_for_points
        self.features_numerical = features_numerical
        self.features_categorical = features_categorical
        self.is_classification = is_classification
        self.classification_objective = classification_objective
        self.fixed_depth = fixed_depth
        self.tree_structure = tree_structure
        self.min_support = min_support_level
        self.min_margin = min_margin

        # Clustering tools
        self.k_clusters = k_clusters
        self.clustering_objective = clustering_objective
        self.is_clustering = is_clustering

        if self.is_clustering or (self.k_clusters is not None):
            self.is_classification = False

        self.epsilon = epsilon
        self.CL_pairs = cl_pairs
        self.ML_pairs = ml_pairs
        self.smart_pairs = smart_pairs

        # return types
        self.model = None
        self.sat_solution = None
        self.min_cost = None
        self.min_depth = None
        self.final_cnf = None
        self.final_literals = None
        self.cluster_assignments = None
        self.cluster_diameters = None

    ##### Categorical Classification Problems ####

    def apply_oblivious_constraints_and_solve(self,
                                              cnf: Union[WCNF, CNF],
                                              features: np.ndarray,
                                              depth: int,
                                              literals: Dict[str, int],
                                              dataset: np.ndarray,
                                              tree: List[Dict[str, Any]],
                                              leaf_nodes: List[int],
                                              labels: List[Any],
                                              use_loandra: bool,
                                              loandra_path: Optional[str] = None,
                                              execution_path: Optional[str] = None) -> Tuple[
        Union[str, List[int]], Optional[int], List[Any]]:
        """
        Applies oblivious tree constraints to the CNF and solves it using either the Loandra-based solver or the standard SAT solver.

        This function augments the original CNF with additional constraints that enforce the oblivious tree structure.
        If Loandra is specified, the CNF is converted to a weighted CNF, written to a file, and solved using Loandra.
        Otherwise, the standard SAT solver (via solve_cnf) is used.

        Args:
            cnf: The CNF represented as a list of clauses.
            features: The array of feature identifiers.
            depth: The current depth of the tree.
            literals: Dictionary mapping literal names to their variable indices.
            dataset: The dataset as a NumPy array.
            tree: The current decision tree structure (list of node dictionaries).
            leaf_nodes: List of indices representing the leaf nodes.
            labels: The list of class labels.
            use_loandra: Flag to indicate whether to use Loandra for solving.
            loandra_path: (Optional) Path to the Loandra executable (required if use_loandra is True).
            execution_path: (Optional) Path to write the CNF file for Loandra (required if use_loandra is True).

        Returns:
            A tuple containing:
              - solution: The SAT solution (as a list of integers) or the string "No solution exists".
              - cost: The cost returned by Loandra, or None if using the standard SAT solver.
              - cnf: The updated CNF after adding oblivious constraints.
        """
        # Add oblivious tree constraints.
        cnf = add_oblivious_tree_constraints(cnf, features, depth, literals, dataset, self.tree_structure)

        if use_loandra:
            # --- LOANDRA PATH ---
            # Convert the CNF into a WCNF (weighted CNF) as expected by Loandra.
            wcnf = WCNF()
            for clause in cnf:
                wcnf.append(clause)
            wcnf.to_file(execution_path)

            solution, cost = run_loandra_and_parse_results(loandra_path, execution_path)
            # If the cost is non-zero, no valid solution was found.
            if cost != 0:
                solution = "No solution exists"

            if solution != "No solution exists":
                # Transform the solution from Loandra to our internal format.
                solution = transform_tree_from_loandra(solution, literals, leaf_nodes, tree, labels, features)
        else:
            # --- STANDARD SAT SOLVER PATH ---
            solution = solve_cnf(cnf, literals, leaf_nodes, tree, labels, features)
            cost = None

        return solution, cost, cnf

    def find_min_depth_tree_categorical_problem(self,
                                                features: np.ndarray,
                                                features_categorical: List[str],
                                                features_numerical: List[str],
                                                labels: List[Any],
                                                true_labels_for_points: List,
                                                dataset: np.ndarray,
                                                use_loandra: bool = False,
                                                loandra_path: Optional[str] = None,
                                                execution_path: Optional[str] = None) -> Tuple[
        Any, Dict[str, int], int, Union[str, List[int]], List[Any]]:
        """
        Finds a decision tree with the minimum depth for a categorical classification problem.

        Starting with a depth of 1, the function iteratively builds a complete tree and its corresponding CNF
        until a solution is found. For each depth, it generates literals and CNF clauses using categorical encoding.
        If a solution is found, it adds thresholds to the tree and produces a visualization.

        Args:
            features: Array of feature identifiers.
            features_categorical: Array of categorical feature indices.
            features_numerical: Array of numerical feature indices.
            labels: List of class labels.
            true_labels_for_points: Array of true labels for the dataset points.
            dataset: The input dataset as a NumPy array.
            use_loandra: Whether to use Loandra to solve the CNF.
            loandra_path: (Optional) Path to the Loandra executable.
            execution_path: (Optional) Path for writing the CNF file (used when use_loandra is True).

        Returns:
            A tuple containing:
              - tree_with_thresholds: The decision tree structure with thresholds added.
              - literals: Dictionary mapping literal names to variable indices.
              - depth: The depth at which a solution was found.
              - solution: The SAT solution as a list of integers, or "No solution exists".
              - cnf: The final CNF used.
        """

        depth = 1  # Start with a depth of 1
        solution = "No solution exists"
        tree_with_thresholds = None
        literals = None
        cnf = None  # Initialize cnf to avoid potential reference before assignment

        # Try increasing depths until a solution is found
        while solution == "No solution exists":
            tree, TB, TL = build_complete_tree(depth)
            literals = create_literals(TB, TL, features, labels, len(dataset), False)[0]
            cnf = build_clauses_categorical(literals, dataset, TB, TL, len(features), features_categorical,
                                            features_numerical, labels, true_labels_for_points)

            # Add oblivious tree constraints (if used)
            solution, cost, cnf = self.apply_oblivious_constraints_and_solve(
                cnf, features, depth, literals, dataset, tree, TL, labels,
                use_loandra, loandra_path, execution_path
            )

            if solution != "No solution exists":
                tree_with_thresholds = add_thresholds_categorical(tree, literals, solution, dataset,
                                                                  features_categorical)
                dot = visualize_tree(tree_with_thresholds)
                folder = Path('../images/min_height/')
                folder.mkdir(parents=True, exist_ok=True)
                if use_loandra:
                    dot.render(
                        folder / f'LOANDRA_SOLVED_binary_decision_tree_min_depth_with_categorical_features_depth_{depth}',
                        format='png', cleanup=True)
                else:
                    dot.render(folder / f'binary_decision_tree_min_depth_with_categorical_features_depth_{depth}',
                               format='png', cleanup=True)
            else:
                print("No solution at depth: ", depth)
                depth += 1  # Increase the depth and try again

        return tree_with_thresholds, literals, depth, solution, cnf

    def find_fixed_depth_tree_categorical_problem(self,
                                                  features: np.ndarray,
                                                  features_categorical: List[str],
                                                  features_numerical: List[str],
                                                  labels: List[Any],
                                                  true_labels_for_points: List,
                                                  dataset: np.ndarray,
                                                  depth: int,
                                                  use_loandra: bool = False,
                                                  loandra_path: Optional[str] = None,
                                                  execution_path: Optional[str] = None) -> Union[
        Tuple[Any, Dict[str, int], int, Union[str, List[int]], Optional[int], Any], str]:
        """
        Constructs a decision tree for a categorical problem with a fixed depth.

        This function builds the complete tree and generates literals using the categorical fixed encoding.
        It then applies any necessary min support constraints and solves the weighted CNF formulation. If a solution
        is found, thresholds are added and the tree is visualized.

        Args:
            features: Array of feature identifiers.
            features_categorical: Array of categorical feature indices.
            features_numerical: Array of numerical feature indices.
            labels: List of class labels.
            true_labels_for_points: Array of true labels for the dataset points.
            dataset: The input dataset as a NumPy array.
            depth: The fixed depth for the decision tree.
            use_loandra: Whether to use Loandra for solving.
            loandra_path: (Optional) Path to the Loandra executable.
            execution_path: (Optional) Path for writing the CNF file (if using Loandra).

        Returns:
            A tuple containing:
              - tree_with_thresholds: The decision tree with thresholds.
              - literals: Dictionary mapping literal names to variable indices.
              - depth: The fixed depth.
              - solution: The SAT solution (list of integers) or "No solution exists".
              - cost: The cost (if applicable) or None.
              - wcnf: The final weighted CNF.
            If no solution is found, returns the string "No solution".
        """

        # Build the complete tree and generate literals
        tree, TB, TL = build_complete_tree(depth)
        literals = create_literals(TB, TL, features, labels, len(dataset), True)[0]

        # Build the CNF clauses using the provided categorical fixed encoding
        wcnf = build_clauses_categorical_fixed(literals, dataset, TB, TL, len(features), features_categorical,
                                               features_numerical, labels, true_labels_for_points)

        # Add the min support constraint if applicable
        solution, cost, wcnf = self.finalize_and_solve_wcnf(
            wcnf, literals, dataset, TL, features, depth, tree, labels,
            use_loandra, loandra_path, execution_path
        )

        # If a solution was found, add thresholds and generate a visualization
        if solution != "No solution exists":
            tree_with_thresholds = add_thresholds_categorical(tree, literals, solution, dataset, features_categorical)
            dot = visualize_tree(tree_with_thresholds)
            folder = Path('../images/fixed_height/')
            folder.mkdir(parents=True, exist_ok=True)
            if use_loandra:
                dot.render(
                    folder / f'LOANDRA_SOLVED_binary_decision_tree_fixed_with_categorical_features_depth_{depth}',
                    format='png', cleanup=True)
            else:
                dot.render(folder / f'binary_decision_tree_fixed_with_categorical_features_depth_{depth}', format='png',
                           cleanup=True)
        else:
            print('could not find solution')
            return 'No solution'

        return tree_with_thresholds, literals, depth, solution, cost, wcnf

    #### Numerical Classification Problems ####

    def find_min_depth_tree_problem(self,
                                    features: np.ndarray,
                                    labels: List[Any],
                                    true_labels_for_points: List[Any],
                                    dataset: np.ndarray,
                                    use_loandra: bool = False,
                                    loandra_path: Optional[str] = None,
                                    execution_path: Optional[str] = None) -> Tuple[
        Any, Dict[str, int], int, Union[str, List[int]], List[Any]]:
        """
        Finds a decision tree with the minimum depth for a numerical classification problem.

        Starting with depth 1, iteratively builds the tree and CNF formulation using numerical encoding.
        The CNF is solved (using Loandra if specified) until a valid solution is found. The tree is then updated
        with thresholds and visualized.

        Args:
            features: Array of feature identifiers.
            labels: List of class labels.
            true_labels_for_points: Array of true labels for the dataset points.
            dataset: The input dataset as a NumPy array.
            use_loandra: Whether to use Loandra for solving.
            loandra_path: (Optional) Path to the Loandra executable.
            execution_path: (Optional) Path for writing the CNF file (if using Loandra).

        Returns:
            A tuple containing:
              - tree_with_thresholds: The decision tree with thresholds.
              - literals: Dictionary mapping literal names to variable indices.
              - depth: The depth at which the solution was found.
              - solution: The SAT solution (list of integers) or "No solution exists".
              - cnf: The final CNF formulation.
        """
        depth = 1  # Start with a depth of 1
        solution = "No solution exists"
        tree_with_thresholds = None
        literals = None
        cnf = None  # Initialize cnf to avoid potential reference before assignment

        while solution == "No solution exists":
            tree, TB, TL = build_complete_tree(depth)
            literals = create_literals(TB, TL, features, labels, len(dataset), False)[0]
            cnf = build_clauses(literals, dataset, TB, TL, len(features), labels, true_labels_for_points)

            # Oblivious Tree Constraints addition if ever used
            solution, cost, cnf = self.apply_oblivious_constraints_and_solve(
                cnf, features, depth, literals, dataset, tree, TL, labels,
                use_loandra, loandra_path, execution_path
            )

            if solution != "No solution exists":
                tree_with_thresholds = add_thresholds(tree, literals, solution, dataset)
                dot = visualize_tree(tree_with_thresholds)
                folder = Path('../images/min_height/')
                folder.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
                if use_loandra:
                    dot.render(folder / f'LOANDRA_SOLVED_binary_decision_tree_min_depth_{depth}', format='png',
                               cleanup=True)
                else:
                    dot.render(folder / f'binary_decision_tree_min_depth_{depth}', format='png', cleanup=True)
            else:
                print('no solution at depth', depth)
                depth += 1  # Increase the depth and try again

        return tree_with_thresholds, literals, depth, solution, cnf

    def finalize_and_solve_wcnf(self,
                                wcnf: WCNF,
                                literals: Dict[str, int],
                                dataset: np.ndarray,
                                leaf_nodes: List[int],
                                features: np.ndarray,
                                depth: int,
                                tree: List[Dict[str, Any]],
                                labels: List[Any],
                                use_loandra: bool,
                                loandra_path: Optional[str] = None,
                                execution_path: Optional[str] = None) -> Tuple[
        Union[str, List[int]], Optional[int], WCNF]:
        """
        Applies the min support and oblivious tree constraints to the given weighted CNF and solves it.

        The CNF is first updated with a min support constraint (if specified), then further augmented with oblivious
        tree constraints. The final CNF is solved using Loandra (if specified) or the standard SAT solver.

        Args:
            wcnf: The weighted CNF (WCNF) to be solved.
            literals: Dictionary mapping literal names to variable indices.
            dataset: The input dataset as a NumPy array.
            leaf_nodes: List of indices corresponding to leaf nodes.
            features: Array of feature identifiers.
            depth: The current depth of the tree.
            tree: The decision tree structure as a list of node dictionaries.
            labels: List of class labels.
            use_loandra: Whether to use Loandra for solving.
            loandra_path: (Optional) Path to the Loandra executable.
            execution_path: (Optional) Path to write the CNF file (if using Loandra).

        Returns:
            A tuple containing:
              - solution: The SAT solution (list of integers) or "No solution exists".
              - cost: The cost (if applicable) or None.
              - wcnf: The final weighted CNF after all constraints are applied.
        """
        # Apply min support constraint if specified.
        if self.min_support > 0:
            wcnf = min_support(wcnf, literals, dataset, leaf_nodes, self.min_support)

        # Add oblivious tree constraints.
        wcnf = add_oblivious_tree_constraints(wcnf, features, depth, literals, dataset, self.tree_structure)

        # Solve using either Loandra or the standard solver.
        if use_loandra:
            # --- LOANDRA BRANCH ---
            wcnf.to_file(execution_path)
            solution, cost = run_loandra_and_parse_results(loandra_path, execution_path)
            solution = transform_tree_from_loandra(solution, literals, leaf_nodes, tree, labels, features)
        else:
            # --- STANDARD SOLVER BRANCH ---
            solution, cost = solve_wcnf(wcnf, literals, leaf_nodes, tree, labels, features)

        return solution, cost, wcnf

    def find_fixed_depth_tree_problem(self,
                                      features: np.ndarray,
                                      labels: List[Any],
                                      true_labels_for_points: List[Any],
                                      dataset: np.ndarray,
                                      depth: int,
                                      use_loandra: bool = False,
                                      loandra_path: Optional[str] = None,
                                      execution_path: Optional[str] = None) -> Union[
        Tuple[Any, Dict[str, int], int, Union[str, List[int]], Optional[int], WCNF], str]:
        """
        Constructs a decision tree for numerical problems with a fixed depth.

        The function builds the complete tree, generates literals using numerical encoding, and applies either a
        min margin constraint (if specified) or a standard fixed tree CNF. It then solves the CNF and, if a solution
        is found, adds thresholds to the tree and visualizes it.

        Args:
            features: Array of feature identifiers.
            labels: List of class labels.
            true_labels_for_points: Array of true labels for each data point.
            dataset: The input dataset as a NumPy array.
            depth: The fixed depth for the decision tree.
            use_loandra: Whether to use Loandra for solving.
            loandra_path: (Optional) Path to the Loandra executable.
            execution_path: (Optional) Path to write the CNF file (if using Loandra).

        Returns:
            A tuple containing:
              - tree_with_thresholds: The decision tree with thresholds added.
              - literals: Dictionary mapping literal names to variable indices.
              - depth: The fixed depth.
              - solution: The SAT solution (list of integers) or "No solution exists".
              - cost: The cost from solving the CNF (or None).
              - wcnf: The final weighted CNF.
            If no solution is found, returns "No solution".
        """

        # Build the complete tree and create the literals.
        tree, TB, TL = build_complete_tree(depth)
        literals = create_literals(TB, TL, features, labels, len(dataset), True)[0]

        # Apply min margin constraint (only for numerical problems).
        if self.min_margin > 1:
            wcnf = build_clauses_fixed_tree_min_margin_constraint_add(
                literals, dataset, TB, TL, len(features), labels, true_labels_for_points, self.min_margin
            )
        else:
            wcnf = build_clauses_fixed_tree(literals, dataset, TB, TL, len(features), labels, true_labels_for_points)

        # Apply min support constraint if specified.
        solution, cost, wcnf = self.finalize_and_solve_wcnf(
            wcnf, literals, dataset, TL, features, depth, tree, labels,
            use_loandra, loandra_path, execution_path
        )

        # Process the solution if one was found.
        if solution != "No solution exists":
            tree_with_thresholds = add_thresholds(tree, literals, solution, dataset)
            dot = visualize_tree(tree_with_thresholds)
            folder = Path('../images/fixed_height/')
            folder.mkdir(parents=True, exist_ok=True)
            if use_loandra:
                dot.render(folder / f'LOANDRA_SOLVED_binary_decision_tree_fixed_depth_{depth}', format='png',
                           cleanup=True)
            else:
                dot.render(folder / f'binary_decision_tree_fixed_depth_{depth}', format='png', cleanup=True)
        else:
            print('could not find solution')
            return 'No solution'

        return tree_with_thresholds, literals, depth, solution, cost, wcnf

    @staticmethod
    def plot_and_save_clusters_to_drive(dataset: np.ndarray, cluster_assignments: Dict[Any, List[int]],
                                        k_clusters: int) -> str:
        """
        Plots the dataset points before and after clustering if the dataset has 1 or 2 features.
        Creates a side-by-side plot showing the dataset before clustering and after with cluster IDs.
        Saves the plot to the specified directory with a filename based on the number of clusters.
        Does not display the plot in the output.

        Args:
            dataset: The original dataset as a NumPy array.
            cluster_assignments: Dictionary mapping cluster IDs to lists of data point indices.
            k_clusters: The number of clusters.

        Returns:
            The filesystem path (as a string) to the saved plot image.
        """
        # Define the directory and filename
        directory = Path('../images/cluster_trees/')
        directory.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        filename = f'cluster_tree_with_cluster_size{k_clusters}.png'
        full_path = directory / filename
        full_path = str(full_path)

        # Create the plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.patch.set_facecolor('white')

        if dataset.shape[1] == 2:  # If 2D dataset
            axes[0].scatter(dataset[:, 0], dataset[:, 1], c='gray', label='Data Points')
            axes[0].set_title('Before Clustering')
            axes[1].scatter(dataset[:, 0], dataset[:, 1], c='gray', label='Data Points')
            axes[1].set_title('After Clustering')
        elif dataset.shape[1] == 1:  # If 1D dataset
            axes[0].scatter(dataset[:, 0], np.zeros_like(dataset[:, 0]), c='gray', label='Data Points')
            axes[0].set_title('Before Clustering')
            axes[1].scatter(dataset[:, 0], np.zeros_like(dataset[:, 0]), c='gray', label='Data Points')
            axes[1].set_title('After Clustering')
        else:
            return 'can only plot 2d or 1d datasets'

        # Assign colors to clusters
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, k_clusters))
        for cluster_id, data_points in cluster_assignments.items():
            if dataset.shape[1] == 2:
                axes[1].scatter(dataset[data_points, 0], dataset[data_points, 1],
                                color=colors[cluster_id], label=f'Cluster {cluster_id}')
            elif dataset.shape[1] == 1:
                axes[1].scatter(dataset[data_points, 0], np.zeros_like(dataset[data_points, 0]),
                                color=colors[cluster_id], label=f'Cluster {cluster_id}')

        # Add legend to the second plot
        axes[1].legend()

        # Save the figure
        fig.savefig(full_path)
        plt.close(fig)  # Close the figure to prevent it from displaying in the output
        return full_path

    def solve_clustering_problem_max_diameter(self,
                                              dataset: np.ndarray,
                                              features: np.ndarray,
                                              k_clusters: int,
                                              depth: int,
                                              epsilon: float,
                                              cl_pairs: np.ndarray,
                                              ml_pairs: np.ndarray,
                                              use_loandra: bool = False,
                                              loandra_path: Optional[str] = None,
                                              execution_path: Optional[str] = None) -> Tuple[
        Dict[Any, List[int]], Dict[Any, float], Dict[str, int], Union[str, List[int]]]:
        """
        Solves the clustering problem where the objective is to maximize the minimum cluster diameter.

        This function creates a decision tree for clustering, generates appropriate literals for the clustering SAT
        formulation, builds the CNF specific to the clustering objective, and solves it. It then decodes the SAT solution
        into cluster assignments and computes the maximum diameter for each cluster.

        Args:
            dataset: The input dataset as a NumPy array.
            features: Array of feature identifiers.
            k_clusters: The number of clusters.
            depth: The depth of the decision tree.
            epsilon: Parameter used in the distance class calculation.
            cl_pairs: Array of must-link pairs.
            ml_pairs: Array of cannot-link pairs.
            use_loandra: Whether to use Loandra for solving.
            loandra_path: (Optional) Path to the Loandra executable.
            execution_path: (Optional) Path to write the CNF file (if using Loandra).

        Returns:
            A tuple containing:
              - cluster_assignments: Dictionary mapping cluster IDs to lists of data point indices.
              - cluster_diameters: Dictionary mapping cluster IDs to their maximum diameter.
              - literals: Dictionary mapping literal names to variable indices.
              - solution: The SAT solution (list of integers) or "No solution exists".
        """
        dataset_size = len(dataset)
        num_features = len(features)
        dist1, dist2, distance_classes = create_distance_classes(dataset, epsilon)
        tree_structure, TB, TL = build_complete_tree(depth)

        literals = create_literals_cluster_tree(TB, TL, features, k_clusters, dataset_size, distance_classes, False)
        wcnf = build_clauses_cluster_tree_md(literals, dataset, TB, TL, num_features, k_clusters,
                                             cl_pairs, ml_pairs, distance_classes)

        if use_loandra:
            # --- LOANDRA PATH ---
            wcnf.to_file(execution_path)
            solution, cost = run_loandra_and_parse_results(loandra_path, execution_path)
        else:
            # --- STANDARD SOLVER PATH ---
            solution = solve_wcnf_clustering(wcnf)

        a_matrix, s_matrix, z_matrix, g_matrix, x_i_c_matrix, bw_m_vector = create_literal_matrices_modular(
            literals=literals,
            solution=solution,
            dataset_size=len(dataset),
            k_clusters=k_clusters,
            branch_nodes=TB,
            leaf_nodes=TL,
            num_features=len(features),
            distance_classes=distance_classes,
            bicriteria=False
        )
        cluster_assignments, cluster_diameters = assign_clusters_and_diameters(x_i_c_matrix, dataset, k_clusters)

        if len(self.features) <= 2:
            self.plot_and_save_clusters_to_drive(dataset, cluster_assignments, k_clusters)

        return cluster_assignments, cluster_diameters, literals, solution

    def solve_clustering_problem_bicriteria(self,
                                            dataset: np.ndarray,
                                            features: np.ndarray,
                                            k_clusters: int,
                                            depth: int,
                                            epsilon: float,
                                            cl_pairs: np.ndarray,
                                            ml_pairs: np.ndarray,
                                            use_loandra: bool = False,
                                            loandra_path: Optional[str] = None,
                                            execution_path: Optional[str] = None) -> Tuple[
        Dict[Any, List[int]], Dict[Any, float], Dict[str, int], Union[str, List[int]]]:
        """
        Solves the bicriteria clustering problem, which integrates clustering objectives with additional co-clustering preferences.

        The function creates literals and builds a weighted CNF that encodes both the clustering assignment constraints
        and soft constraints (e.g., distance classes). Depending on whether Loandra is used, the CNF is solved and the SAT
        solution is post processed to derive final cluster assignments and compute cluster diameters.

        Args:
            dataset: The input dataset as a NumPy array.
            features: Array of feature identifiers.
            k_clusters: The number of clusters.
            depth: The depth of the decision tree.
            epsilon: Parameter used in distance class calculation.
            cl_pairs: Array of must-link pairs.
            ml_pairs: Array of cannot-link pairs.
            use_loandra: Whether to use Loandra for solving.
            loandra_path: (Optional) Path to the Loandra executable.
            execution_path: (Optional) Path to write the CNF file (if using Loandra).

        Returns:
            A tuple containing:
              - cluster_assignments: Dictionary mapping cluster IDs to lists of data point indices.
              - cluster_diameters: Dictionary mapping cluster IDs to their maximum diameter.
              - literals: Dictionary mapping literal names to variable indices.
              - solution: The SAT solution (list of integers) or "No solution exists".
        """
        dataset_size = len(dataset)
        num_features = len(features)
        dist1, dist2, distance_classes = create_distance_classes(dataset, epsilon)
        tree_structure, TB, TL = build_complete_tree(depth)

        # Create the literals with bicriteria flag True.
        literals = create_literals_cluster_tree(TB, TL, features, k_clusters, dataset_size, distance_classes, True)

        # Build the WCNF using smart pairs if enabled.
        if self.smart_pairs:
            wcnf = build_clauses_cluster_tree_md_ms_smart_pair(literals, dataset, TB, TL, num_features, k_clusters,
                                                               cl_pairs, ml_pairs, distance_classes)
        else:
            wcnf = build_clauses_cluster_tree_md_ms(literals, dataset, TB, TL, num_features, k_clusters,
                                                    cl_pairs, ml_pairs, distance_classes)

        if use_loandra:
            # --- LOANDRA BRANCH ---
            wcnf.to_file(execution_path)
            solution, cost = run_loandra_and_parse_results(loandra_path, execution_path)

            # Process the solution by creating literal matrices in bicriteria mode.
            a_matrix, s_matrix, z_matrix, g_matrix, x_i_c_matrix, bw_m_vector, bw_p_vector = create_literal_matrices_modular(
                literals=literals,
                solution=solution,
                dataset_size=len(dataset),
                k_clusters=k_clusters,
                branch_nodes=TB,
                leaf_nodes=TL,
                num_features=len(features),
                distance_classes=distance_classes,
                bicriteria=True
            )
            cluster_assignments, cluster_diameters = assign_clusters_and_diameters(x_i_c_matrix, dataset, k_clusters)
        else:
            # --- STANDARD (NON-LOANDRA) BRANCH ---
            cluster_assignments, cluster_diameters, solution = process_clustering_solution(
                wcnf, literals, dataset, features, k_clusters, TB, TL, distance_classes
            )

        # Optionally plot clusters if there are two or fewer features.
        if len(self.features) <= 2:
            self.plot_and_save_clusters_to_drive(dataset, cluster_assignments, k_clusters)

        return cluster_assignments, cluster_diameters, literals, solution

    #### SAT Solving given problem ####
    def solve(self) -> None:
        """
        Solves the decision tree problem based on the specified objectives and dataset features.

        This method determines the appropriate solving strategy depending on whether the problem is classification
        or clustering. For classification, it further distinguishes between categorical and numerical datasets, and
        between minimum height and maximum accuracy objectives. For clustering, it ensures that the chosen tree depth
        can accommodate the specified number of clusters, and then calls the corresponding clustering solving method.

        Returns:
            None
        """

        if self.is_classification:  # classification problem domain

            if self.features_categorical is not None and len(
                    self.features_categorical) > 0:  # categorical feature dataset

                if self.classification_objective == 'min_height':  # minimum height 100% accuracy on training problem
                    self.model, self.final_literals, self.min_depth, self.sat_solution, self.final_cnf = \
                        self.find_min_depth_tree_categorical_problem(
                            self.features,
                            self.features_categorical,
                            self.features_numerical,
                            self.labels,
                            self.true_labels_for_points,
                            self.dataset,
                            use_loandra=False
                        )
                else:  # Max accuracy problem
                    self.model, self.final_literals, self.fixed_depth, self.sat_solution, self.min_cost, self.final_cnf = \
                        self.find_fixed_depth_tree_categorical_problem(
                            self.features,
                            self.features_categorical,
                            self.features_numerical,
                            self.labels,
                            self.true_labels_for_points,
                            self.dataset,
                            self.fixed_depth
                        )
            else:  # numerical feature dataset strictly
                if self.classification_objective == 'min_height':
                    self.model, self.final_literals, self.min_depth, self.sat_solution, self.final_cnf = \
                        self.find_min_depth_tree_problem(
                            self.features,
                            self.labels,
                            self.true_labels_for_points,
                            self.dataset,
                            use_loandra=False
                        )
                else:  # max accuracy problem
                    self.model, self.final_literals, self.fixed_depth, self.sat_solution, self.min_cost, self.final_cnf = \
                        self.find_fixed_depth_tree_problem(
                            self.features,
                            self.labels,
                            self.true_labels_for_points,
                            self.dataset,
                            self.fixed_depth,
                            use_loandra=False
                        )
        else:
            max_clusters = 2 ** self.fixed_depth
            if self.k_clusters > max_clusters:
                raise ValueError(
                    f"The assigned depth {self.fixed_depth} is not sufficient to accommodate {self.k_clusters} clusters.")

            if self.clustering_objective == 'max_diameter':
                self.cluster_assignments, self.cluster_diameters, self.final_literals, self.sat_solution = \
                    self.solve_clustering_problem_max_diameter(
                        self.dataset,
                        self.features,
                        self.k_clusters,
                        self.fixed_depth,
                        self.epsilon,
                        self.CL_pairs,
                        self.ML_pairs,
                        use_loandra=False
                    )
            else:  # Bicriteria
                # print('solving bicriteria')
                self.cluster_assignments, self.cluster_diameters, self.final_literals, self.sat_solution = \
                    self.solve_clustering_problem_bicriteria(
                        self.dataset,
                        self.features,
                        self.k_clusters,
                        self.fixed_depth,
                        self.epsilon,
                        self.CL_pairs,
                        self.ML_pairs,
                        use_loandra=False
                    )

    ############################## LOANDRA Functionality Support for External SOLVING ###################################

    def export_cnf(self, filename: str = 'dimacs/export_to_solver.cnf') -> None:
        """
        Exports the final CNF formula to a DIMACS format file for external solvers.

        This function is available after the problem has been solved and supports both weighted and non-weighted CNF.

        Args:
            filename: Filesystem path (as a string) where the CNF will be saved.

        Returns:
            None
        """
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        if self.final_cnf:
            self.final_cnf.to_file(filename)
        else:
            raise ValueError("Cannot export CNF. The final CNF is not available. Make sure to solve the problem first.")

    def export_cnf_min_height(self, filename: str = 'dimacs/export_to_solver_min_height.cnf') -> None:
        """
        Exports the final CNF for the minimum height classification problem to a DIMACS file.

        Args:
            filename: Filesystem path (as a string) where the CNF will be saved.

        Returns:
            None
        """
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        if self.final_cnf:
            wcnf = WCNF()
            for clause in self.final_cnf:
                wcnf.append(clause)
            wcnf.to_file(str(filename))
        else:
            raise ValueError("Cannot export CNF. The final CNF is not available")

    def export_cnf_max_accuracy_problem(self, filename: str = 'dimacs/export_to_solver_max_acc_problem.cnf') -> None:
        """
        Exports the CNF for the maximum accuracy classification problem to a DIMACS file.
        This export is available before solving the max accuracy problem.

        Args:
            filename: Filesystem path (as a string) where the CNF will be saved.

        Returns:
            None
        """
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        if self.is_classification:  # classification problem domain
            if self.classification_objective != 'min_height':  # minimum height 100% accuracy on training problem

                tree, TB, TL = build_complete_tree(self.fixed_depth)
                literals = create_literals(TB, TL, self.features, self.labels, len(self.dataset), True)[0]

                if self.features_categorical is not None and len(
                        self.features_categorical) > 0:  # categorical feature dataset
                    wcnf = build_clauses_categorical_fixed(literals, self.dataset, TB, TL, len(self.features),
                                                           self.features_categorical, self.features_numerical,
                                                           self.labels, self.true_labels_for_points)
                else:
                    wcnf = build_clauses_fixed_tree(literals, self.dataset, TB, TL, len(self.features), self.labels,
                                                    self.true_labels_for_points)

                wcnf.to_file(str(filename))
            else:
                raise ValueError("Cannot export CNF without solving for min height problem first.")
        else:
            raise ValueError("Cannot export CNF. The final CNF is not available. Make sure to solve the problem first.")

    def export_cnf_min_height_k(self, depth: int,
                                filename: str = 'dimacs/export_to_solver_min_height_problem_at_given_depth.cnf') -> None:
        """
        Exports the CNF at a given depth k to a DIMACS file for external solving.

        Args:
            depth: The specific tree depth at which the CNF is generated.
            filename: Filesystem path (as a string) where the CNF will be saved.

        Returns:
            None
        """
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        if self.is_classification:  # classification problem domain
            if self.classification_objective == 'min_height':  # minimum height 100% accuracy on training problem
                tree, TB, TL = build_complete_tree(depth)
                literals = create_literals(TB, TL, self.features, self.labels, len(self.dataset), False)[0]

                if self.features_categorical is not None and len(
                        self.features_categorical) > 0:  # categorical feature dataset
                    cnf = build_clauses_categorical(literals,
                                                    self.dataset, TB, TL, len(self.features),
                                                    self.features_categorical, self.features_numerical, self.labels,
                                                    self.true_labels_for_points)
                else:
                    cnf = build_clauses(literals, self.dataset, TB, TL, len(self.features), self.labels,
                                        self.true_labels_for_points)
                wcnf = WCNF()
                for clause in cnf:
                    wcnf.append(clause)
                wcnf.to_file(str(filename))
            else:
                raise ValueError("Must be a min height objective")
        else:
            raise ValueError("Cannot export CNF ")

    def solve_loandra(self, loandra_path: str, execution_path: str = 'dimacs/export_to_solver.cnf') -> None:
        """
        Solves the decision tree problem using the Loandra MaxSAT solver.

        Depending on the problem type (classification or clustering) and the objective,
        this function builds the corresponding CNF, calls the Loandra solver, and updates the internal state
        with the SAT solution and related outputs.

        Args:
            loandra_path: Filesystem path to the Loandra executable directory.
            execution_path: Filesystem path to store the CNF file (as a string).

        Returns:
            None
        """
        execution_path = Path(execution_path)
        execution_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        if self.is_classification:  # classification problem domain

            if self.features_categorical is not None and len(
                    self.features_categorical) > 0:  # categorical feature dataset

                if self.classification_objective == 'min_height':  # minimum height 100% accuracy on training problem
                    self.model, self.final_literals, self.min_depth, self.sat_solution, self.final_cnf = \
                        self.find_min_depth_tree_categorical_problem(
                            self.features,
                            self.features_categorical,
                            self.features_numerical,
                            self.labels,
                            self.true_labels_for_points,
                            self.dataset,
                            use_loandra=True,
                            loandra_path=loandra_path,
                            execution_path=str(execution_path)
                        )
                else:  # Max accuracy problem
                    self.model, self.final_literals, self.fixed_depth, self.sat_solution, self.min_cost, self.final_cnf = \
                        self.find_fixed_depth_tree_categorical_problem(
                            self.features,
                            self.features_categorical,
                            self.features_numerical,
                            self.labels,
                            self.true_labels_for_points,
                            self.dataset,
                            self.fixed_depth,
                            use_loandra=True,
                            loandra_path=loandra_path,
                            execution_path=str(execution_path)
                        )

            else:  # numerical feature dataset strictly
                if self.classification_objective == 'min_height':
                    self.model, self.final_literals, self.min_depth, self.sat_solution, self.final_cnf = \
                        self.find_min_depth_tree_problem(
                            self.features,
                            self.labels,
                            self.true_labels_for_points,
                            self.dataset,
                            use_loandra=True,
                            loandra_path=loandra_path,
                            execution_path=str(execution_path)
                        )
                else:  # max accuracy problem
                    self.model, self.final_literals, self.fixed_depth, self.sat_solution, self.min_cost, self.final_cnf = \
                        self.find_fixed_depth_tree_problem(
                            self.features,
                            self.labels,
                            self.true_labels_for_points,
                            self.dataset,
                            self.fixed_depth,
                            use_loandra=True,
                            loandra_path=loandra_path,
                            execution_path=str(execution_path)
                        )
        else:
            max_clusters = 2 ** self.fixed_depth
            if self.k_clusters > max_clusters:
                raise ValueError(
                    f"The assigned depth {self.fixed_depth} is not sufficient to accommodate {self.k_clusters} clusters.")

            if self.clustering_objective == 'max_diameter':
                self.cluster_assignments, self.cluster_diameters, self.final_literals, self.sat_solution = \
                    self.solve_clustering_problem_max_diameter(
                        self.dataset,
                        self.features,
                        self.k_clusters,
                        self.fixed_depth,
                        self.epsilon,
                        self.CL_pairs,
                        self.ML_pairs,
                        use_loandra=True,
                        loandra_path=loandra_path,
                        execution_path=str(execution_path)
                    )
            else:  # bicriteria
                # print('solving bicriteria')
                self.cluster_assignments, self.cluster_diameters, self.final_literals, self.sat_solution = \
                    self.solve_clustering_problem_bicriteria(
                        self.dataset,
                        self.features,
                        self.k_clusters,
                        self.fixed_depth,
                        self.epsilon,
                        self.CL_pairs,
                        self.ML_pairs,
                        use_loandra=True,
                        loandra_path=loandra_path,
                        execution_path=str(execution_path)
                    )

    ##################################### Auxiliary Helper Functions for User Interface #############################

    @staticmethod
    def create_solution_matrix(literals: Dict[str, int], solution: Union[List[int], str], var_type: str) -> List[
        List[int]]:
        """
        Creates a matrix representation for a specific variable type based on the provided literals and SAT solution.

        Args:
            literals: Dictionary mapping literal names (e.g., 'a_0_1') to variable indices.
            solution: The SAT solution as a list of integers or a string ("No solution exists").
            var_type: The variable type prefix (e.g., 'a', 's', 'z', 'g').

        Returns:
            A 2D list (matrix) where each cell is set to 1 if the corresponding literal is true in the solution, else 0.
        """
        # Find the maximum index for this var_type
        max_index = max(int(key.split('_')[1]) for key, value in literals.items() if key.startswith(var_type)) + 1
        max_sub_index = max(int(key.split('_')[2]) for key, value in literals.items() if key.startswith(var_type)) + 1

        # Initialize the matrix with zeros
        matrix = [[0 for _ in range(max_sub_index)] for _ in range(max_index)]

        # Fill in the matrix with 1 where the literals are true according to the solution
        for key, value in literals.items():
            if key.startswith(var_type):
                index, sub_index = map(int, key.split('_')[1:])
                matrix[index][sub_index] = 1 if value in solution else 0

        return matrix

    def display_solution(self) -> None:
        """
        Displays the SAT solution in a human-readable format by printing matrices for each variable group.

        For classification problems, the variables 'a', 's', 'z', and 'g' are displayed. For non-min_height
        objectives, additional 'p' variables are printed.

        Returns:
            None
        """
        print("\nSolution of Literals")

        if self.classification_objective == 'min_height':
            var_types = ['a', 's', 'z', 'g']
        else:
            var_types = ['a', 's', 'z', 'g', 'p']
        for var_type in var_types:
            if var_type != 'p':
                matrix = self.create_solution_matrix(self.final_literals, self.sat_solution, var_type)
                print(f"{var_type.upper()} Variables:")
                for row in matrix:
                    print(' '.join(map(str, row)))
                print("\n")
            elif var_type == 'p' and self.classification_objective != 'min_height':
                # finish the p_literals
                print("P Variables:")
                for p_literal, value in self.final_literals.items():
                    if p_literal.startswith('p_'):
                        # Convert positive values to 1 and negative to 0
                        clue = 1 if value > 0 else 0
                        print(f"{p_literal}: {clue}")
