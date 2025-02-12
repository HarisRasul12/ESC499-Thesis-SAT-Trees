"""
=========== Module Description ===========

SATreeCraft Python Library for user-oriented approach.

This library provides tools for solving classification problems using SAT-based decision trees. It supports two classification objectives:
1. Minimum height tree with 100% training classification.
2. Maximum accuracy given a fixed depth.

The library works with datasets containing both categorical and numerical features.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pysat.formula import WCNF

from satree.treemodder.builder import build_complete_tree, create_literals

from satree.classification.min_depth_tree import build_clauses, add_thresholds, solve_cnf, visualize_tree
from satree.classification.fixed_depth_tree import build_clauses_fixed_tree, solve_wcnf
from satree.classification.min_depth_tree_categorical import build_clauses_categorical, add_thresholds_categorical
from satree.classification.fixed_depth_tree_categorical import build_clauses_categorical_fixed
from satree.classification.additional_constraints import add_oblivious_tree_constraints, min_support, build_clauses_fixed_tree_min_margin_constraint_add

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
    given a fixed depth. It also supports clusteirng tree objectives such as maximizing minimum split and minimizing maximum diameter

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

    def __init__(self, dataset,features,labels = None, true_labels_for_points = None, features_numerical = None, features_categorical = None,
                 is_classification = True, classification_objective = 'min_height', fixed_depth = None, tree_structure = 'Complete', min_support = 0,
                 min_margin = 1, k_clusters = None, clustering_objective = 'max_diameter', is_clustering = False, epsilon = 0, CL_pairs = np.array([]), ML_pairs = np.array([]),
                 smart_pairs = False
                 ):
        
        """Initializes the SATreeCraft instance with provided dataset and configuration."""

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
        self.min_support = min_support
        self.min_margin = min_margin


        # Clustering tools
        self.k_clusters = k_clusters
        self.clustering_objective = clustering_objective
        self.is_clustering = is_clustering
        
        if self.is_clustering or (self.k_clusters is not None):
            self.is_classification = False
        
        self.epsilon = epsilon
        self.CL_pairs = CL_pairs
        self.ML_pairs = ML_pairs
        self.smart_pairs = smart_pairs

        # return types 
        self.tree_model = None
        self.sat_solution = None
        self.min_cost = None
        self.min_depth = None
        self.final_cnf = None
        self.final_literals = None
        self.cluster_assignments = None
        self.cluster_diameters = None

    ##### Categorical Classfication Problems ####


    def apply_oblivious_constraints_and_solve(self, cnf, features, depth, literals, dataset, tree, TL, labels,
                                              use_loandra, loandra_path=None, execution_path=None):
        """
        Applies oblivious tree constraints to the CNF and then solves it using either the Loandra-based solver
        or the standard SAT solver.

        Parameters:
            cnf: The CNF (list of clauses) to be processed.
            features: The feature set.
            depth: The current depth.
            literals: The literals generated for the tree.
            dataset: The dataset.
            tree: The current tree structure.
            TL: The tree's leaves (or additional tree information needed by the solver).
            labels: The labels.
            use_loandra (bool): Whether to use Loandra for solving.
            loandra_path (optional): Path to the Loandra executable (if use_loandra is True).
            execution_path (optional): Path to write the CNF file (if use_loandra is True).

        Returns:
            solution: The solution (transformed if using Loandra).
            cost: The cost from the Loandra solver (or None for standard solving).
            cnf: The updated CNF (after adding constraints).
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
                solution = transform_tree_from_loandra(solution, literals, TL, tree, labels, features)
        else:
            # --- STANDARD SAT SOLVER PATH ---
            solution = solve_cnf(cnf, literals, TL, tree, labels, features)
            cost = None

        return solution, cost, cnf


    def find_min_depth_tree_categorical_problem(self, features, features_categorical, features_numerical, labels,
                                                true_labels_for_points, dataset, use_loandra=False,
                                                loandra_path=None, execution_path=None):
        depth = 1  # Start with a depth of 1
        solution = "No solution exists"
        tree_with_thresholds = None
        tree = None
        literals = None

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
                tree_with_thresholds = add_thresholds_categorical(tree, literals, solution, dataset, features_categorical)
                dot = visualize_tree(tree_with_thresholds)
                folder = Path('images/min_height/')
                folder.mkdir(parents=True, exist_ok=True)
                if use_loandra:
                    dot.render(folder / f'LOANDRA_SOLVED_binary_decision_tree_min_depth_with_categorical_features_depth_{depth}',
                               format='png', cleanup=True)
                else:
                    dot.render(folder / f'binary_decision_tree_min_depth_with_categorical_features_depth_{depth}',
                               format='png', cleanup=True)
            else:
                print("No solution at depth: ", depth)
                depth += 1  # Increase the depth and try again

        return tree_with_thresholds, literals, depth, solution, cnf


    def find_fixed_depth_tree_categorical_problem(self, features, features_categorical, features_numerical, labels, true_labels_for_points, dataset, depth, use_loandra=False, loandra_path=None, execution_path=None):
        solution = "No solution exists"
        tree_with_thresholds = None
        tree = None
        literals = None
        cost = None

        # Build the complete tree and generate literals
        tree, TB, TL = build_complete_tree(depth)
        literals = create_literals(TB, TL, features, labels, len(dataset), True)[0]

        # Build the CNF clauses using the provided categorical fixed encoding
        wcnf = build_clauses_categorical_fixed(literals, dataset, TB, TL, len(features), features_categorical, features_numerical, labels, true_labels_for_points)

        # Add the min support constraint if applicable
        solution, cost, wcnf = self.finalize_and_solve_wcnf(
            wcnf, literals, dataset, TL, features, depth, tree, labels,
            use_loandra, loandra_path, execution_path
        )

        # If a solution was found, add thresholds and generate a visualization
        if solution != "No solution exists":
            tree_with_thresholds = add_thresholds_categorical(tree, literals, solution, dataset, features_categorical)
            dot = visualize_tree(tree_with_thresholds)
            folder = Path('images/fixed_height/')
            folder.mkdir(parents=True, exist_ok=True)
            if use_loandra:
                dot.render(folder / f'LOANDRA_SOLVED_binary_decision_tree_fixed_with_categorical_features_depth_{depth}', format='png', cleanup=True)
            else:
                dot.render(folder / f'binary_decision_tree_fixed_with_categorical_features_depth_{depth}', format='png', cleanup=True)
        else:
            print('could not find solution')
            return 'No solution'

        return tree_with_thresholds, literals, depth, solution, cost, wcnf

    #### Numerical Classification Problems ####

    def find_min_depth_tree_problem(self, features, labels, true_labels_for_points, dataset,
                                    use_loandra=False, loandra_path=None, execution_path=None):
        depth = 1  # Start with a depth of 1
        solution = "No solution exists"
        tree_with_thresholds = None
        tree = None
        literals = None

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
                folder = Path('images/min_height/')
                folder.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
                if use_loandra:
                    dot.render(folder / f'LOANDRA_SOLVED_binary_decision_tree_min_depth_{depth}', format='png', cleanup=True)
                else:
                    dot.render(folder / f'binary_decision_tree_min_depth_{depth}', format='png', cleanup=True)
            else:
                print('no solution at depth', depth)
                depth += 1  # Increase the depth and try again

        return tree_with_thresholds, literals, depth, solution, cnf


    def finalize_and_solve_wcnf(self, wcnf, literals, dataset, TL, features, depth, tree, labels,
                                use_loandra, loandra_path=None, execution_path=None):
        """
        Applies the min support and oblivious tree constraints to the given WCNF and then solves it.

        Parameters:
          - wcnf: The weighted CNF to be solved.
          - literals: The literals generated for the problem.
          - dataset: The dataset.
          - TL: The tree leaves (or other tree-related info) needed for solving.
          - features: The feature set.
          - depth: The current depth.
          - tree: The tree structure.
          - labels: The label set.
          - use_loandra (bool): Flag indicating whether to use Loandra for solving.
          - loandra_path (optional): The path to the Loandra executable (required if use_loandra is True).
          - execution_path (optional): The path to write the CNF file for Loandra.

        Returns:
          - solution: The solution obtained (or transformed) by the solver.
          - cost: The cost (if applicable; None for the standard solver).
          - wcnf: The final WCNF after constraints have been applied.
        """
        # Apply min support constraint if specified.
        if self.min_support > 0:
            wcnf = min_support(wcnf, literals, dataset, TL, self.min_support)

        # Add oblivious tree constraints.
        wcnf = add_oblivious_tree_constraints(wcnf, features, depth, literals, dataset, self.tree_structure)

        # Solve using either Loandra or the standard solver.
        if use_loandra:
            # --- LOANDRA BRANCH ---
            wcnf.to_file(execution_path)
            solution, cost = run_loandra_and_parse_results(loandra_path, execution_path)
            solution = transform_tree_from_loandra(solution, literals, TL, tree, labels, features)
        else:
            # --- STANDARD SOLVER BRANCH ---
            solution, cost = solve_wcnf(wcnf, literals, TL, tree, labels, features)

        return solution, cost, wcnf


    def find_fixed_depth_tree_problem(self, features, labels, true_labels_for_points, dataset, depth,
                                      use_loandra=False, loandra_path=None, execution_path=None):
        solution = "No solution exists"
        tree_with_thresholds = None
        tree = None
        literals = None
        cost = None

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
            folder = Path('images/fixed_height/')
            folder.mkdir(parents=True, exist_ok=True)
            if use_loandra:
                dot.render(folder / f'LOANDRA_SOLVED_binary_decision_tree_fixed_depth_{depth}', format='png', cleanup=True)
            else:
                dot.render(folder / f'binary_decision_tree_fixed_depth_{depth}', format='png', cleanup=True)
        else:
            print('could not find solution')
            return 'No solution'

        return tree_with_thresholds, literals, depth, solution, cost, wcnf


    # Save the plot to the specified directory with the given filename format
    def plot_and_save_clusters_to_drive(self,dataset, cluster_assignments, k_clusters):
        """
        Plots the dataset points before and after clustering if the dataset has 1 or 2 features.
        Creates a side-by-side plot showing the dataset before clustering and after with cluster IDs.
        Saves the plot to the specified directory with a filename based on the number of clusters.
        Does not display the plot in the output.

        Parameters:
        - dataset (np.ndarray): The original dataset with data points.
        - cluster_assignments (dict): A dictionary with cluster IDs and lists of data points in each cluster.
        - k_clusters (int): The number of clusters.

        Returns:
        - full_path (str): The path to the saved plot image.
        """
        # Define the directory and filename
        directory = Path('images/cluster_trees/')
        directory.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        filename = f'cluster_tree_with_cluster_size{k_clusters}.png'
        full_path = directory / filename

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
        colors = plt.cm.tab10(np.linspace(0, 1, k_clusters))
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

    def solve_clustering_problem_max_diameter(self, dataset, features, k_clusters, depth, epsilon, CL_pairs, ML_pairs,
                                              use_loandra=False, loandra_path=None, execution_path=None):
        dataset_size = len(dataset)
        num_features = len(features)
        dist1, dist2, distance_classes = create_distance_classes(dataset, epsilon)
        tree_structure, TB, TL = build_complete_tree(depth)

        literals = create_literals_cluster_tree(TB, TL, features, k_clusters, dataset_size, distance_classes, False)
        wcnf = build_clauses_cluster_tree_md(literals, dataset, TB, TL, num_features, k_clusters,
                                             CL_pairs, ML_pairs, distance_classes)

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


    def solve_clustering_problem_bicriteria(self, dataset, features, k_clusters, depth, epsilon, CL_pairs, ML_pairs,
                                            use_loandra=False, loandra_path=None, execution_path=None):
        dataset_size = len(dataset)
        num_features = len(features)
        dist1, dist2, distance_classes = create_distance_classes(dataset, epsilon)
        tree_structure, TB, TL = build_complete_tree(depth)

        # Create the literals with bicriteria flag True.
        literals = create_literals_cluster_tree(TB, TL, features, k_clusters, dataset_size, distance_classes, True)

        # Build the WCNF using smart pairs if enabled.
        if self.smart_pairs:
            wcnf = build_clauses_cluster_tree_md_ms_smart_pair(literals, dataset, TB, TL, num_features, k_clusters,
                                                               CL_pairs, ML_pairs, distance_classes)
        else:
            wcnf = build_clauses_cluster_tree_md_ms(literals, dataset, TB, TL, num_features, k_clusters,
                                                    CL_pairs, ML_pairs, distance_classes)

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
    def solve(self):
        """
        Solve the decision tree problem based on specified objectives and dataset features. Classifcation or Clustering 
        It chooses between categorical and numerical feature handling as well as the optimization
        objective (minimum height or maximum accuracy given a fixed depth).
        must set is_classification to false to work on clusteirng porblem or set is_clustering to true 
        """

        if self.is_classification: # classifciation problem domain
            
            if self.features_categorical is not None and len(self.features_categorical) > 0: # categorical feature dataset
                
                if self.classification_objective == 'min_height': # minimum height 100% accuracy on training problem
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
                else: # Max accuracy problem
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
            else: # numerical feature dataset strictly
                if self.classification_objective == 'min_height':
                    self.model, self.final_literals, self.min_depth, self.sat_solution, self.final_cnf = \
                        self.find_min_depth_tree_problem(
                            self.features,
                            self.labels,
                            self.true_labels_for_points,
                            self.dataset,
                            use_loandra=False
                        )
                else: # max accuracy problem
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
                raise ValueError(f"The assigned depth {self.fixed_depth} is not sufficient to accommodate {self.k_clusters} clusters.")
            
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
            else: # Bicriteria
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
    
    
    def export_cnf(self, filename='dimacs/export_to_solver.cnf'):
        """
        Exports the final CNF formula to a file in DIMACS format. This allows for the use
        of the CNF with external solvers. The export is only available after solving the CNF. 
        Supports both weighted and non weighted cnf. 
        """
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        if self.final_cnf:
            self.final_cnf.to_file(filename)
        else:
            raise ValueError("Cannot export CNF. The final CNF is not available. Make sure to solve the problem first.")

    def export_cnf_min_height(self, filename='dimacs/export_to_solver_min_height.cnf'):
        """
        Exports the final CNF formula to a file in DIMACS format. This allows for the use
        of the CNF with external solvers. The export is only available after solving the CNF. 
        Supports both weighted and non weighted cnf. 
        """
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        if self.final_cnf:
            wcnf = WCNF()
            for clause in self.final_cnf:
                    wcnf.append(clause)
            wcnf.to_file(filename)
        else:
            raise ValueError("Cannot export CNF. The final CNF is not available")
    
    def export_cnf_max_accuracy_problem(self, filename='dimacs/export_to_solver_max_acc_problem.cnf'):
        """
        Exports the final CNF formula to a file in DIMACS format. This allows for the use
        of the CNF with external solvers. Export before solving max accuracy problem.
        """
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        if self.is_classification: # classifciation problem domain
                if self.classification_objective != 'min_height': # minimum height 100% accuracy on training problem
                    
                    tree_with_thresholds = None
                    tree = None
                    literals = None
                    cost = None
                    tree, TB, TL = build_complete_tree(self.fixed_depth)
                    literals = create_literals(TB, TL, self.features, self.labels, len(self.dataset), True)[0]
                    
                    if self.features_categorical is not None and len(self.features_categorical) > 0: # categorical feature dataset
                        wcnf = build_clauses_categorical_fixed(literals, self.dataset, TB, TL, len(self.features), 
                                                            self.features_categorical, self.features_numerical, 
                                                            self.labels,self.true_labels_for_points)
                    else:
                        wcnf = build_clauses_fixed_tree(literals, self.dataset, TB, TL, len(self.features), self.labels, self.true_labels_for_points)
                    
                    wcnf.to_file(filename)
                else:
                    ("Cannot export CNF without solving for min height problem first.")
        else:
            raise ValueError("Cannot export CNF. The final CNF is not available. Make sure to solve the problem first.")
    
    def export_cnf_min_height_k(self,depth,filename = 'dimacs/export_to_solver_min_height_problem_at_given_depth.cnf'):
        """
        Exports the CNF formula at a given depth k to a file in DIMACS format. 
        This allows for External solver support of the CNF problem.
        """
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        if self.is_classification: # classifciation problem domain
                if self.classification_objective == 'min_height': # minimum height 100% accuracy on training problem
                    solution = "No solution exists"
                    tree_with_thresholds = None
                    tree = None
                    literals = None
                    tree, TB, TL = build_complete_tree(depth)
                    literals = create_literals(TB, TL, self.features, self.labels, len(self.dataset), False)[0]

                    if self.features_categorical is not None and len(self.features_categorical) > 0: # categorical feature dataset
                        cnf = build_clauses_categorical(literals, 
                                                        self.dataset, TB, TL, len(self.features), 
                                                        self.features_categorical, self.features_numerical, self.labels, self.true_labels_for_points)
                    else:
                        cnf = build_clauses(literals, self.dataset, TB, TL, len(self.features), self.labels, self.true_labels_for_points)
                    wcnf = WCNF()
                    for clause in cnf:
                        wcnf.append(clause)
                    wcnf.to_file(filename)
                else:
                    ("Must be a min height objective")
        else:
            ("Cannot export CNF ")
    
    

    def solve_loandra(self,loandra_path,execution_path='dimacs/export_to_solver.cnf'):
        """
        Solve the decision tree problem based on specified objectives and dataset features. Classifcation or Clustering 
        It chooses between categorical and numerical feature handling as well as the optimization
        objective (minimum height or maximum accuracy given a fixed depth).
        LOANDRA VARIANT - calls external solver support system 
        """
        execution_path = Path(execution_path)
        execution_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        if self.is_classification: # classifciation problem domain
            
            if self.features_categorical is not None and len(self.features_categorical) > 0: # categorical feature dataset
                
                if self.classification_objective == 'min_height': # minimum height 100% accuracy on training problem
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
                            execution_path=execution_path
                        )
                else: # Max accuracy problem
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
                            execution_path=execution_path
                        )

            else: # numerical feature dataset strictly
                if self.classification_objective == 'min_height':
                    self.model, self.final_literals, self.min_depth, self.sat_solution, self.final_cnf = \
                        self.find_min_depth_tree_problem(
                            self.features,
                            self.labels,
                            self.true_labels_for_points,
                            self.dataset,
                            use_loandra=True,
                            loandra_path=loandra_path,
                            execution_path=execution_path
                        )
                else: # max accuracy problem
                    self.model, self.final_literals, self.fixed_depth, self.sat_solution, self.min_cost, self.final_cnf = \
                        self.find_fixed_depth_tree_problem(
                            self.features,
                            self.labels,
                            self.true_labels_for_points,
                            self.dataset,
                            self.fixed_depth,
                            use_loandra=True,
                            loandra_path=loandra_path,
                            execution_path=execution_path
                        )
        else:
            max_clusters = 2 ** self.fixed_depth
            if self.k_clusters > max_clusters:
                raise ValueError(f"The assigned depth {self.fixed_depth} is not sufficient to accommodate {self.k_clusters} clusters.")
            
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
                        execution_path=execution_path
                    )
            else: # bicriteria
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
                        execution_path=execution_path
                    )

 
    ##################################### Auxillary Helper Functions for User Interface #############################


    def create_solution_matrix(self, literals, solution, var_type):
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
    
    
    def display_solution(self):
        '''
        Display solved solution of porblem in readble format of literals
        '''
        print("\nSolution of Literals")
        
        if self.classification_objective == 'min_height':
            var_types = ['a', 's', 'z', 'g']
        else:
            var_types = ['a', 's', 'z', 'g','p']
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
