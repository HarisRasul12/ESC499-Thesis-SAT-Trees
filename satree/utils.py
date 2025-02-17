"""
=========== Module Description ===========

This module converts datasets into proper formats for inputting into tree creation. It includes data loaders
and preprocessing utilities that leverage scikit-learn and pandas frameworks to prepare datasets for tree-based
methods. Specifically, the module provides classes for loading and preprocessing:
  - Binary numerical datasets (via TreeDataLoaderBinaryNumerical), and
  - Datasets with categorical features (via TreeDataLoaderWithCategorical).

Both classes handle file reading (from CSV, Excel, or text files), custom column exclusion, and label encoding.
Additionally, the module offers a k-fold cross-validation utility (k_fold_tester) to evaluate SAT-based decision tree
classifiers using the processed datasets.

To do handle xlsx files, csv files, label on first index, label on last index - give in the extensions do
the processing return the object type with all the stuff
"""

import os
from typing import Optional, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from satree.SATreeClassifier import SATreeClassifier
from satree.SATreeCraft import SATreeCraft


class TreeDataLoaderBinaryNumerical:
    """
    A class to represent a dataset with labeled data for numerical and binary data only for features

    Attributes:
        file_path (str): The path to the dataset file.
        delimiter (str): The delimiter used in the dataset file to separate columns.
        label_position (int): The index of the column containing the labels. By default, it is set to -1, assuming the label is in the last column.
        features (np.ndarray): The names of the features, encoded as strings representing their column index.
        labels (np.ndarray): The unique labels present in the dataset after processing.
        true_labels_for_points (np.ndarray): The array containing the labels for each data point after processing.
        dataset (np.ndarray): The array containing the features for each data point.

    Methods:
        process_data_into_tree_form(): Reads the dataset from the file path, processes it, and populates the attributes with the processed data.

    Usage:
        # To create an instance of the class:
        my_dataset = TreeDataLoaderMinHeightBinaryNumerical('path/to/your/datafile.csv', delimiter=',', label_position=-1)

        # Access the processed data:
        my_dataset.features
        my_dataset.labels
        my_dataset.true_labels_for_points
        my_dataset.dataset
    """

    def __init__(self,
                 file_path: str,
                 delimiter: Optional[str] = None,
                 label_position: int = -1,
                 custom_exclude: Optional[List[int]] = None) -> None:
        """
        Initializes a data loader for binary numerical datasets.

        Args:
            file_path: Path to the dataset file.
            delimiter: Delimiter used in the dataset file (default is ',').
            label_position: Index of the label column (default is -1, i.e., last column).
            custom_exclude: Optional list of column indices to exclude.
        """
        self.file_path = file_path
        self.delimiter = delimiter or ','
        self.label_position = label_position
        self.custom_exclude = custom_exclude or []
        self.features = None
        self.labels = None
        self.true_labels_for_points = None
        self.dataset = None
        self.label_encoder = LabelEncoder()
        self.process_data_into_tree_form()

    def process_data_into_tree_form(self) -> None:
        """
        Reads and processes the dataset file, converting it into a format suitable for tree-based methods.
        """
        file_extension = os.path.splitext(self.file_path)[1].lower()

        if file_extension in ['.csv', '.xls', '.xlsx']:
            self._process_csv_or_excel()
        else:
            self._process_text_file()

    def _process_csv_or_excel(self) -> None:
        """
        Processes CSV or Excel files by reading the data, excluding custom columns, and splitting features and labels.
        """
        if self.file_path.endswith('.csv'):
            df = pd.read_csv(self.file_path)
        else:  # Excel file
            df = pd.read_excel(self.file_path)

        # Exclude custom columns if necessary
        if self.custom_exclude:
            df.drop(df.columns[self.custom_exclude], axis=1, inplace=True)

        # Assign features and labels based on label position
        if self.label_position is not None and self.label_position != -1:
            features_array = df.drop(df.columns[self.label_position], axis=1).values
            labels_array = df.iloc[:, self.label_position].values
        else:
            features_array = df.iloc[:, :-1].values
            labels_array = df.iloc[:, -1].values

        self._finalize_data(features_array, labels_array)

    def _process_text_file(self) -> None:
        """
        Processes a text file by reading each line, splitting by the delimiter, and converting feature values to floats.
        """
        features_list = []
        raw_labels_list = []
        with open(self.file_path, 'r') as file:
            for line in file:
                components = line.strip().split(self.delimiter)

                # Extract label based on the label_position after excluding custom columns
                label = components.pop(self.label_position) if self.label_position is not None else components.pop(-1)
                raw_labels_list.append(label)
                # print(label)
                if len(self.custom_exclude) > 0:
                    for i in self.custom_exclude:
                        components.pop(i)
                # Convert remaining components to float
                try:
                    features = [float(comp) for comp in components]
                except ValueError as e:
                    # Handle conversion error if necessary
                    print(f"Error converting to float: {e}")
                    continue  # Skip this line and continue with the next

                features_list.append(features)

        self._finalize_data(np.array(features_list), np.array(raw_labels_list))

    def _finalize_data(self, features_array: np.ndarray, labels_array: np.ndarray) -> None:
        """
        Finalizes the data processing by encoding labels, generating feature names, and storing the processed dataset.
        """
        # Encode labels to numeric values
        self.true_labels_for_points = self.label_encoder.fit_transform(labels_array)
        self.dataset = features_array
        self.features = np.array([str(i) for i in range(self.dataset.shape[1])])
        self.labels = np.unique(self.true_labels_for_points)


class TreeDataLoaderWithCategorical:
    """
    A class to load and preprocess data from a text file for use in decision tree algorithms.
    
    This class handles both categorical and numerical data and converts text-based labels to 
    numerical labels, while also handling rows with missing values.

    Attributes:
        file_path: The file path to the dataset.
        label_index: The index of the column containing the labels.
        numerical_indices: The indices of columns that contain numerical data.
        categorical_feature_index: The index of the column containing a string of categorical features.
        delimiter: The delimiter used in the text file to separate data columns.
        dataset: The array containing the processed features for each data point.
        features_categorical: The array containing the processed categorical features.
        features_numerical: The array containing the processed numerical features.
        true_labels_for_points: The array containing the processed labels for each data point.
        labels: The array containing the unique labels present in the dataset.
        features: The array containing the names of the features.

    Methods:
        process_data(): Main method to load and process the data from the file path.
        
    Usage:
        # To create an instance of the class:
        data_loader = TreeDataLoaderWithCategorical(
            file_path='path/to/datafile.txt',
            label_index=-1,
            numerical_indices=[1, 2],
            categorical_string_index=3
        )

        # Access the processed data:
        data_loader.dataset
        data_loader.features_categorical
        data_loader.features_numerical
        data_loader.true_labels_for_points
        data_loader.labels
        data_loader.features
    """

    def __init__(self,
                 file_path: str,
                 label_index: int,
                 numerical_indices: Optional[List[int]] = None,
                 categorical_feature_index: Optional[int] = None,
                 delimiter: str = ',') -> None:
        """
        Initializes a data loader for datasets containing both categorical and numerical features.

        Args:
            file_path: Path to the dataset file.
            label_index: Index of the column containing labels.
            numerical_indices: Optional list of indices for numerical columns.
            categorical_feature_index: Optional index of the column containing a string of categorical features.
            delimiter: Delimiter used in the file (default is ',').
        """
        self.file_path = file_path
        self.label_index = label_index
        self.numerical_indices = numerical_indices
        self.categorical_feature_index = categorical_feature_index
        self.delimiter = delimiter
        self.features = None
        self.features_categorical = None
        self.features_numerical = None
        self.labels = None
        self.true_labels_for_points = None
        self.dataset = None
        self.label_encoder = LabelEncoder()
        self.process_data()

    def process_data(self) -> None:
        """
        Loads and processes the dataset from a text file, handling both categorical and numerical features,
        performing label encoding, and managing missing values.
        """

        if self.categorical_feature_index is not None:
            with open(self.file_path, 'r') as file:
                raw_data = [line.strip().split(self.delimiter) for line in file if '?' not in line]

            # Extract labels and encode them
            labels = [row[self.label_index] for row in raw_data]
            self.true_labels_for_points = self.label_encoder.fit_transform(labels)
            self.labels = np.unique(self.true_labels_for_points)

            # Process features
            if self.categorical_feature_index is not None:
                # Split the string at the categorical feature index into individual characters
                features = [list(row[self.categorical_feature_index].strip()) for row in raw_data]
            else:
                # Treat each comma-separated value as a separate feature, excluding the label
                features = [row[:self.label_index] + row[self.label_index + 1:] for row in raw_data]

            self.dataset = np.array(features)

            # Generate feature names
            self.features = np.array([str(i) for i in range(self.dataset.shape[1])])

            # Identify and separate numerical and categorical features
            if self.numerical_indices is not None:
                self.features_numerical = self.dataset[:, self.numerical_indices].astype(float)
                categorical_indices = list(set(range(self.dataset.shape[1])) - set(self.numerical_indices))
                self.features_categorical = np.array([str(i) for i in categorical_indices])
            else:
                self.features_numerical = np.array([], dtype=float).reshape(self.dataset.shape[0], 0)
                self.features_categorical = np.array([str(i) for i in range(self.dataset.shape[1])])
        else:
            # Initialize lists to hold the dataset and labels
            dataset = []
            labels = []

            # Dictionary to map textual labels to numeric labels
            label_mapping = {}
            label_counter = 0

            # Read the file
            with open(self.file_path, 'r') as file:
                for line in file:
                    # Skip if there's a missing value
                    if '?' in line:
                        continue

                    # Split the line into parts and extract the label and features
                    parts = line.strip().split(',')
                    label = parts[self.label_index]
                    parts.pop(self.label_index)
                    features = parts

                    # If the label is new, add it to the label mapping
                    if label not in label_mapping:
                        label_mapping[label] = label_counter
                        label_counter += 1

                    # Add the numeric label and features to their respective lists
                    labels.append(label_mapping[label])
                    dataset.append(features)

            # Convert lists to numpy arrays
            self.dataset = np.array(dataset, dtype=str)  # Assuming features are numeric
            self.true_labels_for_points = np.array(labels, dtype=int)
            # print(self.dataset)
            features_list = [str(i) for i in range(self.dataset.shape[1])]
            self.features = np.array(features_list)
            self.labels = np.unique(self.true_labels_for_points)

            if self.numerical_indices is None:
                self.features_numerical = np.array([])
                self.features_categorical = self.features
            else:
                self.features_numerical = self.features[self.numerical_indices]
                # Get the residual elements
                mask = np.ones(len(self.features), dtype=bool)  # Create a mask of all True values
                mask[self.numerical_indices] = False  # Set the indices in x to False
                self.features_categorical = self.features[mask]  # Y contains el


def k_fold_tester(
        k: int,
        depth: int,
        dataset: np.ndarray,
        true_labels_for_points: np.ndarray,
        labels: List[Any],
        features: np.ndarray,
        features_categorical: Optional[List[str]] = None,
        features_numerical: Optional[List[str]] = None,
        complete_tree: bool = True,
        min_support_level: int = 0,
        min_margin_level: int = 1,
        loandra_path: Optional[str] = None
) -> Tuple[np.ndarray, float]:
    """
    Performs k-fold cross-validation to train a SAT-based decision tree and measure accuracy.

    If `loandra_path` is provided, uses the LOANDRA solver. Otherwise, defaults to
    the built-in SAT solver in `SATreeCraft.solve()`.

    Notes:
    - This function uses KFold from scikit-learn for cross-validation.
    - If `loandra_path` is None, it defaults to the standard solver.
    - If `loandra_path` is a valid path, LOANDRA integration is used.

    Args:
        k: Number of folds for cross-validation.
        depth: Fixed depth for the decision tree.
        dataset: Feature matrix (X). Each row corresponds to one data point.
        true_labels_for_points: Ground truth labels (y) for each row in `dataset`.
        labels: Array of all potential labels for the dataset.
        features: Array of feature names/indices used by the tree.
        features_categorical: Indices/names of categorical features. Default is None.
        features_numerical: Indices/names of numerical features. Default is None.
        complete_tree: If True, uses a Complete (standard) tree structure; if False, uses an Oblivious tree structure. Default is True.
        min_support_level: Minimum support constraint for leaves (default 0 means no constraint).
        min_margin_level: Minimum margin constraint (default 1 means no added margin).
        loandra_path: If provided, indicates the file path/location of the LOANDRA solver.
            The decision tree is then constructed using `solve_loandra(loandra_path)` instead of the default `.solve()` method. Default is None.

    Returns:
        k_accuracies: Array of length k with training accuracies for each fold.
        mean_score: Mean accuracy across all folds.
    """

    # Determine tree structure type
    tree_structure = 'Complete' if complete_tree else 'Oblivious'

    # Prepare arrays for storing fold accuracies
    k_accuracies = []
    kf = KFold(n_splits=k, shuffle=True)

    # Loop over the k folds
    for train_index, test_index in kf.split(dataset):
        X_train, X_test = dataset[train_index], dataset[test_index]
        y_train, y_test = true_labels_for_points[train_index], true_labels_for_points[test_index]

        # Create a SATreeCraft instance
        max_accuracy_problem = SATreeCraft(
            dataset=X_train,
            features=features,
            labels=labels,
            true_labels_for_points=y_train,
            features_categorical=features_categorical,
            features_numerical=features_numerical,
            classification_objective='max_accuracy',
            fixed_depth=depth,
            min_support_level=min_support_level,
            min_margin=min_margin_level,
            tree_structure=tree_structure
        )

        # If LOANDRA path is provided, solve with LOANDRA; otherwise solve normally
        if loandra_path is None:
            max_accuracy_problem.solve()
        else:
            max_accuracy_problem.solve_loandra(loandra_path)

        # Build the classifier from the resulting model
        model = SATreeClassifier(max_accuracy_problem.model)

        # Evaluate on the test set
        acc = model.score(X_test, y_test)
        k_accuracies.append(acc)
        print('Fold complete. Accuracy =', acc)

    # Ensure the accuracies array is of type float
    k_accuracies = np.array(k_accuracies, dtype=float)
    mean_score = float(np.mean(k_accuracies))
    return k_accuracies, mean_score
