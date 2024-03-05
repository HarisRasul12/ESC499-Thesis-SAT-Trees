# Created by: Haris Rasul
# Date: December 22nd 2023
# Python script to convert datasets into proper formats for inputting into tree creation
# Includes data loaders, sklearn frameworks data preporcessing in rder to feed into tree methods.

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from SATreeClassifier import SATreeClassifier
from SATreeCraft import SATreeCraft

from sklearn.model_selection import KFold

#To do handle xlsx files, csv files, label on first index, label on last index - give in the exetnesions do the proessing return the object type with all the stuff
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
    def __init__(self, file_path, delimiter=None, label_position=-1, custom_exclude=None):
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

    def process_data_into_tree_form(self):
        file_extension = os.path.splitext(self.file_path)[1].lower()

        if file_extension in ['.csv', '.xls', '.xlsx']:
            self._process_csv_or_excel()
        else:
            self._process_text_file()

    def _process_csv_or_excel(self):
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

    def _process_text_file(self):
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

        self._finalize_data(np.array(features_list), raw_labels_list)

    def _finalize_data(self, features_array, labels_array):
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
        file_path (str): The file path to the dataset.
        label_index (int): The index of the column containing the labels.
        numerical_indices (list of int): The indices of columns that contain numerical data.
        categorical_string_index (int or None): The index where a single string of categorical 
                                                 features is located, if applicable.
        delimiter (str): The delimiter used in the text file to separate data columns.
        dataset (np.ndarray): The array containing the processed features for each data point.
        features_categorical (np.ndarray): The array containing the processed categorical features.
        features_numerical (np.ndarray): The array containing the processed numerical features.
        true_labels_for_points (np.ndarray): The array containing the processed labels for each data point.
        labels (np.ndarray): The array containing the unique labels present in the dataset.
        features (np.ndarray): The array containing the names of the features.

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
    def __init__(self, file_path, label_index, numerical_indices=None, categorical_feature_index=None, delimiter=','):
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
    

    def process_data(self):

        
        if self.categorical_feature_index != None:
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
                features = [row[:self.label_index] + row[self.label_index+1:] for row in raw_data]

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

            if (self.numerical_indices is None):
                self.features_numerical = np.array([])
                self.features_categorical = self.features
            else:
                self.features_numerical = self.features[self.numerical_indices]
                # Get the residual elements
                mask = np.ones(len(self.features), dtype=bool)  # Create a mask of all True values
                mask[self.numerical_indices] = False  # Set the indices in x to False
                self.features_categorical = self.features[mask]  # Y contains el

def k_fold_tester(k, depth, dataset, true_labels_for_points, labels, features, features_categorical=None, features_numerical=None, complete_tree=True, min_support_level = 0,
                  min_margin_level = 1):
    '''
    Inputs:
    k - number of folds ; (int)
    depth - fixed depth tree created ; (int)
    splitrate - train test split wanted ; (float)
    dataset: dataset X to create folds upon ; (np.ndarray)
    true_labels_for_points: labels y assigned to each x_i datapoint of dataset X ; (np.ndarray)
    labels: all potential labels for dataset X ; (np.ndarray)
    features: all features represented by an index ; (np.ndarray)
    features_categorical: Optional - required for categorical feature dataset ; default = None (np.ndarray)
    features_numerical: Optional - required for numerical feature dataset ; default = None (np.ndarray)
    complete_tree : boolean indicator to see if tree type of construction is complete or oblivious (currently only supports complete) ; (bool) 

    Returns:
    k_fold_array - np.ndarray of k length with all k runs of training accuracies
    mean_accuracy - mean score of the array
    '''
    k_accuracies = []
    kf = KFold(n_splits=k, shuffle=True)

    for train_index, test_index in kf.split(dataset):
        X_train, X_test = dataset[train_index], dataset[test_index]
        y_train, y_test = true_labels_for_points[train_index], true_labels_for_points[test_index]

        # Assuming SATreeCraft and SATreeClassifier are defined elsewhere and work similarly to scikit-learn models
        # print(min_support_level)
        max_accuracy_problem = SATreeCraft(dataset=X_train,
                                           features=features,
                                           labels=labels,
                                           true_labels_for_points=y_train,
                                           features_categorical=features_categorical,
                                           features_numerical=features_numerical,
                                           classification_objective='max_accuracy',
                                           fixed_depth=depth,
                                           min_support = min_support_level,
                                           min_margin= min_margin_level)
        # build model
        max_accuracy_problem.solve()
        model = SATreeClassifier(max_accuracy_problem.model)
        
        # score model on test data
        k_accuracies.append(model.score(X_test, y_test))
        print('Iteration complete')

    k_accuracies = np.array(k_accuracies)
    mean_score = np.mean(k_accuracies)

    return k_accuracies, mean_score

