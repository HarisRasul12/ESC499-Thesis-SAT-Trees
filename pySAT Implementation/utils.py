# Created by: Haris Rasul
# Date: December 22nd 2023
# Python script to convert datasets into proper formats for inputting into tree creation
# Includes data loaders, sklearn frameworks data preporcessing in rder to feed into tree methods.

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

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
    A class to represent a dataset with labeled data for numerical, binary data, and catgeroical data

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