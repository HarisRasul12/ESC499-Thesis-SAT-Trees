# Created by: Haris Rasul
# Date: December 22nd 2023
# Python script to convert datasets into proper formats for inputting into tree creation
# Includes data loaders, sklearn frameworks data preporcessing in rder to feed into tree methods.

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


#To do handle xlsx files, csv files, label on first index, label on last index - give in the exetnesions do the proessing return the object type with all the stuff
class TreeDataLoader:
    def __init__(self, file_path, delimiter=',', label_position=-1):
        self.file_path = file_path
        self.delimiter = delimiter
        self.label_position = label_position  # Default is last column
        self.features = None
        self.labels = None
        self.true_labels_for_points = None
        self.dataset = None
        self.process_data_into_tree_form()  # Automatically process data on initialization
    
    def process_data_into_tree_form(self):
        pass