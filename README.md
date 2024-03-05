# ESC499-Thesis-SAT-Trees Library: SATreeCraft, an Open Source Python SAT Optimization Library for the Construction of Optimal Decision Trees

<p align="center">
<img width="200" alt="Screenshot 2024-03-05 at 1 29 24 AM" src="https://github.com/HarisRasul12/ESC499-Thesis-SAT-Trees/assets/66268214/498cb519-4745-42a2-89b4-2565ad0dfe35">
</p>


 <strong>This contains the Python library code for the Engineering Science Thesis by Haris Rasul 2023-2024. </strong>


# Depedancies:

1. pySAT
2. graphviz
3. Numpy
4. Sklearn
5. openpxyl
6. pandas

# SATreeCraft 

SATreeCraft is a Python Library designed to solve classification problems using SAT-based encodings of decision trees to produce excat optimal solutions. It supports datasets with categorical and/or numerical features and can optimize for minimum tree height or maximum accuracy given a fixed depth. It also supports clustering tree objectives such as maximizing minimum split and minimizing maximum diameter

Currently Supports:
- Classifcation Objectives: Min height, Max accuracy
- Classification Constraints: Cardinality constraint Minimum support, Minimum Margin Constraints
- Clustering Objectives:
- External SAT solver support is offered 


# Main Modules

1. SATreeCraft.py
 - Contans the SAT Encoding tree builder class that creates model ,finds the optimal objective value
 - allows user to specify problem type, tree structure type, optimization objecive
 - can export CNF to external solver

2. SATreeClassifier.py
 - Contains tree classifcation predictor function given built model from SATreeCraft()
 - contains sklearn pipeline and metrics for model evalauation 


3. utils.py - contains dataloader for excel,csv, and .txt file datasets

 - Has been scaled to allow for different types of parsing of fields within files
 - Allows specification of labels and features
 - Numerically encodes features and labels
 - TreeDataLoaderBinaryNumerical is a dataloader created to only handle numerical and binary data features with excel and csv load in capabilities
 - TreeDataLoaderWithCategorical handles categorcal features, numerical, and binary only txt files 

# Submodule Code Description:

1. min_height_tree_module.py - this contains the SAT min-height tree problem model proposed in P. Shati, E. Cohen, and S. McIlraith, “Optimal decision trees for interpretable clustering with constraints,” Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, 2023. doi:10.24963/ijcai.2023/225

- This itertaively builds the tree with a starting depth of 1 and keeps expanding until solution is found
- It also visualizes the tree that was built and stored in the images folder
- Current solver: mm2

2. fixed_height_tree_module.py - this contains the partial MaxSAT problem model for maximiinzg number of corrcet training labels for any given depth proposed in  P. Shati, E. Cohen, and S. McIlraith, “Optimal decision trees for interpretable clustering with constraints,” Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, 2023. doi:10.24963/ijcai.2023/225

 - Must give a fixed depth and uses same logic as min_heightPtree_module.py for building tree structures and dceoding solutions
 - Uses Weighted CNF approach for MaxSAT to tag hard clauses (no weight), and soft clauses (weight = 1)
 - cost = number of correct label assignmnet literals pi's that are true (soft clauses)
 - solver = RC2

3. DataReader.ipynb - allows for testing on different datasets

Please note that categorical feature tree modules have bene created for the min height and max accuarcy optimization problems for datasets that have combination of numerical, binary, and categorical features. Seen in the followin files: 

4. min_height_tree_categorical_module.py - min height tree problem with categorical dataset
5. fixed_height_tree_categorical_module.py - fixed height max accuarcy with categorical dataset


# Constraint modules:
Classification Constraints:
- additional_classification_constraints.py - contains cardinality constraint such as minimum support and other additional constraints user can put onto their tree constructor such as minimum margin.

# Datasets:
Datasets can be accessed via this link: https://archive.ics.uci.edu

# Contributors
1. Haris Rasul, AI Engineering - UofT
