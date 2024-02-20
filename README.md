# ESC499-Thesis-SAT-Trees

# Depedancies:

1. pySAT
2. graphviz
3. Numpy
4. Sklearn
5. openpxyl
6. pandas

# SATreeCraft 

SATreeCraft is a Python Library designed to solve classification problems using SAT-based encodings of decision trees to produce excat optimal solutions. It supports datasets with categorical and/or numerical features and can optimize for minimum tree height or maximum accuracy given a fixed depth. It also supports clusteirng tree objectives such as maximizing minimum split and minimizing maximum diameter

Currently Supports:
- Classifcation Objectives: Min height, Max accuracy
- Clustering Objectives:
- External SAT solver support is offered 

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

3. utils.py - contains dataloader for excel,csv, and .txt file datasets

 - Has been scaled to allow for different types of parsing of fields within files
 - Allows specification of labels and features
 - Numerically encodes features and labels
 - TreeDataLoaderBinaryNumerical is a dataloader created to only handle numerical and binary data features with excel and csv load in capabilities
 - TreeDataLoaderWithCategorical handles categorcal features, numerical, and binary only txt files 

4. DataReader.ipynb - allows for testing on different datasets

Please note that categorical feature tree modules have bene created for the min height and max accuarcy optimization problems for datasets that have combination of numerical, binary, and categorical features. Seen in the followin files: 

5. min_height_tree_categorical_module.py
6. fixed_height_tree_categorical_module.py 

# Datasets:
Datasets can be accessed via this link: https://archive.ics.uci.edu
