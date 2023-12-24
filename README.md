# ESC499-Thesis-SAT-Trees

# Depedancies:

1. pySAT
2. graphviz
3. Numpy
4. Sklearn
5. openpxyl
6. pandas

# Code Description:

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
 - TreeDataLoaderBinaryNumerical is a dataloader created to only handle numerical and binary data features

4. DataReader.ipynb - allows for testing on different datasets


# Datasets:
Datasets can be accessed via this link: https://archive.ics.uci.edu
