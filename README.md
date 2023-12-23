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

2. utils.py - contains dataloader for excel,csv, and .txt file datasets

 - Has been scaled to allow for different types of parsing of fields within files
 - Allows specification of labels and features
 - Numerically encodes features and labels
 - TreeDataLoaderBinaryNumerical is a dataloader created to only handle numerical and binary data features

3. DataReader.ipynb - allows for testing on different datasets


