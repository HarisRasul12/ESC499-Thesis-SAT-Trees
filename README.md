# Engineering Science Undergraduate Thesis: SATreeCraft, an Open Source Python SAT Optimization Library for the Construction of Optimal Decision Trees

<p align="center">
<img width="200" alt="Screenshot 2024-03-05 at 1 29 24 AM" src="https://github.com/HarisRasul12/ESC499-Thesis-SAT-Trees/assets/66268214/498cb519-4745-42a2-89b4-2565ad0dfe35">
</p>


 This is the open source Python library produced for the University of Toronto's Engineering Science Program Undergraduate Thesis by Haris Rasul 2023-2024.

<strong> [Full Paper Accessible in the PDF Report](/Thesis-FinalReport-HarisRasul-2023-24%20copy.pdf) </strong>

Rasul, H., 2024. SATreeCraft: Creating a Flexible Framework for the SAT Encoding of Optimal Decision Trees with Constraints. University of Toronto.





# Table of Contents
1. [Dependencies](#dependencies)
2. [About SATreeCraft](#satreecraft)
3. [Main Modules](#main-modules)
4. [Submodule Code Description](#submodule-code-description)
   - 4.1. [Classification Problems](#classification-problems)
   - 4.2. [Clustering Problems](#clustering-problems)
5. [Constraint Modules](#constraint-modules)
   - 5.1. [Classification Constraints](#classification-constraints)
   - 5.2. [Clustering Constraints](#clustering-constraints)
6. [Datasets](#datasets)
7. [External Solver Support](#external-solver-support-loandra)
8. [Contributors and Acknowledgements](#contributors-and-acknowledgements)

# Dependencies:

<ol>
  <li><a href="https://pysathq.github.io/installation/" target="_blank">pySAT</a></li>
  <li><a href="https://pypi.org/project/graphviz/" target="_blank">graphviz</a></li>
  <li><a href="https://numpy.org/install/" target="_blank">Numpy</a></li>
  <li><a href="https://scikit-learn.org/stable/install.html" target="_blank">Sklearn</a></li>
  <li><a href="https://pypi.org/project/openpyxl/" target="_blank">openpxyl</a></li>
  <li><a href="https://pandas.pydata.org/docs/getting_started/install.html" target="_blank">pandas</a></li>
  <li><a href="https://github.com/jezberg/loandra" target="_blank">LOANDRA</a></li>
 <li><a href="https://matplotlib.org/stable/users/installing/index.html" target="_blank">matplotlib</a></li>
</ol>

# SATreeCraft 

SATreeCraft is a Python Library designed to solve classification/clustering problems using SAT-based encodings of decision trees to produce exact optimal solutions. It supports datasets with categorical and/or numerical features and can optimize for minimum tree height or maximum accuracy given a fixed depth. It also supports clustering tree objectives such as maximizing minimum split and minimizing maximum diameter

Currently Supports:
- Classifcation Objectives: Min height, Max accuracy
- Classification Constraints: Cardinality constraint Minimum support, Minimum Margin Constraints

- Clustering Objectives: Minimize Maximum Diameter
- Clustering Constraints: Pairwise Must-link, Cannot-link constraints
  
- External SAT solver support is offered via LOANDRA


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

## Classification Problems:

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

## Clustering Problems:

1. clustering_advanced.py - This contains the encodings for interpretable constrained clustering using SAT Optimized Decsision trees.The SAT formulation follows the encodins proposed in P. Shati, E. Cohen, and S. McIlraith, “Optimal decision trees for interpretable clustering with constraints,” Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, 2023. doi:10.24963/ijcai.2023/225
2. clustering_minsplit.py - This contains the encodings for minimum split and maximum diameter bi-criteria optimization problem proposed in Shati et al.

version 1.1.0 currently supports Two Clustering objectives

Minimize Maximum Diameter:

<img width="494" alt="Screenshot 2024-03-23 at 3 15 21 AM" src="https://github.com/HarisRasul12/ESC499-Thesis-SAT-Trees/assets/66268214/6442428d-2e83-4933-8c78-806e880b80d0">

Bicriteria Optimization (Max min split; Min max diameter) :

<img width="494" alt="Screenshot 2024-03-23 at 3 15 21 AM" src="https://github.com/HarisRasul12/ESC499-Thesis-SAT-Trees/assets/66268214/6442428d-2e83-4933-8c78-806e880b80d0">

<img width="360" alt="Screenshot 2024-04-27 at 3 00 53 PM" src="https://github.com/HarisRasul12/ESC499-Thesis-SAT-Trees/assets/66268214/6923b5c8-430b-4e85-a033-1c26444a5986">

## Constraint modules:
### Classification Constraints:
- additional_classification_constraints.py - contains cardinality constraint such as minimum support and other additional constraints user can put onto their tree constructor such as minimum margin.

1. min_support(wcnf, literals, X, TL, min_support) -  Add minimum support constraints for decision tree leaf nodes.
2. build_clauses_fixed_tree_min_margin_constraint_add(wcnf, min_margin (int) ) - minumim margin constraint added  to existing problem
3. add_oblivious_tree_constraints(cnf, features, depth, literals) - Add constraints to the CNF for an oblivious tree where all nodes at the same level
    must select the same feature for splitting.

The Oblivious Tree constraint uses the Type 1 Definition (as of now) - all branch nodes on the same level use the same feature as splitting criteria. This is an extension of P. Shati, E. Cohen, and S. McIlraith, “Optimal decision trees for interpretable clustering with constraints,” Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, 2023. doi:10.24963/ijcai.2023/225 :

<img width="536" alt="Screenshot 2024-03-15 at 1 20 26 AM" src="https://github.com/HarisRasul12/ESC499-Thesis-SAT-Trees/assets/66268214/31bd4c51-3ee7-46af-85fe-35adb9ef528a">
<img width="293" alt="Screenshot 2024-03-15 at 1 21 01 AM" src="https://github.com/HarisRasul12/ESC499-Thesis-SAT-Trees/assets/66268214/bcac36b0-7a58-4c60-b964-f26981f77dde">

## Clustering Constraints
<strong>Please note that Clustering Constraints Must-Link Pairs and Cannot-Link Pairs are embedded within encodings for Clustering Objectives.</strong>

Supports user inputs for Pairwise constraints.

- Must-Link Pairs

<img width="399" alt="Screenshot 2024-03-23 at 3 16 47 AM" src="https://github.com/HarisRasul12/ESC499-Thesis-SAT-Trees/assets/66268214/f2f072e4-7595-49e0-893a-37c4633e9625">

- Cannot-Link Pairs

<img width="493" alt="Screenshot 2024-03-23 at 3 16 23 AM" src="https://github.com/HarisRasul12/ESC499-Thesis-SAT-Trees/assets/66268214/ee558fac-0445-47da-a897-0da92c8c40be">

# Datasets:
Datasets can be accessed via this link: https://archive.ics.uci.edu

# External Solver Support: LOANDRA
This library supports External SAT solver for the problems aside from PySATs in built solvers. Our libarary calls the Loandra solver for DIMACS file types: 

Berg, J., Demirović, E. and Stuckey, P.J., 2019. Core-boosted linear search for incomplete MaxSAT. In Integration of Constraint Programming, Artificial Intelligence, and Operations Research: 16th International Conference, CPAIOR 2019, Thessaloniki, Greece, June 4–7, 2019, Proceedings 16 (pp. 39-56). Springer International Publishing.

User must set its solver path in compliance to its solving method.

https://github.com/jezberg/loandra

# Contributors and Acknowledgements
Contributor: Haris Rasul, AI Engineering - UofT

I would like to thank Professor Eldan Cohen(Department of Mechanical and Industrial Engineering, University of Toronto), and Pouya Shati(Vector Institute, Toronto, Canada) for their mentorship in this Thesis for their constant guidance and supervision. This project is an accumulation of all our work to create a powerful and useful library for practioners to use on classification and clustering tasks using unique SAT encodings for exact optimal solutions.

