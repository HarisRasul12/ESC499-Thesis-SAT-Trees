# 1. Critical Files

1. SATreeCraft.py - main moudle that stores tree constructor class
2. SATreeClassifierpy - contains predictor model extension from Tree solution object created by SATreeCraft
3. utils.py - dataloader methods, sklearn metrics overlay on top of tree models

# 2. Jupyter Notebooks Demo

1. Datareader.ipynb - classification tests for all datasets used in original sources (classifacation probelsm only)
2. ExternalSolverSupport.ipynb - LOANDRA solver support showcase demo 
3. ComprehensiveTest.ipynb - data loader, model solution solver, and precstion statitics demo showcase

# 3. dimacs

This folder contains the dimac file type that one can export their solution to produce LOANDRA solutions

# 4. classification_problems

- This folder contains the code modules of all cassication problems - max accuracy and min height problems 
- Supports both categorical and numerical feature datasets


# 5. images 

- This folder contains the tree visualization of the solution found by either PySAT or Loandra for a user's given problem


# 6. loandra_support 

- This folder contains the subprocess code to call one's local isntalled version of Lonadra via python OS and command line
