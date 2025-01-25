# SATreeCraft - An Exact Optimization algorithm Library for Producing interpretable Decision Trees

# Version 1.1 - April 2024

## Table of Contents
- [1. Critical Files](#1-critical-files)
- [2. Jupyter Notebooks Demo](#2-jupyter-notebooks-demo)
- [3. DIMACS Files](#3-dimacs-files)
- [4. Classification Problems](#4-classification-problems)
- [5. Images](#5-images)
- [6. Loandra Support](#6-loandra-support)

## 1. Critical Files
- `SATreeCraft.py`: Main module containing the `TreeConstructor` class for tree data structure manipulation.
- `SATreeClassifier.py`: Extends the tree solution object from `SATreeCraft`, including predictor models.
- `utils.py`: Provides data loader methods, integration with `sklearn` metrics, and a metrics overlay for tree models.

## 2. Jupyter Notebooks Demo
- `Datareader.ipynb`: Performs classificationtests on all datasets, focused solely on classification problems.
- `ExternalSolverSupport.ipynb`: Demonstrates how to integrate with the LOANDRA solver for advanced problem-solving.
- `ComprehensiveTest.ipynb`: A comprehensive demo including data loading, model solution, and statistical analysis with clustering and classification
- `Demo.ipynb`: A high level demo including data loading, model solution, classofocation and clustering using PySAT and LOANDRA solvers

## 3. DIMACS Files
This directory contains `.dimac` files, which allow for the exporting of solutions to integrate with LOANDRA's solver system.

## 4. Classification Problems
- This directory holds all code modules related to classification problems, aimed at maximizing accuracy and minimizing tree height.
- It supports datasets with both categorical and numerical features.

## 4.5 Clustering Problems
- This dircetory holds the modules for the encodings for clustering problems: Minimize Maximum Diameter , Bicriteria Optimization as of now Release ver--1.1

## 5. Images
- Contains visualizations of decision trees, which can be generated using either PySAT or Loandra based on the user's input.

### Classification

<img width="530" alt="Screenshot 2024-04-28 at 1 16 28 PM" src="https://github.com/HarisRasul12/ESC499-Thesis-SAT-Trees/assets/66268214/32893e77-00ff-494e-abf7-240f6d689eef">

### Clustering

<img width="572" alt="Screenshot 2024-04-28 at 1 16 58 PM" src="https://github.com/HarisRasul12/ESC499-Thesis-SAT-Trees/assets/66268214/afa73a76-0db7-4b22-a4bc-a4fb9983755d">

## 6. Loandra Support
- Includes subprocess code for interfacing with a locally installed version of Loandra via Python's `os` module and the command line interface.

## Getting Started
Please ensure PySAT is installed (LOANDRA is optional)

## Usage

<img width="1075" alt="Screenshot 2024-04-28 at 1 13 31 PM" src="https://github.com/HarisRasul12/ESC499-Thesis-SAT-Trees/assets/66268214/e583381b-a152-4bb3-8f9b-0afa077b56a8">

## Contributing
Open Source Library for Classification and Reegression Trees

## License
Under property of The Faculty of Engineering, University of Toronto

## Contact
email: rasul.haris12@gmail.com
