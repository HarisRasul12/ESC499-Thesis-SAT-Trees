# Created by: Haris Rasul
# Date: November 19th 2023
# Python script to build the complete tree and create literals
# for a given depth and dataset. Will attempt to maximize the number of correct labels given to training dataset 
# adding soft clauses for maximizing corrcet solution for a given depth and hard clauses 

from pysat.formula import CNF
from pysat.solvers import Solver
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from graphviz import Digraph
import numpy as np
from classification_problems.min_height_tree_module import *
from pysat.card import CardEnc, IDPool, EncType

def min_support(wcnf, literals, X, TL, min_support):
    # Initialize the variable pool with the highest index plus one to avoid conflicts
    max_var_index = max(literals.values()) + 1
    vpool = IDPool(start_from=max_var_index)

    # Add the minimum support constraints for each leaf node
    for t in TL:
        # Collect all 'z' literals for the current leaf node
        z_literals = [literals[f'z_{i}_{t}'] for i in range(len(X))]

        # Encode the constraint that at least 'min_support' of these literals must be True
        min_support_clauses = CardEnc.atleast(lits=z_literals, bound=min_support, vpool=vpool, encoding=EncType.seqcounter)

        # Add the clauses for the minimum support constraint to the WCNF
        for clause in min_support_clauses.clauses:
            wcnf.append(clause)

        # Update the variable pool for the next available variable index
        max_var_index = vpool.id()
        vpool = IDPool(start_from=max_var_index)

    return wcnf