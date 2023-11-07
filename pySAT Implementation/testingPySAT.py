from pysat.formula import CNF
from pysat.solvers import Glucose3

formula = CNF()
formula.append([1, 2])  # x1 OR x2
formula.append([-1, 3]) # NOT x1 OR x3

with Glucose3(bootstrap_with=formula) as solver:
    result = solver.solve()
    print("Is the formula satisfiable?", result)
    
    if result:
        model = solver.get_model()
        print("Model:", model)