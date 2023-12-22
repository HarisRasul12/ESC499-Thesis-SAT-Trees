from pysat.solvers import Solver,Glucose3
cnf = [[1, 2], [-1, -2], [-1, 3, -4], [-1, -3, 4], [-2, 3, -4], [-5, 3], [-7, 4], [-6, -3], [-8, -4], [5, -3], [6, 3], [7, -4], [8, 4], [-9, -10], [-11, -12], [-1, 3], [-1, -4], [-2, 3], [-2, -4], [-5, 9], [-7, 10], [-6, 11], [-8, 12]]
# Create a SAT solver instance
solver = Glucose3()

# Add the clauses to the solver
for clause in cnf:
    solver.add_clause(clause)

# Attempt to solve the problem
is_solvable = solver.solve()

# Retrieve the solution if one exists
solution = solver.get_model() if is_solvable else None

# Display the results
if is_solvable:
    print("A solution exists:", solution)
else:
    print("No solution exists")

# Don't forget to delete the solver to free up resources
solver.delete()