#%%
import pyomo.environ as pyo
import pyomo.dae as dae
import numpy as np
import matplotlib.pyplot as plt
import time

#%%
# Inspect the model before running
def check_degrees_of_freedom(model):
    """
    This function calculates and prints the degrees of freedom of the given Pyomo model.
    Degrees of freedom are calculated as the difference between the number of variables and the number of constraints.
    """
    num_vars = sum(1 for v in model.component_data_objects(pyo.Var, active=True))
    num_constraints = sum(1 for c in model.component_data_objects(pyo.Constraint, active=True))
    degrees_of_freedom = num_vars - num_constraints
    
    print("=== Actual Degrees of Freedom ===")
    print(f"Number of Variables: {num_vars}")
    print(f"Number of Constraints: {num_constraints}")
    print(f"Degrees of Freedom: {degrees_of_freedom}")
    print("=================================\n")


    return degrees_of_freedom

def print_constraints(model, constraint_name, num_to_print=5):
    """
    This function prints the first few instances of the specified constraint.
    """
    constraint = getattr(model, constraint_name)
    print(f"Constraint: {constraint_name}")
    for i, c in enumerate(constraint.items()):
        if i >= num_to_print:
            break
        print(f"{c[0]}: {c[1].expr}")

def predict_dof_before():
    """
    This function calculates and prints the degrees of freedom before discretization.
    """
    Nt=2 # No discretization specified yet, so it assumes 2 from the set
    Nx=2 # No discretization specified yet, so it assumes 2 from the set

    # only c is counted as Nt * Nx
    num_vars = Nt * Nx

    # Number of Constraints
    num_constraints = Nt * Nx + 2 * Nt + (Nx - 1)  # PDE, bc1, bc2, ic

    # Degrees of Freedom
    dof = num_vars - num_constraints

    # Print results
    print("=== Predicted Degrees of Freedom Before Discretization ===")
    print(f"Number of Variables  : {num_vars}")
    print(f"Number of Equations  : {num_constraints}")
    print(f"Degrees of Freedom   : {dof}")
    print("==============================================\n")
    
    return dof

def predict_dof_after(Nt, Nx):
    """
    This function calculates and prints the degrees of freedom after discretization.
    """
    Nt = Nt
    Nx = Nx
    # Number of Variables
    num_vars = 4 * Nt * Nx  # c, dcdt, dcdx, d2cdx2

    # Number of Constraints
    num_constraints = Nt * Nx + Nt * (Nx - 1) + Nt * (Nx - 2) + (Nt - 1) * Nx + 2 * Nt + (Nx - 1)  # PDE, bc1, bc2, ic, dcdx_disc_eq, d2cdx2_disc_eq, dcdt_disc_eq

    # Degrees of Freedom
    dof = num_vars - num_constraints

    # Print results
    print("=== Predicted Degrees of Freedom After Discretization ===")
    print(f"Number of Variables  : {num_vars}")
    print(f"Number of Equations  : {num_constraints}")
    print(f"Degrees of Freedom   : {dof}")
    print("=============================================\n")
    
    return dof

# A function to inspect the current objective function
def inspect_objective(model):
    """
    This function inspects and prints the details of the objective function in the given Pyomo model.
    """
    objectives = list(model.component_objects(pyo.Objective, active=True))
    
    if not objectives:
        print("No objective function is defined in the model.")
    else:
        for obj in objectives:
            print(f"Objective: {obj.name}")
            print(f"Expression: {obj.expr}")
            print(f"Sense: {'Minimize' if obj.sense == pyo.minimize else 'Maximize'}\n")


#%%

# discretization
Nt = 50
Nx = 100

# parameters
tf = 80
D = 2.68
L = 1.0
KL = 20000.0
Cs = 0.0025
qm = 1.0

m = pyo.ConcreteModel()

m.t = dae.ContinuousSet(bounds=(0, tf))
m.x = dae.ContinuousSet(bounds=(0, L))
m.r = dae.ContinuousSet(bounds=(0, 1))

m.c = pyo.Var(m.t, m.x)
m.dcdt = dae.DerivativeVar(m.c, wrt=m.t)
m.dcdx = dae.DerivativeVar(m.c, wrt=m.x)
m.d2cdx2 = dae.DerivativeVar(m.c, wrt=(m.x, m.x))

@m.Constraint(m.t, m.x)
def pde(m, t, x):
    return m.dcdt[t, x] * (1 + qm*KL/(1 + KL*m.c[t, x])** 2) == D * m.d2cdx2[t, x]

@m.Constraint(m.t)
def bc1(m, t):
    return m.c[t, 0] == Cs

@m.Constraint(m.t)
def bc2(m, t):
    return m.dcdx[t, L] == 0

@m.Constraint(m.x)
def ic(m, x):
    '''Here we skip the constraint for x=0 because it is already defined in the bc1 constraint'''
    if x == 0:
        return pyo.Constraint.Skip
    return m.c[0, x] == 0.0

predict_dof_before()
check_degrees_of_freedom(m)
m.pprint()
#%%

# transform and solve
pyo.TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.x, nfe=Nx-1)
pyo.TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t, nfe=Nt-1)

#inspect_objective(m)
predict_dof_after(Nt, Nx)
check_degrees_of_freedom(m)
m.pprint()
# Inspect added constraints
print_constraints(m, 'dcdx_disc_eq')
print_constraints(m, 'd2cdx2_disc_eq')
print_constraints(m, 'dcdt_disc_eq')
#%%
# start timer
start = time.time()
pyo.SolverFactory('ipopt').solve(m).write()
print('Elapsed time:', time.time()-start)

#%%
#inspect_objective(m)
check_degrees_of_freedom(m)
#m.pprint()

def model_plot(m):
    t = sorted(m.t)
    x = sorted(m.x)

    xgrid = np.zeros((len(t), len(x)))
    tgrid = np.zeros((len(t), len(x)))
    cgrid = np.zeros((len(t), len(x)))

    for i in range(0, len(t)):
        for j in range(0, len(x)):
            xgrid[i,j] = x[j]
            tgrid[i,j] = t[i]
            cgrid[i,j] = m.c[t[i], x[j]].value

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('Distance x')
    ax.set_ylabel('Time t')
    ax.set_zlabel('Concentration C')
    p = ax.plot_wireframe(xgrid, tgrid, cgrid)

# visualization
model_plot(m)
# %%
