
#%%

from pyomo.environ import *
from pyomo.dae import *
import numpy as np
import time
import matplotlib.pyplot as plt

#%%
def print_model_variables(model):
    print("=== Model Variables ===")
    for v in model.component_objects(Var, active=True):
        print(f"{v.name}: {v}")
    print("========================\n")

def print_model_constraints(model):
    print("=== Model Constraints ===")
    for c in model.component_objects(Constraint, active=True):
        print(f"{c.name}: {c}")
    print("========================\n")

def check_degrees_of_freedom(model):
    """
    This function calculates and prints the degrees of freedom of the given Pyomo model.
    Degrees of freedom are calculated as the difference between the number of variables and the number of constraints.
    """
    num_vars = sum(1 for v in model.component_data_objects(Var, active=True))
    num_constraints = sum(1 for c in model.component_data_objects(Constraint, active=True))
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
    Nz=2 # No discretization specified yet, so it assumes 2 from the set

    # only c, q are counted as Nt * Nz, none of their derivatives are counted yet
    num_vars = Nt * Nz #TODO: Put the correct number of variables

    # Number of Constraints
    num_constraints = Nt * Nz + 2 * Nt + (Nz - 1)  #TODO: Put the correct number of constraints

    # Degrees of Freedom
    dof = num_vars - num_constraints

    # Print results
    print("=== Predicted Degrees of Freedom Before Discretization ===")
    print(f"Number of Variables  : {num_vars}")
    print(f"Number of Equations  : {num_constraints}")
    print(f"Degrees of Freedom   : {dof}")
    print("==============================================\n")
    
    return dof

def predict_dof_after(Nt, Nz):
    """
    This function calculates and prints the degrees of freedom after discretization.
    """
    Nt = Nt
    Nz = Nz
    # Number of Variables
    num_vars = 8 * Nt * Nz + 7

    # Number of Constraints
    num_constraints = 8 * Nt * Nz - 3 * Nt + 2

    # Degrees of Freedom
    dof = num_vars - num_constraints

    # Print results
    print("=== Predicted Degrees of Freedom After Discretization ===")
    print(f"Number of Variables  : {num_vars}")
    print(f"Number of Equations  : {num_constraints}")
    print(f"Degrees of Freedom   : {dof}")
    print("=============================================\n")
    

def get_degrees_of_freedom(model):
    """
    This function calculates and prints the degrees of freedom of the given Pyomo model.
    Degrees of freedom are calculated as the difference between the number of variables and the number of constraints.
    """
    num_vars = sum(1 for v in model.component_data_objects(Var, active=True))
    num_constraints = sum(1 for c in model.component_data_objects(Constraint, active=True))
    degrees_of_freedom = num_vars - num_constraints
    
    return num_vars, num_constraints, degrees_of_freedom

def get_predicted_dof_after(Nt, Nz):
    """
    This function calculates and prints the degrees of freedom after discretization.
    """
    num_vars = 8 * Nt * Nz + 7 
    num_vars = num_vars -2 # assuming known k_eff and K_d

    num_constraints = 8*Nt*Nz - 3*Nt + 4
    dof = num_vars - num_constraints
    
    return num_vars, num_constraints, dof


def compare_dof_after(model, Nt, Nz):
    """
    This function compares the actual and predicted degrees of freedom,
    number of variables, and number of constraints, and prints a table with the results.
    """
    # Get actual values
    actual_vars, actual_constraints, actual_dof = get_degrees_of_freedom(model)
    
    # Get predicted values
    predicted_vars, predicted_constraints, predicted_dof = get_predicted_dof_after(Nt, Nz)
    
    # Calculate differences
    diff_vars = actual_vars - predicted_vars
    diff_constraints = actual_constraints - predicted_constraints
    diff_dof = actual_dof - predicted_dof
    
    # Print the comparison table
    print("=== Comparison of Predicted vs Actual Degrees of Freedom ===")
    print(f"{' ':<25} {'Predicted':<15} {'Actual':<15} {'Difference':<15}")
    print(f"{'-'*70}")
    print(f"{'Number of Variables':<25} {predicted_vars:<15} {actual_vars:<15} {diff_vars:<15}")
    print(f"{'Number of Constraints':<25} {predicted_constraints:<15} {actual_constraints:<15} {diff_constraints:<15}")
    print(f"{'Degrees of Freedom':<25} {predicted_dof:<15} {actual_dof:<15} {diff_dof:<15}")
    print("===========================================================\n")


# NOTATION
# - Variables: v_{name}
# - Parameters: p_{name}
# - Equations: e_{name}
# - Initial conditions: ic_{name}
# - Boundary conditions: bc_{name}

#%%
# ------------------------------
# OPTIONS AND INPUTS
# ------------------------------

# Parameters to be estimated
p_k_eff = 0.1  # effective mass transfer coefficient (m/s)
p_K_d = 1e-4  # adsorption isotherm constant

# Fixed Parameters
p_epsilon_b = 0.4  # bed voidage
p_r_p = 40e-6  # particle radius (m)
p_L = 0.10  # column length (m)
p_q_max = 5.5  # saturation capacity (mol/m^3 particle)
p_rho_l = 1000  # liquid density
p_eta = 0.001  # dynamic viscosity
p_D_eff = 1e-9  # effective diffusivity (m^2/s)

# Conversion functions
mm_diameter_to_m2_area = lambda d: np.pi * (d / 2)**2 / 1e6
ml_per_min_to_m3_per_s = lambda ml: ml / 1e6 / 60
min_to_s = lambda t: t * 60
mg_mAb_per_mL_to_moles_per_m3 = lambda c, MW_mAb: c * 1e3 / MW_mAb

# Converted Parameters
A_col = mm_diameter_to_m2_area(8)  # column cross-sectional area (m^2)
Q_in = ml_per_min_to_m3_per_s(2.0)  # inlet volumetric flowrate (m^3/s)
MW_mAb = 145E3  # molecular weight of mAb (g/mol)
C_in = mg_mAb_per_mL_to_moles_per_m3(10, MW_mAb)  # inlet concentration (mol/m^3)
duration = min_to_s(40)  # duration of the simulation (s)
nu = Q_in / (A_col * (1 - p_epsilon_b))

# Time and space discretization
N_t = 16  # number of time points
N_z = 21  # number of spatial points in z

tiny_value = 1e-10

#%%

# TODO: 
# - Remove superfluous variables
# - Check units
# - Consider using tiny values for initial conditions


# Create a Pyomo model
model = ConcreteModel()

model.t = ContinuousSet(bounds=(0, duration))
model.z = ContinuousSet(bounds=(0, p_L))

# Parameters in model
model.p_k_eff = Param(initialize=p_k_eff, within=PositiveReals, doc='Effective mass transfer coefficient [m/s]')
model.p_K_d = Param(initialize=p_K_d, within=PositiveReals, doc='Adsorption isotherm constant')
model.p_epsilon_b = Param(initialize=p_epsilon_b, within=PercentFraction, doc='Bed voidage')
model.p_r_p = Param(initialize=p_r_p, within=PositiveReals, doc='Particle radius [m]')
model.p_L = Param(initialize=p_L, within=PositiveReals, doc='Column length [m]')
model.p_q_max = Param(initialize=p_q_max, within=PositiveReals, doc='Saturation capacity [mol/m^3 particle]')
model.p_rho_l = Param(initialize=p_rho_l, within=PositiveReals, doc='Liquid density [kg/m^3]')
model.p_eta = Param(initialize=p_eta, within=PositiveReals, doc='Dynamic viscosity [kg/m/s]')
model.p_A_col = Param(initialize=A_col, within=PositiveReals, doc='Column cross-sectional area [m^2]')
model.p_Q_in = Param(initialize=Q_in, within=PositiveReals, doc='Inlet volumetric flowrate [m^3/s]')
model.p_C_in = Param(initialize=C_in, within=NonNegativeReals, doc='Inlet concentration [mol/m^3]')

# Variables
model.v_C = Var(model.z, model.t, within=NonNegativeReals, initialize=0, doc='Concentration of mAb in the column [mol/m^3]')
model.v_q = Var(model.z, model.t, within=NonNegativeReals, initialize=0, doc='Concentration of mAb in the particle [mol/m^3]')
model.v_dCdz = DerivativeVar(model.v_C, wrt=model.z)
model.v_dCdt = DerivativeVar(model.v_C, wrt=model.t)
model.v_dqdt = DerivativeVar(model.v_q, wrt=model.t)
model.v_dCdz2 = DerivativeVar(model.v_C, wrt=(model.z, model.z))

model.v_a_v = Var(within=PositiveReals, initialize=(3 * (1 - p_epsilon_b) / p_r_p), doc='Specific interfacial area [m^2/m^3]')
model.v_nu = Var(within=PositiveReals, initialize=nu , doc='Interstitial flowrate velocity [m/s]')
model.v_Q = Var(within=PositiveReals, initialize=Q_in, doc='Volumetric flowrate [m^3/s]')
model.v_q_star = Var(model.z, model.t, within=NonNegativeReals, initialize=0, doc='Equilibrium concentration of mAb in the particle [mol/m^3]')
model.v_R_abs = Var(model.z, model.t, initialize=1e-6, within=Reals, doc='Adsorption rate [mol/m^2/s]')
model.v_Re = Var(within=PositiveReals, initialize=(2 * p_r_p * nu * p_rho_l / p_eta), doc='Reynolds number computed from particle radius and interstitial velocity [dimensionless]')
model.v_D_ax = Var(within=NonNegativeReals, initialize=(2 * p_r_p * nu * p_epsilon_b / (0.339 + 0.033 * model.v_Re**0.48)), doc='Axial dispersion coefficient [m^2/s]')

# ------------------------------
# EQUATIONS
# ------------------------------

#TODO: 
# - Verify the equations are correct
# - Above all, check skipping conditions

# Interstitial liquid material balance
@model.Constraint(model.z, model.t)
def e_interstitial_liquid_material_balance(m, z, t):
    if z == 0 or z == m.p_L:
        return Constraint.Skip
    return m.p_epsilon_b * m.v_dCdt[z, t] == m.v_D_ax * m.v_dCdz2[z, t] - m.v_nu * m.v_dCdz[z, t] - m.v_R_abs[z, t] * m.v_a_v

print("=== Interstitial liquid material balance ===")
print_model_constraints(model)
#print_constraints(model, 'e_interstitial_liquid_material_balance')

# Specific interfacial area
@model.Constraint
def e_specific_interfacial_area(m):
    return m.v_a_v == 3 * (1 - m.p_epsilon_b) / m.p_r_p

print("=== Specific interfacial area ===")
print_model_constraints(model)

# Axial dispersion
@model.Constraint
def e_axial_dispersion(m):
    return m.v_D_ax == 2 * m.p_r_p * m.v_nu * m.p_epsilon_b / (0.339 + 0.033 * m.v_Re**0.48)

print("=== Axial dispersion ===")
print_model_constraints(model)

# Reynolds number
@model.Constraint
def e_reynolds_number(m):
    return m.v_Re == 2 * m.p_r_p * m.v_nu * m.p_rho_l / m.p_eta

print("=== Reynolds number ===")
print_model_constraints(model)

# Particle phase material balance
@model.Constraint(model.z, model.t)
def e_particle_phase_material_balance(m, z, t):
    return (1 - m.p_epsilon_b) * m.v_dqdt[z, t] == m.v_R_abs[z, t] * m.v_a_v

# Adsorption rate
@model.Constraint(model.z, model.t)
def e_adsorption_rate(m, z, t):
    return m.v_R_abs[z, t] == m.p_k_eff * (m.v_q_star[z, t] - m.v_q[z, t])

# Adsorption isotherm
@model.Constraint(model.z, model.t)
def e_adsorption_isotherm(m, z, t):
    return m.v_q_star[z, t] == m.p_q_max * m.v_C[z, t] / (m.p_K_d + m.v_C[z, t])

# Flowrate velocity relation
@model.Constraint
def e_flowrate_velocity_relation(m):
    return m.v_nu == m.v_Q / (m.p_A_col * (1 - m.p_epsilon_b))

# Flowrate inlet control
@model.Constraint
def e_flowrate_inlet_control(m):
    return m.p_Q_in == m.v_Q

# Column inlet boundary condition
@model.Constraint(model.t)
def bc_column_inlet(m, t):
    return m.v_nu * m.p_C_in == m.v_nu * m.v_C[0, t] - m.v_D_ax * m.v_dCdz[0, t]

# Column outlet boundary condition
@model.Constraint(model.t)
def bc_column_outlet(m, t):
    return m.v_dCdz[m.p_L, t] == 0

# Initial Conditions
@model.Constraint(model.z)
def ic_C(m, z):
    if z == 0:
        return Constraint.Skip
    return m.v_C[z, 0] == 0

@model.Constraint(model.z)
def ic_q(m, z):
    return m.v_q[z, 0] == 0

print("=== Initial Conditions ===")
print_model_constraints(model)

# Discretize the model using finite differences
TransformationFactory('dae.finite_difference').apply_to(model, nfe=N_t-1, wrt=model.t)
TransformationFactory('dae.finite_difference').apply_to(model, nfe=N_z-1, wrt=model.z)

compare_dof_after(model, N_t, N_z)

# %%

# Scale model for better performance
print_model_variables(model)
scaled_model = TransformationFactory('core.scale_model').create_using(model)

print_model_variables(scaled_model)

start = time.time()
SolverFactory('ipopt').solve(scaled_model).write()
print('Elapsed time:', time.time()-start)

#TransformationFactory('core.scale_model').propagate_solution(scaled_model, model)

# %%

def contour_plot(m, scaled=False):
    t = sorted(m.t)
    z = sorted(m.z)

    zgrid, tgrid = np.meshgrid(z, t)
    cgrid = np.zeros(zgrid.shape)
    qgrid = np.zeros(zgrid.shape)

    for i, z_val in enumerate(z):
        for j, t_val in enumerate(t):
            if scaled:
                cgrid[j, i] = m.scaled_v_C[z_val, t_val].value
                qgrid[j, i] = m.scaled_v_q[z_val, t_val].value
            else:
                cgrid[j, i] = m.v_C[z_val, t_val].value
                qgrid[j, i] = m.v_q[z_val, t_val].value

    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True, sharey=True)

    cax1 = axes[0].contourf(tgrid, zgrid, cgrid, cmap='viridis')
    axes[0].set_title('Concentration of mAb in the column (v_C)')
    axes[0].set_ylabel('Distance z')
    fig.colorbar(cax1, ax=axes[0], label='Concentration C')

    cax2 = axes[1].contourf(tgrid, zgrid, qgrid, cmap='plasma')
    axes[1].set_title('Concentration of mAb in the particle (v_q)')
    axes[1].set_xlabel('Time t')
    axes[1].set_ylabel('Distance z')
    fig.colorbar(cax2, ax=axes[1], label='Concentration q')

    plt.tight_layout()
    plt.show()

# visualization
contour_plot(scaled_model, scaled=True)

# %%

def smooth_plot(m):
    t = sorted(m.t)
    z = sorted(m.z)

    zgrid, tgrid = np.meshgrid(z, t)
    cgrid = np.zeros(zgrid.shape)
    qgrid = np.zeros(zgrid.shape)

    for i, z_val in enumerate(z):
        for j, t_val in enumerate(t):
            cgrid[j, i] = m.v_C[z_val, t_val].value
            qgrid[j, i] = m.v_q[z_val, t_val].value

    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True, sharey=True)

    im1 = axes[0].imshow(cgrid, aspect='auto', origin='lower', extent=[t[0], t[-1], z[0], z[-1]], cmap='viridis')
    axes[0].set_title('Concentration of mAb in the column (v_C)')
    axes[0].set_ylabel('Distance z')
    fig.colorbar(im1, ax=axes[0], label='Concentration C')

    im2 = axes[1].imshow(qgrid, aspect='auto', origin='lower', extent=[t[0], t[-1], z[0], z[-1]], cmap='plasma')
    axes[1].set_title('Concentration of mAb in the particle (v_q)')
    axes[1].set_xlabel('Time t')
    axes[1].set_ylabel('Distance z')
    fig.colorbar(im2, ax=axes[1], label='Concentration q')

    plt.tight_layout()
    plt.show()

# visualization
smooth_plot(scaled_model)

# %%
