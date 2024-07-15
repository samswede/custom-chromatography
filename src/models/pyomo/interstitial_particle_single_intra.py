from pyomo.environ import *
from pyomo.dae import *
import numpy as np
import matplotlib.pyplot as plt

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
p_D_eff = 1e-9  # effective diffusivity (m^2/s)
p_K_d = 1e-4  # adsorption isotherm constant

# Fixed Parameters
p_epsilon_b = 0.4  # bed voidage
p_r_p = 40e-6  # particle radius (m)
p_L = 0.10  # column length (m)
p_q_max = 5.5  # saturation capacity (mol/m^3 particle)
p_rho_l = 1000  # liquid density
p_eta = 0.001  # dynamic viscosity

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
N_t = 10  # number of time points
N_z = 20  # number of spatial points in z
N_r = 5  # number of spatial points in r

#%%
# Create a Pyomo model
model = ConcreteModel()

model.t = ContinuousSet(bounds=(0, duration))
model.z = ContinuousSet(bounds=(0, p_L))
model.r = ContinuousSet(bounds=(0, p_r_p))

# Parameters in model
model.p_k_eff = Param(initialize=p_k_eff, within=PositiveReals, doc='Effective mass transfer coefficient [m/s]')
model.p_D_eff = Param(initialize=p_D_eff, within=PositiveReals, doc='Effective diffusivity [m^2/s]')
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
model.v_q = Var(model.z, model.r, model.t, within=NonNegativeReals, initialize=0, doc='Concentration of mAb in the particle [mol/m^3]')
model.v_dCdz = DerivativeVar(model.v_C, wrt=model.z)
model.v_dCdt = DerivativeVar(model.v_C, wrt=model.t)
model.v_dqdr = DerivativeVar(model.v_q, wrt=model.r)
model.v_dqdt = DerivativeVar(model.v_q, wrt=model.t)
model.v_dCdz2 = DerivativeVar(model.v_C, wrt=(model.z, model.z))
model.v_dqdr2 = DerivativeVar(model.v_q, wrt=(model.r, model.r))

model.v_a_v = Var(within=PositiveReals, initialize=(3 * (1 - p_epsilon_b) / p_r_p), doc='Specific interfacial area [m^2/m^3]')
model.v_nu = Var(within=PositiveReals, initialize=Q_in / (A_col * (1 - p_epsilon_b)) , doc='Interstitial flowrate velocity [m/s]')
model.v_Q = Var(within=PositiveReals, initialize=Q_in, doc='Volumetric flowrate [m^3/s]')
model.v_q_star = Var(model.z, model.r, model.t, within=NonNegativeReals, initialize=0, doc='Equilibrium concentration of mAb in the particle [mol/m^3]')
model.v_R_abs = Var(model.z, model.r, model.t, initialize=1e-6, within=Reals, doc='Adsorption rate [mol/m^2/s]')
model.v_Re = Var(within=PositiveReals, initialize=(2 * p_r_p * nu * p_rho_l / p_eta), doc='Reynolds number computed from particle radius and interstitial velocity [dimensionless]')
model.v_D_ax = Var(within=NonNegativeReals, initialize=(2 * p_r_p * nu * p_epsilon_b / (0.339 + 0.033 * model.v_Re**0.48)), doc='Axial dispersion coefficient [m^2/s]')

# ------------------------------
# EQUATIONS
# ------------------------------

# Interstitial liquid material balance
@model.Constraint(model.z, model.t)
def e_interstitial_liquid_material_balance(m, z, t):
    if z == 0 or z == m.p_L:
        return Constraint.Skip
    return m.p_epsilon_b * m.v_dCdt[z, t] == m.v_D_ax * m.v_dCdz2[z, t] - m.v_nu * m.v_dCdz[z, t] - m.v_R_abs[z, m.p_r_p, t] * m.v_a_v

# Column inlet boundary condition
@model.Constraint(model.t)
def bc_column_inlet(m, t):
    return m.v_nu * m.p_C_in == m.v_nu * m.v_C[0, t] - m.v_D_ax * m.v_dCdz[0, t]

# Column outlet boundary condition
@model.Constraint(model.t)
def bc_column_outlet(m, t):
    return m.v_dCdz[m.p_L, t] == 0

# Particle phase material balance
@model.Constraint(model.z, model.r, model.t)
def e_particle_phase_material_balance(m, z, r, t):
    if r == 0 or r == m.p_r_p:
        return Constraint.Skip
    return (1 - m.p_epsilon_b) * m.v_dqdt[z, r, t] == m.v_R_abs[z, r, t] * m.v_a_v

# Specific interfacial area
@model.Constraint
def e_specific_interfacial_area(m):
    return m.v_a_v == 3 * (1 - m.p_epsilon_b) / m.p_r_p

# Standard spherical diffusion
@model.Constraint(model.z, model.r, model.t)
def e_standard_spherical_diffusion(m, z, r, t):
    if r == 0 or r == m.p_r_p:
        return Constraint.Skip
    return m.v_dqdt[z, r, t] == m.p_D_eff * (m.v_dqdr2[z, r, t] + (2 / r) * m.v_dqdr[z, r, t])

# Particle surface boundary condition
@model.Constraint(model.z, model.t)
def bc_particle_surface(m, z, t):
    return m.p_D_eff * m.v_dqdr[z, m.p_r_p, t] == m.v_R_abs[z, m.p_r_p, t]

# Particle center boundary condition
@model.Constraint(model.z, model.t)
def bc_particle_center(m, z, t):
    return m.v_dqdr[z, 0, t] == 0 # Is this working correctly?

# Adsorption rate
@model.Constraint(model.z, model.r, model.t)
def e_adsorption_rate(m, z, r, t):
    return m.v_R_abs[z, r, t] == m.p_k_eff * (m.p_q_max * m.v_C[z, t] / (m.p_K_d + m.v_C[z, t]) - m.v_q[z, r, t])

# Axial dispersion
@model.Constraint
def e_axial_dispersion(m):
    return m.v_D_ax == 2 * m.p_r_p * m.v_nu * m.p_epsilon_b / (0.339 + 0.033 * m.v_Re**0.48)

# Reynolds number
@model.Constraint
def e_reynolds_number(m):
    return m.v_Re == 2 * m.p_r_p * m.v_nu * m.p_rho_l / m.p_eta

# Flowrate velocity relation
@model.Constraint
def e_flowrate_velocity_relation(m):
    return m.v_nu == m.v_Q / (m.p_A_col * (1 - m.p_epsilon_b))

# Flowrate inlet control
@model.Constraint
def e_flowrate_inlet_control(m):
    return m.p_Q_in == m.v_Q

# Initial Conditions
@model.Constraint(model.z)
def ic_C(m, z):
    if z == 0: # Trying this out...
        return Constraint.Skip
    return m.v_C[z, 0] == 0 # Is this working correctly?

@model.Constraint(model.z, model.r)
def ic_q(m, z, r):
    if r == 0 or r == m.p_r_p or z == 0: # Trying this out...
        return Constraint.Skip
    
    return m.v_q[z, r, 0] == 0 # Is this working correctly?


# Discretize the model using finite differences
#TransformationFactory('dae.collocation').apply_to(model, nfe=N_t, wrt=model.t)
TransformationFactory('dae.finite_difference').apply_to(model, nfe=N_t, wrt=model.t)
TransformationFactory('dae.finite_difference').apply_to(model, nfe=N_z, wrt=model.z)
TransformationFactory('dae.finite_difference').apply_to(model, nfe=N_r, wrt=model.r)

#%%
# Inspect the model before running
def check_degrees_of_freedom(model):
    """
    This function calculates and prints the degrees of freedom of the given Pyomo model.
    Degrees of freedom are calculated as the difference between the number of variables and the number of constraints.
    """
    num_vars = sum(1 for v in model.component_data_objects(Var, active=True))
    num_constraints = sum(1 for c in model.component_data_objects(Constraint, active=True))
    degrees_of_freedom = num_vars - num_constraints
    
    print(f"Number of Variables: {num_vars}")
    print(f"Number of Constraints: {num_constraints}")
    print(f"Degrees of Freedom: {degrees_of_freedom}")

    return degrees_of_freedom

def my_expected_degrees_of_freedom(model, N_z, N_r, N_t):
    """
    This function calculates and returns the expected degrees of freedom of the given Pyomo model.
    The expected degrees of freedom are calculated as the number of differential equations in the model.
    """
    expected_eqs = 2*N_z * N_r + 2*N_z +8
    expected_vars = 2*N_z * N_r +2*N_z + 5

    print(f"MY Expected Number of Equations: {expected_eqs}")
    print(f"MY Expected Number of Variables: {expected_vars}")
    pass

def calculate_expected_dof(N_z, N_r, N_t):
    """
    Calculates and prints the expected degrees of freedom for the model based on
    discretization parameters N_z, N_r, and N_t.
    """
    # Calculate the total number of variables
    num_vars = 4 * N_z * N_r * N_t + 4 * N_z * N_t + 4
    
    # Calculate the total number of constraints
    num_constraints = 2*N_z*N_r*N_t + N_z*N_t + N_z + N_r*N_z +3
    
    # Calculate the degrees of freedom
    dof = num_vars - num_constraints
    
    # Print the results
    print(f"GPT Expected Number of Variables: {num_vars}")
    print(f"GPT Expected Number of Constraints: {num_constraints}")
    print(f"GPT Expected Degrees of Freedom: {dof}")
    
    return dof
    
check_degrees_of_freedom(model)
my_expected_degrees_of_freedom(model, N_z, N_r, N_t)
calculate_expected_dof(N_z, N_r, N_t)

def summarize_model_components(model):
    """
    This function prints a summary of the components in the given Pyomo model, including parameters, variables, and constraints.
    """
    num_params = len(list(model.component_data_objects(Param, active=True)))
    num_vars = len(list(model.component_data_objects(Var, active=True)))
    num_constraints = len(list(model.component_data_objects(Constraint, active=True)))
    
    print(f"Number of Parameters: {num_params}")
    print(f"Number of Variables: {num_vars}")
    print(f"Number of Constraints: {num_constraints}")

#summarize_model_components(model)

def check_uninitialized_variables(model):
    """
    This function checks for uninitialized variables in the given Pyomo model.
    """
    uninitialized_vars = [v for v in model.component_data_objects(Var, active=True) if v.value is None]
    if uninitialized_vars:
        print("Uninitialized Variables:")
        for v in uninitialized_vars:
            print(f"Variable: {v}")
    else:
        print("All variables are initialized.")

#check_uninitialized_variables(model)



#%%

# Solve the model
solver = SolverFactory('ipopt')
results = solver.solve(model, tee=True)

# Modular Plotting Functions
def plot_outlet_concentration(model, duration, N_t):
    t_values = np.linspace(0, duration, N_t)
    C_outlet = [model.v_C[model.p_L, t].value for t in model.t]
    plt.figure()
    plt.plot(t_values, C_outlet, label='Outlet Concentration')
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration at Outlet (mol/m^3)')
    plt.legend()
    plt.title('Outlet Concentration Over Time')
    plt.show()

def plot_heatmap_concentration(model, L, duration, N_z, N_t):
    z_values = np.linspace(0, L, N_z)
    t_values = np.linspace(0, duration, N_t)
    C_values = np.array([[model.v_C[z, t].value for t in model.t] for z in model.z])
    T, Z = np.meshgrid(t_values, z_values)
    
    plt.figure()
    plt.contourf(T, Z, C_values, levels=50, cmap='viridis')
    plt.colorbar(label='Concentration (mol/m^3)')
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.title('Column Concentration Heatmap')
    plt.show()

# Example usage
plot_outlet_concentration(model, duration, N_t)
plot_heatmap_concentration(model, model.p_L, duration, N_z, N_t)
