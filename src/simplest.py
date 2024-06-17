import numpy as np
import matplotlib.pyplot as plt
from pde import CartesianGrid, FieldCollection, PDEBase, ScalarField, UnitGrid

# Parameters and constants
A_col = 1.0e-4  # m^2, Column cross-sectional area
C_in = lambda t: 1.0  # mol/m^3, Inlet concentration of component
K = 1.0e-3  # mol/m^3, Inverse adsorption equilibrium constant
k_eff = 1.0e-3  # 1/s, Overall mass transport coefficient
L = 0.3  # meters, Axial length of column
q_max = 5.47  # mol/m^3 particle, Maximum adsorption capacity
Q_in = lambda t: 1.0e-3  # m^3/s, Inlet volumetric flow rate
r_p = 45e-6  # meters, Radius of solid particles
eta = 1.0e-3  # Pa.s, Dynamic viscosity of the liquid
rho = 1000.0  # kg/m^3, Density of the liquid

# Calculated parameters
epsilon_b = 0.36  # Bed voidage
epsilon_p = 0.88 # Intraparticle voidage
epsilon_tot = epsilon_b + (1-epsilon_b)*epsilon_p  # Total voidage
Re = lambda nu: 2 * r_p * nu * rho / eta  # Reynolds number
nu = lambda Q: Q / (A_col * (1 - epsilon_b))  # Superficial velocity

# Derived equations
a_v_ads = 3 * (1 - epsilon_b) / r_p
D_ax = lambda nu: 2 * r_p * nu * epsilon_b / (0.339 + 0.033 * Re(nu) ** 0.48)

# PDE definition
class ChromatographyPDE(PDEBase):
    def __init__(self, grid, params):
        super().__init__()
        self.grid = grid
        self.params = params

    def evolution_rate(self, state, t):
        C, q = state  # Unpack the state variables
        nu_t = nu(Q_in(t))
        D_ax_t = D_ax(nu_t)
        
        R_ads = self.params['k_eff'] * (self.params['q_max'] * C / (self.params['K'] + C) - q)
        
        # Inlet boundary condition
        def inlet_bc(value, dx, x, t):
            return nu_t * C_in(t) - D_ax_t * (value - C_in(t)) / dx
        
        # Boundary conditions
        bc = [{"value": inlet_bc, "type": "value_expression"}, {"derivative": 0}]
        
        dC_dt = (D_ax_t * C.laplace(bc=bc)
                 - nu_t * C.gradient(bc=bc)
                 - R_ads * self.params['a_v_ads']) / self.params['epsilon_tot']
        dq_dt = R_ads * self.params['a_v_ads'] / (1 - self.params['epsilon_tot'])
        
        return FieldCollection([dC_dt, dq_dt])

# Grid setup
grid = UnitGrid([100])  # One-dimensional grid with 100 points
C_initial = ScalarField(grid, 0)
q_initial = ScalarField(grid, 0)
state = FieldCollection([C_initial, q_initial])

# Parameters dictionary
params = {
    'a_v_ads': a_v_ads,
    'epsilon_tot': epsilon_tot,
    'k_eff': k_eff,
    'q_max': q_max,
    'K': K
}

# Solve the PDE
pde = ChromatographyPDE(grid, params)
result = pde.solve(state, t_range=10, dt=1e-2)

# Plot the results
plt.figure()
result[0].plot(title="Concentration C(z,t)")
plt.figure()
result[1].plot(title="Solid Phase Concentration q(z,t)")
plt.show()

