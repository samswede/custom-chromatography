
#%%

import numpy as np
from pde import PDEBase, FieldCollection, ScalarField, CartesianGrid, FileStorage, MemoryStorage, PlotTracker
import matplotlib.pyplot as plt
import imageio
import time

#%%
def constant_injection(t, C0, injection_duration):
    """
    Models a constant injection of solute.

    Parameters:
    - t: Time (s)
    - C0: Initial concentration of solute (mol/m^3)
    - injection_duration: Duration of injection (s)

    Returns:
    - Concentration of solute at time t (mol/m^3)
    """
    return C0 if t <= injection_duration else 0

def pulse_injection(t, C0, pulse_start, pulse_duration):
    """
    Models a pulse injection of solute.

    Parameters:
    - t: Time (s)
    - C0: Peak concentration of solute (mol/m^3)
    - pulse_start: Start time of the pulse (s)
    - pulse_duration: Duration of the pulse (s)

    Returns:
    - Concentration of solute at time t (mol/m^3)
    """
    return C0 if pulse_start <= t <= (pulse_start + pulse_duration) else 0


# Conversion functions
mm_diameter_to_m2_area = lambda d: np.pi * (d / 2)**2 / 1e6
ml_per_min_to_m3_per_s = lambda ml: ml / 1e6 / 60
min_to_s = lambda t: t * 60
mg_mAb_per_mL_to_moles_per_m3 = lambda c, MW_mAb: c * 1e3 / MW_mAb

# Physical parameters with unit conversion
epsilon_b = 0.4  # Bed porosity
epsilon_p = 0.3  # Particle porosity
r_p = 5e-4  # Particle radius in meters
column_diameter_mm = 10  # Column diameter in mm
column_length_cm = 10  # Column length in cm

A_col = mm_diameter_to_m2_area(column_diameter_mm)  # Cross-sectional area of the column in m^2
L_col = column_length_cm / 100  # Column length in meters

Q_in = lambda t: ml_per_min_to_m3_per_s(10)  # Inlet flow rate as a function of time in m^3/s
C_in = lambda t: constant_injection(t, mg_mAb_per_mL_to_moles_per_m3(2, 150000), injection_duration=0.2)  # Inlet concentration in mol/m^3 (MW of mAb is 150,000 g/mol)
k_eff = 1e-3  # Effective mass transfer coefficient in 1/s
q_max = mg_mAb_per_mL_to_moles_per_m3(1, 150000)  # Maximum adsorption capacity in mol/m^3
K = mg_mAb_per_mL_to_moles_per_m3(0.5, 150000)  # Adsorption equilibrium constant in mol/m^3

# Derived quantities
rho = 1000  # Fluid density in kg/m^3
eta = 1e-3  # Fluid viscosity in Pa.s
epsilon_tot = epsilon_b + (1 - epsilon_b) * epsilon_p  # Total voidage
Re = lambda nu: 2 * r_p * nu * rho / eta  # Reynolds number
nu = lambda Q: Q / (A_col * (1 - epsilon_b))  # Superficial velocity in m/s

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
grid_points = 200  # Number of grid points
grid = CartesianGrid([[0, L_col]], [grid_points])  # One-dimensional grid from 0 to L_col with grid_points points
C_initial = ScalarField(grid, 0)  # Initial concentration field
q_initial = ScalarField(grid, 0)  # Initial solid phase concentration field
state = FieldCollection([C_initial, q_initial])  # State containing both fields

# Parameters dictionary
params = {
    'a_v_ads': a_v_ads,
    'epsilon_tot': epsilon_tot,
    'k_eff': k_eff,
    'q_max': q_max,
    'K': K
}

# Solve the PDE over time range from 0 to 10 minutes
t_range = min_to_s(1)  # Convert time range to seconds
dt = 1e-3  # Time step in seconds
pde = ChromatographyPDE(grid, params)

# initialise storage
file_write = FileStorage("chromatography.hdf")

memory_storage = MemoryStorage()

start_time = time.time()

# Solve and store trajectory in storage
final_state = pde.solve(
        state, 
        t_range=t_range, 
        dt=dt, 
        adaptive=True,
        tracker=[
            #file_write.tracker(2), 
            memory_storage.tracker(1), 
            "progress"
            ]
        )

end_time = time.time()
cpu_time = end_time - start_time


# %%

# Load the data from storage
#storage = FileStorage("chromatography.hdf")
storage = memory_storage

#%%

def smooth_plot(storage, grid, title_c='Concentration of mAb in the column (v_C)', title_q='Concentration of mAb in the particle (v_q)'):
    # Get the number of time steps and spatial points from the storage and grid
    n_time_steps = len(storage)
    spatial_points = grid.axes_coords[0]
    time_points = storage.times

    cgrid = np.zeros((n_time_steps, len(spatial_points)))
    qgrid = np.zeros((n_time_steps, len(spatial_points)))

    for i in range(n_time_steps):
        cgrid[i, :] = storage[i][0].data  # Access the concentration field
        qgrid[i, :] = storage[i][1].data  # Access the solid phase concentration field

    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True, sharey=True)

    im1 = axes[0].imshow(cgrid, aspect='auto', origin='lower', extent=[time_points[0], time_points[-1], spatial_points[0], spatial_points[-1]], cmap='viridis')
    axes[0].set_title(title_c)
    axes[0].set_ylabel('Distance z')
    fig.colorbar(im1, ax=axes[0], label='Concentration C')

    im2 = axes[1].imshow(qgrid, aspect='auto', origin='lower', extent=[time_points[0], time_points[-1], spatial_points[0], spatial_points[-1]], cmap='plasma')
    axes[1].set_title(title_q)
    axes[1].set_xlabel('Time t')
    axes[1].set_ylabel('Distance z')
    fig.colorbar(im2, ax=axes[1], label='Concentration q')

    plt.tight_layout()
    plt.show()

smooth_plot(storage, grid)
# Create frames and save them as images
#%%

# Create frames and save them as images for Concentration
frames_concentration = []
for i in range(len(storage)):
    fig, ax = plt.subplots()
    storage[i][0].plot(ax=ax)  # Access the concentration field 'c'
    plt.title(f"Time: {storage.times[i]:.2f}")
    filename = f"frame_concentration_{i:03d}.png"
    plt.savefig(filename)
    frames_concentration.append(imageio.imread(filename))
    plt.close(fig)

# Create a GIF from the frames
imageio.mimsave('chromatography_concentration_evolution.gif', frames_concentration, duration=0.2)


# Create frames and save them as images for Solid Phase Concentration
frames_solid_phase = []
for i in range(len(storage)):
    fig, ax = plt.subplots()
    storage[i][1].plot(ax=ax)  # Access the solid phase concentration field 'q'
    plt.title(f"Time: {storage.times[i]:.2f}")
    filename = f"frame_solid_phase_{i:03d}.png"
    plt.savefig(filename)
    frames_solid_phase.append(imageio.imread(filename))
    plt.close(fig)

# Create a GIF from the frames
imageio.mimsave('chromatography_solid_phase_evolution.gif', frames_solid_phase, duration=0.2)

# %%
