import matplotlib.pyplot as plt
import numpy as np

def plot_outlet_concentration(model, duration, N_t):
    t_values = np.linspace(0, duration, N_t)
    C_outlet = [model.C[0.3, t].value for t in model.t]  # Assuming L = 0.3 as the outlet
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
    C_values = np.array([[model.C[z, t].value for t in model.t] for z in model.z])
    T, Z = np.meshgrid(t_values, z_values)
    
    plt.figure()
    plt.contourf(T, Z, C_values, levels=50, cmap='viridis')
    plt.colorbar(label='Concentration (mol/m^3)')
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.title('Column Concentration Heatmap')
    plt.show()
