import numpy as np
import matplotlib.pyplot as plt
import imageio
from pde import AllenCahnPDE, FileStorage, ScalarField, UnitGrid

# Initialize the model
state = ScalarField.random_uniform(UnitGrid([128, 128]), -0.01, 0.01)
eq = AllenCahnPDE()

# Initialize storage
file_write = FileStorage("allen_cahn.hdf")

# Store trajectory in storage
final_state = eq.solve(
    state,
    t_range=100,
    adaptive=True,
    tracker=[file_write.tracker(2)]
)

# Load the data from storage
storage = FileStorage("allen_cahn.hdf")

# Create frames and save them as images
frames = []
for i in range(len(storage)):
    fig, ax = plt.subplots()
    storage[i].plot(ax=ax, colorbar=True)
    plt.title(f"Time: {storage.times[i]:.2f}")
    filename = f"frame_{i:03d}.png"
    plt.savefig(filename)
    frames.append(imageio.imread(filename))
    plt.close(fig)

# Create a GIF from the frames
imageio.mimsave('allen_cahn_evolution.gif', frames, duration=0.2)
