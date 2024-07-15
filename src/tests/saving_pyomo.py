import numpy as np # must be 1.26.4, not 2.0.0
from pyomo.environ import ConcreteModel, Var, Objective


# Create a simple Pyomo model
model = ConcreteModel()
model.x = Var()
model.obj = Objective(expr=model.x**2)

# Write the model to an NL file
model.write('model.nl', format='nl')

# Write the model to an LP file
model.write('model.lp', format='lp')

# Write the model to an MPS file
model.write('model.mps', format='mps')

# Write the model to a BAR file
model.write('model.bar', format='bar')

# Save the model to a pickle file
model.write('model.pkl', format='pkl')

# Apparently, GAMS works too but I can't figure it out...
# model.write('model.gms', format='gms')
