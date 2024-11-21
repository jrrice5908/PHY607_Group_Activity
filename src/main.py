import sys
import os
import matplotlib.pyplot as plt
import numpy as np

from box import Box  # Import from box.py
from particle import Particle  # Import from particle.py


vel_data = np.loadtxt('particle_velocities.txt', skiprows=1)
cow_data = vel_data[:,1]

cow_vels = []

for i in range(len(cow_data)):
	if cow_data[i] == 0:
		cow_vels.append(i)

drag_force_per_step = []

def force_of_drag(n_particles, box_size, drag_coeff, cow_area, cow_vels):
	density = n_particles / box_size
	for i in cow_vels:
		force = (1/2) * density * (i**2) * drag_coeff * cow_area
		drag_force_per_step.append(force)
	return drag_force_per_step

forces = force_of_drag(500, 100, 5, 5, cow_vels)

#print(force_of_drag(500, 100, 5, 5, cow_vels))

plt.plot(cow_vels, forces)
plt.show()


