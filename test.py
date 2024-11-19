import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FFMpegWriter

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import from Monte_sim
from Monte_sim.box import Box  # Import from box.py
from Monte_sim.particle import Particle  # Import from particle.py

# Set random seed for reproducibility
np.random.seed(42)

# Function to run the simulation and measure physical quantities at each time step
def run_simulation_and_measure(n_steps, dt, box_size, n_particles, initial_temp, particle_radius, mode="temperature", plot=True, video_output=True):
    all_velocities = []  # To track all velocities over time
    avg_velocities = []  # To track average velocities per time step
    obs_file = open("observables.txt", "w")  # Open file to record observables at every step
    obs_file.write("TimeStep\tTemperature(K)\tPressure(Pa)\tEnergy(J)\tAverage Velocity\n")  # Header

    velocity_file = open("particle_velocities.txt", "w")  # Open file to record particle velocities
    velocity_file.write("TimeStep\tParticle\tVelocity_X\tVelocity_Y\tVelocity_Z\n")  # Header

    # Initialize the box without predefined temperature
    box = Box(box_size)
    particles = [Particle(np.random.rand(3) * box_size, np.random.randn(3), radius=particle_radius) for _ in range(n_particles)]
    for particle in particles:
        box.add_particle(particle)

    # Initialize velocities based on the chosen mode (temperature, uniform, rejection_sampling, inverse_cdf)
    box.initialize_velocities(temperature=initial_temp, mode=mode)

    if video_output:
        # Setup for saving the video
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([0, box_size])
        ax.set_ylim([0, box_size])
        ax.set_zlim([0, box_size])

        writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        video_filename = f"particle_motion_temp_{int(initial_temp)}.mp4"
        with writer.saving(fig, video_filename, 100):
            # Run the simulation for a certain number of steps
            for step in range(n_steps):
                box.step(dt)  # Perform the simulation step

                # Collect velocities and write to file
                velocities = [np.linalg.norm(p.velocity) for p in box.particles]
                all_velocities.extend(velocities)
                avg_velocities.append(np.mean(velocities))  # Calculate and store average velocity at each step

                # Measure and write observables for each time step
                pressure = box.pressure_over_time[-1]
                energy = box.energy_over_time[-1]
                temperature = box.temperature_over_time[-1]
                avg_velocity = avg_velocities[-1]

                # Write observables to file for each time step
                obs_file.write(f"{step}\t{temperature:.6f}\t{pressure:.6f}\t{energy:.6f}\t{avg_velocity:.6f}\n")

                # Write all particle velocities to the velocity file
                for i, particle in enumerate(box.particles):
                    vx, vy, vz = particle.velocity
                    velocity_file.write(f"{step}\t{i}\t{vx:.6f}\t{vy:.6f}\t{vz:.6f}\n")

                # Update the 3D plot
                x_data = [p.position[0] for p in box.particles]
                y_data = [p.position[1] for p in box.particles]
                z_data = [p.position[2] for p in box.particles]

                ax.clear()
                ax.set_xlim([0, box_size])
                ax.set_ylim([0, box_size])
                ax.set_zlim([0, box_size])
                ax.scatter(x_data, y_data, z_data, c='blue', s=100)
                writer.grab_frame()  # Save each frame to the video

    obs_file.close()  # Close the observables file
    velocity_file.close()  # Close the velocity file
    return all_velocities, avg_velocities

# Function to plot velocity distribution
def plot_velocity_distribution(all_velocities):
    plt.figure()
    plt.hist(all_velocities, bins=30, density=True, alpha=0.7, label="Velocity Distribution")
    plt.xlabel("Velocity (units)")
    plt.ylabel("Probability Density")
    plt.title("Velocity Distribution of Particles")
    plt.legend()
    plt.show()

# Function to save average velocities to a text file
def save_average_velocities_to_file(avg_velocities, filename="average_velocities.txt"):
    with open(filename, "w") as f:
        f.write("Step\tAverage Velocity\n")
        for step, avg_velocity in enumerate(avg_velocities):
            f.write(f"{step}\t{avg_velocity:.6f}\n")
    print(f"Average velocities saved to {filename}")

if __name__ == "__main__":
    # Define parameters for the simulation
    N_PARTICLES = 300  # Number of particles
    BOX_SIZE = 1.0  # Size of the simulation box
    N_STEPS = 200  # Number of simulation steps
    DT = 0.001  # Time step size
    PARTICLE_RADIUS = 0.1  # Set particle radius

    # Initial temperature to use
    initial_temperature = 100  # Set to 100K

    # Choose mode: "temperature" for Maxwell-Boltzmann velocities, or "uniform", "rejection_sampling", or "inverse_cdf"
    velocity_mode = "uniform"

    # Run the simulation for the initial temperature and track velocities
    all_velocities, avg_velocities = run_simulation_and_measure(
        N_STEPS, DT, BOX_SIZE, N_PARTICLES, initial_temperature, PARTICLE_RADIUS, mode=velocity_mode, plot=True, video_output=True)

    # Plot the velocity distribution at the end of the simulation
    plot_velocity_distribution(all_velocities)

    # Save the average velocities to a text file
    save_average_velocities_to_file(avg_velocities, filename="average_velocities.txt")
