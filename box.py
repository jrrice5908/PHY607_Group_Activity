import numpy as np
from .particle import Particle
from .utils import rejection_sampling, inverse_cdf_sampling, velocity_pdf, sample_velocity_with_rejection


class Box:
    """
    Class representing the 3D box in which particles are simulated.

    Attributes
    ----------
    size : float
        The size of the cubic box.
    particles : list
        A list of Particle objects in the box.
    k_B : float
        Boltzmann constant approximation for simplicity.
    """

    def __init__(self, size=10.0):
        self.size = size
        self.particles = []
        self.volume = size ** 3  # Volume of the box
        self.k_B = 1  # Approximation of Boltzmann constant for simplicity

        # Lists to store measurements over time
        self.pressure_over_time = []
        self.temperature_over_time = []
        self.energy_over_time = []

    def add_particle(self, particle):
        """
        Add a particle to the simulation box.

        Parameters
        ----------
        particle : Particle
            The particle to add to the box.
        """
        self.particles.append(particle)

    def initialize_velocities(self, temperature=None, mode="temperature"):
        """
        Initialize particle velocities based on the chosen mode.

        Parameters
        ----------
        temperature : float or None
            The initial temperature for velocity distribution (used if mode is "temperature").
        mode : str
            The mode for velocity initialization. Options are "temperature" (Maxwell-Boltzmann), "uniform", or "rejection_sampling".
        """
        if mode == "temperature" and temperature is not None:
            # Maxwell-Boltzmann distribution for velocity magnitude
            for particle in self.particles:
                velocity_magnitude = np.sqrt(3 * self.k_B * temperature / particle.mass)
                particle.velocity = np.random.normal(0, velocity_magnitude, size=3)
            #print(f"Initialized velocities using Maxwell-Boltzmann distribution with temperature: {temperature}")

        elif mode == "uniform":
            # Random uniform velocity distribution
            for particle in self.particles:
                particle.velocity = np.random.uniform(-10, 10, size=3)
            #print(f"Initialized velocities using uniform distribution")

        elif mode == "rejection_sampling":
            # Sample velocities for each particle using rejection sampling
            for particle in self.particles:
                # Most probable speed (v_mp) for the Maxwell-Boltzmann distribution
                v_mp = np.sqrt(2 * self.k_B * temperature / particle.mass)  # Most probable speed
                #print(f"Most probable speed v_mp: {v_mp}")

                # Set xmax and ymax based on v_mp and the velocity PDF at v_mp
                xmax = 3 * v_mp  # Choose a range around 3 times the most probable speed
                ymax = velocity_pdf(v_mp, temperature, particle.mass)  # Peak value of the PDF
                #print(f"xmax: {xmax}, ymax: {ymax}")

                # Perform rejection sampling to sample a velocity magnitude
                sampled_velocity = sample_velocity_with_rejection(temperature, particle.mass, xmax, ymax, velocity_pdf)

                # Assign the sampled velocity magnitude in a random direction
                direction = np.random.randn(3)  # Random direction
                direction /= np.linalg.norm(direction)  # Normalize to unit vector
                particle.velocity = sampled_velocity * direction
                #print(f"Sampled velocity for particle: {particle.velocity}")

    def handle_wall_collisions(self, particle):
        """
        Handle collisions with the walls of the box.

        If a particle hits a wall, its velocity in the corresponding direction is reversed.
        Adjust the particle's position if it exceeds the boundary (taking its radius into account).

        Parameters
        ----------
        particle : Particle
            The particle to check for wall collisions.
        """
        for i in range(3):  # Check x, y, z directions
            if particle.position[i] - particle.radius <= 0:
                # Particle has hit the wall in the negative direction
                particle.position[i] = particle.radius  # Keep particle inside the box
                particle.velocity[i] *= -1  # Reverse velocity direction
            elif particle.position[i] + particle.radius >= self.size:
                # Particle has hit the wall in the positive direction
                particle.position[i] = self.size - particle.radius  # Keep particle inside the box
                particle.velocity[i] *= -1  # Reverse velocity direction

    def handle_particle_collisions(self, particle1, particle2):
        """
        Handle elastic collisions between two particles.

        Parameters
        ----------
        particle1 : Particle
            The first particle.
        particle2 : Particle
            The second particle.
        """
        distance = np.linalg.norm(particle1.position - particle2.position)

        # Check if the distance between particles is less than or equal to the sum of their radii
        if distance <= particle1.radius + particle2.radius:
            # Elastic collision between two particles
            v1, v2 = particle1.velocity, particle2.velocity
            m1, m2 = particle1.mass, particle2.mass
            pos_diff = particle1.position - particle2.position
            vel_diff = v1 - v2

            # Update velocities (elastic collision formula)
            particle1.velocity = v1 - (2 * m2 / (m1 + m2)) * (
                np.dot(vel_diff, pos_diff) / np.dot(pos_diff, pos_diff)) * pos_diff
            particle2.velocity = v2 - (2 * m1 / (m1 + m2)) * (
                np.dot(-vel_diff, -pos_diff) / np.dot(pos_diff, pos_diff)) * -pos_diff

    def calculate_temperature(self):
        """
        Calculate the temperature of the system based on the kinetic energy of the particles.

        Returns
        -------
        float : Temperature of the system.
        """
        total_kinetic_energy = sum(0.5 * p.mass * np.linalg.norm(p.velocity) ** 2 for p in self.particles)
        avg_kinetic_energy_per_particle = total_kinetic_energy / len(self.particles)

        # T = (2/3) * (avg kinetic energy) / k_B
        return (2 / 3) * avg_kinetic_energy_per_particle / self.k_B

    def calculate_kinetic_energy(self):
        """
        Calculate the total kinetic energy of the particles in the system.

        Returns
        -------
        float : Total kinetic energy of the system.
        """
        return sum(0.5 * p.mass * np.linalg.norm(p.velocity) ** 2 for p in self.particles)

    def calculate_pressure(self):
        """
        Calculate the pressure of the system based on the kinetic energy and particle density.

        Returns
        -------
        float : Pressure of the system.
        """
        N = len(self.particles)
        total_kinetic_energy = self.calculate_kinetic_energy()
        avg_kinetic_energy_per_particle = total_kinetic_energy / N

        # P = (2/3) * (N/V) * (average kinetic energy)
        return (2 / 3) * (N / self.volume) * avg_kinetic_energy_per_particle

    def step(self, dt):
        """
        Perform one time step of the simulation.

        Parameters
        ----------
        dt : float
            The time step of the simulation.
        """
        for i, particle in enumerate(self.particles):
            particle.update_position(dt)
            self.handle_wall_collisions(particle)

        # Check for particle-particle collisions
        for i in range(len(self.particles)):
            for j in range(i + 1, len(self.particles)):
                self.handle_particle_collisions(self.particles[i], self.particles[j])

        # Measure pressure, temperature, and energy
        pressure = self.calculate_pressure()
        energy = self.calculate_kinetic_energy()
        temperature = self.calculate_temperature()

        # Store measurements over time
        self.pressure_over_time.append(pressure)
        self.energy_over_time.append(energy)
        self.temperature_over_time.append(temperature)
