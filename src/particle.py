import numpy as np

class Particle:
    """
    Class representing a particle in the simulation.

    Attributes
    ----------
    position : np.ndarray
        The particle's position in 2D space.
    velocity : np.ndarray
        The particle's velocity in 2D space.
    mass : float
        The particle's mass.
    radius : float
        The radius of the particle.
    """

    def __init__(self, position, velocity, mass=1.0, radius=1.0):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.acceleration = np.array([0, -10])  # Gravity in the negative y-direction
        self.mass = mass
        self.radius = radius

    def update_position(self, dt):
        """
        Update the particle's position and velocity due to gravity.

        Parameters
        ----------
        dt : float
            Time step for position update.
        """
        # Update position
        self.position += self.velocity * dt + (self.acceleration * dt * dt) / 2
        # Update velocity
        self.velocity += self.acceleration * dt

    def apply_force(self, force, dt):
        """
        Update the particle's velocity based on the applied force.

        Parameters
        ----------
        force : np.ndarray
            The force applied to the particle.
        dt : float
            Time step for velocity update.
        """
        acceleration = force / self.mass
        self.velocity += acceleration * dt
