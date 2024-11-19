import numpy as np

class Particle:
    """
    Class representing a particle in the simulation.

    Attributes
    ----------
    position : np.ndarray
        The particle's position in 3D space.
    velocity : np.ndarray
        The particle's velocity in 3D space.
    mass : float
        The particle's mass.
    radius : float
        The radius of the particle.
    """

    def __init__(self, position, velocity, mass=1.0, radius=1.0):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.mass = mass
        self.radius = radius

    def update_position(self, dt):
        """
        Update the particle's position based on its velocity.

        Parameters
        ----------
        dt : float
            Time step for position update.
        """
        self.position += self.velocity * dt

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
