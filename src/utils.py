import numpy as np


def sample_velocity_with_rejection(temperature, mass, xmax, ymax, velocity_pdf, k_B=1):
    """
    Sample a velocity magnitude using rejection sampling.

    Parameters
    ----------
    temperature : float
        The temperature in Kelvin.
    mass : float
        The mass of the particle.
    xmax : float
        Maximum value for the velocity sampling range.
    ymax : float
        Maximum value for the PDF (used for rejection sampling).
    velocity_pdf : callable
        The probability density function for the velocity.
    k_B : float, optional
        Boltzmann constant (default is 1 for simplified units).

    Returns
    -------
    float : Sampled velocity magnitude.
    """
    while True:
        # Sample a candidate velocity from a uniform distribution
        velocity = np.random.uniform(0, xmax)

        # Sample a uniform y-value between 0 and ymax
        y = np.random.uniform(0, ymax)

        # Calculate the probability density for the candidate velocity
        pdf_value = velocity_pdf(velocity, temperature, mass, k_B)

        # Accept the candidate if the y-value is under the PDF curve
        if y < pdf_value:
            return velocity  # Return the accepted velocity magnitude

def velocity_pdf(v, temperature, mass, k_B=1):
    """
    Maxwell-Boltzmann velocity probability density function in 3D.

    Parameters
    ----------
    v : float
        The velocity magnitude.
    temperature : float
        The temperature in Kelvin.
    mass : float
        The mass of the particle.
    k_B : float
        Boltzmann constant (can be adjusted for unit consistency).

    Returns
    -------
    float
        The probability density for the given velocity magnitude.
    """
    factor = mass / (2 * k_B * temperature)
    return v**2 * np.exp(-factor * v**2)  # Maxwell-Boltzmann distribution for velocity magnitude


def rejection_sampling(pdf, xmin, xmax, ymax, size=1):
    """
    Perform rejection sampling to generate samples from a probability distribution.

    Parameters
    ----------
    pdf : callable
        The probability distribution function from which to sample.
    xmin : float
        The minimum x-value for the sampling range.
    xmax : float
        The maximum x-value for the sampling range.
    ymax : float
        The maximum y-value (height) of the probability distribution.
    size : int
        The number of samples to generate.

    Returns
    -------
    samples : np.ndarray
        An array of generated samples.
    """
    samples = []
    while len(samples) < size:
        x = np.random.uniform(xmin, xmax)  # Sample from the uniform distribution over x-range
        y = np.random.uniform(0, ymax)  # Sample a uniform y-value between 0 and ymax
        if y < pdf(x):  # Accept if under the PDF curve
            samples.append(x)
    return np.array(samples)

def inverse_cdf_sampling(cdf_inv, size=1):
    """
    Perform inverse CDF sampling to generate samples from a probability distribution.

    Parameters
    ----------
    cdf_inv : callable
        The inverse of the cumulative distribution function (CDF) to sample from.
    size : int
        The number of samples to generate.

    Returns
    -------
    samples : np.ndarray
        An array of generated samples.
    """
    u = np.random.uniform(0, 1, size)  # Generate random values between 0 and 1
    return cdf_inv(u)  # Map those values using the inverse CDF



