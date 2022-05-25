import numpy as np
import matplotlib.pyplot as plt

"""This code takes preprocessed experimental 4D-STEM data, and simplifies and organizes it for input into a structural 
simulation of polymer chains."""

def map_to_spherical_harmonics(integrated_intensity, verbose=False):
    """Maps intensity data as a function of angle onto the n=2 spherical harmonics.  Assumes all data is in-plane."""
    
    # Unpack input data
    n_images, n_angles = integrated_intensity.shape
    n_grid = int(np.round(n_images**0.5))
    angle_step = int(np.round(180 / n_angles))
    
    # Normalize input data
    norm_intensity_matrix = (integrated_intensity / np.mean(integrated_intensity)).reshape(n_images, n_angles)

    # Map onto spherical harmonic basis set by multiplying by the basis functions and integrating
    angle_values = np.arange(-90, 90, angle_step)
    rad_values = angle_values * np.pi / 180
    sin_values = np.sin(rad_values)
    cos_values = np.cos(rad_values)

    mn2_factors = sin_values * cos_values
    m_2_factors = sin_values**2 - cos_values**2

    mn2 = np.matmul(norm_intensity_matrix, mn2_factors) * np.sqrt(15/4/np.pi) / n_angles
    mn1 = np.zeros(n_grid**2)
    m_0 = np.ones(n_grid**2) * -1 * np.sqrt(5/16/np.pi)
    m_2 = np.matmul(norm_intensity_matrix, m_2_factors)* np.sqrt(15/16/np.pi) / n_angles
    m_1 = np.zeros(n_grid**2)
    
    # Organize Results
    harmonics = np.stack((mn2, mn1, m_0, m_1, m_2))
    
    if verbose:
        plt.imshow(mn2.reshape(n_grid, n_grid))
        plt.title('m = -2')
        plt.show()
        plt.imshow(m_2.reshape(n_grid, n_grid))
        plt.title('m = 2')
        plt.show()
        
    return harmonics