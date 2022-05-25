import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import scipy.stats as stats
import scipy.optimize as opt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


#######################################################################################################################
# User-Facing Functions
#######################################################################################################################


def distance_angle_autocorrelation(matrix, max_grid_distance, grid_distance_step, n_slices=10):
    """This function compares every pair of points in the intensity matrix in order to determine their aggregate
    relationships with each other as a function of distance.  To start, picture that we have an n x n array, where
    n is the number of images and each axis represents image number.  We fill in this array with the distance between
    each pair of images.  Then, for each pair of images, we calculate the intensity product
    II = (I(theta) * I(theta - d_theta), integrated over all theta, for each shift d_theta.  This gives the angle
    correlation between these two points.  Since all of this cannot be done at once for memory reason, the original
    (n x n) array is broken into slices.  Once slice is treated at a time, so the array we deal with is actually
    (m x n), where n is the number of images and m is the slice size.  The results will be aggregated later.  )
    
    Inputs:
        matrix: 2D matrix of intensity as a function of (image, angle).
        max_grid_distance: the function will compare every pair of points separated by this distance or less.  Index units.
        grid_distance_step: bin size along the x-axis.  Index units.
        
    Outputs:
        Autocorrelation: Array of conditional probabilities.  Suppose two units of diffracting material are drawn at random.
            What will the angle between them be, given their distance apart?  Each column is normalized to one and
            therefore represents a probability distribution.  
    """

    print('Setting Up...')

    # Get dimensions from input arguments
    if len(matrix.shape) == 2:
        n_images, n_angles = matrix.shape
        n_grid = int(np.round(n_images ** 0.5))
    else:
        n_grid, _, n_angles = matrix.shape
        n_images = n_grid**2
        matrix = matrix.reshape(n_images, n_angles)
    image_numbers = np.arange(n_images)
    rows, cols = np.divmod(image_numbers, n_grid)
    n_shift = int(np.floor(n_angles / 2) + 1)  # Number of delta theta values to be considered.
    distance_values = np.arange(1, max_grid_distance, grid_distance_step)
    angle_indices = np.arange(n_angles)

    # Set up output matrices
    intensity_count = np.zeros((len(distance_values), n_shift))
    intensity_sum = np.zeros((len(distance_values), n_shift))

    # Create slices. These reduce the impact of the calculation on RAM by breaking it up into smaller chunks,
    # while staying highly parallel.
    slice_size = int(n_images / n_slices)
    slices = [np.arange(i * slice_size, (i + 1) * slice_size, 1) for i in range(n_slices)]

    progress_bar = ProgressBar(n_slices + 1)

    for s, slice in enumerate(slices):
        print('Computing data slice number %d...' % s)

        # Calculate the distance between each pair of points. Points from the current slice are compared to all points in the data set.
        dy = np.subtract.outer(rows, rows[slice], dtype=float)
        dx = np.subtract.outer(cols, cols[slice], dtype=float)
        dr = np.sqrt(dx ** 2 + dy ** 2)

        # Use masks to group points by distance apart.
        d_masks = []
        for i, d in enumerate(distance_values):
            r_min = distance_values[i - 1] if i != 0 else 0
            r_max = distance_values[i]
            d_masks.append(np.where((dr >= r_min) & (dr < r_max)))

        # Compute correlations between each pair of points as a function of angle offset, which is here called "shift"
        correlations = np.zeros((n_images, slice_size, n_shift))
        for shift in range(n_shift):
            backward_shift = np.mod(angle_indices - shift,
                                    n_angles)  # Gives the angle indices when shifted by a certain number of angles
            forward_shift = np.mod(angle_indices + shift, n_angles)
            shifted_slice = (matrix[slice, :][:, forward_shift] + matrix[slice, :][:,
                                                                  backward_shift]) / 2  # Due to linearity, we can treat the forward-shifted and backward-shifted values simultaneously
            product = matrix @ np.transpose(
                shifted_slice)  # This product is an (n x m) array, where n is the number of images and m is the slice size.  Each node gives the correlation between one pair
            # of images at the current angle offset.
            correlations[:, :, shift] = product[:, :]  # The results are recorded as a function of (image1, image2, angle_shift)

        # Bin Results
        for d, mask in enumerate(d_masks):
            for a in range(n_shift):
                intensity_count[d, a] += len(mask[0])  # Number of points that fall in this distance range
                correlation_subset = correlations[:, :, a][mask]  # All correlation values at this angle and distance
                intensity_sum[d, a] += np.sum(correlation_subset)  # Total intensity for this distance category.  If this causes overflow, rescale your pi matrix beforehand

        progress_bar.update('+1')

    print('Wrapping up...')

    # Find correlations for each (distance, angle) pair.  For distances that weren't present, place a zero for all values.
    autocorrelation = np.divide(intensity_sum, intensity_count,
                                out=np.zeros_like(intensity_sum),
                                where=intensity_count != 0)  # Replace divide by 0 with 0
    print(autocorrelation.shape)
    autocorrelation = np.squeeze(autocorrelation)  # Squeeze removes the length-1 dimension that appears.

    print('done')
    return autocorrelation


def resultant_autocorrelation(intensity_matrix, step_size, distances, verbose=False):

    """This function compares every pair of points in the intensity matrix in order to determine their aggregate relationships with each other as a function of distance.
    To start, picture that we have an n x n array, where n is the number of images and each axis represents image number.  We fill in this array with the distance between each
    pair of images.

    This function uses liquid crystal theory, which requires only one director at each point.  If there are multiple peaks at a point, the resultant of the (angle, intensity)
    vectors will be used as the peak for that point.
    
    Inputs:
        intensity_matrix: 2d matrix of intensity at each (image_number, angle) node
        step_size: distance between images from the intensity matrix (in nm)
        distances: 1d array of bin edges (in nm).  The smallest value should be equal to or greater than the step size.  
    
    """

    print('Setting Up...')

    # Get dimensions from input arguments
    n_images, n_angles = intensity_matrix.shape
    angle_step = int(np.round(180 / n_angles))
    n_grid = int(np.round(n_images ** 0.5))
    image_numbers = np.arange(n_images)
    rows, cols = np.divmod(image_numbers, n_grid)

    # Clip Negative Values
    intensity_matrix = np.clip(intensity_matrix, 0, 10**8)

    # Get resultant angle at each point
    angle_values = np.arange(0, 180, angle_step)
    rad_values = angle_values * np.pi / 180
    sin_values = np.sin(2 * rad_values)  # Domain of zero to pi
    cos_values = np.cos(2 * rad_values)  # Domain of zero to pi
    sin_sum = np.sum(intensity_matrix*sin_values[None, :], axis=1)
    cos_sum = np.sum(intensity_matrix*cos_values[None, :], axis=1)

    resultant_angles = np.arctan2(sin_sum, cos_sum) / 2 # Radians.  Switched back to normal domain.

    # Get scalar order parameter at each point
    relative_angles = np.subtract.outer(resultant_angles, rad_values) % np.pi
    order_parameter_prefactors = 0.5 * (3 * np.cos(relative_angles)**2 - 1)
    intensity_sums = np.sum(intensity_matrix, axis=1)[:, None]
    norm_intensity_matrix = np.divide(intensity_matrix, intensity_sums,
                                      out=np.zeros_like(intensity_matrix),
                                      where=(intensity_sums != 0))
    scalar_order_parameters = np.sum(order_parameter_prefactors*norm_intensity_matrix, axis=1)

    # Show results so far
    if verbose:
        plt.imshow(resultant_angles.reshape(n_grid, n_grid))
        plt.title('Resultant Angles')
        plt.colorbar()
        plt.show()
        
        plt.imshow(scalar_order_parameters.reshape(n_grid, n_grid), vmin=-0.5, vmax=1.0)
        plt.title('Scalar Order Parameters')
        plt.colorbar()
        plt.show()
        

    # Calculate the distance between each pair of points. Points from the current slice are compared to all points in the data set.
    print('Computing Distances...')
    dy = np.subtract.outer(rows, rows, dtype=float)
    dx = np.subtract.outer(cols, cols, dtype=float)
    dr = np.sqrt(dx ** 2 + dy ** 2) * step_size  # nm

    # Calculate the cosine of the difference in resultant angle between each pair of points
    print('Computing Angle Differences...')
    d_theta_raw = np.subtract.outer(resultant_angles, resultant_angles, dtype=float) % np.pi
    d_theta = np.min((d_theta_raw, np.pi - d_theta_raw), axis=0)

    # Calculate the tensor covariance between each pair of points
    print('Computing Covariances...')
    combined_order_parameters = np.multiply.outer(scalar_order_parameters, scalar_order_parameters)
    tensor_covariance = combined_order_parameters * (2 * np.cos(d_theta)**2 - 1)
    
    # Set up output matrices
    n_distances = len(distances)
    count = np.zeros(n_distances)
    autocorrelation = np.zeros(n_distances)  # Effectively includes intensity information through the scalar order parameter
    stdev = np.zeros(n_distances)
    skew = np.zeros(n_distances)
    percentiles = [np.zeros(n_distances) for i in range(21)]

    # Use masks to group points by distance apart.
    d_masks = []
    for i, d in enumerate(distances):
        r_min = distances[i - 1] if i != 0 else 0
        r_max = distances[i]
        d_masks.append(np.where((dr >= r_min) & (dr < r_max)))

    # Bin Results
    print('Binning Results...')
    for d, mask in enumerate(d_masks):
        count[d] = len(mask[0])
        if count[d] > 0:
            autocorrelation[d] = np.mean(tensor_covariance[mask])
            stdev[d] = np.std(tensor_covariance[mask])
            skew[d] = stats.skew(tensor_covariance[mask])
            for i in range(21):
                percentiles[i][d] = np.percentile(tensor_covariance[mask], i*5)
        else:
            autocorrelation[d] = None
            stdev[d] = None
            skew[d] = None

    transmittance_values = np.zeros(n_angles)
    radians_values = angle_values * np.pi / 180
    for i, a in enumerate(radians_values):
        rotated_angles = np.mod(resultant_angles - a, np.pi)
        cos2_values = np.cos(rotated_angles)**2
        transmittance_values[i] = np.mean(cos2_values)
    dichoric_ratio = np.max(transmittance_values) / np.min(transmittance_values)
    print('Dichoric Ratio:', dichoric_ratio)

    return autocorrelation, stdev, skew, percentiles, scalar_order_parameters


def single_site_autocorrelation(integrated_intensity):
    """Multiplies diffraction intensity as a function of angle at each point by a version rotated by an angle theta,
    then plots the average result as a function of theta.  Useful for checking for preferred crossing angles
    in polymer samples with overlapping crystallites."""
    
    # Unpack input data
    n_images, n_angles = integrated_intensity.shape
    n_grid = int(np.round(n_images**0.5))
    angle_step = int(np.round(180 / n_angles))
    
    
    # Single-site intensity correlation
    center_matrix = (integrated_intensity - np.mean(integrated_intensity)) / np.mean(integrated_intensity)
    correlation_matrix = np.zeros(n_angles)
    for a in range(n_angles):
        correlation_matrix[a] = (np.mean(center_matrix * np.roll(center_matrix, a, axis=1)) + 
                                 np.mean(center_matrix * np.roll(center_matrix, -a, axis=1)))

    correlation_matrix = correlation_matrix / np.max(correlation_matrix)
    angle_values = np.linspace(0, 180, n_angles, endpoint=False)
    first_quadrant_mask = np.where(angle_values <= 90)
    plt.plot(angle_values[first_quadrant_mask], correlation_matrix[first_quadrant_mask], linewidth=0, marker='.')
    plt.xlabel('Angle Difference (Degrees)')
    plt.ylabel('Single-Location Intensity Correlation')
    plt.title('Single-Site')
    plt.xlim(0, 90)
    plt.axhline(y=0, c=(0.8, 0.8, 0.8))
    figure_image = plt.gcf()
    plt.show()
    
    return figure_image


#######################################################################################################################
# Plotting Functions
#######################################################################################################################

def plot_distance_angle_autocorrelation(autocorrelation, step_size, max_grid_distance, grid_distance_step, interpolate=False):
    """Plots the outputs of the function distance_angle_autocorrelation()."""
    n_distances, n_unique_angles = autocorrelation.shape
    if n_distances < max_grid_distance / grid_distance_step:
        max_grid_distance = n_distances * grid_distance_step
    max_distance = max_grid_distance * step_size

    if interpolate:
        autocorrelation = interpolate_zeros(autocorrelation)  # This produces a new array

    # Normalize Values
    final_distances = np.where(np.sum(autocorrelation, axis=1) != 0)[0]

    probability_distribution = np.zeros(autocorrelation.shape)
    probability_distribution[final_distances, :] = autocorrelation[final_distances, :] / (
    np.sum(autocorrelation[final_distances, :], axis=1)[:, None])
    versus_random = probability_distribution * n_unique_angles

    # Make custom color range
    top_range = 1
    bottom_range = np.max(versus_random) - 1
    resolution = 100
    top_indices = int(np.round(top_range * resolution))
    bottom_indices = int(np.round(bottom_range * resolution))

    top = cm.get_cmap('Blues_r', top_indices)
    bottom = cm.get_cmap('Reds', bottom_indices)

    newcolors = np.vstack((top(np.linspace(0, 1, top_indices)),
                           bottom(np.linspace(0, 1, bottom_indices))))
    newcmp = ListedColormap(newcolors, name='RedBlue')

    # Set range and aspect ratio
    aspect = max_distance / 100
    max_steps = max_grid_distance

    # Create plot
    plt.matshow(np.transpose(versus_random[:max_steps, :]), origin='lower',
                extent=(0, max_distance, 0, 90), aspect=aspect, cmap=newcmp, vmin=0)
    plt.ylabel('Angular Difference (degrees)')
    plt.xlabel('Separation Distance (nm)')
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Ratio to Random Distribution', verticalalignment='bottom', rotation=270)
    image = plt.gcf()

    return image

def interpolate_zeros(array):
    """Replaces zeros in an array by interpolating adjacent values in the row.  Used for heat plots."""
    n_rows, n_cols = array.shape
    inactive_rows = np.where(np.sum(array, axis=1) == 0)[0]
    active_rows = np.where(np.sum(array, axis=1) != 0)[0]
    new_array = np.array(array)

    for i, row in enumerate(inactive_rows):

        # Skip rows with insufficient information to interpolate
        if row < np.min(active_rows):
            # There are no active rows above. Skip this row.
            continue
        elif row > np.max(active_rows):
            # There are no active rows below. Skip this row.
            continue

        # Find the nearest active neighbors of the current row
        lower_indices = active_rows[np.where(active_rows < row)]
        lower_neighbor = np.max(lower_indices)

        upper_indices = active_rows[np.where(active_rows > row)]
        upper_neighbor = np.min(upper_indices)

        # Find the distance from each neighbor, then take a weighted average
        lower_distance = row - lower_neighbor
        upper_distance = upper_neighbor - row
        total = lower_distance + upper_distance
        lower_weight = 1 - lower_distance / total
        upper_weight = 1 - upper_distance / total

        new_array[row, :] = array[lower_neighbor, :] * lower_weight + array[upper_neighbor, :] * upper_weight

    return new_array


def plot_lc_covariance(lc_covariance, percentiles, distances):
    """Plots the outputs of the function resultant_autocorrelation()"""

    plt.scatter(distances, lc_covariance, s=1)

    # Percentiles are spaced by 5, so there are 21 total and the center is #10.
    center = 10
    spacing = 5  # Center 50% of population
    low = percentiles[center - spacing]
    high = percentiles[center + spacing]
    plt.fill_between(distances, low, high, alpha=0.1)
    

    plt.legend(['Mean', '50% of Population'])
    plt.axhline(y=0, c=(0.8, 0.8, 0.8))
    plt.xlabel('Distance (nm)')
    plt.ylabel('Covariance')
    plt.xlim(0, np.max(distances))
    plt.ylim(-0.5, 1)

    return plt.gcf()



#######################################################################################################################
# Utility Functions
#######################################################################################################################


class ProgressBar():
    def __init__(self, total_tasks, interval=5):
        self.total_tasks = total_tasks
        self.milestone = 0
        self.gap = interval - 1
        self.tic = time.time()
        self.current_task = 0
        self.finished = False
        print('0%')

    def update(self, current_task, verbose=False, clear_when_finished=False):
        if current_task == '+1':
            self.current_task += 1
        else:
            self.current_task = current_task
        progress = int(self.current_task / self.total_tasks * 100)
        if progress - self.gap > self.milestone or (verbose and progress > 0):
            self.milestone = progress

            # Estimate time remaining
            self.toc = time.time()
            average_rate = (self.toc - self.tic) / progress
            time_remaining = (100 - progress) * average_rate
            m, s = divmod(time_remaining, 60)
            time_string = "%d:%02d" % (m, s)
            print(progress, '%', ' ' * 10, time_string, ' remaining')
            if progress > 99:
                total_time = self.toc - self.tic
                m, s = divmod(total_time, 60)
                print('Total Time ', "%d:%02d" % (m, s))
        if self.current_task == self.total_tasks - 1:
            total_time = self.toc - self.tic
            m, s = divmod(total_time, 60)
            time_string = "%d:%02d" % (m, s)
            print('Finished in ', time_string)

    def get_total_time(self):
        return self.toc - self.tic
