import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import sys
import os
import time
import importlib as imp  # Only used in example workflow


#######################################################################################################################
# Overview
#######################################################################################################################
"""

Parameters:
    rotate: True if polmer chains are expected to be perpendicular to diffraction peaks, as is true for the pi-
        stacking peak.  False if polymer chains are expected to be parallel to the diffraction peaks, as is true
        for backbone peaks.  
    peaks_per_site: To reduce visual business, only the brightest peaks are drawn.  The program chooses an intensity
        threshold for which peaks should be drawn using a percentile that yields this number of peaks per location.
    peak_width: Minimum distance between two peaks.  This is used to determine whether a pair of local maxima
        are one peak or two.  In Degrees.
    bend_tolerance: Maximum amount that a flow line can bend between two grid squares.  For example, if the bend
        tolerance is 20 and the angle difference between two grid squares is 25, the line will be broken at that 
        point instead of connecting. In degrees.
    line_spacing: minimum distance between two lines for both lines to be drawn. This has no physical significance
        and is only for visualization purposes.  In grid units (i.e. 2 would be one line per 2 grid squares)
    seed_density: line seeds created per diffraction peak.  When only one line is seeded per peak, geometric 
        artifacts sometimes occur that make diagonal crystals look different than orthogonal ones.  Increasing 
        seed density helps remove these artifacts but takes more time.  2 is usually the best compromise.  
    curve_resolution: The length of line that is drawn before checking for a change in angle.  Higher values 
        create more accurate maps when high-curvature features are present.  Default is 2.  
    spacing_resolution: the resolution when determining the spacing between lines to draw.  Low values lead 
        to geometric artifacts that make horizontal crystals look different than vertical crystals.  Default 
        is 5.  
    angle_spacing_degrees: Lines can cross on the flow plot if the difference between their angles is larger
        than the angle spacing.
    max_overlap_fraction: Lines will be drawn if their overlap with other lines that have already been drawn
        is less than the max_overlap_fraction. High values can lead to line "bunching," while low values create 
        shorter, choppier flow lines.  Default is 0.5.
    preview_sparsity: plots of propagated lines, before trimming, take a lot of time to render.  preview_sparsity
        thins these plots by only drawing 1/preview_sparsity of the lines. Default is 20, which means that 1 in 20
        lines is drawn in the preview of propagated lines.  
    format_codes: list of which types of line plots to draw, which can be defaults or user-defined.  User-defined
        format codes can be any data type, including strings.
    brightness, contrast, alpha: these determine the mapping of a data category like intensity onto a
        display property like linewidth.  See the scale_values function for more details.
    
"""

#######################################################################################################################
# Functions specific to 4D-STEM data
#######################################################################################################################

def preview_intensity_vs_q(data, q_min, q_max, q_step, plot_vs_k=False):
    """Plots the mean intensity and average local median intensity as a function of diffraction vector q, as well as 
    the difference between them.  Because the noise profile is symmetric, 
    subtracting the average local median works well for finding peaks.  
    Optionally, the data can be plotted against reciprocal distance k instead of q.
    
    Input data matrix should be three-dimensional with axes (image, q, angle)."""
    
    mean_intensity = np.mean(data, axis=(0, 2))
    median_intensity = np.mean(np.median(data, axis=2), axis=0)
    q = np.linspace(q_min, q_max, int(np.round((q_max - q_min) / q_step)), endpoint=False)
    k = q * 10 / (2 * np.pi)  # q = 2*pi*k.  Also changes from inverse angstroms to inverse nanometers.
    
    plt.figure(figsize=(4, 4))
    if plot_vs_k:
        plt.plot(k, mean_intensity)
        plt.plot(k, median_intensity)
        plt.xlabel('Inverse Distance k($nm^{-1}$)')
        plt.xlim(q_min * 10 / (2 * np.pi), q_max * 10 / (2 * np.pi))
    else:
        plt.plot(q, mean_intensity)
        plt.plot(q, median_intensity)
        plt.xlabel('Diffraction Vector q ($\mathrm{\AA}^{-1}$)')
        plt.xlim(q_min, q_max)
    plt.legend(['Mean', 'Median'])
    plt.ylabel('Aggregate Intensity (arb. units)')
    plt.gca().set_box_aspect(1)
    aggregate_plot = plt.gcf()
    plt.show()

    data_subtracted = data - np.median(data, axis=2)[:, :, None]  # Same as mean - median
    difference = np.mean(data_subtracted, axis=(0, 2))
    
    plt.figure(figsize=(4, 4))
    if plot_vs_k:
        plt.plot(k, difference)
        plt.xlim(q_min * 10 / (2 * np.pi), q_max * 10 / (2 * np.pi))
        plt.xlabel('Diffraction Vector q ($\mathrm{\AA}^{-1}$)')
    else:
        plt.plot(q, difference)
        plt.xlim(q_min, q_max)
        plt.xlabel('Inverse Distance k($nm^{-1}$)')
    plt.ylim(0)
    plt.gca().set_box_aspect(1)
    plt.ylabel('Net Intensity (arb. units)')
    difference_plot = plt.gcf()
    plt.show()
    
    return aggregate_plot, difference_plot


def integrate_peaks(data, q_min, q_max, q_step, q_min_peak, q_max_peak):
    """Subtracts the local median and integrates intensity between two q-values (inclusive)."""  
    data_subtracted = data - np.median(data, axis=2)[:, :, None]  # Subtract local median
    q = np.linspace(q_min, q_max, int(np.round((q_max - q_min) / q_step)), endpoint=False)
    peak_mask = np.where((q >= q_min_peak) & (q <= q_max_peak))[0]
    peak_intensity = np.mean(data_subtracted[:, peak_mask, :], axis=1)  # Intensity as a function of image number and angle
    return peak_intensity


def format_flow_inputs(matrix, rotate=False):
    """Reshapes the matrix of diffraction intensities to I(row, column, angle).  Normalizes
    the data so that the RMS intensity is 1.  Optionally rotates all angles by 90 degrees, 
    which is useful for peaks such as the pi-stacking peak that lie perpendicular to the 
    polymer chains."""
    
    # Reshape 
    n_images, n_angles = matrix.shape
    n_rows = int(np.round(n_images**0.5))
    matrix = matrix.reshape((n_rows, n_rows, n_angles))
    
    # Normalize
    rms = np.mean(matrix**2)**0.5
    matrix = matrix / rms
    
    # If the diffraction peaks are perpendicular to the chain direction, rotate the matrix 90 degrees
    if rotate:
        n_angles = matrix.shape[2]
        matrix = np.roll(matrix, int(n_angles/2), axis=2)
        
    return matrix


#######################################################################################################################
# Functions for Creating Flow Plots
# These are organized in order of appearance in the process.
#######################################################################################################################

# Find Peaks in Intensity as a function of angle

def select_peaks(intensity_matrix, peak_width, peaks_per_site):
    """Finds peaks in an array of intensity as a function of position and angle. Creates an array of "peak/no_peak"
    as a function of position and angle.
    
    A site will be identified as a peak if it is 
    1) a local maximum as a function of angle, 
    2) greater than any other local maxima within the peak width, and
    3) has a high enough intensity to justify being drawn.  
    
    Before running this function, look at the raw electron diffraction data for your sample and estimate 
    the average number of peaks at each location.  For example, if most sites have only one peak but a few 
    have two, a value of 1.2 might be used for peaks_per_site. 
    """
    
    print('Finding Peaks...')

    # Extract Values from Arguments
    n_rows, n_col, n_angles = intensity_matrix.shape
    angle_step = int(180 / n_angles)

    # Use peak width to omit points that are dimmer than their neighbors in the theta direction
    matrix_peak_width = int(np.floor(peak_width / 2 / angle_step))
    intensity_matrix_peaks_only = np.array(intensity_matrix)
    for i in range(1, matrix_peak_width + 1):
        non_peak_mask = np.where(intensity_matrix < np.roll(intensity_matrix, i, axis=2))
        intensity_matrix_peaks_only[non_peak_mask] = 0
        non_peak_mask = np.where(intensity_matrix < np.roll(intensity_matrix, -i, axis=2))
        intensity_matrix_peaks_only[non_peak_mask] = 0
        
    # Determine intensity threshold and remove peaks too small to be drawn
    percentile = (1 - float(peaks_per_site) / n_angles) * 100
    cutoff = np.percentile(intensity_matrix_peaks_only, percentile)
    peak_mask = np.where(intensity_matrix_peaks_only > cutoff)

    # Create peaks matrix
    peak_matrix = np.zeros((intensity_matrix.shape))
    peak_matrix[peak_mask] = 1
    n_peaks = int(np.sum(peak_matrix))
    print("Cutoff = %.2f" % cutoff)
    print("Peak Width = %d" % peak_width)
    print("%d total peaks, which is an average of %.2f peaks per grid square" % (n_peaks, n_peaks / (n_rows * n_col)))

    return peak_matrix

# Seed Lines

def define_subgrid(a1_shape, a1_spacing, grid_density):
    """This function returns six functions that allow for easy conversion between cartesian coordinates,
    row-column coordinates, and subgrids within row-column coordinates. The properties of the grid only need to be
    entered once, and functions are returned that already know the grid's shape.

    These functions are written to be jit-compatible.  Jit is a package that speeds up your code but can make
    debugging more difficult. Using jit created negligible speed gains, so I would recommend not using it for now,
    but it's here for if the code becomes more demanding in the future.

    Inputs:
        a1_shape is the shape of a matrix, or "grid", that you want to map onto cartesian coordinates.  
        a1_spacing is the step size of the grid in cartesian units, i.e. 10nm.  2d array or tuple.
        grid_density is the number of subgrid units per grid unit.  For example, a shape of (4, 5) yields a
            subgrid that has 4 times as many rows and 5 times as many columns as the original grid.

    Outputs:
        grid_to_subgrid: function that takes matrix coordinates and returns coordinates on the finer subgrid
        subgrid_to_grid: inverse of grid_to_subgrid
        cart_to_grid: function that takes cartesian coordinates and returns the matching grid square as well as the
            coordinates within that grid square
        grid_to_cart: inverse of cart_to_grid
        cart_to_subgrid: function that takes cartesian coordinates and returns the matching subgrid square as well
            as the coordinates within that subgrid square
        subgrid_to_cart: inverse of cart_to_subgrid

        """
    a2_shape = np.array(a1_shape) * grid_density # Added the type change during debugging.  Not sure if it's
    # jit compatible or not.  Otherwise this line is treated as tuple multiplication, which just repeats the tuple.
    a2_spacing = a1_spacing / grid_density

    # Set variable types (necessary when using jit, which must infer the types of the variables from the code).
    row, col, new_row, new_col = (1, 1, 1, 1)
    x, y, new_x, new_y = (1.0, 1.0, 1.0, 1.0)

    def grid_to_subgrid(row, col, x, y):
        new_row = row * grid_density - np.floor_divide(y, a2_spacing[0]) + (grid_density - 1)
        new_col = col * grid_density + np.floor_divide(x, a2_spacing[1])
        new_y = np.mod(y, a2_spacing[0])
        new_x = np.mod(x, a2_spacing[1])
        return new_row, new_col, new_x, new_y

    def subgrid_to_grid(row, col, x, y):
        new_row = np.floor_divide(row, grid_density)
        new_col = np.floor_divide(col, grid_density)
        new_y = y - np.mod(row, grid_density) * a2_spacing[0] + a2_spacing[0] * (grid_density - 1)
        new_x = x + np.mod(col, grid_density) * a2_spacing[1]
        return new_row, new_col, new_x, new_y

    def cart_to_grid(x, y):
        new_row = a1_shape[0] - np.floor_divide(y, a1_spacing[0]) - 1
        new_col = np.floor_divide(x, a1_spacing[1])
        new_y = np.mod(y, a1_spacing[0])
        new_x = np.mod(x, a1_spacing[1])
        return new_row, new_col, new_x, new_y

    def grid_to_cart(row, col, x, y):
        new_y = y + (a1_shape[0] - row - 1) * a1_spacing[0]
        new_x = x + col * a1_spacing[1]
        return new_x, new_y

    def cart_to_subgrid(x, y):
        new_row = a2_shape[0] - np.floor_divide(y, a2_spacing[0]) - 1
        new_col = np.floor_divide(x, a2_spacing[1])
        new_y = np.mod(y, a2_spacing[0])
        new_x = np.mod(x, a2_spacing[1])
        return new_row, new_col, new_x, new_y

    def subgrid_to_cart(row, col, x, y):
        new_y = y + (a2_shape[0] - row - 1) * a2_spacing[0]
        new_x = x + col * a2_spacing[1]
        return new_x, new_y

    return grid_to_subgrid, subgrid_to_grid, cart_to_grid, grid_to_cart, cart_to_subgrid, subgrid_to_cart


def seed_lines(peak_matrix, step_size, seed_density=1, verbose=False):
    """Uses the peak matrix to produce a 3xn matrix of x, y, angle values to act as the start of lines that will be
    drawn. Lines are seeded in every grid square with a peak.  The number of line seeds per peak is equal to
    seed_density**2.  Line seeds are distributed evenly in the grid square.  If grid density is one, the line seed is
    in the center of the grid square.  Line seeds point in both directions, so angles are on a domain of 360 degrees.
    
    Inputs:
        peak_matrix: binary matrix showing the positions of peaks
        step_size: spacing between matrix nodes
        seed_density: larger grid density means more line seeds.  The line seeds will be distributed evenly about
            the grid square.
        
    Outputs:
        3xn array of (x, y, angle) values.  Angles are on a domain of 360 degrees.  
    """
    print('Seeding Lines...')

    # Expand pi matrix and peak matrix to 360 degrees to track directionality
    peak_matrix = np.concatenate((peak_matrix, peak_matrix), axis=2)

    # Extract Values from Arguments
    n_rows, n_cols, n_angles = peak_matrix.shape
    angle_step = int(360 / n_angles)

    # Create matrices that enumerate the dimensions of the peak matrix
    rows, cols = np.divmod(
        np.arange(n_rows * n_cols).reshape((n_rows, n_cols))[:, :, None] * np.ones(peak_matrix.shape), n_cols)
    angles = np.arange(0, 360, angle_step)[None, None, :] * np.ones(peak_matrix.shape)

    # For each peak in the peak matrix, extract the position in row, column, angle_index coordinates
    peak_mask = np.where(peak_matrix == 1)
    base_line_seeds = np.stack((rows[peak_mask], cols[peak_mask], angles[peak_mask]))  # Grid Coordinates

    # Expand line seeds to a finer grid
    expanded_line_seeds = base_line_seeds * np.array((seed_density, seed_density, 1))[:, None]  # Subgrid Coordinates
    # These are the same seeds, just in different coordinates
    offset_line_seeds = []
    for i in range(seed_density):
        for j in range(seed_density):
            new_line_seed = np.array(expanded_line_seeds)  # Copy of line seeds in new coordinate system
            # new_line_seed[0, :] += i * base_line_seeds[0, :]  # Shift row value
            # new_line_seed[1, :] += j * base_line_seeds[1, :]  # Shift column value
            new_line_seed[0, :] += i  # Shift row value
            new_line_seed[1, :] += j  # Shift column value
            offset_line_seeds.append(new_line_seed)
    all_line_seeds = np.concatenate(offset_line_seeds,
                                    axis=1)  # Subgrid coordinates. This simply combines the line seeds from the for loop.

    # Convert to cartesian coordinates with angles in degrees
    a1_spacing = np.array((float(step_size), float(step_size)))
    grid_to_subgrid, subgrid_to_grid, cart_to_grid, grid_to_cart, cart_to_subgrid, subgrid_to_cart = define_subgrid(
        peak_matrix.shape, a1_spacing, seed_density)
    cart_line_seeds = np.zeros(all_line_seeds.shape)
    center_of_grid_square = np.ones(
        all_line_seeds.shape[1]) * 0.5  # Used to place line seeds at the center of their respective grid squares
    cart_line_seeds[:2, :] = subgrid_to_cart(all_line_seeds[0, :], all_line_seeds[1, :], center_of_grid_square,
                                             center_of_grid_square)
    cart_line_seeds[2, :] = all_line_seeds[2, :]

    if verbose:
        plt.scatter(base_line_seeds[0, :], base_line_seeds[1, :], s=0.1)
        plt.show()

        plt.scatter(expanded_line_seeds[0, :], expanded_line_seeds[1, :], s=0.1)
        plt.show()

        plt.scatter(all_line_seeds[0, :], all_line_seeds[1, :], s=0.1)
        plt.show()

        plt.scatter(cart_line_seeds[0, :], cart_line_seeds[1, :], s=0.1)
        plt.show()

    return cart_line_seeds


# Propagate Lines

def get_vector(theta, length):
    """Returns the 2d vector of the specified length at the specified angle in degrees"""
    return np.array((np.cos(np.radians(theta)), np.sin(np.radians(theta)))) * length


def propagate_lines(line_seeds, peak_matrix, matrix_step_size, bend_tolerance, curve_resolution=2, max_grid_length=100, silent=False):
    """Lines are extended in both direction from each line seed to create a more readable image.  When lines encounter 
    a small change in angle, they bend to the new angle. If the angle difference is larger than the angle tolerance, 
    the line ends instead.  Lines also end if they reach the edge of the figure or they reach the maximum number 
    of points.  At each step, the lines are extended by a fixed unit length using a test line. If the test line finds a 
    compatible location, the line is extended; otherwise it is terminated.  Lines extend in both directions from each 
    line seed, but the two directions are treated as separate lines.
    
    Inputs:
        line_seeds: as created by the seed_lines function, this is a 3xn matrix of (x, y, theta) values.
        peak_matrix: as created by the determine_peaks function, this is a binary matrix showing the positions 
            and angles of peaks.
        matrix_step_size: the real space distance between grid squares, i.e. 10nm
        bend_tolerance: the maximum bending permitted in a line spanning two grid squares
        curve_resolution: the number of times per unit length that the angle of the line is updated
        max_grid_length: the maximum length a line can be extended, in grid units

        
    Outputs:
        line_matrix: (4 x m x n) matrix, which contains the (x, y, theta, is_active) values for each point m along each 
            line n.  is_active is a binary value which is true if the 
            line is still being drawn at that point.
        """
    print('Propogating Lines...')

    # Expand peak matrix to 360 degrees
    peak_matrix = np.concatenate((peak_matrix, peak_matrix), axis=2)

    # Extract Values from Arguments
    n_grid, _, n_angles = peak_matrix.shape
    n_lines = line_seeds.shape[1]
    angle_step = int(360 / n_angles)
    grid_bend_tolerance = int(np.floor(bend_tolerance / angle_step))
    segment_length = matrix_step_size / curve_resolution
    max_points = max_grid_length * curve_resolution

    # Create tolerance matrix, which shows a 1 at a given point and angle if a line can continue at that point from an 
    # adjacent point with that angle.
    tolerance_matrix = np.zeros(peak_matrix.shape)
    for i in range(-grid_bend_tolerance, grid_bend_tolerance + 1):
        tolerance_matrix += np.roll(peak_matrix, i, axis=2)
    tolerance_matrix = np.clip(tolerance_matrix, 0, 1)

    # Create best angle matrix, which shows the best angle for a line to continue at at a given point, given the angle 
    # it was coming from.
    # Try angles gradually farther away, and fill them in if the space is empty.  
    angle_indices = np.arange(n_angles)[None, None, :] * np.ones(peak_matrix.shape)
    peak_angle_matrix = angle_indices * peak_matrix
    best_angle_matrix = np.empty(peak_matrix.shape)
    best_angle_matrix[:, :, :] = np.nan  # so empty won't be confused with zero degrees
    for i in range(grid_bend_tolerance + 1):  # Both sides at once so that distance gradually increases
        for j in [1, -1]:
            offset_peaks = np.roll(peak_matrix, i * j, axis=2)
            offset_angles = np.roll(peak_angle_matrix, i * j, axis=2)
            match_mask = (np.isnan(best_angle_matrix) & (offset_peaks == 1.0))
            best_angle_matrix[match_mask] = offset_angles[match_mask]

    # Set up line matrices and add first point
    line_matrix = np.zeros((4, max_points, n_lines))  # x, y, theta, is_active.  
    line_matrix[:2, 0, :] = line_seeds[:2, :]
    line_matrix[2, 0, :] = line_seeds[2, :]
    line_matrix[3, 0, :] = 1  # All lines are active

    # Set up subgrid
    a1_shape = np.array((n_grid, n_grid))
    a1_spacing = np.array((matrix_step_size, matrix_step_size))
    grid_density = np.array((1, 1))  # No need for extra precision when propogating lines
    grid_to_subgrid, subgrid_to_grid, cart_to_grid, grid_to_cart, cart_to_subgrid, subgrid_to_cart = define_subgrid(
        a1_shape, a1_spacing, grid_density)

    # propagate
    if not silent:
        progress_bar = ProgressBar(max_points)
    for p in range(1, max_points):
        # Create test lines to see if lines can advance
        test_vector = get_vector(line_matrix[2, p - 1, :], segment_length)  # This is 2 because it refers to theta.
        test_lines_matrix = line_matrix[:2, p - 1, :] + test_vector  # This is :2 because it refers to x and y.

        # Get grid location of new position.  This will be used to look up data in matrices.  
        # Create clipped version to avoid lookup errors.
        grid_positions = np.stack(cart_to_grid(test_lines_matrix[0, :], test_lines_matrix[1, :])[:2]).astype(np.int16)
        clipped_positions = np.clip(grid_positions, 0, n_grid - 1)

        # Mark in bounds lines as active.  Others are marked inactive by default.
        on_grid_mask = np.where(
            (grid_positions[0, :] >= 0) & (grid_positions[0, :] < n_grid) & (grid_positions[1, :] >= 0) & (
                    grid_positions[1, :] < n_grid))[0]
        line_matrix[3, p, on_grid_mask] = 1

        # Check tolerance matrix to see if lines can advance.  Mark lines that can't advance as inactive.  
        angle_indices = (line_matrix[2, p - 1, :] / angle_step).astype(int)  # Angles at previous step
        can_advance = np.squeeze(tolerance_matrix[clipped_positions[0, :], clipped_positions[1, :], angle_indices])
        advancement_mask = np.where(can_advance == 1)
        terminate_mask = np.where(can_advance == 0)
        line_matrix[3, p, terminate_mask] = 0

        # Mark previously inactive lines as inactive
        inactive_mask = np.where(line_matrix[3, p - 1, :] == 0)[0]
        line_matrix[3, p, inactive_mask] = 0

        # Get preferred angles for advancement
        new_angles = np.squeeze(best_angle_matrix[clipped_positions[0, :], clipped_positions[1, :], angle_indices])

        # Tally currently active lines 
        final_active_mask = np.where(line_matrix[3, p, :] == 1)[0]

        # Use current position and angle to advance active lines one step
        vector = get_vector(line_matrix[2, p - 1, final_active_mask], segment_length)  # 2 refers to theta
        line_matrix[:2, p, final_active_mask] = line_matrix[:2, p - 1,
                                                final_active_mask] + vector  # :2 refers to x and y.  
        line_matrix[2, p, final_active_mask] = new_angles[final_active_mask] * angle_step
        
        if not silent:
            progress_bar.update('+1')

    # Remove directionality information, leaving angles on a domain of 180 degrees
    line_matrix[2, :, :] = np.mod(line_matrix[2, :, :], 180)

    return line_matrix


def plot_solid_lines(propagated_lines, min_length=2, sparsity=1):
    """Plots lines with a constant color and width.
    Inputs:
        propagated_lines: (4 x m x n) matrix, which contains the (x, y, theta, is_active) values for each point m along each
            line n.  is_active is a binary value which is true if the
            line is still being drawn at that point.
        min_length: lines shorter than this will not be drawn
        sparsity: values greater than 1 reduce the number of lines drawn for speed and readability purposes.
    Outputs:
        fig: matplotlib.pyplot figure produces by this function.
    """

    print('Plotting Solid Lines...')
    n_dims, max_length, n_lines = propagated_lines.shape
    for i in range(0, n_lines, sparsity):
        active = propagated_lines[3, :max_length, i]
        active_mask = np.where(active == 1)
        length = len(active_mask[0])
        if length > min_length:
            x = propagated_lines[0, :max_length, i][active_mask]
            y = propagated_lines[1, :max_length, i][active_mask]
            plt.plot(x, y, linewidth=0.1, c=(0.3, 0.3, 0.7))
    fig = plt.gcf()
    fig.set_size_inches((12, 12))

    return fig


# Trim lines

def shift_mask(mask, target_array_shape, shift):
    """Moves a mask by an integer number of spaces in each direction.  This allows a shape such as an ellipse to
    be drawn at any point on a matrix, and is faster than methods like np.roll.
    Inputs:
        mask: output of an np.where() function that you would like to move
        target_array_shape: shape of the array that was used in the np.where() function
        shift: tuple or numpy array of the number of spaces to move in each direction.
    Ouputs:
        formatted_mask: shifted mask in the same format as the original mask
    """
    new_masks = []
    for i, submask in enumerate(mask):
        new = np.mod(submask + shift[i], target_array_shape[i])
        new_masks.append(new)
    formatted_mask = tuple(new_masks)
    return formatted_mask


def trim_lines(propagated_lines, intensity_matrix_shape, step_size, line_spacing, spacing_resolution,
               angle_spacing_degrees, max_overlap_fraction=0.5, min_length=2, verbose=False, silent=False):
    """Trims lines created by the propagate_lines function for purposes of readability, rendering speed, and
    homogeneous line density.
    Inputs:
        propagated_lines: (4 x m x n) matrix, which contains the (x, y, theta, is_active) values for each point m along
            each line n.  is_active is a boolean value which is true if the
            line is still being drawn at that point.
        intensity_matrix_shape: dimensions provided by intensity_matrix.shape
        step_size: real space distance  between points in the intensity matrix, i.e. 10nm
        line_spacing: minimum distance between two lines before they start to overlap.  Note that some overlaps
            may be allowed.
        spacing_resolution: higher values reduce geometric artifacts that cause greater spacing at certain angles.  Spacing resolution should be about 
        angle_spacing_degrees: minimum angular distance between two crossing lines before those lines count as
            overlapping.
        max_overlap_fraction: fraction of line that can overlap with already-drawn lines.  If the line overlaps
            by more than this it is not drawn.
        min_length: lines with less points than the minimum length are not drawn
        verbose: Boolean. When true, extra troubleshooting information is shown, including the space-filling process.
    Outputs:
        trimmed_lines: matplotlib.pyplot figure drawn by the function

     """

    print('Trimming Lines...')
    # Extract values from inputs
    n_dims, max_length, n_lines = propagated_lines.shape
    n_grid, _, n_angles = intensity_matrix_shape
    angle_step = int(np.round(180 / n_angles))
    point_separation = int(line_spacing * spacing_resolution)  # Subgrid units.
    angle_spacing = int(np.round(angle_spacing_degrees / angle_step))
    grid_density = spacing_resolution  # nickname

    # Make subgrid. This helps spacings between diagonal lines and spaces between horizontal/vertical lines be treated 
    # more equally.  
    shape = np.array((n_grid, n_grid))
    spacing = np.array((step_size, step_size))
    grid_to_subgrid, subgrid_to_grid, cart_to_grid, grid_to_cart, cart_to_subgrid, subgrid_to_cart = define_subgrid(
        shape, spacing, grid_density)

    # Make Canvas Matrix which tracks space available for lines
    n_subgrid = n_grid * grid_density
    canvas_matrix = np.zeros((n_subgrid, n_subgrid, n_angles))

    # Make Exclusion Matrix with ellipse to denote space filling
    point_separation_steps = int(np.ceil(point_separation))
    angle_spacing_steps = int(np.ceil(angle_spacing))  # Angles don't need extra visual resolution
    exclusion_width = 2 * point_separation_steps + 1
    angle_exclusion_width = 2 * angle_spacing_steps + 1
    exclusion_matrix = np.zeros(canvas_matrix.shape)
    r1 = point_separation_steps
    r2 = angle_spacing_steps
    rows = np.arange(n_subgrid)[:, None, None] * np.ones(exclusion_matrix.shape)
    cols = np.arange(n_subgrid)[None, :, None] * np.ones(exclusion_matrix.shape)
    angles = np.arange(n_angles)[None, None, :] * np.ones(exclusion_matrix.shape)
    for x_shift in (0, n_subgrid):
        for y_shift in (0, n_subgrid):
            for z_shift in (0, n_angles):
                distances = np.sqrt(
                    ((rows - y_shift) / r1) ** 2 + ((cols - x_shift) / r1) ** 2 + ((angles - z_shift) / r2) ** 2)
                elipse_mask = np.where(distances <= 1)
                exclusion_matrix[elipse_mask] = 1
    exclusion_mask = np.where(exclusion_matrix == 1)

    # Sort lines by length
    # Since we're looking at the subset of the matrix that only includes row 3, axis number zero is point number
    line_lengths = np.sum(propagated_lines[3, :, :], axis=0)  
    by_length = np.argsort(line_lengths * -1)  # Descending
    sorted_lines = propagated_lines[:, :, by_length]

    # Choose lines to draw using a space-filling algorithm
    lines_to_draw = []
    if not silent:
        progress_bar = ProgressBar(n_lines)
    for i in range(n_lines):
        # Identify active region of line
        active_mask = np.where(sorted_lines[3, :max_length, i] == 1)
        active_length = len(active_mask[0])

        # Get line in cartesian coordinates
        x_values = sorted_lines[0, :max_length, i][active_mask].astype(int)
        y_values = sorted_lines[1, :max_length, i][active_mask].astype(int)

        # Get line in subgrid coordinates
        row, col, x, y = cart_to_subgrid(x_values, y_values)
        row, col = row.astype(int), col.astype(int)
        a = (sorted_lines[2, :max_length, i] / angle_step)[active_mask].astype(int)


        # Use line coordinates as mask to determine if there is room for the new line
        n_overlaps = np.sum(canvas_matrix[(row, col, a)])
        overlap_fraction = n_overlaps / active_length

        if overlap_fraction <= max_overlap_fraction and active_length >= min_length:
            # Record line
            lines_to_draw.append(i)

            # Fill space
            for shift in zip(row, col, a):
                shifted_mask = shift_mask(exclusion_mask, canvas_matrix.shape, shift)
                canvas_matrix[shifted_mask] = 1

            if verbose:
                if len(lines_to_draw) <= 20:
                    plt.imshow(np.sum(canvas_matrix, axis=2))
                    plt.colorbar()
                    plt.show()
        if not silent:
            progress_bar.update('+1')

    trimmed_lines = sorted_lines[:, :, np.array(lines_to_draw)]
    return trimmed_lines


# Plot Lines

def prepare_line_data(trimmed_lines, norm_intensity_matrix, step_size):
    """Combines data from trimmed lines and normalized intensities to give the angle, intensity, and other paramters at 
    each point in the trimmed lines.

    Inputs:
        trimmed_lines: (4 x m x n) matrix, which contains the (x, y, theta, is_active) values for each point m along 
            each line n.  This is the output of the trim_lines function.
        norm_intensity_matrix: 3d matrix where the three dimensions are (real space x position, real space y position, angle).  
            Normalizing values is recommended for ease-of-use.

    Outputs:
        line_data: (7 x m x n) matrix, which contains the (x, y, theta, is_active, intensity, _, _) values for each 
        point m along each line n.  The last two values are placeholders, for example if you wanted to add in peak 
        width or diffraction angle.
        """
    _, n_points, n_lines = trimmed_lines.shape
    n_grid, _, n_angles = norm_intensity_matrix.shape
    angle_step = np.int(180 / n_angles)
    line_data = np.zeros(
        (7, n_points, n_lines))  # 7 rows are x, y, angle, is_active, intensity, linewidth, q.  Feel free to add more.
    line_data[:4, :, :] = trimmed_lines

    # Set up subgrid
    a1_shape = np.array((n_grid, n_grid))
    a1_spacing = np.array((step_size, step_size))
    grid_density = np.array((1, 1))  # No need for extra precision when propogating lines
    grid_to_subgrid, subgrid_to_grid, cart_to_grid, grid_to_cart, cart_to_subgrid, subgrid_to_cart = define_subgrid(
        a1_shape, a1_spacing, grid_density)

    for i in range(n_lines):
        # Identify active region of line
        active_mask = np.where(line_data[3, :, i] == 1)
        active_length = len(active_mask[0])

        # Get line in grid coordinates
        x_values = line_data[0, :, i][active_mask].astype(int)
        y_values = line_data[1, :, i][active_mask].astype(int)
        row, col, x, y = cart_to_grid(x_values, y_values)
        row, col = row.astype(int), col.astype(int)
        a = (line_data[2, :, i] / angle_step)[active_mask].astype(int)

        # Look up intensity
        # square_intensity_matrix = intensity_matrix.reshape(n_grid, n_grid, n_angles)
        intensity_values = norm_intensity_matrix[(row, col, a)]
        line_data[4, :active_length, i] = intensity_values

        # Linewidth is not present for now - feel free to add
        # q is not present for now - feel free to add

    return line_data


def color_by_angle(theta):
    """Assigns r, g, b values based on input angle.
    Inputs:
        theta: angle in degrees.  Can be a single value or an array.
    Outputs:
        tuple of r, g, b values, each of which has the same shape as the input values."""
    radians = theta * 3.14 / 180
    red = np.cos(radians) ** 2  # Range is max 1 not max 255
    green = np.cos(radians + 3.14 / 3) ** 2 * 0.7  # Makes green darker - helps it stand out equally to r and b
    blue = np.cos(radians + 2 * 3.14 / 3) ** 2
    return (red, green, blue)


def scale_values(values, vmin=0, brightness=1, contrast=1, gamma=1):
    """Rescales data such as intensity, q, or similar to values suitable for linewidth, alpha, or similar.  There are 
    many possible versions of this function - this is the one that works best for my data so far.  My guess is that 
    some users will write a different version of this function to fit their needs.

    Inputs:
        values: the values to be scaled.  Typically an array.
        vmin: any values below vmin are set equal to vmin before scaling.
        brightness: the highest value output by this function.  Note that when mapping to RGBA values, brightness 
            generally can't be greater than 1.
        contrast: the value of input data that yields the brightest possible signal.
        gamma: the nonlinearity of the mapping curve.  If gamma > 1, strong signals will be spaced more than weak 
            signals, and vice versa for gamma < 1.
        If gamma = 0, all inputs will be mapped to the same value."""
    return ((np.clip(values, vmin, contrast) / contrast) ** gamma) * brightness


def plot_graded_lines(trimmed_lines, red, green, blue, alpha, width, outline=False, background=None, background_cmap=None, background_vmin=None, background_vmax=None, colorbar=True):
    """Plots lines with non-constant RGBA or Width values.
    Inputs:
        trimmed_lines: (4 x m x n) matrix, which contains the (x, y, theta, is_active) values for each point m along
            each line n.  This is the output of the trim_lines function.
        red: color value between 0 and 1
        green: color value between 0 and 1
        blue: color value between 0 and 1
        alpha: transparency value between 0 and 1, where 1 is fully opaque and 0 is fully transparent. Interestingly,
            transparency does not seem to increase rendering time.
        width: line width.  The value is not capped but is usually on the order of 1.
    """

    print('Plotting Graded Lines...')

    n_dims, max_length, n_lines = trimmed_lines.shape
    # To make graded lines, lines must be divided into line segments.  
    # Lines will be considered connected along axis 0 and disconnected along axis 1.
    # The number of line segments in each line will be one less than the number of points in that line.
    # Color, alpha and width values will be averaged between each pair of points
    r_avg = red[:-1] + np.diff(red, axis=0) / 2
    g_avg = green[:-1] + np.diff(green, axis=0) / 2
    b_avg = blue[:-1] + np.diff(blue, axis=0) / 2
    a_avg = alpha[:-1] + np.diff(alpha, axis=0) / 2
    w_avg = width[:-1] + np.diff(width, axis=0) / 2

    # Divide lines into line segments.  One segment is recorded [(x1, y1), (x2, y2)].  
    # Coordinates, colors, and widths are all recorded as lists.
    segments = []
    colors = []
    widths = []
    for i in range(n_lines):
        # Identify active region of line
        active_mask = np.where(trimmed_lines[3, :, i] == 1)
        active_length = len(active_mask[0])

        # Record each segment of the line
        for j in range(active_length - 1):
            x1, y1 = trimmed_lines[:2, j, i]
            x2, y2 = trimmed_lines[:2, j + 1, i]
            segments.append([(x1, y1), (x2, y2)])
            colors.append((r_avg[j, i], g_avg[j, i], b_avg[j, i], a_avg[j, i]))
            widths.append(w_avg[j, i])

    # Plot Lines
    fig, ax = plt.subplots(figsize=(12, 12))
    if background is not None:
        plt.imshow(np.transpose(background), origin='lower', cmap=background_cmap, extent=(0, 800, 0, 800), vmin=background_vmin, vmax=background_vmax)
        if colorbar:
            plt.colorbar()
    
    line_plot = mc.LineCollection(segments, linewidth=widths, colors=colors)
    ax.add_collection(line_plot)
    plt.autoscale(enable=True, axis='y')
    plt.autoscale(enable=True, axis='x')


def smooth(array, smoothing_length):
    """Smooth a 2D array along axis 1 using a moving average.  This function assumes that the edges of the array
    terminate, as opposed to looping around like for an angular value.

    Inputs:
        array: 2d array to be smoothed along axis 1
        smoothing_length: number of points to average.
    Outputs:
        smoothed_array: array with the same shape as the original, after applying the moving average
    """

    # Extract Values from Inputs
    max_shift = int((smoothing_length - 1) / 2)
    n_points, n_lines = array.shape

    # Replace intensities in inactive region with the last active intensity
    filled_array = np.array(array)
    inactive_mask = np.where(filled_array == 0)
    for i in range(n_lines):
        # Identify active region of line
        active_mask = np.where(array[:,
                               i] != 0)  # Different from previous method.  Since intensities are real and continuous 
        #                                  it's incredibly unlikely they would be zero naturally.
        active_length = len(active_mask[0])
        final_intensity = array[active_length - 1, i]
        filled_array[active_length:, i] = final_intensity

    # Pad sides of array with end values
    padded_array = np.pad(filled_array, max_shift, mode="edge")[:, max_shift:n_lines + max_shift]

    # Add shifted matrices and divide
    shifted_matrices = [padded_array[shift:n_points + shift, :] for shift in range(smoothing_length)]
    smoothed_array = np.sum(shifted_matrices, axis=0) / smoothing_length
    # Make inactive regions zero again
    smoothed_array[inactive_mask] = 0
    return smoothed_array


# Other Miscellaneous Functions

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