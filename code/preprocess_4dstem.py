cklimport numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import collections as mc
import scipy.signal as sig
import scipy.optimize as opt
import sys
import os
import time
from ncempy.io import dm
from tqdm.notebook import tqdm, trange
import importlib as imp  


#######################################################################################################################
# Sample Workflow
#######################################################################################################################
"""
This section shows recommended code to copy into a Jupyter notebook in order to use this script.  All constants in this 
workflow can be changed based on your data set.

    
Import this module using "import preprocess_4dstem as prep" to use the sample code below without modification.  
If you make changes to this code, you can load your saved changes into Jupyter using:
importlib.reload(prep)

    
Preprocessing 4D-STEM data can use a lot of RAM.  This program is organized in a way that reduces RAM use, so the total
RAM use should only be 2-3x the size of the original file.  To do this, instead of saving a new set of images to memory
after each step (centering, artifact subtraction, etc), the modifications are tabulated and only applied when necessary.  
For example, Box 1 calculates the centers of each image, but doesn't save the centered images to memory - it simply
records the image centers so that they can be used as needed.  The same is done for artifact subtraction.  This method also
improves the accuracy of the final data since there is less rounding along the way.  


For the sample workflow below, copy each section into a different box in Jupyter.  You can use the section headings as
markdown in Jupyter if you like.  


    
    
###################################################################################################################
# Define Inputs
###################################################################################################################    

# These can be defined manually, or captured from your DM4 file or filename using automated
# scripts.  Be aware that metadata in DM4 files is organized inconsistently.  

# File System
dm_path = "/home2/luke/tem_data/2018-06-09/dm4/3_80x80_ss=10nm_spot9_alpha=p48_cl=480_RT_300kV_33ms_bin=4.dm4"
output_directory = "/home2/luke/projects/pbttt/test/"
nickname = dm_path.split('/')[-1][:10]

# Diffraction Data
n_grid =   # number of rows of images in 4D-STEM scan
n_pixels =  # number of rows of pixels in one image
step_size =  # spacing between 4D-STEM images, in nm
camera_length = # in mm

print("Reading from File %s\nWriting to Directory %s\nNickname %s" % (dm_path, output_directory, nickname))

# Calculate values from input data
n_images = n_grid**2
q_per_pixel = 1152 * np.pi / (camera_length * n_pixels) # At 256 pixels, 1/d = 6nm^-1 for CL=480 (GMS 3 shows 5.996).  
                                                        # This is equivalent to 1.2*pi nm^-1 from center to edge. 


###################################################################################################################
# Prepare Image Centers
###################################################################################################################   

x_offset, y_offset = prep.calculate_centroids(dm_path, n_grid, n_pixels)

max_error = 2  # Pixels.  Images farther than this from the fit are omitted.  
n_cycles = 10  # Number of times to omit high-error images and re-fit
x_centers, y_centers, istcklist = prep.iterative_fit(x_offset, y_offset, n_pixels, max_error, n_cycles)


###################################################################################################################
# Prepare Vertical Stripe Subtraction
###################################################################################################################   

# Note: this section quantifies an artifact commonly found on the TitanX microscope at NCEM.  If you use a different
# microscope, there may be different artifacts to remove.  If you need high precision, you may need to remove additional
# artifacts.

average_image = prep.get_average_centered_image(dm_path, n_pixels, x_centers, y_centers, blocklist, verbose=True)
streak_image = prep.get_streak(average_image, verbose=True)

###################################################################################################################
# Conservative Background Subtraction
###################################################################################################################   
# This section offers an approximate background subtraction that helps to view the quality of the data and artifact
# removal before the slow integration step.  A more thorough background subtraction is done after integration.  

current_image = average_image - streak_image
background_image = prep.get_background_image(current_image, reduction_factor = 0.9, fit_start=40, fit_end=300, verbose=False)

###################################################################################################################
# View corrected diffraction patterns
###################################################################################################################   

total_artifact = streak_image + background_image
prep.stitch_diffraction_images(dm_path, n_pixels, x_centers, y_centers, blocklist, artifact=total_artifact, 
                               low_percentile=80, high_percentile=95, alpha=0.2,
                               save=False, output_directory="/home2/luke/simulations_paper/figures/", 
                               max_images=1, start_image=0, figsize=(12, 12), filter_size=None,
                               patterns_per_block=4, colorbar=False, cmap='viridis')


###################################################################################################################
# Integrate and Convert to q vs chi
###################################################################################################################   

# Set range and resolution
q_min, q_max, q_step = (1.0, 3.0, 0.5) 
angle_step = 20

plt.rc('axes', titlesize=BIGGER_SIZE)
q_theta_matrix = prep.integrate_diffraction(dm_path, n_pixels, x_centers, y_centers, 
                                  q_min, q_max, q_step, angle_step, q_per_pixel,
                                  artifact=streak_image, background=background_image, 
                                  blocklist=blocklist, check_images=15, medfilt_size=3, verbose=False, 
                                  crosshairs=False, check_vmin=0, check_vmax=20, colorbar=False, big=False)

###################################################################################################################
# Save Results
################################################################################################################### 

path = os.path.join(output_directory, 'big_datasets', filename.split('.')[0] + '__q %.2f %.2f %.2f a %.2f.npy' % (
        q_min, q_max, q_step, angle_step))
print(path)
np.save(path, q_theta_matrix)

"""

###################################################################################################################
# Heavy-Lifting Functions for Integrating Data
###################################################################################################################   


def integrate_diffraction(dm_path, n_pixels, x_centers, y_centers, 
                          q_min, q_max, q_step, angle_step, q_per_pixel,
                          artifact=None, background=None, blocklist=[], check_images=5, medfilt_size=0, verbose=False, 
                          check_vmin=0, check_vmax=20, crosshairs=True, colorbar=False, big=False, 
                          save_check_images=False, save_path=None):
    """Loops over a 4D-STEM file using lazy access and converts it to radial coordinates using the given bin sizes.  
    Also applies centering and removes provided artifacts and background.  Also displays check images for the first 
    few diffraction patterns that do the analysis 'the long way' so that users can understand the process and check
    their results.  Note that Inputing a q_step or angle_step of zero will cause the function to ignore that dimension.
    """

    # Expand input data
    dm_file = FileDM(dm_path)
    n_images = dm_file.n_images
    n_grid = int(np.round(n_images**0.5))
    n_pixels = 512
    
    if q_step:
        n_q = int(np.round(((q_max - q_min) / q_step)))
        q_values = np.linspace(q_min, q_max, n_q, endpoint=False)
    else:
        n_q = 1
        q_values = np.array([q_min])
        q_step = q_max - q_min
    if angle_step:
        n_angles = int(180 / angle_step)
        angle_values = np.arange(-90, 90, angle_step)
    else:
        n_angles = 1
        angle_values = np.array([angle_min])
        angle_step = 180
        
    # Set up position matrices
    rows, cols = np.divmod(np.arange(n_pixels**2), n_pixels)
    
    # Set up results matrices
    sum_matrix = np.zeros((n_images, n_q, n_angles))
    count_matrix = np.zeros((n_images, n_q, n_angles))
    
    # Prepare centered masks for preview images (provided by check_diffraction)
    yc = n_pixels/2 - 0.5
    xc = yc
    angle_map = (np.arctan( (cols-xc) / (rows-yc)) * 180 / np.pi).reshape((n_pixels, n_pixels))
    q_map = (np.sqrt( (rows-yc)**2 + (cols-xc)**2 ) * q_per_pixel).reshape((n_pixels, n_pixels))
    centered_masks = []
    for j, q in enumerate(q_values):
        for k, a in enumerate(angle_values):
            mask = np.where( (a <= angle_map) & (angle_map < a + angle_step) & (q <= q_map) & (q_map < q + q_step))
            centered_masks.append((j, k, mask))
    
    # Iterate through images and integrate
    progress_bar = tqdm(total=dm_file.n_images, desc='Integrating Diffraction Patterns', leave=False)
    for i, image_data in enumerate(dm_file.lazy_access()):
        image, image_number = image_data
        
        if image_number in blocklist:
            progress_bar.update()
            continue  # Skip blocklisted data, leave intensity marked as zero
            
        # Apply median filter
        if medfilt_size:
            image = sig.medfilt(image, medfilt_size)
            
        # Prepare references for masks
        flat_image = image.reshape(n_pixels**2)
        xc, yc = x_centers[image_number], y_centers[image_number] 
        angle_map = np.arctan( (cols-xc) / (rows-yc)) * 180 / np.pi
        q_map = np.sqrt( (rows-yc)**2 + (cols-xc)**2 ) * q_per_pixel
        
        if verbose:
            if i < 5:
                plt.imshow(angle_map.reshape((n_pixels, n_pixels)))
                plt.title('Angle Relative to Center')
                plt.colorbar()
                plt.show()
                plt.imshow(q_map.reshape((n_pixels, n_pixels)))
                plt.title('Distance Relative to Center')
                plt.colorbar()
                plt.show()
        
        masks = [] # Saved temporarily to check the integration results
        for j, q in enumerate(q_values):
            for k, a in enumerate(angle_values):
                mask = np.where( (a <= angle_map) & (angle_map < a + angle_step) & (q <= q_map) & (q_map < q + q_step))
                masks.append((j, k, mask))
                image_slice = flat_image[mask]
                sum_matrix[image_number, j, k] = np.sum(image_slice)
                count_matrix[image_number, j, k] = image_slice.shape[0]
        if image_number < check_images:
            print(image_number)
            check_diffraction(image, image_number, x_centers, y_centers, centered_masks, background=background, artifact=artifact, 
                              crosshairs=crosshairs, vmin=check_vmin, vmax=check_vmax, colorbar=colorbar, big=big, 
                              save=save_check_images, path=save_path)
        progress_bar.update()
        
    q_theta_matrix = np.divide(sum_matrix, count_matrix, 
                          out=np.zeros_like(sum_matrix), where=count_matrix!=0) # Replaces divide by zero with zero
    
    # Subtract background and artifacts from final answer
    if artifact is not None:  
        integrated_artifacts = integrate_artifacts(artifact, q_min, q_max, q_step, angle_step, q_per_pixel, verbose=False)
        q_theta_matrix -= integrated_artifacts
    if background is not None:
        integrated_background = integrate_artifacts(background, q_min, q_max, q_step, angle_step, q_per_pixel, verbose=False)
        q_theta_matrix -= integrated_background
    
    return q_theta_matrix


###################################################################################################################
# Functions for Removing Artifacts
################################################################################################################### 



def get_average_centered_image(dm_path, n_pixels, x_centers, y_centers, blocklist, n_preview=10, verbose=False):
    """Loops over a 4D-STEM file using lazy access, centers each image to the nearest pixel using pre-computed 
    image centers, and averages those centered images.  The resulting image is useful for computing the power
    law background of the image and impact of certain artifacts."""
    
    centered_image_sum = np.zeros((n_pixels, n_pixels))
    target_center = (n_pixels - 1) / 2
    dm_file = FileDM(dm_path)
    if verbose:
        print('Displaying up to %d preview images' % n_preview)
    progress_bar = tqdm(total=dm_file.n_images, desc='Summing Images', leave=False)
    for i, image_data in enumerate(dm_file.lazy_access()):
        image, image_number = image_data
        if image_number in blocklist:
            progress_bar.update()
            continue  # Skip blocklisted data, leave intensity marked as zero

        # Center image to nearest integer pixel
        yc, xc = y_centers[i], x_centers[i]
        shift = (int(np.round(target_center - yc)), int(np.round(target_center - xc)))

        centered_image = np.roll(
            np.roll(
            image, shift[0], axis=0), # Vertical direction in stitched image, positive moves image farther down
            shift[1], axis=1)  # Horizontal direction in stitched image, positive moves image farther right
        
        if shift[0] < 0:
            centered_image[shift[0]:, :] = 0
        else:
            centered_image[:shift[0], :] = 0
        if shift[1] < 0:
            centered_image[:, shift[1]:] = 0
        else:
            centered_image[:, :shift[1]] = 0
            
        if verbose:
            if i < n_preview:
                plt.imshow(centered_image, vmax=np.percentile(centered_image, 80))
                plt.title('(%d, %d)' % (shift[0], shift[1]))
                plt.show()
        
        centered_image_sum += centered_image
        progress_bar.update()
    average_image = centered_image_sum / dm_file.n_images
    
    if verbose:
        plt.imshow(average_image, vmax=np.percentile(average_image, 95))
        plt.title('Average Centered Image')
        plt.show()
    
    return average_image


def get_streak(average_image, streak_width=100, reference_width=50, verbose=False):
    """Gives the profile of a vertical streak present in an image, using intensity near the
    edges of the image.  This profile can later be subtracted from the original data to remove
    the streak artifact."""
    
    n_pixels = average_image.shape[0]
    
    edges = np.concatenate([np.arange(reference_width), np.arange(512 - reference_width, 512, 1)])
    reference_with_streak = average_image[edges, :]
    reference_without_streak = average_image[:, edges]
    
    baseline_with_streak = np.median(reference_with_streak, axis=0)
    baseline_without_streak = np.median(reference_without_streak, axis=1)
    
    start, stop = int(np.round((512 - streak_width) / 2)), int(np.round((512 + streak_width) / 2))
    streak_1d = np.zeros(n_pixels)
    streak_1d[start:stop] = baseline_with_streak[start:stop] - baseline_without_streak[start:stop]
    streak_2d = streak_1d[None, :] * np.ones((n_pixels, n_pixels))
    image_without_streak = average_image - streak_2d
    
    if verbose:        
        plt.imshow(reference_with_streak, vmax=np.percentile(average_image, 95))
        plt.title('Reference With Streak')
        plt.show()
        
        plt.imshow(reference_without_streak, vmax=np.percentile(average_image, 95))
        plt.title('Reference Without Streak')
        plt.show()
        
        plt.imshow(streak_2d, vmax=np.percentile(average_image, 95))
        plt.title('Streak')
        plt.show()
        
        plt.imshow(image_without_streak, vmax=np.percentile(average_image, 95))
        plt.title('Average Image Without Streak')
        plt.show()
        
    return streak_2d


def integrate_artifacts(image, q_min, q_max, q_step, angle_step, q_per_pixel, verbose=False):
    """Converts a single image to radial coordinates using the specified bin sizes."""
    n_pixels = image.shape[0]
    
    # Set up position variables
    rows, cols = np.divmod(np.arange(n_pixels**2), n_pixels)
    xc = (n_pixels - 1) / 2
    yc = (n_pixels - 1) / 2
    
    # Set up integration ranges
    q_values = np.arange(q_min, q_max - q_step/2, q_step)  # Avoids accidentally including last value
    n_q = len(q_values)
    angle_values = np.arange(-90, 90, angle_step)
    n_angles = len(angle_values)

    # Set up output matrices
    sum_matrix = np.zeros((n_q, n_angles))
    count_matrix = np.zeros((n_q, n_angles))

    # Prepare references for masks
    flat_image = image.reshape(n_pixels**2)
    angle_map = np.arctan( (cols-xc) / (rows-yc)) * 180 / np.pi
    q_map = np.sqrt( (rows-yc)**2 + (cols-xc)**2 ) * q_per_pixel

    # Integrate artifacts
    for j, q in enumerate(q_values):
        for k, a in enumerate(angle_values):
            mask = np.where( (a <= angle_map) & (angle_map < a + angle_step) & (q <= q_map) & (q_map < q + q_step))
            image_slice = flat_image[mask]
            sum_matrix[j, k] = np.sum(image_slice)
            count_matrix[j, k] = image_slice.shape[0]

    integrated_artifacts = np.divide(sum_matrix, count_matrix, 
                              out=np.zeros_like(sum_matrix), where=count_matrix!=0) # Replaces divide by zero with zero

    if verbose:
        plt.imshow(integrated_artifacts, vmax=100, extent=(-90, 90, 0, q_max - q_min), aspect=(180 / (q_max - q_min)))
        plt.title('Integrated Artifacts')
        plt.xlabel('Angle (Degrees)')
        plt.ylabel('q (inverse Angstroms)')
        plt.show()
        
    return integrated_artifacts


def get_background_image(image, n_fit_steps=50, fit_start=20, fit_end=300, reduction_factor = 0.5, verbose=False):
    """Performs a conservativive background subtraction on an electron diffraction pattern to help with artifact
    removal and troubleshooting.  The image should be the average centered version of the data with major artifacts
    removed.  Image data is fit to the function I(r) = A * r**b, where I is the intensity, r is the distance
    from the center of the image.  The resulting model is multiplied by a reduction factor to avoid over-subtracting
    since this method is approximate."""
    
    # Get radial coordinates
    n_pixels = image.shape[0]
    rows, cols = np.divmod(np.arange(n_pixels**2).reshape((n_pixels, n_pixels)), n_pixels)
    center = (n_pixels - 1.0) / 2
    r = np.sqrt((rows - center)**2 + (cols - center)**2)
    
    # Create masks
    r_masks = []
    r_thresholds = np.linspace(0, n_pixels/2, n_fit_steps)
    n_r = len(r_thresholds) - 1
    for i in range(n_r):
        r_min = r_thresholds[i]
        r_max = r_thresholds[i + 1]
        new = np.where((r >= r_min) & (r < r_max))
        r_masks.append(new)
        
    # Integrate
    intensity = np.zeros(n_r)
    for i, mask in enumerate(r_masks):
        if len(mask) > 0:
            intensity[i] = np.mean(image[mask])
    
    # Fit to power law, omitting asymptote and possibly other values
    r_centers = (r_thresholds[1:] + r_thresholds[:-1]) / 2
    fit_mask = np.where((r_centers >= fit_start) & (r_centers < fit_end))[0]
    x = r_centers[fit_mask].flatten()
    y = intensity[fit_mask].flatten()
    params, extras = opt.curve_fit(power_law, x, y, p0=[100, -1])
    
    # Remove asymptote and plot
    fit = power_law(r_centers, *params)
    cutoff = power_law(fit_start, *params)
    fit = np.clip(fit, -np.inf, cutoff)

    plt.plot(r_centers, intensity, linewidth=0, marker='.')
    plt.plot(r_centers, fit)
    plt.plot(r_centers, fit * reduction_factor)
    plt.yscale('log')
    plt.show()
    
    background_image = power_law(r, *params)
    background_image = np.clip(background_image, -np.inf, cutoff)
    background_image = background_image * reduction_factor
    
    vmax=40

    
    if verbose:
        plt.imshow(image, vmax=vmax)
        plt.colorbar()
        plt.show()
        
        plt.imshow(background_image, vmax=vmax)
        plt.colorbar()
        plt.show()
    
    corrected_image = image - background_image
    plt.imshow(corrected_image, vmax=vmax)
    plt.colorbar()
    plt.show()
        
    return background_image


###################################################################################################################
# Math Functions
###################################################################################################################  

def get_centroid(image):
    """Returns the (y, x) centroid of a single image."""
    y_pixels, x_pixels = image.shape
    x_weights = np.arange(x_pixels)  # For weighted average for centroid
    y_weights = np.arange(y_pixels)
    x_condensed = np.sum(image, axis=0)
    y_condensed = np.sum(image, axis=1)
    xc = np.dot(x_condensed, x_weights) / sum(x_condensed)
    yc = np.dot(y_condensed, y_weights) / sum(y_condensed)

    return yc, xc   # Row before column


def calculate_centroids(dm_path, n_grid, n_pixels, verbose=False):
    """Loops over 4D-STEM file using lazy access and calculates the centroid of each diffraction pattern."""
    x_offset = np.zeros(n_grid**2)
    y_offset = np.zeros(n_grid**2)

    dm_file = FileDM(dm_path)
    progress_bar = tqdm(total=dm_file.n_images, desc='Calculating Centers', leave=False)
    for i, image_data in enumerate(dm_file.lazy_access()):
        image, image_number = image_data
        yc, xc = get_centroid(image)
        y_offset[image_number] = yc - (n_pixels / 2 - 0.5)
        x_offset[image_number] = xc - (n_pixels / 2 - 0.5)
        
        if verbose:
            print(image_number)
            # plt.imshow(image, vmax=np.percentile(image, 80))
            plt.imshow(image, vmin=0, vmax=100)
            plt.gca().axis('off')
            plt.show()
        
        progress_bar.update()
        
    return x_offset, y_offset


def linear_model(xy, a, b, c):
    return a*xy[0, :] + b*xy[1, :] + c


def power_law(x, a, b):
    return a * x ** b


def iterative_fit(x_offset, y_offset, n_pixels, threshold, n_cycles):
    """Fits the centroids of individual diffraction patterns to track the sway of the beam and return more
    accurate beam positions for each pattern.  Diffraction patterns that deviate from the fit by more than
    a given threshold are added to a blocklist and omitted from future fits.  The blocklist is returned so
    that future analysis can avoid using the errant diffraction patterns."""
    n_images = len(x_offset)
    n_grid = int(np.round(n_images**0.5))
    image_numbers = np.arange(n_images)
    rows, cols = np.divmod(image_numbers, n_grid)
    xy = np.stack((rows, cols))
    blocklist = []
    
    for i in range(n_cycles):
        keeplist = np.delete(np.arange(n_images), blocklist)
        keep_images = np.ones(n_grid**2)
        
        y_guess = [0, 0, np.mean(y_offset[keeplist])]
        y_params, y_extras = opt.curve_fit(linear_model, xy[:, keeplist], y_offset[keeplist], p0=y_guess)
        y_fit = linear_model(xy, *y_params)
        y_res = y_offset - y_fit

        x_guess = [0, 0, np.mean(x_offset[keeplist])]
        x_params, x_extras = opt.curve_fit(linear_model, xy[:, keeplist], x_offset[keeplist], p0=x_guess)
        x_fit = linear_model(xy, *x_params)
        x_res = x_offset - x_fit
        
        blocklist = []  # Indicates points that are suspect and should be excluded from analysis
        keep_images = np.ones(n_images)
        for i in range(n_images):
            if np.abs(x_res[i]) > threshold or np.abs(y_res[i]) > threshold:
                blocklist.append(i)
                keep_images[i] = 0
        keeplist = np.delete(np.arange(n_images), blocklist)
        
        x_centers = x_fit + n_pixels/2 - 0.5
        y_centers = y_fit + n_pixels/2 - 0.5
        
        
    plt.imshow(np.reshape(keep_images, (n_grid, n_grid)), vmin=0, vmax=1)
    plt.title('Keeplist vs blocklist')
    plt.show()
        
    return x_centers, y_centers, blocklist


###################################################################################################################
# Plotting Functions
###################################################################################################################  


def bin_2D_parallel(array, binning):
    """Reduces the size of an image through binning."""
    n_rows, n_col = array.shape
    y_bin, x_bin = (int(binning[0]), int(binning[1]))
    assert n_rows % binning[0] == 0 # Will give wrong answer for unequal bins
    assert n_col % binning[1] == 0 # Will give wrong answer for unequal bins
    new_array = np.zeros((int(n_rows / binning[0]), int(n_col / binning[1])))
    for i in range(y_bin):
        for j in range(x_bin):
            new_part = array[i::y_bin, j::x_bin]
            new_array = new_array + new_part
    return new_array / (binning[0] * binning[1])

def center_image(image, image_number, x_centers, y_centers, half_width=256, binning=1):
    """Centers image to nearest pixel."""
    x_shift = int(np.round(half_width * binning + 0.5 - x_centers[image_number]))
    y_shift = int(np.round(half_width * binning + 0.5 - y_centers[image_number]))
    centered_image = np.roll(image, (y_shift, x_shift), axis=(0,1))
    if x_shift > 0:
        centered_image[:, :x_shift] = 0
    else:
        centered_image[:, x_shift:] = 0
    if y_shift > 0:
        centered_image[:, :y_shift] = 0
    else:
        centered_image[:, y_shift:] = 0
    return centered_image

def prep_image_display(centered_image, crosshairs=True, binning=1, linewidth=1, half_width=256):
    """Prepares a diffraction pattern to be combined with other diffraction patterns and saved as
    a signle image file.  This includes binning the image to reduce size and adding optional outlines
    and crosshairs."""
    border = linewidth * 2
    binned_image = bin_2D_parallel(centered_image, (binning, binning))
    if crosshairs:
        binned_image[:, half_width - linewidth:half_width + linewidth] = 0
        binned_image[half_width - linewidth:half_width + linewidth, :] = 0
    new = np.pad(binned_image, border, mode='constant')
    return new

def stitch_diffraction_images(dm_path, n_pixels, x_centers, y_centers, blocklist, 
                              patterns_per_block=5, start_image=0, max_images=5, binning=4, border=2, low_percentile=20, high_percentile=90, alpha=1,  
                              filter_size=None, crosshairs=False, artifact=None, dpi=300, cmap='hot', colorbar=False, figsize=False, 
                              save=False, output_directory = None, subdirectory='diffraction_images/', base_filename=''):
    """Plots multiple diffraction patterns next to one another in a grid.  Since plotting single images is tedious and reproducing an
    entire 4D-STEM scan is infeasible, the image is broken into square blocks, generally with 4-100 diffraction patterns each."""
    
    
    dm_file = FileDM(dm_path)
    n_images = dm_file.n_images
    n_grid = int(np.round(n_images**0.5))

    n_blocks = int(np.ceil(n_grid / patterns_per_block))**2
    n_block_rows = int(np.ceil(n_grid / patterns_per_block))

    binned_pattern_size = int(n_pixels / binning)

    block_pixels = patterns_per_block * binned_pattern_size + (patterns_per_block + 1) * border
    blocks = [np.zeros((block_pixels, block_pixels)) for  i in range(n_blocks)]

    target_center = (n_pixels - 1) / 2

    
    
    progress_bar = tqdm(total=n_images, desc='Stitching Images', leave=False)
    for i, image_data in enumerate(dm_file.lazy_access()):
        image, image_number = image_data
        if image_number in blocklist:
            progress_bar.update()
            continue  # Skip blocklisted data, leave intensity marked as zero

        # Determine location of this image
        scan_row,  scan_col = np.divmod(image_number, n_grid)
        extra_block_row, intra_block_row = np.divmod(scan_row, patterns_per_block)
        extra_block_col, intra_block_col = np.divmod(scan_col, patterns_per_block)
        
        block_number = extra_block_row * n_block_rows + extra_block_col
        if block_number < start_image or block_number >= start_image + max_images:
            progress_bar.update()
            continue  # This image won't be needed

        # Center image to nearest integer pixel
        yc, xc = y_centers[i], x_centers[i]
        shift = (int(np.round(target_center - yc)), int(np.round(target_center - xc)))
        centered_image = np.roll(
            np.roll(
            image, shift[0], axis=0), # Vertical direction in stitched image, positive moves image farther down
            shift[1], axis=1)  # Horizontal direction in stitched image, positive moves image farther right
        
        # Apply median filter
        if filter_size is not None:
            centered_image = sig.medfilt(centered_image, kernel_size=filter_size)
        
        # Remove Artifact
        if artifact is not None:
            centered_image = centered_image - artifact
        
        # Add crosshairs
        if crosshairs:
            start = int(target_center - 5.5)
            end = int(target_center + 6.5)
            centered_image[:, start:end] = 10**10
            centered_image[start:end, :] = 10**10
        binned_image = bin_2D_parallel(centered_image, (binning, binning))

        # Add image to block
        block_number = extra_block_row * n_block_rows + extra_block_col
        block_y = intra_block_row * (binned_pattern_size + border) + border
        block_x = intra_block_col * (binned_pattern_size + border) + border
        
        # Clip at zero
        binned_image = np.clip(binned_image, 0, 10**10)
        
        # Add nonlinear scaling
        binned_image = binned_image**alpha
        
        # Write block to list
        try:
            blocks[block_number][block_y:block_y + binned_pattern_size, 
                                 block_x:block_x + binned_pattern_size] = binned_image
        except IndexError:
            pass
        
        progress_bar.update()
    
    
    vmax = np.percentile(blocks[start_image:start_image + max_images], high_percentile)
    vmin = np.percentile(blocks[start_image:start_image + max_images], low_percentile)

    for i, b in enumerate(blocks):
        if i < start_image:
            continue
        if i >= max_images + start_image:
            break
        if figsize:
            plt.figure(figsize=figsize)  # Should be a length 2 tuple
        plt.imshow(b, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.title('Block %d' % i)
        plt.axis('off')
        if colorbar:
            plt.colorbar()
        if save:
            directory = os.path.join(output_directory, subdirectory)
            try:
                os.mkdir(directory)
            except FileNotFoundError:
                print('Make sure you have created the parent folder!!!')
                raise
            except FileExistsError:
                pass
            full_path = output_directory + subdirectory + base_filename + '%dx%d Block %d %s.png' % (
                patterns_per_block, patterns_per_block, i, 'with crosshairs'*crosshairs)
            plt.gcf().savefig(full_path, bbox_inches='tight', dpi=dpi)
        plt.show()
        
def check_diffraction(image, image_number, x_centers, y_centers, centered_masks, binning=4, filter_size=None, background=None, artifact=None, 
                      crosshairs=True, vmin=0, vmax=20, colorbar=False, big=False, save=False, path=None):
        """This function uses the inputs of integrate_diffraction to show the step-by-step process of preprocessing the data for a small number of 
        diffraction patterns.  The patterns shown should mostly match the output of integrate_diffraction, though there may be some rounding error
        since this function rounds to the nearest pixel when centering.  This way of preprocessing the data is more intuitive and shows more intermediate
        steps, but is overall slower.  Thus, it is mostly used for spot checks and troubleshooting."""
    
        n_rows = image.shape[0]
        half_width = int(n_rows / binning / 2)
        
        if background is None:
            background = np.zeros(image.shape)
        if artifact is None:
            artifact = np.zeros(image.shape)
        
        # Item 0: Raw Image
        item_0 = prep_image_display(image, half_width=half_width, binning=binning, crosshairs=crosshairs)
        
        # Item 1: Centered Image
        centered_image = center_image(image, image_number, x_centers, y_centers, half_width=half_width, binning=binning)
        item_1 = prep_image_display(centered_image, half_width=half_width, binning=binning, crosshairs=crosshairs)
        
        # Item 2: Background Subtracted Image
        item_2 = prep_image_display(centered_image - background, half_width=half_width, binning=binning, crosshairs=crosshairs)
        
        # Item 3: Artifact Subtracted Image
        item_3 = prep_image_display(centered_image - background - artifact, half_width=half_width, binning=binning, crosshairs=crosshairs)
        
        # Item 4: Median Filter
        if filter_size:
            filtered_image = sig.medfilt(centered_image, kernel_size=filter_size)  - background - artifact
        else:
            filtered_image = centered_image - background - artifact
        item_4 = prep_image_display(filtered_image, half_width=half_width, binning=binning, crosshairs=crosshairs)
        
        # Item 5: Region of Interest
        region_image = np.zeros((n_rows, n_rows))
        for i, j, mask in centered_masks:
            region_image[mask] = filtered_image[mask]
        item_5 = prep_image_display(region_image, half_width=half_width, binning=binning, crosshairs=crosshairs)
        
        
        # Item 6: Integrated
        integrated_image = np.zeros((n_rows, n_rows))
        for i, j, mask in centered_masks:
            integrated_image[mask] = np.mean(filtered_image[mask])
        item_6 = prep_image_display(integrated_image, half_width=half_width, binning=binning, crosshairs=crosshairs)
        
        subplots = []
        if big:
            fig, axs = plt.subplots(2, 4, figsize=(20, 10))
            for i in range(2):
                for j in range(4):
                    subplots.append(axs[i, j])
        else:
            fig, axs = plt.subplots(1, 7, figsize=(18, 4))
            for i in range(7):
                subplots.append(axs[i])
            
        subplots[0].imshow(item_0, vmin=vmin, vmax=vmax)
        subplots[1].imshow(item_1, vmin=vmin, vmax=vmax)
        subplots[2].imshow(item_2, vmin=vmin, vmax=vmax)
        subplots[3].imshow(item_3, vmin=vmin, vmax=vmax)
        subplots[4].imshow(item_4, vmin=vmin, vmax=vmax)
        subplots[5].imshow(item_5, vmin=vmin, vmax=vmax)
        subplots[6].imshow(item_6, vmin=vmin, vmax=vmax)

        subplots[0].title.set_text('Raw Image')
        subplots[1].title.set_text('Center')
        subplots[2].title.set_text('Remove \n Background')
        subplots[3].title.set_text('Remove Streak')
        subplots[4].title.set_text('Median Filter')
        subplots[5].title.set_text('Set Range')
        subplots[6].title.set_text('Integrate')
        
        if colorbar:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm))
        
        for a in subplots:
            a.axis('off')
            
        if save:
            plt.savefig(path + 'check_diffraction%.0f.png' % time.time())
            
        plt.show()
        
        
###################################################################################################################        
# Class for dm3 and dm4 files
###################################################################################################################

class FileDM(dm.fileDM):  # Builds upon a parent class with the same name from ncempy module
    """This class derives from the ncempy fileDM class.  It adds a lazy_access method to reduce RAM use by reading one image at a time.  
    It also adds a wrap option that corrects for an error on the TitanX microscope at NCEM."""
    def __init__(self, full_path):
        super().__init__(full_path)
        
        self.n_images = 0
        self.index = 0
        
        self.parseHeader() # Ncempy method to get metadata from file.  Note that conventions are inconsistent.
        self.interpret_header()

        
        
    def interpret_header(self, verbose=False):
        # Different instruments/scans appear to use different conventions for creating dm4 files.
        # This routine accounts for those differences.
        self.head = {'ValidNumberElements': 2} # Avoids a bug from some obsolete code in ncempy

        if '.ImageList.2.ImageTags.Series.stepsizex' in self.allTags:
            # Use convention #1, which works for TitanX
            self.y_pixels = int(self.allTags['.ImageList.2.ImageData.Dimensions.1'])
            self.x_pixels = int(self.allTags['.ImageList.2.ImageData.Dimensions.2'])

            self.y_grid = int(self.allTags['.ImageList.2.ImageTags.Series.nimagesy'])
            self.x_grid = int(self.allTags['.ImageList.2.ImageTags.Series.nimagesx'])
            # self.n_images = self.allTags['.ImageList.2.ImageData.Dimensions.3']

            self.y_step_size = self.allTags['.ImageList.2.ImageTags.Series.stepsizex']
            self.x_step_size = self.allTags['.ImageList.2.ImageTags.Series.stepsizey']
        else:
            # Use contention #2, which works for TEAM I
            self.y_grid = self.allTags['.ImageList.2.ImageData.Dimensions.1']
            self.x_grid = self.allTags['.ImageList.2.ImageData.Dimensions.2']

            self.y_pixels = self.allTags['.ImageList.2.ImageData.Dimensions.3']
            self.x_pixels = self.allTags['.ImageList.2.ImageData.Dimensions.4']

            self.y_step_size = self.allTags['.ImageList.2.ImageTags.Session Info.Items.1.Data Type']
            self.x_step_size = self.allTags['.ImageList.2.ImageTags.Session Info.Items.2.Data Type']
        self.image_size = int(self.x_pixels * self.y_pixels)
        self.n_images = int(self.x_grid * self.y_grid)

        if verbose:
            print('Images are %d x %d pixels' % (self.x_pixels, self.y_pixels))
            print('Scan was made over a %d x %d rectangle with (%d, %d)nm step size.'% (self.x_grid, self.y_grid, self.x_step_size, self.y_step_size))
            
            
    def readSingleImage(self, index, image_number=0, next_image=False):
        '''Retrieves a single image from a dm3/dm4 file.  Header must already be parsed.'''

        # The first dataset is usually a thumbnail. Test for this and skip the thumbnail automatically
        if self.numObjects == 1:
            ii = index
        else:
            ii = index + 1

        # Position Reader
        if next_image:
            pass  # Already in place - no need to seek
        else:
            self.seek(self.fid, self.dataOffset[ii] + image_number*self.image_size * 2, 0) #Seek to start of dataset from beginning of the file

        # Read Data
        data = self.fromfile(self.fid, count=self.image_size, dtype=self._DM2NPDataType(
            self.dataType[ii])).reshape((self.y_pixels, self.x_pixels))

        return data

                
    def lazy_access(self, wrap=-2):
        """Generator that reads one image at a time from a DM3 or DM4 file.  Setting 'wrap=-2' corrects a known error with the TitanX
        microscope at NCEM where the first two rows of images are placed at the end instead of at the start.  Yields one image from the stack with its
        corresponding image number after correction."""
        
        # Position read head
        self.seek(self.fid, self.dataOffset[self.index], 0)  # Seek to start of dataset from beginning of the file
        
        for image_number in range(self.n_images):
            shifted_image_number = np.divmod(image_number + wrap, self.n_images)[1]  # 2 rows on wrong side

            square_array = self.readSingleImage(self.index, image_number)
                    
            yield square_array, shifted_image_number  # If you're not familiar with the yield command, google Python Generators.
            
                