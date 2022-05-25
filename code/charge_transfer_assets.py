# Import External Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import pathlib
import random
import importlib as imp
import scipy.stats as stats
import scipy.optimize as opt
import scipy.linalg as linalg
import itertools as iter
import math
from tqdm.notebook import tqdm, trange
# from tqdm import tqdm, trange
import ipywidgets
import datetime
import time
import json
import lmfit
import psutil
import imageio  # for gifs

# Import Custom Libraries
import flow_fields as flow
for mod in [flow]:
    imp.reload(mod)

##########################################################################################
# Chain Set Class
##########################################################################################

class ChainSet:
    """A set of descritized polymer chains defined by the xyz coordinates of their subunits. The class has four main utilities:
    1. Organizes polymer beads into bins.  Users can quickly find the contents of each bin and therefore the neighbors of each bead
    2. Organizes functions for visualizing, analyzing and troubleshooting the set of chains
    3. Supports charge transport simulations, especially lookups of chains, beads, and bins, and selections of neighbors for interchain hops.  
    4. Performs simple modifications of chain structures
    
    Requirements:
        Chains must all be the same length.  This class freqently makes use of the fact that (bead_number // chain_length == chain_number), and generalizing
        the class to polydisperse chains would take some time.  
    
    Inputs:
        xyz: (n x 3) array of polymer beads.  Beads on the same chain should be ordered and consecutive.  
        chain_length: number of beads in each chain
        xyz_full_path: full path of input file.  This is used for bookkeeping purposes only.  
        
        The following keyword arguments modify the chain structure for specific types of simulations:
        shuffle: If True, randomly rearrange some chains to disrupt long-range structure
        align: If True, align chains so that the direction of their Q-tensors all point in the same direction
        make_rigid: If True, extend some fraction of chains into rigid rods while keeping their centroids constant
        modify_fraction: fraction of chains modified for shuffle, align and/or make_rigid.  If multiple options are chosen, the same chains will be modified for all modifications.
    """
    
    def __init__(self, xyz, chain_length, xyz_full_path, box_size=(800, 800, 30), shuffle=False, align=False, make_rigid=False, modify_fraction=1.0):
        self.xyz = np.array(xyz)  # Copy so that xyz can be modified for this instance without changing variable
        self.chain_length = chain_length
        self.xyz_full_path = xyz_full_path
        self.file_number = int(self.xyz_full_path.split('v')[-1]) - 1  # For my input files, the input files end with an integer that describes their degree of alignment
        self.box_size = np.array(box_size)
        
        # Get Data Dimensions
        self.n_beads = xyz.shape[0]
        self.n_chains = int(np.round(self.n_beads / chain_length))  
        self.all_chains = self.xyz.reshape((self.n_chains, self.chain_length, 3))
        self.chain_starts = ((np.arange(self.n_beads) // self.chain_length) * self.chain_length).astype(int)
        
        # Mark out-of-bounds beads and create alternate array where all beads are forced in-bounds
        self.xyz_clipped = np.array(self.xyz)
        for dim in range(3):
            self.xyz_clipped[:, dim] = np.clip(self.xyz[:, dim], 0, box_size[dim] - 0.001)
        self.in_bounds = np.where(self.xyz_clipped == self.xyz)[0]
        self.in_bounds_2d = np.where((self.xyz_clipped[:, 0] == self.xyz[:, 0]) & (self.xyz_clipped[:, 1] == self.xyz[:, 1]))[0]
        self.out_bounds_2d = np.where((self.xyz_clipped[:, 0] != self.xyz[:, 0]) | (self.xyz_clipped[:, 1] != self.xyz[:, 1]))[0]
        
        # Set up arrays for binning data
        self.box_size = box_size
        self.shape = None
        self.default_shape = None
        self.bin_size = None
        self.n_bins = None
        self.x_bins = None  # number of bins in x dimension.  
        self.max_dim = None  # nm
        
        self.bins_by_bead_3d = None
        self.numbering_system_3d = None
        self.numbering_system_2d = None
        self.bins_by_bead = None
        
        self.by_bin = None
        self.beads_by_bin = None
        self.bin_sizes = None
        self.bin_ends = None
        self.bin_starts = None
        
        #######################################
        # Optional Structural Modifications
        #######################################
        
        self.shuffle = shuffle
        self.align = align
        self.make_rigid = make_rigid
        self.modify_fraction = modify_fraction
        
        # Choose chains to modify
        rnd = np.random.random(self.n_chains)
        n_chains_to_modify = int(self.n_chains * modify_fraction)
        chains_to_modify = np.argsort(rnd)[:n_chains_to_modify]
        
        # Compute chain centroids and relative bead positions
        chain_centroids = np.mean(self.all_chains, axis=1)
        chains_minus_centroids = self.all_chains - chain_centroids[:, None, :]
        
        # Apply modifications
        if shuffle:
            # Randomly reorganize chains to destroy mesoscale order while keeping single-chain geometries
            
            # Compute chain centroids and relative bead positions - should be computed before each modification in case they change
            chain_centroids = np.mean(self.all_chains, axis=1)
            chains_minus_centroids = self.all_chains - chain_centroids[:, None, :]
            
            # Apply Shuffle
            rnd_2 = np.random.random(n_chains_to_modify)
            new_order = np.argsort(rnd_2)
            new_indices = chains_to_modify[new_order]
            
            # Swap Chains
            self.all_chains[chains_to_modify, :, :] = (chain_centroids[chains_to_modify, None, :] + 
                                                        chains_minus_centroids[chains_to_modify, :, :][new_order, :, :])
            
        if make_rigid:
            # Extend chains into rigid rods along their current Q-tensor angle
            
            # Compute chain centroids and relative bead positions - should be computed before each modification in case they change
            chain_centroids = np.mean(self.all_chains, axis=1)
            chains_minus_centroids = self.all_chains - chain_centroids[:, None, :]
            chain_angles = self.get_chain_angles()
            
            # Apply make_rigid
            bead_distances = np.linspace(-(self.chain_length - 1) / 2.0, (self.chain_length - 1) / 2.0, self.chain_length, endpoint=True)
            self.all_chains[chains_to_modify, :, 0] = chain_centroids[chains_to_modify, None, 0] + np.cos(chain_angles[chains_to_modify])[:, None] @ bead_distances[None, :]
            self.all_chains[chains_to_modify, :, 1] = chain_centroids[chains_to_modify, None, 1] + np.sin(chain_angles[chains_to_modify])[:, None] @ bead_distances[None, :]
            self.all_chains[chains_to_modify, :, 2] = chain_centroids[chains_to_modify, None, 2]
            
        if align:
            # Set the Q-tensor angles of chains to zero degrees
            
            # Compute chain centroids and relative bead positions - should be computed before each modification in case they change
            chain_centroids = np.mean(self.all_chains, axis=1)
            chains_minus_centroids = self.all_chains - chain_centroids[:, None, :]
            chain_angles = self.get_chain_angles()

            # Apply Alignment
            rotation_matrix = np.transpose(np.stack([np.cos(chain_angles), -np.sin(chain_angles), np.sin(chain_angles), np.cos(chain_angles)])).reshape((self.n_chains, 2, 2))
            rotated_chains_minus_centroids = np.zeros((self.n_chains, self.chain_length, 2))
            for i in range(self.n_chains):
                rotated_chains_minus_centroids[i, :, :] = chains_minus_centroids[i, :, :2] @ rotation_matrix[i, :, :]
            self.all_chains[chains_to_modify, :, :2] = chain_centroids[chains_to_modify, None, :2] + rotated_chains_minus_centroids[chains_to_modify, :, :]   
        
        # Clean up changes
        if shuffle or align or make_rigid:
            self.xyz = self.all_chains.reshape((self.n_beads, 3))
            
    
    #############################################################################################
    # Setup Methods - Should be run after creating the class, and again to modify if needed
    #############################################################################################
        
    def create_bins(self, shape, force_in_bounds=True):
        """Set up grid and numbering system. This can be used to analyze the structure (i.e. RMS density variation)
        and to find neighbors of each bead in charge transport simulations.  Running this method again can overwrite 
        with a new grid.
        
        Inputs:
            shape: Length 3 Tuple with desired number of bins in each dimension.
            force_in_bounds: out-of-bounds beads are assigned to the nearest bin
        """
        self.shape = np.array(shape)  # Number of bins in each of three dimensions
        self.bin_size = self.box_size / self.shape
        self.n_bins = np.product(self.shape)
        self.max_dim = np.max(self.shape * self.bin_size)
        self.x_bins = self.shape[0]
        
        # If this is the first time this function has been run for this ChainSet, set this shape as default
        if self.default_shape is None:
            self.default_shape = np.array(self.shape)
            
        # Choose how to treat out-of-bounds beads
        coordinates = self.xyz_clipped if force_in_bounds else self.xyz
        
        # Number bins
        self.bins_by_bead_3d = np.floor_divide(coordinates, self.bin_size[None, :]).astype(int)
        self.numbering_system_2d = (self.shape[1], 1, 0)
        self.numbering_system_3d = (self.shape[2]*shape[1], self.shape[2], 1)
        self.bins_by_bead = self.bins_by_bead_3d @ self.numbering_system_3d # rasters over pillar, then column, then row
        
        # Set up tables for binning data
        self.by_bin = np.argsort(self.bins_by_bead)
        self.beads_by_bin = np.arange(self.n_beads)[self.by_bin]
        self.bin_sizes = np.bincount(self.bins_by_bead, minlength=self.n_bins)
        self.bin_ends = np.cumsum(self.bin_sizes)
        self.bin_starts = np.roll(self.bin_ends, 1)
        self.bin_starts[0] = 0  
        
    #############################################################################################
    # Methods used in Charge Transport simulations
    #############################################################################################
            
    def get_random_neighbor(self, beads): 
        """Select and return a random neighbor of each bead in an imput array. 
        
        A "neighbor" is any bead in the same bin as the input bead.  
        It is possible for the random neighbor to be the original bead, or a bead on the same chain as the original bead.  
        
        Prerequisites:
            Run the create_bins method to set up bins and numbering system.
        Inputs:
            beads -- array of arbitrary size, filled with bead numbers as integers
        Outputs:
            new_beads -- array of the same shape as the input array, filled with bead numbers as integers.
        """
        
        bins = self.bins_by_bead[beads]
        bin_sizes = self.bin_sizes[bins]
        rnd = np.random.rand(*bins.shape)
        new_bead_indices = (self.bin_starts[bins] + np.floor(rnd * bin_sizes)).astype(int)
        new_beads = self.beads_by_bin[new_bead_indices]
        return new_beads
        
    def get_random_bead(self, bins):
        """Select and return a random bead from each of the specified bins.
        
        Prerequisites:
            Run the create_bins method to set up bins and numbering system.
        Inputs:
            bins -- array of bin numbers as integers
        Outputs
            new_beads -- array of beads as integers, with the same shape as the input array
        """
        
        bin_sizes = self.bin_sizes[bins]
        rnd = np.random.rand(*bins.shape)
        new_bead_indices = (self.bin_starts[bins] + np.floor(rnd * bin_sizes)).astype(int)
        new_beads = self.beads_by_bin[new_bead_indices]
        return new_beads
    
    #############################################################################################
    # Methods for Getting Geometry Data
    #############################################################################################
        
    def bin_data(self, data, fun=np.mean, n_output_dims=2):
        """Transforms data as a function of bead into data as a function of bin.
        
        Prerequisites:
            Run the create_bins method to set up bins and numbering system.
        Inputs:
            data -- 1D array with size equal to the number of beads.  For example, this might be
            the number of times a charge visited each bead.
            fun -- function for aggregating the data (default is np.mean)
            n_output_dims -- defines whether bins at the same (x, y) position but different (z) positions
                will be treated together or separately.  Options: 2, 3
        Outputs
            output -- 1D array of size equal to the number of bins, where each value is the aggregated data
                for that bin.  
        """
        
        # Set up bins (this could be combined with similar code elsewhere)
        if n_output_dims == 2: # Return data binned into 2d 
            n_bins = self.shape[0] * self.shape[1]
            numbering_system = (self.shape[1], 1, 0)
        else:
            n_bins = self.shape[0] * self.shape[1] * self.shape[2]
            numbering_system = (shape[2]*shape[1], shape[2], 1) # rasters over pillar, then column, then row
            
            
        bins_by_bead = self.bins_by_bead_3d @ numbering_system  # Gives each bin a unique bin number 
        bins_by_bead[self.out_bounds_2d] = n_bins  # set out-of-bounds beads to higher bead number so they can be excluded
        bin_sizes = np.bincount(bins_by_bead, minlength=n_bins)
        
        if len(bin_sizes) > n_bins:  # Out-of-bounds bins are present
            bin_sizes = bin_sizes[:-1]  # Trim last value, which is the number of out-of-bounds beads
        bin_ends = np.cumsum(bin_sizes)
        bin_starts = np.roll(bin_ends, 1)
        bin_starts[0] = 0
        
        output = np.zeros(n_bins)
        sorted_data = data[np.argsort(bins_by_bead)]
        for i, start, end in zip(iter.count(), bin_starts, bin_ends):
            if end != start:  # Non-empty slice - zeros will remain zero
                output[i] = fun(sorted_data[start:end])

        return output   
    
    def bin_points(self, xyz, dims=3):
        """Accepts an arbitrary set of xyz coordinates, and returns the corresponding bin for each point.
        
        Prerequisites:
            Run the create_bins method to set up bins and numbering system.
        Inputs:
            xyz -- nx3 array of xyz coordinates
            n_output_dims -- defines whether bins at the same (x, y) position but different (z) positions
                will be treated together or separately.  Options: 2, 3
        Outputs
            bins_by_point -- 1D array of size equal to the number of input coordinates, containing the bin
                number corresponding to each coordinate point.  
        """

        bins_by_point_3d = xyz // self.bin_size[None, :]
        bins_by_point_3d = np.clip(bins_by_point_3d, 0, 79)  # Shouldn't be needed, but the loop below isn't working
        for dim in range(3):
            bins_by_point_3d[:, dim] = np.clip(bins_by_point_3d[:, dim], 0, self.shape[dim] - 1)
        if dims == 3:
            bins_by_point = bins_by_point_3d @ self.numbering_system_3d
        else:
            bins_by_point = bins_by_point_3d @ self.numbering_system_2d
        
        return bins_by_point.astype(int)
    
    def bin_density(self):
        """Return the number of beads in each column of bins."""
        return self.bin_data(np.ones(self.n_beads), fun=np.sum, n_output_dims=2)
    
    def get_chain_angles(self):
        """Return the average angle of each chain.
        
        Returns:
            1D Array of length n_chains containing the average angle of each chain (in radians)
        
        Note: Computing an average of angles is tricky because of the periodicity present.  For example, the average of 
        1 degree 359 degrees and  should really be 0 degrees, not 180 degrees as would be given by a standard average.
        This problem is addressed in Circular Statistics, which defines a Circular Mean that gives the intuitive result.
        Our problem is one step trickier - since diffraction patterns are rotationally symmetric, the average of 1 degree
        and 179 degrees should really be 0 degrees, not 90 degrees.  This program treats this problem in two ways, which are
        mathematically equivalent to each other.  The method get_chain_angles uses the circular mean with a change of 
        coordintes to account for symmetry, while the method get_bin_angles uses tensors to calculate the average angle.  
        In two dimensions, these approaches are mathematically equivalent."""
        
        # Determine angles of each line segment
        dx = np.diff(self.all_chains[:, :, 0])
        dy = np.diff(self.all_chains[:, :, 1])
        theta = np.arctan2(dy, dx)
        
        # Change periodicity for quadrupole symmetry
        theta_2 = 2 * theta
        dx_2 = np.cos(theta_2)
        dy_2 = np.sin(theta_2)
        dx_2_avg = np.mean(dx_2, axis=1)
        dy_2_avg = np.mean(dy_2, axis=1)
        theta_2_avg = np.arctan2(dy_2_avg, dx_2_avg)
        theta_avg = theta_2_avg / 2
        
        return theta_avg
    
    def get_bin_angles(self):
        """Return the average angle of each bin.
        
        Returns:
            1D Array of length n_bins containing the average angle of each bin (in radians)
        
        Note: Computing an average of angles is tricky because of the periodicity present.  For example, the average of 
        1 degree 359 degrees and  should really be 0 degrees, not 180 degrees as would be given by a standard average.
        This problem is addressed in Circular Statistics, which defines a Circular Mean that gives the intuitive result.
        Our problem is one step trickier - since diffraction patterns are rotationally symmetric, the average of 1 degree
        and 179 degrees should really be 0 degrees, not 90 degrees.  This program treats this problem in two ways, which are
        mathematically equivalent to each other.  The method get_chain_angles uses the circular mean with a change of 
        coordintes to account for symmetry, while the method get_bin_angles uses tensors to calculate the average angle.  
        In two dimensions, these approaches are mathematically equivalent."""
        
        # Find position and bin of each segment
        segment_positions = (self.all_chains[:, 1:, :] + self.all_chains[:, :-1, :]) / 2
        
        
        # Find angle of each chain segment
        dx = np.diff(self.all_chains[:, :, 0])
        dy = np.diff(self.all_chains[:, :, 1])
        theta = np.arctan2(dy, dx)

        # Construct tensor
        direction_tensor = np.zeros((self.n_chains, self.chain_length-1, 2, 2))
        direction_tensor[:, :, 0, 0] = np.cos(theta)**2 - 0.5
        direction_tensor[:, :, 1, 1] = np.sin(theta)**2 - 0.5
        direction_tensor[:, :, 0, 1] = np.sin(theta) * np.cos(theta)
        direction_tensor[:, :, 1, 0] = np.sin(theta) * np.cos(theta)
        
        # Individual Beads
        direction_tensor_padded = np.pad(direction_tensor, [(0, 0), (1, 1), (0, 0), (0, 0)], mode='edge')
        direction_tensor_by_chain = (direction_tensor_padded[:, 1:, :, :] + direction_tensor_padded[:, :-1, :, :]) / 2
        direction_tensor_by_bead = direction_tensor_by_chain.reshape((self.n_beads, 2, 2))

        # Number bins in 2d
        n_bins_2d = self.shape[0] * self.shape[1]
        numbering_system = (self.shape[1], 1, 0)
        bins_by_bead = self.bins_by_bead_3d @ numbering_system

        # Get average direction tensor by bin and convert to angle
        theta_by_bin = np.zeros(n_bins_2d)
        for i in range(n_bins_2d):
            beads_to_average = np.where(i == bins_by_bead)[0]
            bin_tensor = np.mean(direction_tensor_by_bead[beads_to_average, :, :], axis=0)
            sin_2_theta = bin_tensor[1, 0]
            cos_2_theta = bin_tensor[0, 0] - bin_tensor[1, 1]
            theta_by_bin[i] = np.arctan2(sin_2_theta, cos_2_theta) / 2
        
        return theta_by_bin

    
    def in_bounds(self, points, n_dims=3):
        """Returns True for points in an array that are inside the box defined by self.box_size, otherwise returns False.
        
        Inputs:
            points: nx3 array of xyz coordinates
            n_dims: number of dimensions to check.  For example, if n_dims=2, points that are out-of-bounds in the 3rd dimension still return True.  
        
        Outputs:
            Array of boolean values"""
        
        

        
    #############################################################################################
    # Plotting Methods
    #############################################################################################
        
    def plot_radius_of_gyration(self, **kwargs):
        """Computes the radius of gyration of each individual chain.  Plots a histogram of these values as well as the mean.  
        This function was broken in the past - please test before using."""
        chain_centroids = np.mean(self.all_chains, axis=1)
        r = self.all_chains - chain_centroids[:, None, :]
        d = np.sum(r**2, axis=2)**0.5
        rg = np.sqrt(np.sum(d**2, axis=1) / self.chain_length)
        hist, bins = np.histogram(rg)
        mean = np.mean(rg)
        bin_centers = (bins[1:] + bins[:-1]) / 2
        plt.bar(bin_centers, hist / self.n_chains)
        plt.axvline(x=mean, color=(1, 0, 0))
        plt.xlabel('Radius of Gyration (nm)')
        plt.ylabel('Fraction of Chains')
        plt.show()
        
    def plot_density(self, **kwargs):
        """Creates a heat plot of the number of beads in each bin.  Also shows the mean and standard deviation of beads per bin."""
        output = self.bin_density()
        mean_density = np.mean(output)
        stdev = np.std(output)
        if 'vmin' not in kwargs:
            kwargs['vmin'] = mean_density - stdev
        if 'vmax' not in kwargs:
            kwargs['vmax'] = mean_density + stdev
        plt.imshow(np.transpose(output.reshape((self.x_bins, self.x_bins))), origin='lower', **kwargs)
        plt.colorbar()
        plt.axis('off')
        plot=plt.gcf()

        print('Density: %.0f +/- %.1f beads/bin (%.1f%%)' % (mean_density, stdev, stdev / mean_density * 100))  # Note: stdev and rms variation are equivalent here
        return plot
    
    def plot_chain_ends(self, **kwargs):
        """Creates a map of all chain ends.  Clusters of chain ends could be interesting if they are present."""
        start_coordinates = self.all_chains[:, 0, :2] 
        end_coordinates = self.all_chains[:, -1, :2]
        all_ends = np.concatenate([start_coordinates, end_coordinates], axis=0)
        print(all_ends.shape)
        plt.plot(all_ends[:, 0], all_ends[:, 1], linewidth=0, marker='.', **kwargs)
        plt.xlim(0, self.max_dim)
        plt.ylim(0, self.max_dim)
        return plt.gca()
    
    def plot_lines(self, grid=None, thinning=1, linewidth=0.05, method='Color by Bin', bin_size=10, alpha=1): 
        """Creates map of polymer chains, with a variety of formatting options.  
        
        Inputs:
            thinning -- reduces the number of chains drawn to improve visualization.  For example, thinning=20 would draw 1/20 of chains
            grid -- reduces the number of chains drawn to improve visualization.  Overrides "thinning."  Instead of pseudo-randomly selecting
                    chains to draw, creates a grid with one chain drawn at each grid point.  Should be a tuple of ints, length 2.  
                    For example, grid=(10,10) would draw 100 chains, evenly distributed.  
            linewidth -- weight of lines drawn.  Int or float.  
            method -- the method for using color in the figure.  There are three options: 'Color by Chain', 'Color by Bead', or 'Color by Bin'.
            bin_size -- selects the size for bins if "method=Color by Bin" is chosen.  
            alpha -- selects the bin color transparency if "method=Color by Bin" is chosen.  
        
        Recommended Format Settings:
            Default settings - draws all polymer chains
            (grid=(15, 15), linewidth=2, method='Color by Bin', alpha=0.5) - draws grid of polymer chains with background showing average alignment
        """        
        
        # Determine which chains to draw
        if grid is not None:
            # Draw a small number of evenly distributed chains
            x_targets = self.shape[0] // (grid[0] + 1) * np.arange(1, grid[0] + 1) * self.bin_size[0]
            y_targets = self.shape[1] // (grid[1] + 1) * np.arange(1, grid[1] + 1) * self.bin_size[1]
            x_centers = np.mean(self.all_chains[:, :, 0], axis=1)
            y_centers = np.mean(self.all_chains[:, :, 1], axis=1)
            chains_to_draw = []
            for x in x_targets:
                for y in y_targets:
                    objective = (x_centers - x)**2 + (y_centers - y)**2
                    nearest_chain = np.argmin(objective)
                    chains_to_draw.append(nearest_chain)
        else:
            # Pseudo-randomly choose chains to draw
            chains_to_draw = [i for i in range(self.n_chains) if i % thinning == 0]
        
        # Determine line colors
        if method == 'Color by Chain':
            theta_avg = self.get_chain_angles[chains_to_draw]
            chain_colors = flow.color_by_angle(np.ndarray.flatten(theta_avg) * 180/np.pi)
            chain_color_tuples = [i for i in zip(*chain_colors)]
        elif method == 'Color by Bead':
            # Color each segment individually
            dx = np.diff(self.all_chains[:, :, 0])
            dy = np.diff(self.all_chains[:, :, 1])
            theta = np.arctan2(dy, dx)
            segment_colors = flow.color_by_angle(np.ndarray.flatten(theta) * 180/np.pi)
            segment_color_tuples = [i for i in zip(*segment_colors)]
        elif method == 'Color by Bin':
            theta_by_bin = self.get_bin_angles()
                
            length, width, height = self.shape
            r, g, b = flow.color_by_angle(theta_by_bin * 180/np.pi)
            bin_colors = np.stack([np.transpose(r.reshape((length, width))), 
                                   np.transpose(g.reshape((length, width))),
                                   np.transpose(b.reshape((length, width))),
                                   np.ones((self.x_bins, self.x_bins)) * alpha], axis=2) # Alpha
                                   # np.transpose(S_by_bin.reshape((length, width)))], axis=2)
            
            # Set line color to dark grey
            chain_color_tuples = [(0.1, 0.1, 0.1)] * self.n_chains          
        
        # Break chains into segments and add color        
        segments_to_draw = []
        colors_to_draw = []     
        for c in chains_to_draw:
            for b in range(self.chain_length - 1):
                new_line = [(self.all_chains[c, b,     0], self.all_chains[c, b,     1]), 
                            (self.all_chains[c, b + 1, 0], self.all_chains[c, b + 1, 1])]
                segments_to_draw.append(new_line)

                if method == 'Color by Chain' or method == 'Color by Bin':
                    colors_to_draw.append(chain_color_tuples[c]) 
                elif method == 'Color by Bead':
                    segment_number = c * (self.chain_length - 1) + b
                    colors_to_draw.append(segment_color_tuples[segment_number])   
        
        # Plot Lines
        fig, ax = plt.subplots(figsize=(12, 12))
        if method == 'Color by Bin':
            plt.imshow(bin_colors, origin='lower', extent=[0, self.max_dim, 0, self.max_dim])
        line_plot = mc.LineCollection(segments_to_draw, colors=colors_to_draw, linewidth=linewidth)
        ax.add_collection(line_plot)
        plt.autoscale(enable=True, axis='y')
        plt.autoscale(enable=True, axis='x')
        plt.axis('off')
        
        return plt.gcf()
        
    def plot_segment_angle_distribution(self, n_bins=12):
        """Creates a histogram of segment angles present in chains.  Also gives ratio of segments which are more horizontal to more vertical, i.e. <45 degrees or no. """ 
        dx = np.diff(self.all_chains[:, :, 0])
        dy = np.diff(self.all_chains[:, :, 1])
        theta = np.arctan2(dy, dx)
        degrees = theta * 180 / np.pi
        degrees = ((degrees + 90) % 180) - 90  # Domain of -90 to 90
        hist, bin_edges = np.histogram(degrees, bins=n_bins)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        bin_size = bin_edges[1] - bin_edges[0]
        
        group_1 = np.where((bin_centers > -45) & (bin_centers < 45))
        group_2 = np.where((bin_centers < -45) | (bin_centers > 45))
        
        plt.bar(bin_centers[group_1], hist[group_1], width=0.9*bin_size)
        plt.bar(bin_centers[group_2], hist[group_2], width=0.9*bin_size)
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Number of Chain Segments')
        plt.xlim(-90, 90)
        plt.xticks(np.arange(-90, 90, 30))
        plt.legend(['More Horizontal', 'More Vertical'])
        plt.ylim(0, np.max(hist) * 1.5)
        
        fraction_more_horizontal = np.sum(hist[group_1]) / np.sum(hist)
        plt.show()
        
        print('Fraction of More Horizontal Segments: %.2f' % fraction_more_horizontal)
        
    def plot_chain_angle_distribution(self, n_bins=12):
        """Creates a histogram of chain angles.  Also gives ratio of chains which are more horizontal to more vertical, i.e. <45 degrees or no. """ 
        chain_angles = self.get_chain_angles()
        degrees = chain_angles * 180 / np.pi
        degrees = ((degrees + 90) % 180) - 90  # Domain of -90 to 90
        hist, bin_edges = np.histogram(degrees, bins=n_bins)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        bin_size = bin_edges[1] - bin_edges[0]
        
        group_1 = np.where((bin_centers > -45) & (bin_centers < 45))
        group_2 = np.where((bin_centers < -45) | (bin_centers > 45))
        
        plt.bar(bin_centers[group_1], hist[group_1], width=0.9*bin_size)
        plt.bar(bin_centers[group_2], hist[group_2], width=0.9*bin_size)
        plt.xlabel('Circular Mean Angle (degrees)')
        plt.ylabel('Number of Chains')
        plt.xlim(-90, 90)
        plt.xticks(np.arange(-90, 90, 30))
        plt.legend(['More Horizontal', 'More Vertical'])
        plt.ylim(0, np.max(hist) * 1.5)
        
        fraction_more_horizontal = np.sum(hist[group_1]) / np.sum(hist)
        plt.show()
        
        print('Fraction of More Horizontal Chain: %.2f' % fraction_more_horizontal)
        
    def plot_segment_angle_distribution_shifted(self, n_bins=24, y_max=None):
        """Creates a histogram of the angles of segments relative to the average angles of their bins."""
        
        # Get segment angles
        dx = np.diff(self.all_chains[:, :, 0])
        dy = np.diff(self.all_chains[:, :, 1])
        theta = np.arctan2(dy, dx)
        
        # Get bin angles
        bin_angles = self.get_bin_angles()
        
        # Get segment bins
        segment_centers = (self.all_chains[:, 1:, :] + self.all_chains[:, :-1, :]) / 2
        bins_by_segment = self.bin_points(segment_centers, dims=2)
        print(np.max(bins_by_segment))
        
        # Find angle difference
        bin_angles_by_segment = bin_angles[bins_by_segment]
        d_theta = (theta - bin_angles_by_segment) 

        d_theta = (d_theta + np.pi/2) % np.pi - np.pi/2  # Domain of -pi/2 to pi/2
        d_theta_degrees = d_theta * 180 / np.pi
        
        # bin and plot
        hist, bin_edges = np.histogram(d_theta_degrees, bins=n_bins)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        bin_size = bin_edges[1] - bin_edges[0]
        
        plt.bar(bin_centers, hist / np.sum(hist), width=0.8*bin_size)
        plt.xlabel('Relative Angle (degrees)')
        plt.ylabel('Fraction of Segments')
        plt.xlim(-90, 90)
        plt.xticks(np.arange(-90, 90, 30))
        
        if y_max:
            plt.ylim(0, y_max)
        
        plt.gca().set_aspect(180 / plt.gca().get_ylim()[1])
        
        plt.show()
        

##########################################################################################
# Path Experiment Class
##########################################################################################
    
    
class PathExperiment:
    """Runs charge transport experiments, organizes metadata and analyzes results.
    
    Inputs:
        silent -- if True, does not print default outputs to screen
        save_big_arrays -- saves all results from the compute_rates method and all charge moves in the propogate_charges_method.  This adds 10GB or more to file size.  
        
    Primary Methods:
        compute_rates -- pre-compute probabilities of on-chain moves.  This greatly improves performance, since computing probablities on an as-needed basis requires
            ~1ms per calculation.  
        propagate_charges -- place test charges in the region and track their motion
        unpack_results -- perform basic calculations on the output data that is required for some analysis methods. 
        make_default_figures -- Make the most common plots using default values.  Good starting point for analysis or troubleshooting.
        to_file -- save simulation data
        from_file -- load saved simulation data
    """
    def __init__(self, silent=False, save_big_arrays=False):
        self.silent = silent
        self.save_big_arrays = save_big_arrays  # If false, skips saving items in the list self.big_arrays to save memory
        self.big_arrays = ['expm_results', 'all_positions', 'K', 'K_0', 'P_by_bead', 'P_cumulative'] 
        
        # Results of To File/From File methods:
        self.full_path = None
        
        # Inputs of Compute Rates method
        self.chains = None
        self.field = None
        self.angle = None
        self.radians = None
        self.sign = None
        self.energetic_disorder = 0.0  
        
        # Results of Compute Rates method
        self.charge_density = None
        self.binned_potential = None
        self.K = None
        self.K_0 = None  # K for s=0 case (infinite time)
        self.interchain_rate = None
        self.on_chain_rate = None
        self.forward_rates = None
        self.backward_rates = None
        self.short_time_mobilities = None
        self.expm_times = None
        self.expm_results = None
        
        self.P_by_bead = None
        self.P_cumulative = None
        
        # Results of plot_rates_given_time method
        self.ps_mobility = None
        self.ns_mobility = None
        
        # Inputs of Propagate Charges method
        self.track_single_moves = None
        self.bin_size = None
        
        # Results of Propogate Charges method
        self.bead_progress = None
        self.block_times = False
        self.block_lengths = None
        self.time_seconds = False
        self.completion_block = False
        self.fraction_finished = False
        self.positions_with_time = None
        self.density_vs_time = None
        self.finish_blocks = None
        self.finish_distance = None
        self.n_hops = None
        self.finish_line = None
        self.multi_finish_lines = None
        self.multi_finish_distributions = []
        self.end = None
        self.forward_finish_distributions = []
        self.reverse_finish_distributions = []
        self.all_positions = None  
        
        # Results of Unpack Results method
        self.avg_progress = None
        self.avg_velocity = None
        self.grand_avg_velocity = None
        self.distance_per_hop = None
        self.estimated_mobility = None
        self.estimated_mobility_vs_time = None
        
        # Results of Single Move Analysis method
        self.consecutive_moves = None
        self.consecutive_moves_same_bead = None
        self.consecutive_moves_same_chain = None  # Can also be used to find fraction of moves landing on same bead/chain
        self.move_distances = None  # x-axis of histogram, in nm
        self.move_frequencies = None  # y-axis of histogram
        self.avg_forward_distance = None  # Conceptually the same as "distance_per_hop" but computed differently
        self.avg_abs_distance = None
        
        
    #############################################################################################
    # File System Methods
    #############################################################################################
        
    def to_file(self, path, name=None, batch_name=None):
        """Packages and saves data from this simulation.  Simple types are saved as JSON and numpy arrays are saved 
        independently.  The same is done for the accompanying ChainSet object."""
        
        # Create new directory to store this experiment
        dir_name = ''
        if batch_name is not None:
            dir_name += batch_name
        if name is not None:
            dir_name += name
        else:
            dir_name += timestamp()
        folder_path = path + dir_name
        os.mkdir(folder_path)
        self.full_path = folder_path
        
        # Save ChainSet attributes
        chains_simple_attributes = {}
        for key, value in self.chains.__dict__.items():
            if type(value).__name__ == 'ndarray':
                full_path = folder_path + '/' + key
                np.save(full_path, getattr(self.chains, key))
            elif is_jsonable(value):
                chains_simple_attributes[key] = value
        with open(folder_path + '/' + 'chains_simple_attributes.txt', 'w') as outfile:
            json.dump(chains_simple_attributes, outfile)
            
        # Save ChainSet matrix atttributes
        np.save(folder_path + '/' + 'chains_xyz', self.chains.xyz)
        
        # Save PathExperiment Attributes
        simple_attributes = {}
        for key, value in self.__dict__.items():
            if key in self.big_arrays and not self.save_big_arrays:
                # Skip saving these attributes to reduce file size
                continue
            if type(value).__name__ == 'ndarray':
                full_path = folder_path + '/' + key
                np.save(full_path, getattr(self, key))
            elif is_jsonable(value):
                simple_attributes[key] = value
        with open(folder_path + '/' + 'simple_attributes.txt', 'w') as outfile:
            json.dump(simple_attributes, outfile)
        
    
    def from_file(self, folder_path, skip_attrs=None):
        """Loads saved simulation data and copies it to this PathExperiment object and
        the associated ChainSet object."""
        
        # Load simple attributes
        with open(folder_path + '/' + 'simple_attributes.txt', 'r') as file:
            simple_attributes = json.load(file)
        for key, value in simple_attributes.items():
            setattr(self, key, value)
        
        # Handle attributes that will be skipped (usually to save memory)
        if skip_attrs is not None:
            for name in skip_attrs:
                if name not in self.__dict__:
                    print('Cannot skip attribute %s: attribute unknown' % name)
                    raise
        else:
            skip_attrs = []
        
        # Get all np arrays in folder
        for i in os.walk(folder_path):
            numpy_files = [j for j in i[2] if '.npy' in j]
            break
        
        # Loop over arrays and put them in the correct place
        for f in numpy_files:
            try:
                name = f[:-4]
                if name in skip_attrs:
                    continue
                if name in ['xyz', 'chains_xyz', 'shape', 'default_shape']:
                    # These are for the ChainSet class
                    continue

                try:
                    full_path = folder_path + '/' + name + '.npy'
                    value = np.load(full_path, allow_pickle=True)
                    setattr(self, name, value)
                except OSError:
                    # Attribute not present in this version of the file
                    # print('Could not load attribute %s' % name)
                    pass
            except AttributeError:
                # This value was probably saved by an earlier version of the simulation code, and is no longer used.
                print('Unexpected numpy array discovered: %s.  Skipping...' % f)
        
        # Load ChainSet data
        with open(folder_path + '/' + 'chains_simple_attributes.txt', 'r') as file:
            chain_data = json.load(file)
        try:
            chains_xyz = np.load(folder_path + '/' + 'xyz.npy', allow_pickle=True)
        except OSError:
            # Temporary reverse-compatibility
            chains_xyz = np.load(folder_path + '/' + 'chains_xyz.npy', allow_pickle=True)
            
        try:
            shape = np.load(folder_path + '/' + 'default_shape.npy', allow_pickle=True)
        except OSError:
            # Temporary reverse-compatibility
            shape = chain_data['shape']
            
        self.chains = ChainSet(chains_xyz, chain_data['chain_length'], chain_data['xyz_full_path'])
        self.chains.create_bins(shape)
        
        for key, value in chain_data.items():
            setattr(self.chains, key, value)
        
        # Force data types
        self.completion_block = int(self.completion_block)
        
        # Temporary Reverse-Compatibility
        self.full_path = folder_path

        return self
    
    
    def rename(self, new_name):
        """Changes the name of the directory with output data to the new name.  Changes full_path of object.  
        Does not change path in simple_attributes.txt"""
        
        old_path = self.full_path
        path_as_list = old_path.split('/')
        path_as_list[-1] = new_name
        new_path = '/'.join(path_as_list)
        os.rename(old_path, new_path)
        self.full_path = new_path
        print('Changed path from %s to %s' % (old_path, self.full_path))
        
    
    
    #############################################################################################
    # Simulation Workflow
    #############################################################################################
        
    def compute_rates(self, chains, field, angle, n_times=20, verbose=False, images=True, energetic_disorder=0):
        """Pre-tabulates probabilities for on-chain moves to improve performance during a charge transport simulation.
        
        Inputs:
            chains -- ChainSet object (with bins created)
            field -- applied field strength in V/m.  Can be negative.
            angle -- angle of the applied field.  Any angle can be used, though the propagate_charges method only supports 
                0 or 90 degrees.
            n_times -- the probability density as a function of time is pre-tabulated for this many time values.  Higher values 
                take more time up-front but could be more accurate.  5 is sufficient for most simulations.
            verbose -- If True, display additional intermediate steps and troubleshooting information in Jupyter while running.
            images -- If True, show a heat plot of the average potential in each bin, which can be a good sanity check.
            energetic_disorder -- If larger than 0, applies a Gaussian density of states, with each bead's potential 
                determined independently. The given value is the standard deviation of that Gaussian, in eV.  
        """
        # Unpack Inputs
        self.chains = chains  # positions in nm
        self.field = field  # V/m
        self.field_strength = np.abs(self.field)
        self.angle = angle  # In Degrees
        self.radians = angle * np.pi / 180
        self.sign = -1 if field < 0 else 1
        self.energetic_disorder = energetic_disorder
        unit_potential_vector = np.array((np.cos(self.radians), np.sin(self.radians), 0))
        self.bead_progress = self.chains.xyz @  unit_potential_vector # nm
        
        # Estimate average rates
        self.on_chain_rate = get_on_chain_rate_spiro(field) # per second
        self.interchain_rate = get_interchain_rate_spiro(field)  # per second
        self.k_hop = self.interchain_rate / self.on_chain_rate  # Unitless ratio
        assert self.k_hop < 0.01  # Assume that charge has time to explore the entire chain
        print('Estimated rates: On-chain = %.2E per second, Interchain = %.2E per second' % (self.on_chain_rate, self.interchain_rate))
        
        # Compute potential of each bead
        potential_vector = -1 * unit_potential_vector * self.field * 10**-9  # V/nm.  Positive on left side -> charges move right to left
        
        bead_potential = self.chains.all_chains @ potential_vector  # V
        if self.energetic_disorder:
            energy_modifier = np.random.normal(scale=self.energetic_disorder, size=bead_potential.shape)  # scale is standard deviation in eV
            bead_potential += energy_modifier
        potential_difference = np.diff(bead_potential, axis=1) 

        # Compute rates between each pair of adjacent beads using marcus theory
        self.forward_rates = get_on_chain_rate(potential_difference) # per s
        self.backward_rates = get_on_chain_rate(-potential_difference)  # per s     
        
        # Visualizing the potential is a good spot-check to make sure the field is pointed the direction you expect.  
        self.binned_potential = self.chains.bin_data(np.ndarray.flatten(bead_potential))
        if images:
            self.plot_binned_potential()
            plt.show()

        # Fill In Rates Matrix K 
        # (units of per s)
        self.K = np.zeros((self.chains.n_chains, self.chains.chain_length, self.chains.chain_length)) # Chain, bead, bead.  
        # Fill Main Diagonal
        for i in range(self.chains.chain_length):
            backward_loss = self.backward_rates[:, i - 1] if i > 0 else 0
            forward_loss = self.forward_rates[:, i] if i < self.chains.chain_length - 1 else 0
            self.K[:, i, i] = backward_loss + forward_loss
        # Fill Side Terms
        for i in range(self.chains.chain_length - 1):
            self.K[:, i + 1, i] = -self.forward_rates[:, i]
            self.K[:, i, i + 1] = -self.backward_rates[:, i] 
        
        # Tabulate values for eigenvalue method
        print('Diagonalizing Rates...')
        n_chains = self.chains.n_chains
        chain_length = self.chains.chain_length
        chain_eigenvalues = np.zeros((n_chains, chain_length))
        chain_eigenvectors = np.zeros((n_chains, chain_length, chain_length))
        chain_eigenvectors_inv = np.zeros((n_chains, chain_length, chain_length))
        for i in range(n_chains):
            w, v = np.linalg.eig(self.K[i, :, :])
            # v is The normalized (unit “length”) eigenvectors, such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
            chain_eigenvalues[i, :] = w[:]
            chain_eigenvectors[i, :, :] = v[:, :]
            chain_eigenvectors_inv[i, :, :] = np.linalg.inv(v[:, :])
        # Estimate range of eigenvalues.  High accuracy not needed, abs prevents error for occasional negative value.  
        self.smallest_eigenvalue = np.min(np.sort(np.abs(chain_eigenvalues))[:, 1:]) # Per s. 
        self.largest_eigenvalue = np.max(chain_eigenvalues) # Per s
                
        # Tabulate values for precomputed expm method
        # Dimensions are Chain, Time, End, Start to match matrix equation
        min_time = 1/self.largest_eigenvalue/10  # s
        max_time = 1/self.smallest_eigenvalue * 10 # s
        print('Pre-computing matrix exponentials for Times %.2E to %.2E s' % (min_time, max_time))
        self.expm_times = np.zeros(n_times)  # s
        self.expm_times[1:] = np.logspace(np.log10(min_time), np.log10(max_time), n_times-1)  # s
        self.expm_results = np.zeros((n_chains, n_times, chain_length, chain_length)).astype(np.uint16)  # Largest uint16 is 65535
        print('Expected RAM use', self.expm_results.nbytes/10**6, 'MB')
        
        progress_bar = tqdm(total=n_chains, desc='Exponentiating Matrices...', leave=False)
        for i in range(n_chains):
            K = self.K[i, :, :] # Per s
            for t, time in enumerate(self.expm_times):  # s
                P = linalg.expm(-K * time)
                # Convert to unsigned integers to reduce memory use
                P_norm = P / np.max(P, axis=0)[None, :]  # Normalized to max of 1, not sum of 1.  Re-normalize for probabilities.  
                P_int = np.round(P_norm * 65535).astype(np.uint16)  # Largest uint16 is 65535
                self.expm_results[i, t, :, :] = P_int[:, :]
            
            progress_bar.update()
            
        # Check for nan, inf
        not_finite = np.sum(np.invert(np.isfinite(self.expm_results)))
        if not_finite:
            print('%d non-finite values detected' % not_finite)
            raise
        
            
        # Tabulate values for the infinite-time case
        
        # Set Initial Conditions
        # Set up a stack of all initial conditions where the charge starts on a single bead.  This happens to be an identity matrix.  
        all_starts = np.identity(self.chains.chain_length)

        # Calculate probabilities of end bead given start bead
        self.K_0 = self.K + (self.k_hop * np.identity(self.chains.chain_length))[None, :, :]  # Adds k_hop to main diagonal
        K_inv = np.linalg.inv(-self.K_0)
        P_raw = K_inv @ all_starts  # Dimensions are chain, end bead, start bead.  That means start bead is given by "columns" of the square matrix.  
        P = P_raw / np.sum(P_raw, axis=1)[:, :, None]  
        
        # Reshape and rework to to bead vs. relative change
        self.P_by_bead = np.concatenate([np.transpose(P[i, :, :]) for i in range(self.chains.n_chains)], axis=0)
        
        # Change to cumulative sum for randomizing
        self.P_cumulative = np.cumsum(self.P_by_bead, axis=1) 
        
        # Record short-time mobilities by position
        print('Recording Short-Time Mobilities...')
        self.ps_mobility = self.plot_rates_given_time(10**-12, plot=False)
        self.ns_mobility = self.plot_rates_given_time(10**-9, plot=False)
        
        
    def propagate_charges(self, charge_density=1, t_min=10**-6, t_max=10**-2, n_blocks=1000, linear_spacing=False, 
                          start_padding=50, end_padding=50, start_width=10, verbose=False, bin_size=10, track_single_moves=False,
                          track_performance=False, largest_time_interval=10**-4):  
        """Performs a simulation in which charges along and between chains under the influence of an applied field.
        
        Inputs:
            charge_density -- number of charges initiated in each start location.  The total number of charges is given by 
                charge_density * start_width * (height in bins).
            t_min -- time of first save point in seconds
            t_max -- time of last save point in seconds.  The simulation ends at this time if it hasn't already.  
            n_blocks -- Number of save points.  The time between two save points is called a block.  
            linear_spacing -- if True, the save points will be spaced linearly, which is good for making gifs or videos.  
                By default, save points are spaced logarithmically to capture behavior at multiple time scales.  
            start_padding -- distance between charges' starting points and the reflecting boundary at the starting wall
                of the simulation.  Larger numbers give less error in quantitative analysis, especially at low field 
                strengths.  For qualitative analysis, it can be better to use zero so that the entire simulation is visible.
                Distance in nm.  
            end_padding -- offset between the absorbing boundary and the far wall of the simulation.  Prevents a problem where 
                charges are sometimes unable to leave and build up at the absorbing boundary.  Distance in nm.  
            start_width -- spread in charge starting points in the forward/backward direction.  Increasing this value improves 
                sampling at short time scales.  Distance in nm.  
            verbose -- If True, show additional intermediate steps and troubleshooting information.
            bin_size -- two beads are considered neighbors if they are in the same bin, so increasing bin size increases the
                possible number of neighbors.  This is useful for sensitivity analysis to show the impact of the simple model
                of interchain hopping.
            track_single_moves -- If True, record every location a charge occupies, not just its location when a save point is 
                reached.  This increases RAM use greatly and may require special planning,  but doens't otherwise affect simulation 
                speed.  This data is not saved to disc unless self.save_big_arrays is True.  However, statistics on the moves can 
                be collected and saved using the single_move_analysis method.  
            largest_time_interval - if the time between save points is greater than this value, the calculation between those save
                points is broken into smaller pieces to improve performance.  
                
        """
        
        #########################################################
        # Set Up Variables
        #########################################################
        
        print('Propagating Charges. Setting Up...')
        
        # Record inputs
        self.charge_density = charge_density
        self.bin_size = bin_size
        self.track_single_moves = track_single_moves
        
        # Set up save points, logarithmically spaced
        if linear_spacing:
            self.time_seconds = np.linspace(t_min, t_max, n_blocks)  # Times of save points in seconds
        else:
            self.time_seconds = np.logspace(np.log10(t_min), np.log10(t_max), n_blocks)  # Times of save points in seconds
        self.block_lengths = np.diff(self.time_seconds, prepend=self.time_seconds[0])  # first value will be zero
        
        # Create bins for placing initial charges
        self.chains.create_bins(self.chains.default_shape) 
        length, _, height = self.chains.shape
        step_size = self.chains.bin_size[0]
        max_distance = self.chains.max_dim
        n_bins = self.chains.n_bins
        n_beads = self.chains.n_beads
        n_charges = length * height * self.charge_density * int(np.round(start_width / step_size))
        
        # Set up hop counter
        self.n_hops = np.zeros(n_charges)  # Total interchain hops by each charge
        
        # Place initial charges 
        all_bins = np.arange(self.chains.n_bins).reshape(self.chains.shape)  
        
        if self.angle == 0 and self.field >= 0:
            low = int(np.round(start_padding / step_size))
            high = int(np.round((start_padding + start_width) / step_size))
            starting_bins = all_bins[low:high, :, :]
        elif self.angle == 0 and self.field < 0:
            low = int(np.round((max_distance - start_padding - start_width) / step_size))
            high = int(np.round((max_distance - start_padding) / step_size))
            starting_bins = all_bins[low:high, :, :]
        elif self.angle == 90 and self.field >= 0:
            low = int(np.round(start_padding / step_size))
            high = int(np.round((start_padding + start_width) / step_size))
            starting_bins = all_bins[:, low:high, :]
        elif self.angle == 90 and self.field < 0:
            low = int(np.round((max_distance - start_padding - start_width) / step_size))
            high = int(np.round((max_distance - start_padding) / step_size))
            starting_bins = all_bins[:, low:high, :]
        else:
            # Trying to force code to run at an unusual angle
            low = int(np.round((max_distance - start_padding - start_width) / step_size))
            high = int(np.round((max_distance - start_padding) / step_size))
            starting_bins = all_bins[:, low:high, :]
        current_bins = starting_bins.flatten().repeat(self.charge_density)
        charge_locations = self.chains.get_random_bead(current_bins)
        
        # Set up finish line
        self.finish_line = self.chains.max_dim - start_padding - end_padding  # nm
        
        # Set up positions matrix
        if self.track_single_moves:
            estimated_hops = int(np.round((t_max - t_min) * self.interchain_rate))
            print('Estimated Hops per charge: %.0f' % estimated_hops)
            print('Initializing Array with %d elements to track single moves' % (estimated_hops * 2 * n_charges * 2))
            self.all_positions = np.empty((estimated_hops * 2, n_charges, 2))
            self.all_positions[:, :, :] = np.nan
            hop_counter = 0
        
        #########################################################
        # Perform Simulation
        #########################################################
        
        # Create smaller bins for charge propagation
        n_bins = int(np.round(self.chains.max_dim / bin_size))
        self.chains.create_bins(self.chains.default_shape)
        
        # Record Step 0
        self.positions_with_time = np.zeros((n_blocks, n_charges), dtype=np.int32)  # Holds final results
        self.positions_with_time[0, :] = charge_locations[:]
        charge_progress_baseline = self.bead_progress[charge_locations]
        
        if track_performance:
            print("Estimating Largest Possible Array...")
            # Initialize largest possible array to see RAM hit.  Only true if "track_single_moves" is disabled.  
            all_t, n_intervals = get_poisson_times(self.interchain_rate, largest_time_interval, n_charges, verbose=True) 
            ram_impact = all_t.size * all_t.itemsize / 10.0**9  # GB
            print('Size of Largest Possbile Array: %.1f' % ram_impact)
        
        
        progress_bar = tqdm(total=t_max*10**6 + n_blocks, desc='Propagating Charges', leave=False)
        for b in range(1, n_blocks): # One block is one save point
            if track_performance:
                print('Block Number: %d' % b)
                print('RAM memory percent used:', psutil.virtual_memory()[2])
                print('RAM memory Available: %.1f GB' % (int(psutil.virtual_memory()[1]) / 10.0**9))
                print('CPU Used: %.0f percent' % psutil.cpu_percent(5))
            
            # Check if Simulation is finished
            current_progress = self.bead_progress[charge_locations] - charge_progress_baseline
            completed_charges = np.where(np.abs(current_progress) >= self.finish_line)[0]
            self.fraction_finished = len(completed_charges) / float(n_charges)
            if self.fraction_finished == 1.0:
                # End simulation early since all charges have finished
                self.completion_block = b
                print('Simulation Finished Early')
                break
            
            # if block is too large to compute all at once, split into smaller pieces.  It is still one save point.  
            time_remaining = self.block_lengths[b]
            while time_remaining > 0:
                sub_block_length = np.min((time_remaining, largest_time_interval))
                time_remaining -= largest_time_interval
            
                # Get times between hopping events for this time interval
                all_t, n_intervals = get_poisson_times(self.interchain_rate, sub_block_length, n_charges) 
                max_intervals = all_t.shape[0] # Using max(n_intervals) here gives round-off errors that lead to index errors

                for i in range(max_intervals):

                    # Determine which beads will make an on-chain move and/or an interchain move this sub-loop
                    on_chain_moves = np.where((i < n_intervals) & (np.abs(current_progress) < self.finish_line))[0]
                    interchain_moves = np.where((i < n_intervals - 1) & (np.abs(current_progress) < self.finish_line))[0] 
                    self.n_hops[interchain_moves] += 1

                    # Phase 1 - On-Chain Transport
                    before = np.array(charge_locations)
                    charge_locations[on_chain_moves] = self.on_chain_move(charge_locations[on_chain_moves], all_t[i, on_chain_moves])
                    after = np.array(charge_locations)

                    # Phase 2 - Interchain Transport
                    before_2 = np.array(charge_locations[interchain_moves])
                    charge_locations[interchain_moves] = self.chains.get_random_neighbor(charge_locations[interchain_moves])
                    after_2 = np.array(charge_locations[interchain_moves])

                    # Record positions if saving maximum information
                    if self.track_single_moves:
                        # Omit start of block since this is a partial move (except for the first block)
                        if (b == 0) or (i != 0):
                            # Use interchain_moves mask to omit on-chain moves at the end of a block.  
                            self.all_positions[hop_counter, interchain_moves, 0] = before[interchain_moves]
                            self.all_positions[hop_counter, interchain_moves, 1] = after[interchain_moves]
                            hop_counter += 1
                    
                
            mem_use = psutil.virtual_memory()[2]
            if mem_use > 90:
                print('Loop %d Memory Use %.1f' % (b, mem_use))
                # raise
                    
                
            # Record Results
            self.positions_with_time[b, :] = charge_locations[:]
            progress_bar.update(np.round(self.block_lengths[b] * 10**6 + 1))
        else:
            # Not all charges finished - set finish block as max # of blocks
            self.completion_block = n_blocks
            
        print('Simulation Complete')
            
        # Reset Bins
        self.chains.create_bins(self.chains.default_shape)  # This is my default shape - needs to be generalized
        
        # Trim all_positions array
        if self.track_single_moves:
            self.all_positions = self.all_positions[:hop_counter + 1, :, :]
            
            
    def unpack_results(self):
        """Performs a number of basic analysis steps that are used by other analysis methods.  Should be run after running a simulation or 
        loading a file."""
        
        # Update Shorthand
        self.field_strength = np.abs(self.field)
        
        
        n_blocks, n_charges = self.positions_with_time.shape
        length, width, height = self.chains.shape
        charge_numbers = np.arange(n_charges)
        end = self.completion_block if self.completion_block < n_blocks else self.completion_block - 1
        self.end = end

        # Compute Charge Progress in Field Direction
        charge_progress = self.bead_progress[self.positions_with_time]  # (time, charge), nm
        charge_progress = charge_progress - charge_progress[0, :][None, :]  # Subtracts exact position each charge started at
        charge_progress[end:] = charge_progress[end - 1, :][None, :] # Charges stay at the same position after simulation finishes
        
        
        # Get aggregate metrics of charge motion
        charge_velocity = np.diff(charge_progress, axis=0) / np.diff(self.time_seconds)[:, None] * 10**-7  # cm/s
        
        past_finish_line = np.heaviside(np.abs(charge_progress) - self.finish_line, 1)  # Second argument is what to return if x == 0
        self.active_fraction = 1 - np.mean(past_finish_line, axis=1) # vs time
        self.avg_progress = np.mean(charge_progress, axis=1) # nm
        self.avg_velocity = np.mean(charge_velocity, axis=1)[:end] / self.active_fraction[:end]  # Average at each time step
        self.estimated_mobility_vs_time = self.avg_velocity / (self.field / 100)  # cm2/Vs.  INACURRATE FOR LOW FIELDS. 
        
        # Get average velocity of each charge
        self.finish_blocks = np.argmax(past_finish_line, axis=0)
        finish_times = self.time_seconds[self.finish_blocks]
        finish_times[np.where(self.finish_blocks == 0)] = self.time_seconds[-1]  # Charges that didn't finish are set to the longest time
        finish_distances = charge_progress[-1, :]
        velocity_by_charge = finish_distances / finish_times
        
        # Get grand average of velocity
        # Neither system works.  Average of equally weighted charges gives values 7 orders of magnitude higher than before.  
        
        
        
        # Old system was total distance over total time, which overrepresents traps.  Average across equally-weighted charges is better
        # because 20% traps results in a 20% drop in velocity.  
        self.grand_avg_velocity = np.mean(velocity_by_charge)
        self.estimated_mobility = self.grand_avg_velocity / (self.field_strength / 100)  # cm2/Vs.  INACURRATE FOR LOW FIELDS.  
        self.interchain_rate = get_interchain_rate_spiro(self.field)  # Older simulations didn't save this value.  Redundant for newer simulations.
        self.distance_per_hop = self.grand_avg_velocity / self.interchain_rate
        
        # Old method for grand avgerage velocity
        # weight = self.block_lengths[:end] * self.active_fraction[:end]
        # value = self.avg_velocity[:end]
        # self.grand_avg_velocity = np.nansum(weight * value) / np.nansum(weight) # cm/s
        
 
        
        #########################################################
        # Bin Density and Progress
        #########################################################
        self.density_vs_time = np.zeros((self.completion_block, length**2))
        self.active_density_vs_time = np.zeros((self.completion_block, length**2))
        bins_by_bead = self.chains.bins_by_bead_3d @ self.chains.numbering_system_2d
        bins_by_bead[self.chains.out_bounds_2d] = length**2  # Mark out-of-bounds beads
        for i in range(self.completion_block):
            position_snapshot = np.squeeze(self.positions_with_time[i, :])
            active_mask = np.where(np.abs(charge_progress[i, :]) < self.finish_line)[0]
            occupied_beads = position_snapshot[active_mask]

            occupied_bins = bins_by_bead[occupied_beads]
            count_by_bin = np.bincount(occupied_bins, minlength=length**2)
            if len(count_by_bin) > length**2:  # Out-of-bounds beads present
                count_by_bin = count_by_bin[:-1]
            self.density_vs_time[i, :] = count_by_bin[:]
            
        # Do Single-Move Analysis
        if self.track_single_moves:
            self.single_move_analysis()
            
            
    #############################################################################################
    # Simulation Helper Functions
    #############################################################################################
    
    def on_chain_move(self, current_beads, t):
        """Uses pre-tabulated probabilites and random numbers to determine charges' positions based
        on their previous positions and the time spent on that polymer chain."""
        # Set up
        chain_starts = self.chains.chain_starts[current_beads] # Index of first bead in each bead's chain
        current_chains, start_positions = np.divmod(current_beads, self.chains.chain_length)
        rnd = np.random.random(len(current_beads))
        
        if not len(current_beads): # empty array was passed as current_beads
            return current_beads  # return empty array
        
        p = self.interpolate_probabilities(current_chains, start_positions, t)
        
        # Convert to cumulative probability
        p_cumulative = np.cumsum(p, axis=1)
        
        # Subtract RND and get index of first positive value
        end_positions = np.argmax(np.sign(p_cumulative - rnd[:, None]), axis=1)
        
        # Move charges
        new_beads = chain_starts + end_positions
        
        return new_beads.astype(int)
    
    
    def interpolate_probabilities(self, chains, start_positions, t):
        """Uses linear interpolation to determine probability densities for lengths of time between the tabulated values."""
        
        # Set up time variables
        t_clipped = np.clip(t, -10**20, self.expm_times[-1] - 10**-20)  # Set highest t to right below highest tabulated value
        t_difference = self.expm_times[None, :] - t_clipped[:, None] 
        clipped_array = np.clip(t_difference, -10**20, 0)

        i_upper = np.argmax(clipped_array, axis=1)  # Gives index of first tabulated time greater than t for each charge
        i_lower = i_upper - 1 
        
        # Load Probabilities
        low = self.expm_results[chains, i_lower, :, start_positions]
        high = self.expm_results[chains, i_upper, :, start_positions]
            
        # Renormalize and apply linear interpolation
        low_norm = high.astype(int) / np.sum(high, axis=1)[:, None]  # Probability distributions at lower bound of time
        high_norm = high.astype(int) / np.sum(high, axis=1)[:, None]  # Probability distributions at upper bound of time
        weight = (t - self.expm_times[i_lower]) / (self.expm_times[i_upper] - self.expm_times[i_lower]) # This is giving div/0 warning for 0/0 
        p = low_norm * weight[:, None] + high_norm * (1 - weight)[:, None]
        
        return p
    
    
    #############################################################################################
    # Analysis: Visualize Inputs
    #############################################################################################    
    
    
    def plot_binned_potential(self):
        """Plots binned potential.  Good for sanity check when computing rates."""
        plt.imshow(np.transpose(self.binned_potential.reshape((self.chains.shape[0], self.chains.shape[1]))), origin='lower')
        cbar = plt.colorbar()
        cbar.set_label('Potential (V/m)', rotation=270, labelpad=15)
    
    
    #############################################################################################
    # Analysis: Short-Range Transport
    #############################################################################################    
    
    def plot_chain_probability_distribution(self, n_chains=10, start=0, alpha=0.5, bin_size=2, fill_fraction=0.9, histogram=True, scatter=True, 
                                            colorbar=True, connect_dots=True, show=True):
        """Plots histograms of 'starting' and 'ending' positions along a single chain, both in terms of bead number and distance. 
        Also plots geometry of that chain.  Multiple chains are
        plotted successively.  Requires that compute_rates has been completed."""
        
        for chain_number in range(start, start + n_chains):
            
            # Get chain geometry
            xy = self.chains.all_chains[chain_number, :, :2]
            centroid = np.mean(xy, axis=0)
            relative_xy = xy - centroid[None, :]
            x = relative_xy[:, 0]
            y = relative_xy[:, 1]
            
            # Load Probailities and re-normalize
            P_start = np.ones(self.chains.chain_length) / self.chains.chain_length
            P_end = self.expm_results[chain_number, -1, :, 0]
            P_end = P_end / np.sum(P_end)
            
            # Get Relative Progress By Position
            bead_numbers = np.arange(self.chains.chain_length * chain_number, self.chains.chain_length * (chain_number + 1))
            progress = self.bead_progress[bead_numbers]
            relative_progress = progress - np.mean(progress)
            
            P_ratio = np.max(P_end) / np.min(P_end)
            potential_difference = (np.max(x) - np.min(x)) * self.field
            dy = np.max(y) - np.min(y)
    
            if histogram:
                # Set Up Histogram
                low = (np.min(x) // bin_size) * bin_size
                high = (np.max(x) // bin_size + 1) * bin_size
                n_bins = int(np.round((high - low) / bin_size))

                start_progress, bin_edges = np.histogram(relative_progress, range=(low, high), bins=n_bins, weights=P_start, density=True)
                end_progress, bin_edges = np.histogram(relative_progress, range=(low, high), bins=n_bins, weights=P_end, density=True)
                bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

                bar_width = (bin_centers[1] - bin_centers[0]) * fill_fraction
                plt.bar(bin_centers, start_progress, width=bar_width, alpha=alpha)
                plt.bar(bin_centers, end_progress, width=bar_width, alpha=alpha)
                plt.legend(['Start', 'End'])
                plt.xlim(low, high)
                plt.ylim(0)
                plt.xlabel('Position (nm)')
                plt.ylabel('Probability')
                plt.show()
                
            if scatter:
                # Scatterplot Format
                cm = plt.cm.get_cmap('cool')
                plt.scatter(x, y, c=P_end, cmap=cm, s=100, zorder=5)
                if colorbar:
                    plt.colorbar(orientation='horizontal', pad=0.2, label='Occupancy Probability')
                
                if connect_dots:
                    plt.plot(x, y, c=(0, 0, 0.8))
                # plt.xlabel('Distance in Field Direction (nm)')
                # plt.ylabel('Perpendicular Distance (nm)')
                # plt.axis('equal')
                
                if show:  # Turning off "show" and viewing one image lets you edit outside the function
                    plt.show()
                else:
                    plt.axis('equal')
            print('Chain Number %d ^^^' % chain_number)
            print('Potential Difference %.0f mV' % np.abs(potential_difference * 10**-6))
            print('%.1f to %.1f' % (np.min(x), np.max(x)))
            print('Probability Ratio %.1f' % P_ratio)
            
            
    def plot_rate_results_by_chain(self, n_chains=None, **kwargs):
        """Plots the results of the compute_rates method for the given number of chains, all on one set of axes."""

        # self.expm_results is chain, time, end, start
            
        n_chains = self.n_chains  # These used to be different due to a reverse-compatibility issue
            
        # Compute Progress
        results = self.expm_results[:n_chains, :, :, :]
        norm_results = results.astype(int) / np.sum(results, axis=2)[:, :, None, :]
        start_positions = np.mean(norm_results, axis=2) # Average across all end positions - should all be equal but in the same shape as end positions
        end_positions = np.mean(norm_results, axis=3)  # Average across all start positions
        progress_by_chain = self.bead_progress[:n_chains * self.chains.chain_length].reshape((n_chains, self.chains.chain_length))
        start_progress = np.sum(start_positions * progress_by_chain[:, None, :], axis=2)  # Sum to avoid dividing by length twice
        end_progress = np.sum(end_positions * progress_by_chain[:, None, :], axis=2)
        net_progress = end_progress - start_progress
        
        n_chains_to_plot = np.min((n_chains, 100))
        for i in range(n_chains_to_plot):
            plt.plot(self.expm_times, net_progress[i, :], **kwargs)
        plt.xscale('log')
        plt.xlabel('Time (s)')
        plt.ylabel('Expected Progress (nm)')
        plt.show()
        
        # Plot average progress
        avg_progress = np.mean(net_progress[:n_chains, :], axis=0) * self.sign  # both field directions give positive progress
        plt.plot(self.expm_times, avg_progress, **kwargs)
        plt.xscale('log')
        plt.xlabel('Time (s)')
        plt.ylabel('Expected Progress (nm)')
        plt.show()
        
        # Plot average velocity
        avg_v = np.diff(avg_progress) / np.diff(self.expm_times) * 10**-7  # cm/s
        time_centers = (self.expm_times[1:] + self.expm_times[:-1]) / 2
        plt.plot(time_centers, avg_v)
        plt.xscale('log')
        plt.xlabel('Time (s)')
        plt.ylabel('Expected Velocity (cm/s)')
        plt.show()
    
        # Plot expected mobility.  I think since this starts on every bead it's allowed without accounting for diffusion.  
        avg_mobility = avg_v / (self.field * 10**-2)
        plt.plot(time_centers, avg_mobility)
        plt.xscale('log')
        plt.xlabel('Time (s)')
        plt.ylabel('Expected Mobility (cm2/Vs)')
        plt.show()
        
        
    def plot_rates_given_time(self, time, plot=True, **kwargs):
        """Computes and plots the average short-time mobility as a function of position.  
        
        Inputs:
            time -- the average short-time mobility is computed from zero seconds to this value.  Should be much less than the
                    interchain hopping time, since interchain hopping is not included in this analysis. 
            plot -- if False, just returns the computed values.
            kwargs -- passed directly to the plt.imshow function
        
        Outputs
            mobility_2d - average short-time 2d mobility in each bin.  This is ok to use at low fields since it isn't affected by
                          asymmetric diffusion."""
        
        # self.expm_results is chain, time, end, start
        # norm_results is chain, end, start
        
        # Select time
        end_index = len(np.where(self.expm_times <= time)[0]) - 1
        results = self.expm_results[:, end_index, :, :]
        norm_results = results.astype(float) / np.sum(results, axis=1)[:, None, :]
        
        # Get distances of beads in field direction
        progress_by_chain = self.bead_progress.reshape((self.chains.n_chains, self.chains.chain_length))
    
        # Compute Progress
        start_positions = np.mean(norm_results, axis=1) # Average across all end positions - should all be equal but in the same shape as end positions
        end_positions = np.mean(norm_results, axis=2)  # Average across all start positions
        start_progress = np.sum(start_positions * progress_by_chain[:, :], axis=1)  # Sum to avoid dividing by length twice
        end_progress = np.sum(end_positions * progress_by_chain[:, :], axis=1)
        net_progress = end_progress - start_progress
        
        net_progress_cm = net_progress * 10**-7  # cm
        velocity = net_progress_cm / self.expm_times[end_index]  # cm/s
        mobility = velocity / (self.field * self.sign * 10**-2)  # cm2/Vs
        
        # Assign mobility values to beads so that each bead in a chain has the same value.  This isn't a perfect solution but I think it's better than 
        # chain ends playing a huge role.  
        mobility_by_bead = np.stack([mobility]*self.chains.chain_length, axis=1).flatten()
        
        # Assign beads to bins and find mean mobility per bin
        self.chains.create_bins(self.chains.default_shape)  
        mobility_2d = self.chains.bin_data(mobility_by_bead).reshape((self.chains.x_bins, self.chains.x_bins))
        
        if plot:
            plt.imshow(np.transpose(mobility_2d), origin='lower', **kwargs)
            plt.axis('off')
            plt.colorbar()
            plt.title('Moblity %.0EV/cm  %.0Es  %d\N{DEGREE SIGN}' % (self.field * self.sign * 10**-2, time, self.angle))
            plt.show()
        
        return mobility_2d
    
    
    #############################################################################################
    # Single Move Analysis
    #############################################################################################
    
    def single_move_analysis(self, clear_memory=False):
        """Aggregate single-move data to get useful information, then optionally delete it."""
        if self.all_positions is not None:
            self.get_move_distribution()

            if clear_memory:
                self.all_positions = None
        
    
    def get_move_distribution(self, bins=120, distance_range=(-60, 60)):
        """Aggregate single-move data to get distributions of single moves and distributions of consecutive moves."""
        
        # Preprocess: Remove NAN values and concatenate results for different charges
        data = self.all_positions
        n_steps, n_charges, _ = data.shape  # Third dimension is 2: position before and after the move
        data_sets = []
        for i in range(n_charges):
            active_mask = np.invert(np.isnan(data[:, i, 0]))
            trimmed = data[active_mask, i, :].astype(int)
            data_sets.append(trimmed)
        all_data = np.concatenate(data_sets, axis=0)

        all_progress = self.bead_progress[all_data]  
        start_progress = all_progress[:, 0]
        end_progress = all_progress[:, 1]
        net_progress = end_progress - start_progress
        
        hist, bin_edges = np.histogram(net_progress, bins=bins, range=distance_range) 
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        
        # Record results as attributes
        self.move_distances = bin_centers
        self.move_frequencies = hist
        self.avg_forward_distance = np.mean(net_progress)
        self.avg_abs_distance = np.mean(np.abs(net_progress))
        
        #######################
        # Consecutive Moves
        #######################
        
        # This section could be useful for finding correlations between consecutive moves.  It hasn't been tested in its current version.
        if False:

            # Define positions a, b, c, d.  A charge moves along a chain from a to b, hops to c, then continues to d
            a = all_data[:-1, 0]
            b = all_data[:-1, 1]
            c = all_data[1:, 0]
            d = all_data[1:, 1]

            # Get distances of the two moves
            first_distance = self.bead_progress[b] - self.bead_progress[a]
            second_distance = self.bead_progress[d] - self.bead_progress[c]

            # Sort the data based on whether b and c are the same bead, on the same chain, or neither
            same_bead_mask = np.where(b == c)
            same_chain_mask = np.where(b // self.chains.chain_length == c // self.chains.chain_length)

            self.consecutive_moves = np.histogram2d(first_distance, second_distance, bins=bins, range=_range)[0]
            self.consecutive_moves_same_bead = np.histogram2d(first_distance, second_distance, bins=bins, range=range)[0]
            self.consecutive_moves_same_chain = np.histogram2d(first_distance, second_distance, bins=bins, range=range)[0]
            
    #############################################################################################
    # Long-Range Transport: General Analysis
    #############################################################################################      
    
    def make_default_figures(self):
        """Calls the most commonly used plotting methods using their default values."""
        # Set up reused variables
        n_blocks, n_charges = self.positions_with_time.shape
        length, width, height = self.chains.shape
        charge_numbers = np.arange(n_charges)
        end = self.completion_block if self.completion_block < n_blocks else self.completion_block - 1
        time_centers = ((self.time_seconds[1:] + self.time_seconds[:-1]) / 2)
        
        # Plot total density
        self.plot_total_density(vmax=self.charge_density/200)
        
        # Plot density vs time
        self.plot_density_vs_time(vmax=(self.charge_density/5))
        
        # Plot Lorenz Curve
        self.plot_sorted_density()
        
        # Plot total progress
        plt.plot(self.time_seconds[:end], self.avg_progress[:end] * self.sign)
        plt.xlabel('Time (s)')
        plt.ylabel('Average Progress (nm)')
        plt.xscale('log')
        # plt.yscale('log')
        plt.show()
        
        # Plot average velocity
        plt.plot(time_centers[:end], self.avg_velocity[:end] * self.sign, linewidth=0, marker='.')
        plt.xlabel('Time (s)')
        plt.ylabel('Time (s)')
        plt.ylabel('Average Velocity (cm/s)')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        
        # Plot fraction finished
        charge_progress = self.bead_progress[self.positions_with_time]  # (time, charge), nm
        charge_progress = charge_progress - charge_progress[0, :][None, :]
        charge_progress[end:] = charge_progress[end - 1, :][None, :]
        forward_finished = np.sum(np.heaviside(charge_progress - self.finish_line, 1), axis=1) / n_charges
        reverse_finished = np.sum(np.heaviside(-charge_progress - self.finish_line, 1), axis=1) / n_charges
        total_finished = forward_finished + reverse_finished
        plt.plot(self.time_seconds, total_finished)
        plt.plot(self.time_seconds, forward_finished)
        plt.plot(self.time_seconds, reverse_finished)
        plt.xlabel('Time (s)')
        plt.ylabel('Fraction of Charges Finished')
        plt.xscale('log')
        # plt.xlim(10**-6, 10**-1)
        plt.ylim(0, 1.1)
        plt.legend(['Total', 'Forward', 'Reverse'])
        plt.show()

        
    def show_stats(self):
        """Prints a variety of simple results of the simulation."""
        self.on_chain_rate = get_on_chain_rate_spiro(self.field) # per second
        self.interchain_rate = get_interchain_rate_spiro(self.field)  # per second
        total_distance = 750  # Automate this
        
        print('Field Strength: %.2fV/um' % (self.field * self.sign * 10**-6))
        print('Total Distance: %.dnm' % total_distance)
        print('On Chain Rate: %.1Es^-1' % self.on_chain_rate)
        print('Interchain Rate: %.1Es^-1' % self.interchain_rate)
        print('Grand Average Velocity: %.2Ecm/s' % self.grand_avg_velocity)
        grand_avg_time = total_distance / (self.grand_avg_velocity * 10**7)
        print('Grand Average Time: %.2Es' % grand_avg_time)
        
        distance_per_hop = self.grand_avg_velocity / self.interchain_rate * 10**7  # nm
        average_hops = total_distance / distance_per_hop
        print('Average Hops: %.0f' % average_hops)
        print('Distance per hop: %.4fnm' % distance_per_hop)
        print('\n')
        
        
    def plot_distance_vs_time(self, **kwargs):
        """Plots the aggregate distance charges have moved away from their starting points as a function of time."""
        # Display charge progress vs time
        plt.plot(self.time_seconds[:self.completion_block], self.avg_progress[:self.completion_block], linewidth=0, marker='.', **kwargs)
        plt.xscale('log')
        plt.xlabel('Time (s)')
        plt.ylabel('Avg Bidirectional Distance From Start(nm)')
        image = plt.gcf()
        plt.show()
        return image
        
        
    #############################################################################################
    # Long-Range Transport: Charge Density Maps
    ############################################################################################# 
    
    
    def integrated_charge_density(self):
        """Converts Density(Position, Time) to Density(Position) by integrating over the length of the simulation."""
        result = np.nansum(self.density_vs_time[:, :] * self.block_lengths[:self.completion_block, None], axis=0) / np.sum(self.block_lengths[:self.completion_block]) 
        return result
    
    def plot_total_density(self, normalize_by_time=False, normalize_by_mean=False, colorbar=False, title=True, figsize=None, **kwargs):
        """Plots the total time charges spent in each bin over the entire simulation."""
        # Display Density Integrated over All Time
        fig, ax = plt.subplots(figsize=figsize)
        if title:
            plt.title('Total Density')
        length, _, height = self.chains.shape
        total_density = self.integrated_charge_density()
        
        if normalize_by_time:
            crossing_times = self.time_seconds[self.finish_blocks]

            total_density = total_density / np.mean(crossing_times)
            
        if normalize_by_mean:
            total_density = total_density / np.mean(total_density)
            

        
        im = ax.imshow(np.transpose(total_density.reshape((length, length))), origin='lower', **kwargs)
        
        plt.axis('off')
        if colorbar:
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.2)
            plt.colorbar(im, cax=cax)
        image = plt.gcf()
        plt.show()  
        return image
    
    
    def plot_density_vs_time(self, n_images=5, **kwargs):
        # Display Density vs Time
        length, _, height = self.chains.shape
        display_cycles = int(np.ceil(self.completion_block / n_images))
        images = []
        for i in range(0, self.completion_block, display_cycles):
            print(' '*5 + 'Time=%.1f \u03BCs' % ((self.time_seconds[i] - self.time_seconds[0]) * 10**6))  # Microseconds
            consistent_imshow(self.density_vs_time[i, :].reshape((length, length)), **kwargs)
            plt.colorbar()
            plt.axis('off')
            images.append(plt.gcf())
            plt.show()
        return images
    
    
    def plot_density_specific_time(self, time, normalize_by_mean=False, normalize_by_total_mean=False, colorbar=True, title=True, **kwargs):
        """Finds the save point nearest to the specified time (in seconds) and plot the corresponding charge density."""
        nearest_index = np.argmin((self.time_seconds - self.time_seconds[0] - time)**2)
        
        length, _, height = self.chains.shape
        print(' '*5 + 'Time=%.1f \u03BCs' % ((self.time_seconds[nearest_index] - self.time_seconds[0]) * 10**6))  # Microseconds
        print('%.0f%% Error' % ((self.time_seconds[nearest_index] - self.time_seconds[0] - time) / time * 100))
        density = self.density_vs_time[nearest_index, :].reshape((length, length))
        
        if normalize_by_total_mean:  # Makes scale match the plot of total density integrated across all time
            density = density / np.mean(self.integrated_charge_density())
        
        if normalize_by_mean:
            density = density / np.mean(density)
        
        im = consistent_imshow(density, **kwargs)
        
        plt.axis('off')
        
        if colorbar:
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.2)
            plt.colorbar(im, cax=cax)
        
        if title:
            plt.title('%.0f\u03BCs' % (time * 10**6))
        
        
        image = plt.gcf()
        plt.show()
        return image
    
    
    #############################################################################################
    # Long-Range Transport: Charge Density Distributions
    ############################################################################################# 
    
    
    def get_charge_density_distribution(self):
        """Charges spend more time in some bins than others.  This function creates and returns a histogram of integrated charge density in various bins."""
        density = self.integrated_charge_density().reshape((self.chains.x_bins, self.chains.x_bins)) / self.chains.bin_density().reshape((self.chains.x_bins, self.chains.x_bins))  # Normalize for density of bin
        # Manual trimming for now - generalize
        start_padding, end_padding = (5, 5) # Bins
        side_padding = 2 # Bins
        if self.angle == 0:
            values = density[start_padding:-end_padding, side_padding:-side_padding].flatten()
        else:
            values = density[side_padding:-side_padding, start_padding:-end_padding].flatten()
           
        hist, bin_edges = np.histogram(values, bins=100)
        hist = hist / np.sum(hist)
        return hist, bin_edges

    
    def get_lorenz_curve(self):
        """Creates a Lorenz curve charge density as a function of spatial bin"""
        values = self.trim_charge_density_by_bin()
           
        lorenz_curve = (np.cumsum(np.sort(values)) / np.sum(values))
        gini_coefficient = np.sum(lorenz_curve) / (len(lorenz_curve) / 2)
        return lorenz_curve, gini_coefficient
    
    
    def trim_charge_density_by_bin(self):
        """Removes bins from the Lorenz curve calculation that are excluded by the boundary conditions of the simulation."""
        density = self.integrated_charge_density().reshape((self.chains.x_bins, self.chains.x_bins)) / self.chains.bin_density().reshape((self.chains.x_bins, self.chains.x_bins))  # Normalize for density of bin
        start_padding, end_padding = (5, 5) # Bins
        side_padding = 2 # Bins
        if self.angle == 0:
            values = density[start_padding:-end_padding, side_padding:-side_padding].flatten()
        else:
            values = density[side_padding:-side_padding, start_padding:-end_padding].flatten()
        return values
    
    
    def plot_sorted_density(self, show=True, **kwargs):
        """Plots the output of get_lorenz_curve in a format typical for Lorenz curves"""
        lorenz_curve, gini = self.get_lorenz_curve()
        
        x = np.linspace(0, 100, len(lorenz_curve), endpoint=False)
        plt.plot(x, lorenz_curve)
        plt.xlabel('Percentile of Charge Density')
        plt.ylabel('Cumulative Fraction')
        plt.xlim(0, 100)
        plt.ylim(0, 1)
        plt.gca().set_aspect(100)
        
        image = plt.gcf()
        if show:
            plt.show()  
        return image
    
    
    def get_lorenz_curve_2(self):
        """Creates a Lorenz curve charge density as a function of polymer chain"""
        
        # Use chains instead of bins
        n_blocks = len(self.block_lengths)
        n_chains = self.chains.n_chains
        start_distance, end_distance = (10, 740)
        
        # Get extreme points of chains 
        chain_xyz = self.chains.all_chains  # n_chains, chain_length, 3
        min_x = np.min(chain_xyz[:, :, 0], axis=1)
        max_x = np.max(chain_xyz[:, :, 0], axis=1)
        min_y = np.min(chain_xyz[:, :, 1], axis=1)
        max_y = np.max(chain_xyz[:, :, 1], axis=1)
        
        
        # Get extreme points of chains.  If any points are past the finish line, omit those chains.  
        progress_by_chain = self.bead_progress.reshape((self.chains.n_chains, self.chains.chain_length))
        chain_max = np.max(progress_by_chain, axis=1)
        chain_min = np.min(progress_by_chain, axis=1)
        in_bounds_mask = np.where((chain_max < end_distance) & (chain_min > start_distance))[0]
        
        # Convert bead locations to chain locations.  Set charges that have already finished to a throw-away value.  
        chains_with_time = self.positions_with_time // self.chains.chain_length  #  [Blocks, Charges]
        
        # Bin Positions
        chains_with_time_binned = np.zeros((n_blocks, n_chains))
        for i in range(n_blocks):
            chains_with_time_binned[i, :] = np.bincount(chains_with_time[i, :].astype(int), minlength=n_chains)
        time_by_chain = np.sum(chains_with_time_binned[:self.completion_block, :] * self.block_lengths[:self.completion_block, None], axis=0)[in_bounds_mask]
        lorenz_curve = (np.cumsum(np.sort(time_by_chain)) / np.sum(time_by_chain))
        
        gini_coefficient = np.sum(lorenz_curve) / (len(lorenz_curve) / 2)
        # print('Gini Coefficeint:', gini_coefficient)
        
        return lorenz_curve, gini_coefficient
    
    
    #############################################################################################
    # Long-Range Transport: Charge Mobility
    #############################################################################################     
        
        
    def get_plateau_mobility(self, t_min=None, t_max=None, verbose=False):
        """This method computes a weighted average estimated mobility. More importantly, the verbose option plots the raw values and weights
        for learning and troubleshooting purposes.  
        
        WARNING: DOES NOT ACCOUNT FOR ASYMMETRIC DIFFUSION!  Error could be large for applied fields below 0.1 V/um.  
        """
        
        end = int(self.end) if self.end is not None else self.n_blocks
        if t_max:
            if self.time_seconds[end] > t_max:
                end = np.where(self.time_seconds > t_max)[0][0]
        start = np.where(self.time_seconds <= t_min)[0][0] if t_min else 0
        weights = self.block_lengths[:end] * self.active_fraction[start:end]  # charges * seconds
        values = self.estimated_mobility_vs_time[start:end]
        avg_mobility = np.nansum(weights * values) / np.sum(weights)  # Final average velocity will be nan
        
        
        if verbose:
            # Plot mobility vs time
            plt.plot(self.time_seconds[start:end], self.estimated_mobility_vs_time[start:end], linewidth=0, marker='.')
            plt.xlabel('Time (s)')
            plt.ylabel('Estimated Mobility ($cm^2$/Vs)')
            plt.xscale('log')
            plt.yscale('log')
            plt.ylim(10**-7, 10**-2)
            plt.axhline(y=avg_mobility, color=(1, 0, 0))
            plt.legend(['$\mu$(t)', 'Weighted Average'])
            plt.show()
            
            # Plot Active Fraction and Weights
            fig, ax1 = plt.subplots()
            red = (1, 0, 0)
            blue = (0, 0, 1)
            
            ax1.plot(self.time_seconds[start:end], self.active_fraction[start:end], linewidth=0, marker='.', color=blue)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Fraction of Charges Active', color=blue)
            ax1.tick_params(axis='y', labelcolor=blue)
            ax1.set_ylim((0, 1.05))
            plt.xscale('log')
            
            # Plot Weights
            weights = self.block_lengths[start:end] * self.active_fraction[start:end]  # charges * seconds
            ax2 = plt.gca().twinx()
            ax2.plot(self.time_seconds[start:end], weights[start:end], linewidth=0, marker='.', color=red)
            ax2.set_ylabel('Weight (charges * seconds)', color=red)
            ax2.tick_params(axis='y', labelcolor=red)
            plt.show()
        return avg_mobility  # cm2/Vs.  ESTIMATE!  NOT ACCURATE AT LOW FIELDS!


    #############################################################################################
    # GIF creation methods
    #############################################################################################

    def prep_gif_images(self, start_frame=0, max_frames=100, step_size=1, save=False, **kwargs):
        """Step one of making GIFS: make some individual frames and put them in a directory."""
        # Unpack Inputs
        max_frames = np.min((max_frames, self.density_vs_time.shape[0]))
        
        try:
            # Create directory
            if save:
                gif_path = self.full_path + '/gif_images/'
                os.mkdir(gif_path)

            # Make Images
            time_indices = np.arange(start_frame, max_frames, step_size)
            print(time_indices)
            time_us = self.time_seconds * 10**6
            for i in time_indices:
                density = self.density_vs_time[i, :].reshape((self.chains.x_bins, self.chains.x_bins))
                plt.figure(figsize=(4, 4))
                plt.imshow(np.transpose(density), origin='lower', **kwargs)
                plt.gca().axis('off')
                plt.colorbar()

                plt.title('%.0f \u03BCs' % time_us[i])

                if save:
                    plt.gcf().savefig(gif_path + 'frame%d.png' % i, dpi=300)

                plt.show()
        except FileExistsError:
            print('Directory Exists - Skipping Image Creation')
            pass
        
    def make_gif(self, filename, duration=1, max_frames=None):
        """Step 2 of making GIFs: set an end point, choose image duration, and combine into a single file."""
        gif_dir = self.full_path + '/gif_images/'
        gif_path = gif_dir + '%s.gif' % filename
        
        for step in os.walk(gif_dir):
            image_names = [i for i in step[2] if '.png' in i]
            break
        image_numbers = [int(name[5:-4]) for name in image_names]
        sorted_names = [x for _, x in sorted(zip(image_numbers, image_names))]
        
        if max_frames:
            sorted_names = sorted_names[:max_frames]
        
        images = [imageio.imread(gif_dir + name) for name in sorted_names]
        
        imageio.mimsave(gif_path, images, duration=duration)

        print('Done')
                  
            
#############################################################################################
# Global Constants
#############################################################################################
# Physical Constants
K_B = 1.38 * 10.0**-23
T_ROOM = 300
H_BAR = 6.626 * 10.0**-34 / (2 * 3.14)
E_CHARGE = 1.6 * 10.0**-19
PI = 3.14159

# Constants from Spirobifluorene paper 
SPIRO_L_0 = 0.9 * 10**-9
SPIRO_J_0 = 46 * 10**-3 * E_CHARGE  # J
SPIRO_LAMBDA_0 = 644 * 10**-3 * E_CHARGE  # J
SPIRO_ALPHA = 5.6
SPIRO_J_HOP = 0.34 * 10**-3 * E_CHARGE  # J
SPIRO_LAMBDA_HOP = 644 * 10**-3 * E_CHARGE  # J


#############################################################################################
# Marcus Theory functions
#############################################################################################
def avg_marcus_rate(L, J, Lambda, field, alpha=1):
    """Returns the rate of charge transfer between two points using Marcus theory in a constant field.  
    
    Inputs:
        L -- characteristic spacing in meters
        J --  Coupling energy in Joules
        Lambda -- Reorganization energy in Joules
        field -- strength of applied electric field in V/m
        alpha -- Ratio between point spacing and characteristic spacing 
    Returns
        transfer rate in s^-1
    """
    
    delta_G = L * field * E_CHARGE * alpha
    term_1 = 2 * PI / H_BAR * J**2
    term_2 = (4 * PI * Lambda * K_B * T_ROOM)**-0.5 
    term_3 = np.exp(-(Lambda + delta_G)**2 / (4 * Lambda * K_B * T_ROOM))
    return term_1 * term_2 * term_3

def get_on_chain_rate_spiro(field):
    """Uses values from Spirobifluorene paper to get an average rate."""
    return avg_marcus_rate(SPIRO_L_0, SPIRO_J_0, SPIRO_LAMBDA_0, -1 * np.abs(field))

def get_interchain_rate_spiro(field):
    """Uses values from Spirobifluorene paper to get an average rate."""
    return avg_marcus_rate(SPIRO_L_0, SPIRO_J_HOP, SPIRO_LAMBDA_HOP, -1 * np.abs(field), SPIRO_ALPHA) 

def get_on_chain_rate(potential_difference):
    """Uses values from Spirobifluorene paper for Spiro except for the characteristic distance, instead getting delta_G from potential difference."""
    delta_G = potential_difference * E_CHARGE  # Joules
    Lambda = SPIRO_LAMBDA_0
    J = SPIRO_J_0
    term_1 = 2 * PI / H_BAR * J**2
    term_2 = (4 * PI * Lambda * K_B * T_ROOM)**-0.5 
    term_3 = np.exp(-(Lambda + delta_G)**2 / (4 * Lambda * K_B * T_ROOM))
    return term_1 * term_2 * term_3
            

#############################################################################################
# Helper Functions
#############################################################################################

def get_poisson_times(rate, interval, n_sets, verbose=False):
    """Given an time interval and a mean time between events, uses the Poisson distribution to randomly determine
    the number of events that occur and the times of those events.  Repeats for n_sets duplicate experiments."""
    
    # Determine how many random values are needed, using conservative estimate from standard deviation
    mean_events = interval * rate
    stdev = np.sqrt(mean_events) # This is stdev of the number of values, not of the waiting time
    max_values = int(np.ceil(mean_events + stdev*5)) + 1
    
    if verbose:
        print('Number of Random Numbers: %.0f Million' % (max_values * n_sets / 10**6))
    
    # Generate random values
    rnd = np.random.random((max_values, n_sets))
    t = - np.log(rnd) / rate
    
    # Trim to interval
    t_cumulative = np.cumsum(t, axis=0)
    t_clipped = np.clip(t_cumulative, 0, interval)
    t[1:, :] = np.diff(t_clipped, axis=0)
    
    # Find length of each column
    n_intervals = np.argmin(t, axis=0)
    
    # Where no events occur, set the first interval time equal to the total interval.  
    no_events_mask = np.where(n_intervals == 1)[0]
    t[0, no_events_mask] = interval
    
    # Truncate extra zeros
    first_empty_row = np.argmin(np.sum(t, axis=1))
    t_trimmed = t[:first_empty_row, :]
    
    return t_trimmed, n_intervals

def get_poisson_times_2(rate, interval, n_sets, verbose=False):
    """Given an time interval and a mean time between events, uses the Poisson distribution to randomly determine
    the number of events that occur and the times of those events.  Repeats for n_sets duplicate experiments.
    
    This version of the functions creates fewer arrays for intermediate steps, improving performance."""
    
    # Determine how many random values are needed, using conservative estimate from standard deviation
    mean_events = interval * rate
    stdev = np.sqrt(mean_events) # This is stdev of the number of values, not of the waiting time
    max_values = int(np.ceil(mean_events + stdev*5)) + 1
    
    if verbose:
        print('Number of Random Numbers: %.0f Million' % (max_values * n_sets / 10**6))
    
    # Generate random values
    event_times = - np.log(np.random.random((max_values, n_sets))) / rate # Time of each event after the preceding event
    
    # Trim to interval
    event_times[1:, :] = np.diff(np.clip(np.cumsum(event_times, axis=0), 0, interval), axis=0)
    
    # Find length of each column
    n_intervals = np.argmin(event_times, axis=0)
    
    # Where no events occur, set the first interval time equal to the total interval.  
    no_events_mask = np.where(n_intervals == 1)[0]
    event_times[0, no_events_mask] = interval
    
    # Truncate extra zeros
    first_empty_row = np.argmin(np.sum(event_times, axis=1))
    event_times = event_times[:first_empty_row, :]
    
    return event_times, n_intervals



def timestamp():
    """Returns a string with the current date and time"""
    return str(datetime.datetime.now()).replace(' ', '__').replace('.', '__').replace(':', ".")
    
#############################################################################################
# Graphing Functions
#############################################################################################

def plot_color_wheel(alpha=1, length=1000):
    """Creates key for plots of color by direction, such as chain_set.plot_lines"""
    center = length / 2 + 0.5
    cols, rows = np.meshgrid(np.arange(length), np.arange(length))
    theta = np.arctan2(rows - center, cols - center)
    d = np.sqrt((rows - center)**2 + (cols - center)**2)
    
    r, g, b = flow.color_by_angle(theta * 180/np.pi)
    
    rgba = np.stack([
        r.reshape((length, length)), 
        g.reshape((length, length)),
        b.reshape((length, length)), 
        np.ones((length, length))], axis=2)
    
    # Add outline and transparent color
    r1 = 0.90 * length / 2
    r2 = 0.95 * length / 2
    ring = np.where((d > r1) & (d <= r2))
    
    rgba[ring[0], ring[1], :] = np.array([0.5, 0.5, 0.5, 1.0])
    background = np.where(d > r2)
    rgba[background[0], background[1], :] = np.array([0, 0, 0, 0])
    
    plt.imshow(rgba)
    plt.axis('off')
    
    return plt.gcf()


def consistent_imshow(array, **kwargs):
    """Rotates and flips heat plot to match scatter plot data."""
    return plt.imshow(np.transpose(array), origin='lower', **kwargs)


def reduce_spacing(array: np.ndarray, factor: int, fun=np.mean) -> np.ndarray:
    """Reduces the size of a 1D array by an integer factor by averaging adjacent points"""
    new_length = len(array) // factor
    new_array = np.zeros(new_length)
    for i in range(new_length):
        start = factor * i
        end = factor * (i + 1)
        new_array[i] = fun(array[start:end])
    return new_array

#############################################################################################
# Math Functions
#############################################################################################

def integrate(x, y):
    slice_sizes = np.abs(np.diff(x))
    mean_values = (y[:-1] + y[1:]) / 2
    return np.sum(slice_sizes * mean_values)

def normalize(x, y):
    assert len(x) == len(y)
    return y / integrate(x, y)

def x_centroid(x, y):
    return integrate(x, x*y) / integrate(x, y)

def gaussian_model(x, M, S):
    return 1 / (2 * np.pi) * np.exp(-(x-M)**2 / (2 * S**2))

def lognormal_model(x, M, S):
    return 1 / (2 * np.pi * x) * np.exp(-(np.log(x)-M)**2 / (2 * S**2))

def lognormal_fit(x, y, n_points=100, guess_width=1):
    guess = [np.log(x[np.argmax(y)]), guess_width]
    fit_range = np.where(y > np.max(y) / 10)
    params, extras = opt.curve_fit(lognormal_model, x[fit_range], y[fit_range], p0=guess)
    
    
    new_x = np.logspace(np.log10(x[0]), np.log10(x[-1]), n_points)
    new_y = lognormal_model(new_x, *params)
    return new_x, new_y

def scale_to_int(a, vmin, vmax, n_steps):
    a = np.array(a)
    normalized_data = (a - vmin) / (vmax - vmin)
    rescaled_data = normalized_data * n_steps
    floor_data = np.floor(rescaled_data)
    final = floor_data.astype(int)
    return final


#############################################################################################
# Artificial Chains
#############################################################################################


def make_chains(unit_length, n_units, n_chains, method, angle=0, box_x=800, box_y=800, box_z=3, silent=False):
    # Angle is in degrees.  Choose a method from those listed below.  
    radians = np.radians(angle)
    
    all_chains = np.zeros((n_units * n_chains, 3))
    progress_bar = tqdm(total=n_chains, desc='Generating Chains...', leave=False)
    
    if method == 'Straight':
        # Create single chain
        single_chain = np.zeros((n_units, 3))
        single_chain[:, 0] = np.arange(n_units) * unit_length * np.cos(radians)
        single_chain[:, 1] = np.arange(n_units) * unit_length * np.sin(radians)
        x_size = n_units * unit_length * np.cos(radians)
        y_size = n_units * unit_length * np.sin(radians)
        
        # Place Chains
        for chain_number in range(n_chains):
            first_index, last_index = (chain_number * n_units, (chain_number + 1) * n_units)
            
            chain_start = np.zeros(3)
            chain_start[0] = np.random.random() * (box_x - x_size)  # Prevents being placed outside the box
            chain_start[1] = np.random.random() * (box_y - y_size)
            chain_start[2] = np.random.random() * box_z
            new_chain = single_chain + chain_start[None, :]
            
            all_chains[first_index:last_index, :] = new_chain
            progress_bar.update(chain_number)
            
    if method == 'Boundary':
        # Create single chain
        single_chain = np.zeros((n_units, 3))
        single_chain[:, 0] = np.arange(n_units) * unit_length * np.cos(radians)
        single_chain[:, 1] = np.arange(n_units) * unit_length * np.sin(radians)
        x_size = n_units * unit_length * np.cos(radians)
        y_size = n_units * unit_length * np.sin(radians)
        
        # Create mirrored chain
        mirrored_chain = np.array(single_chain)
        mirrored_chain[:, 0] = single_chain[:, 0] * -1 + np.max(single_chain[:, 0])
        
        # Place chains
        for chain_number in range(n_chains):
            first_index, last_index = (chain_number * n_units, (chain_number + 1) * n_units)
            
            chain_start = np.zeros(3)
            chain_start[0] = np.random.random() * (box_x - x_size)  # Prevents being placed outside the box
            chain_start[1] = np.random.random() * (box_y - y_size)
            chain_start[2] = np.random.random() * box_z
            if chain_start[0] < 0.5 * (box_x - x_size):
                new_chain = single_chain + chain_start[None, :]
            else:
                new_chain = mirrored_chain + chain_start[None, :]
            
            all_chains[first_index:last_index, :] = new_chain
            progress_bar.update(chain_number)
        
            
    return all_chains

def plot_chains_simple(xyz, chain_length, n_chains, sparsity=10, **kwargs):

    progress_bar = tqdm(total=n_chains, desc='Plotting Chains...', leave=False)
    
    for chain_number in range(n_chains):
        if chain_number % sparsity == 0:
            first_index, last_index = (chain_number * chain_length, (chain_number + 1) * chain_length)
            x = xyz[first_index:last_index, 0]
            y = xyz[first_index:last_index, 1]
            plt.plot(x, y, **kwargs)
            progress_bar.update(chain_number)
            
#############################################################################################
# Time functions
#############################################################################################
            
def date_string():
    return str(datetime.date.today())

def time_string():
    return (str(datetime.datetime.today())[11:16]).replace(' ', '_')

def date_time_string():
    return (str(datetime.datetime.today())[:16]).replace(' ', '_')


#############################################################################################
# File System Functions
#############################################################################################

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

def get_xyz(path, sorted_files, file_number, bin_size, rotate=False, max_dim=800):
    xyz_filename = sorted_files[file_number]
    xyz_full_path = path + '/' + xyz_filename
    xyz = np.loadtxt(xyz_full_path) * bin_size
    print('Chosen File: ', xyz_filename)
    if rotate:
        new_xyz = np.array(xyz)
        new_xyz[:, 1] = max_dim - xyz[:, 1] 
        xyz = new_xyz
        
    return xyz, xyz_full_path


class FileManager:
    """This class helps to load PathExperiment functions, select them based on their parameters, track memory use, and modify files to reflect any changes in this module."""
    def __init__(self, parent_dir):
        self.parent_dir = parent_dir
        self.files = []  # List of loaded experiment objects
        
    def show_options(self):
        """Shows directories in the parent directory that may contain simulation data."""
        for i in os.walk(self.parent_dir):
            options = [name for name in i[1] if 'ipynb' not in name]
            break    
        print(options)
        
    def add_files(self, directory, skip_attrs=None, **kwargs):
        """Loads simulation data from the chosen directory. 
        Inputs:
            Directory -- Place to look for files, relative to the parent_directory attribute of the FileManager object.  String.
            skip_attrs -- Attributes to NOT load in order to reduce RAM impact.  List of strings.  
            **kwargs -- each key/value pair provided is compared against the simple_attributes.txt and chains_simple_attributes.txt 
                        files for each simulation.  Only simulations that match the given parameters are loaded.
        """
        
        path = self.parent_dir + directory
        
        # Get subdirectories (one per experiment)
        for i in os.walk(path):
            experiment_dirs = [name for name in i[1] if 'ipynb' not in name]
            break
        else:
            raise FileNotFoundError('Parent Directory does not exist')
        # Add experiments that match kwargs
        for exp in sorted(experiment_dirs):
            # Get basic data
            try:
                with open(path + exp + '/simple_attributes.txt') as file_1:
                    with open(path + exp + '/chains_simple_attributes.txt') as file_2:
                        exp_attributes = json.load(file_1) | json.load(file_2)
            except FileNotFoundError:
                print('Missing Simple Attributes for', exp)
                continue

            # Skip experiments that don't match requirements
            for key, value in kwargs.items():
                if key in exp_attributes:
                    if exp_attributes[key] != value:
                        break
                else:
                    print('%s not present for file %s in set %s' % (key, exp, path)) 
                    break
            else:
                # If no problems, load file.  See "for.. else" syntax for details.  
                try:
                    new = PathExperiment().from_file(path + exp, skip_attrs=skip_attrs)
                except ValueError:
                    print(exp)
                    raise
                except FileNotFoundError:
                    print('Could not load', exp)
                    continue
                
                self.files.append(new)
                print('File Added: %d Total Files' % len(self.files))
        print('RAM memory % used:', psutil.virtual_memory()[2])
        print('RAM memory Available: %.1f GB' % (int(psutil.virtual_memory()[1]) / 10.0**9))
        
        
    def get_files_where(self, **kwargs):
        """Returns a subset of self.files with attributes that match the provided key/value pairs."""
        return [f for f in self.files if match(kwargs, (f.__dict__ | f.chains.__dict__))]
                    
                    
    def update(self, exp_class):
        """Normally, if a user made changes to this module, PathExperiment objects would have to be re-initialized and the data
        re-loaded to reflect changes.  This method gets around that problem by initializing new PathExperiment objects using
        the updated module, then copying the attributes from the old objects to the new.  This saves a lot of time in 
        troubleshooting and development.  The input should be the PathExperiment class after it is reloaded.
        
        Currently, this only works for changes to the PathExperiment class.  Changes to the ChainSet child class are not included."""

        
        for i, f in enumerate(self.files):
            source = f
            target = exp_class()
            
            for key, value in source.__dict__.items():
                try:
                    setattr(target, key, value)
                except AttributeError:
                    print('Could not update %s' % key)
                    
            self.files[i] = target
            
        print('Files Updated')
                    
    def clear(self):
        self.files = []
        
def match(small_dict, large_dict):
    """Returns True if small_dict is a matching subset of large_dict, otherwise returns False"""
    for key in small_dict:
        if key in large_dict:
            if small_dict[key] != large_dict[key]:
                return False
        else:
            return False
        
    return True

#############################################################################################
# Testing Functions
#############################################################################################

def test_get_poisson_times():
    rate, interval, trials = (1, 100, 5000)
    times, n_intervals = get_poisson_times(rate, interval, trials)
    bin_size = np.mean(times) / 10
    hist = np.bincount(np.floor(times.flatten() / bin_size).astype(int))[1:]
    plt.plot(np.arange(len(hist)) * bin_size, hist, marker='.', linewidth=0)
    plt.yscale('log')
    plt.title('Rate %.2f Interval %.2f Trials %d' % (rate, interval, trials))
    plt.xlabel('Interval Time (Input Units)')
    plt.ylabel('Number of Occurances')
    plt.show()

    print(np.sum(times, axis=0))
    print(n_intervals)
    print(times.astype(int))
    

def test_get_rate_eff():
    distance = 10 * 10**-7 # cm
    field_values = 10**np.arange(1, 6, 0.5) # V/cm
    desc_values = np.array((1, 2, 5, 10, 50))  # Descritization
    n_fields = len(field_values)
    n_desc = len(desc_values)
    
    rates = np.zeros((n_fields, n_desc))
    reverse_rates = np.zeros((n_fields, n_desc))
    
    for i, f in enumerate(field_values):
        for j, d in enumerate(desc_values):
            rates[i, j] = get_rate_eff(f, distance, d)  # Per second
            reverse_rates[i, j] = get_rate_eff(f, -distance, d)  # Per second

    corrections = rates / rates[:, 0][:, None]
    reverse_corrections = reverse_rates / reverse_rates[:, 0][:, None]
            
            
    # Plot results
    for j, d in enumerate(desc_values):
        plt.plot(field_values, rates[:, j], marker='.')
    plt.title('Forward Rates')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(desc_values)
    plt.xlabel('Field (V/cm)')
    plt.ylabel('Rate (s^-1)')
    plt.show()    
    
    for j, d in enumerate(desc_values):
        plt.plot(field_values, reverse_rates[:, j], marker='.')
    plt.title('Reverse Rates')
    plt.xscale('log')
    plt.yscale('log')
    # plt.ylim(0, 1.2)
    plt.legend(desc_values)
    plt.xlabel('Field (V/cm)')
    plt.ylabel('Rate (s^-1)')
    plt.show()
    
    for j, d in enumerate(desc_values):
        plt.plot(field_values, corrections[:, j], marker='.')
    plt.title('Forward Rates')
    plt.xscale('log')
    # plt.yscale('log')
    plt.legend(desc_values)
    plt.xlabel('Field (V/cm)')
    plt.ylabel('k_eff / k_0')
    plt.show()
    
    for j, d in enumerate(desc_values):
        plt.plot(field_values, corrections[:, j], marker='.')
    plt.title('Forward Rates')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(desc_values)
    plt.xlabel('Field (V/cm)')
    plt.ylabel('k_eff / k_0')
    plt.show()
    
    for j, d in enumerate(desc_values):
        plt.plot(field_values, reverse_corrections[:, j], marker='.')
    plt.title('Reverse Rates')
    plt.xscale('log')
    # plt.yscale('log')
    plt.ylim(0, 1.2)
    plt.legend(desc_values)
    plt.xlabel('Field (V/cm)')
    plt.ylabel('k_eff / k_0')
    plt.show()
    
    for j, d in enumerate(desc_values):
        plt.plot(field_values, reverse_corrections[:, j], marker='.')
    plt.title('Reverse Rates')
    plt.xscale('log')
    plt.yscale('log')
    # plt.ylim(0, 1.2)
    plt.legend(desc_values)
    plt.xlabel('Field (V/cm)')
    plt.ylabel('k_eff / k_0')
    plt.show()
    
    
    
