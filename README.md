# 4DSTEM_charge_transport
 Combines microscopy and simulation to study charge transport in semiconducting polymers.


#     Basic info

Authors: 
Luke Balhorn
Contact: lk.balhorn@gmail.com

#     Contents

Code:  
-The Python files preprocess_4dstem.py, flow_fields.py, and charge_transfer_assets.py 
contain most of the analysis code, which can be imported into scripts or Jupyter notebooks 
to perform the analysis.  
-Sample Jupyter notebooks preprocess_demo.ipynb, flow_fields_demo.ipynb, and 
charge_transport_demo.ipynb provide a demonstration of how to use the tools in these 
modules.  Additionally, paper_figures.ipynb shows the actual code used for analysis for 
a forthcoming paper.  
-Each of these Jupyter notebooks was also saved as HTML to show sample outputs.  

Data:
-The 4DSTEM directory contains preprocessed 4DSTEM data at two different resolutions.  
This sample data is created by centering and cleaning the 4D-STEM output, then integrating
over cake slices to convert to Q vs. Chi.
-Raw 4DSTEM available is not included at this time because the file sizes are too large for 
GitHub. 
-The chain_geometries directory contains output xyz coordinates from structural simulations.
Each chain is comprised of 54 xyz positions.  Increasing numbers in the filenames indicate 
increasing degrees of alignment.  
-The transport_simulations directory contains outputs of a charge transport simulation using
one of the chain_geometries as its input.


#     Sample Study

To see the motivations behind this code and the results of a recent study, see the thesis
"Structure-function relationships in semiconducting polymers : new methods combining 
transmission electron microscopy and Monte Carlo simulations" by Luke Balhorn, available
online through Stanford University.


#     Analysis Steps

Analysis begins with the output of a 4D-STEM experiment.  The preprocessing package and
notebook are used to clean and center diffraction data, then integrate and convert to 
radial coordinates.  

This data can then be visualized using the flow_fields package and notebook.  

Expected alignment at each point is expressed on a spherical harmonic basis set, which is
used as an input to the structural simulation.  Structural simulation code is available at
https://github.com/SpakowitzLab/wlcsim.  

Outputs of the structural simulation are available in the chain_geometries directory and 
are used as inputs to charge transport simulations.  These simulations are performed and
analyzed using the charge_transfer package and notebook.  
