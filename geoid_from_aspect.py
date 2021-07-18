#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 11:59:48 2021

This code will load output from ASPECT, calculate the geoid, and plot it.
This only works for 2D, Cartesian models.

@author: max
"""

#%% imports
import numpy as np
#from vtureader import *
import h5py as h5
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from geoid_functions import *

#%% parse command-linne arguments
# usage of this program: ./groid_from_aspect.py solution_foldergraphics_name
# graphics_name is the filename for graphics, not including the file extension (.svg)
if len(sys.argv) < 3: 
    raise SystemExit("Usage: ./geoid_from_aspect.py output_directory graphics_name")
output_folder = sys.argv[1]
graphics_file = sys.argv[2]
    
#%% Do the geoid calculation
mesh_file = output_folder+'/solution/mesh-00000.h5'
file0 = output_folder+'/solution/solution-00000.h5'
cells,x,y,rho,topo = load_data(mesh_file,file0) 
ind_surf,x_surf,ind_botm,x_botm = get_boundary_arrays(x,y)
x_obs,y_obs = get_observation_vector(x,y,use_nodes=False)
delta_rho = get_density_anomaly(cells,x,y,rho)

N_interior = interior_geoid(cells,x,y,delta_rho,x_obs,y_obs,method='line')
#N_interior_patch = interior_geoid(cells,x,y,delta_rho,x_obs,y_obs,method='patch') # very slow.
N_surface = surface_geoid(ind_surf,x,rho,topo,x_obs)
N_cmb = cmb_geoid(ind_botm,x,y,rho,topo,x_obs)
N_total = N_surface + N_interior + N_cmb

#%% plot the geoid:
fig, (ax1,ax2)=plt.subplots(2,1,figsize=(4,8))
ax1.plot(x[ind_botm],10*topo[ind_botm],label='CMB Topography (x10)')
ax1.plot(x[ind_surf],topo[ind_surf],label='Surface Topography')
ax1.legend()
ax1.set_title('Topography')
ax1.set_ylabel('Topography (m)')
ax2.set_title('Geoid')
ax2.plot(x_obs,N_interior,label='interior')
try:
    ax2.plot(x_obs,N_interior_patch,'--',label='interior-patch')
except:
    pass
ax2.plot(x_obs,N_surface,label='surface')
ax2.plot(x_obs,N_cmb,label='cmb')
# plt.plot(x_obs,N_intfast,'--',label='quick interior')
ax2.plot(x_obs,N_total,'k',label='Total')
ax2.set_ylabel('Geoid anomaly (m)')
ax2.legend()
#plt.show()
plt.savefig(graphics_file + '_geoid.svg')