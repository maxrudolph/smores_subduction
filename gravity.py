#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 08:55:28 2021

@author: max rudolph
"""
#%% imports
import numpy as np
#from vtureader import *
import h5py as h5
import scipy.interpolate as interp
import matplotlib.pyplot as plt

#%% physical constants
G = 6.67430e-11 # Universal constant of gravitation, in N*m^2/kg^2

#%% Load data
mesh_file = 'output_boxslab_topcrust_h5/solution/mesh-00000.h5'
file0 = 'output_boxslab_topcrust_h5/solution/solution-00000.h5'
f = h5.File(file0, 'r')
mf = h5.File(mesh_file, 'r')
x=mf['nodes'][:,0]
y=mf['nodes'][:,1]
rho = f['density'][:].flatten()
eta = np.array(f['viscosity']).flatten()
syy = np.array(f['stress_yy'])
topo = np.array(f['dynamic_topography'])
cells = mf['cells'][:]
cellx = x[cells]
celly = y[cells]

#%% Calculate gravity anomaly
# calculate the area of the cells - first calculate a dx and dy
dx = np.max(cellx,axis=1)-np.min(cellx,axis=1)
dy = np.max(celly,axis=1)-np.min(celly,axis=1)
dA = dx*dy
cx = np.mean(cellx,axis=1)
cy = np.mean(celly,axis=1)
crho = np.mean(rho[cells],axis=1)
# make a vector of observation points on the surface
ind_surf = np.argwhere(np.isclose(y,np.max(y))).flatten()
ind_tmp = x[ind_surf].argsort()
ind_surf = ind_surf[ind_tmp]

obs_x = x[ind_surf][:,None]
obs_y = y[ind_surf][:,None]
# calculate the distance from each point on the surface to each subsurface point
dx,dy = obs_x-cx, obs_y-cy
# calculate the background density profile, using values along the left boundary of the domain
ind_left = np.argwhere(np.isclose(x,0.0))
rho_left = rho[ind_left] # density values along left boundary
y_left = y[ind_left] # y-values corresponding to rho values along left boundary
rho_f = interp.interp1d(y_left.flatten(),rho_left.flatten()) # creates a 1D interpolating function for the density values along the left boundary
rho_ref = rho_f(cy) # evaluate the interpolant everywhere in the domain
# calculate the gravity anomaly due to each element at each observation point.
# Use Turcotte and Schubert Equation 5.104
dgamma = (crho-rho_ref)*dA # mass anomaly per unit distance in/out of plane

dg = 2.*G*dgamma[:,None]*(dy/(dx**2+dy**2)).T # dg is (number of cells)x(number of observation points)
dg_interior = np.sum(dg,axis=0)
# use the Bouguer gravity formula to calculate influence of surface.
dg_surf = 2.*np.pi*G*rho[ind_surf].flatten()*topo[ind_surf].flatten()

#%% Calculate gravity anomalies associated with bottom topography
# because the burial depth of the CMB is great, we cannot use the bouguer formula.
# Gravity anomalies here are computed via summation of cell-wise mass anomalies
rho_core = 9903.#-5566 # This must match the value used in ASPECT.

ind_botm = np.argwhere(np.isclose(y,np.min(y))).flatten()
ind_tmp = np.argsort(x[ind_botm]) # sort x-values into increasing order
ind_botm = ind_botm[ind_tmp] # ind bottom is now ordered by increasing x-values
x_botm, y_botm = x[ind_botm].flatten(), y[ind_botm].flatten()
topo_botm = topo[ind_botm].flatten()
rho_botm = rho[ind_botm].flatten()

# calculate coordinates of cell centers along bottom
xc_botm, yc_botm = x_botm[:-1] + np.diff(x_botm)/2., y_botm[:-1] + np.diff(y_botm)/2.
dx_botm = np.diff(x_botm) # cell widths
# get values of density and topography at cell centers.
rhoc_botm = rho_botm[:-1] + np.diff(rho_botm)/2.
topoc_botm = topo_botm[:-1] + np.diff(topo_botm)/2.

# calculate the mass excess/deficit accorting to (topography)*(delta_rho_cmb)
# From the ASPECT manual:
# "For the bottom surface we chose the convection that positive values are up (out) 
# and negativevalues are in (down)" -> positive topography means mass excess
gamma = (rho_core-rhoc_botm)*topoc_botm*dx_botm
dx,dy = obs_x-xc_botm, obs_y-yc_botm
# Calculate the gravity anomaly using Turcotte and Schubert 5.104
dg = 2.*G*gamma[:,None]*(dy/(dx**2+dy**2)).T
dg_botm = np.sum(dg,axis=0)

# compare this with the bouguer formula
plt.figure()
plt.plot(obs_x,dg_botm,label='complicated')
plt.plot(x_botm,2*np.pi*G*(rho_core-rho_botm)*topo_botm,label='bouguer formula')
plt.legend()
plt.show()

#%% Calculate and plot the total geoid anomaly



#%%
plt.figure()
plt.plot(obs_x.flatten()/1e3,dg_interior,label='internal')
#plt.plot(obs_x/1e3,dg_surf,label='surface')
#plt.plot(obs_x/1e3,dg_botm,label='bottom')
plt.legend()
plt.ylabel('Gravity anomaly (m/s$^2$)')
plt.xlabel('Position (km)')
plt.show()

#%%
plt.figure()
plt.scatter(cx,cy,c=crho-rho_ref,s=0.1)
plt.colorbar()
plt.title('density anomaly')
plt.show()

# plt.figure()
# plt.scatter(x,y,c=rho,s=0.1)
# plt.show()

# ind = np.argwhere(np.isclose(y,max(y)))
# x_surf = x[ind]
# syy_surf = syy[ind]
# topo_surf = topo[ind]

# ind = np.argwhere(np.isclose(y,min(y)))
# topo_btm = topo[ind]
# x_btm = x[ind]

# plt.figure()
# plt.scatter(x_surf,topo_surf.flatten(),s=0.1,label='surface')
# plt.scatter(x_btm,topo_btm.flatten(),s=0.1,label='bottom')

# plt.legend()
# plt.xlabel('Horizontal Position (m)')
# plt.ylabel('Dynamic topography (m)')
# plt.ylim((-10000,10000))
# plt.draw()

# plt.figure()
# plt.scatter(x,y,c=np.log10(eta),s=0.1)
# plt.colorbar()
# plt.show()




# cells = vtu_celldata['connectivity'].reshape((-1,4))
# density = vtu_pointdata['density'].flatten()
# x=vtu_points[0,:]
# y=vtu_points[1,:]

# plt.scatter(x,y,c=density,s=1,vmin=2000,vmax=3300)
# #plt.tricontourf(ptri,density,vmin=2000,vmax=3300)

# plt.colorbar()
# plt.show()

# import vtk as vtk
# for i in range(0,32):
#     file0 = 'boxslab_spcrust/solution/solution-00000.{:04d}.vtu'.format(i)
#     reader = vtk.vtkXMLUnstructuredGridReader() 
#     reader.SetFileName(file0)
#     reader.Update()
#     points = reader.GetOutput().GetPointData().GetArray('coordinates')
#     rho = reader.GetOutput().GetPointData().GetArray('density')
#     a=numpy_support.vtk_to_numpy(rho)