#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 10:10:01 2021

Helper functions for geoid calculation

@author: max
"""
#%% physical constants
G = 6.67430e-11 # Universal constant of gravitation, in N*m^2/kg^2
g0 =  9.80665 

#%% introduce routines for calculating geoid anomaly associated with a subsurface 'patch'
# calculate geoid anomaly associated with cell
def G2(y,z,yp,zp):
    # Chapman (1979) equation 18. y,z are coordinates of the corner of the patch.
    # yp,zp are evaluation/measurement point coordinates.
    term1 = np.zeros_like(zp)
    term1[z-zp==0]=0
    mask = (z-zp != 0)
    term1[mask] = (z-zp[mask])*np.log((y-yp[mask])**2 +(z-zp[mask])**2)
    return (y-yp) * (term1 - 2*z +2*np.abs(y-yp)*np.arctan2((z-zp),np.abs(y-yp)))\
        - 2*y*z +((z-zp)**2+(y-yp)**2)*np.arctan2((y-yp),(z-zp)) + (z-zp)*(y-yp)
        
def N_patch(y1,y2,z1,z2,rho,x,z):
    # Calculate the geoid anomaly for a 'patch' of anomalous density. Assumes x2>x1 and z2>z1
    # Assumes that z is the vertical coordinate, increasing DOWNWARD
    return -G*rho/g0*(G2(y2,z2,x,z) - G2(y1,z2,x,z) - G2(y2,z1,x,z) + G2(y1,z1,x,z))

def G3(y0,z0,yp):
    # Calculate the integrand required for a horizontal sheet anomaly.
    # Based on integration of Chapman (1979) equation 13-15
    # y increases to the right in this subroutine. z increases downward.
    # y,z are the coordinates of the endpoints of the line source
    # yp,zp are the evaluation points.
    mask = (yp-y0)**2+z0**2 != 0.
    term1 = np.zeros_like(yp)
    term1[mask] = (yp[mask]-y0)*np.log((yp[mask]-y0)**2+z0**2)
    return term1 - 2*z0*np.arctan2(yp-y0,z0)-2*y0

def N_sheet(x1,x2,z0,sigma,x):
    # Calculate the geoid anomaly for a sheet of anomalous density.
    # the sheet is buried at a depth z0.
    # x1 and x2 are the start and end x-coordinates of the sheet (x increasing rightward)
    # x,z are the locations of the observation points (y increasing upward)
    return -G*sigma/g0 * (G3(x2,z0,x)-G3(x1,z0,x))

#%% Load data
def load_data(meshfile,datafile):
    f = h5.File(datafile, 'r')
    mf = h5.File(meshfile, 'r')
    x = mf['nodes'][:,0]
    y = mf['nodes'][:,1]
    cells = mf['cells'][:]
    rho = f['density'][:].flatten()
    topo = f['dynamic_topography'][:].flatten()
    return cells,x,y,rho,topo

def get_dA(cells,x,y):
    cellx = x[cells]
    celly = y[cells]
    # calculates area of each finite element cell 
    dx = np.max(cellx,axis=1)-np.min(cellx,axis=1)
    dy = np.max(celly,axis=1)-np.min(celly,axis=1)
    dA = dx*dy
    return dA

def get_boundary_arrays(x,y):
    ind_surf = np.argwhere(np.isclose(y,np.max(y))).flatten()
    ind_tmp = x[ind_surf].argsort()
    ind_surf = ind_surf[ind_tmp]
    x_surf = x[ind_surf]
    
    ind_botm = np.argwhere(np.isclose(y,np.min(y))).flatten()
    ind_tmp = x[ind_botm].argsort()
    ind_botm = ind_botm[ind_tmp]
    x_botm = x[ind_botm]
    return ind_surf,x_surf,ind_botm,x_botm

def get_observation_vector(x,y,use_nodes=False):
    if use_nodes:
        obs_x = x[ind_surf][:,None]
        obs_y = y[ind_surf][:,None]
    else:
        obs_x = np.linspace(x.min(),x.max(),101)[:,None]
        obs_y = y.max()*np.ones_like(obs_x)
    return obs_x,obs_y

def get_density_anomaly(cells,x,y,rho):
    # calculate the background density profile, using values along the left boundary of the domain
    ind_left = np.argwhere(np.isclose(x,0.0))
    rho_left = rho[ind_left] # density values along left boundary
    y_left = y[ind_left] # y-values corresponding to rho values along left boundary
    rho_f = interp.interp1d(y_left.flatten(),rho_left.flatten()) # creates a 1D interpolating function for the density values along the left boundary
    cy = np.mean(y[cells],axis=1)
    rho_ref = rho_f(cy) # evaluate the interpolant everywhere in the domain
    crho = np.mean(rho[cells],axis=1)
    # Use Turcotte and Schubert Equation 5.104
    drho = crho-rho_ref # density anomaly
    dA = get_dA(cells, x, y)
    drho = drho - np.sum(drho*dA)/np.sum(dA) # make average density anomaly 0.
    return drho

def interior_geoid(cells,x,y,drho,obs_x,obs_y,method='line'):
    # calculate the geoid contribution from internal density anomalies.
    # 'line' method is faster and treats each element as an infinite line source
    # 'patch' method is slower and uses formulae for 2D rectangle.
    cellx,celly = x[cells],y[cells] # nodal coordinates for each cell.
    cx,cy = np.mean(cellx,axis=1),np.mean(celly,axis=1) # locations of cell centers

    if method == 'line':
        dA = get_dA(cells,x,y)
        # using Chapman (1979) equation 13:
        Rterm = np.log((obs_x-cx)**2 + (obs_y-cy)**2)
        N =  -G*(drho*dA)[:,None]*Rterm.T/g0
        N = np.sum(N,axis=0)
    elif method == 'patch':
        cellz = np.max(celly)-celly # depth of each node.
        N = np.zeros((len(obs_x),len(cells)))
        for i in tqdm(range(0,len(cells))): #irange(len(cells)):
            x1,x2 = np.min(cellx[i]),np.max(cellx[i])
            z1,z2 = np.min(cellz[i]),np.max(cellz[i])
            if drho[i] != 0.0:
                N[:,i]=( N_patch(x1,x2,z1,z2,drho[i],obs_x,np.zeros_like(obs_y)) ).flatten() # note - last argument is depth, which is zero at surface.
        print(N.shape)
        N = np.sum(N,axis=1)
    else:
        raise NotImplementedError
    return N

def surface_geoid(ind_surf,x,rho,topo,obs_x):
    x_surf = x[ind_surf]
    sigma_surf = -rho[ind_surf]*topo[ind_surf]
    N = np.zeros((len(sigma_surf)-1,len(obs_x)))
    for i in range(len(sigma_surf)-1):
        sigma_tmp = 0.5*(sigma_surf[i+1]+sigma_surf[i])
        N[i,:] = N_sheet(x_surf[i],x_surf[i+1],0.0,sigma_tmp,obs_x).flatten()
    N = np.sum(N,axis=0)
    return N
    
def cmb_geoid(ind_botm,x,y,rho,topo,obs_x,rho_below=9900.):
    # From the ASPECT manual:
    # "For the bottom surface we chose the convection that positive values are up (out)     
    # and negativevalues are in (down)" -> positive topography means mass excess
    sigma_botm = -(rho_below-rho[ind_botm])*topo[ind_botm]
    x_botm = x[ind_botm]
    N = np.zeros((len(sigma_botm)-1,len(obs_x)))
    botm_depth = np.max(y)-np.min(y)
    for i in range(len(sigma_botm)-1):
        sigma_tmp = 0.5*(sigma_botm[i+1]+sigma_botm[i])
        # kludge = place density anomaly at 1m depth until singularities are sorted out.
        N[i,:] = N_sheet(x_botm[i],x_botm[i+1],botm_depth,sigma_tmp,obs_x).flatten()
    N = np.sum(N,axis=0)
    return N