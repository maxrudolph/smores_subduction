#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 09:24:24 2021

@author: max
"""
import numpy as np
import subprocess
from geoid_functions import *
from copy import deepcopy

def run_aspect(parameters,base_input_file):
    prm_filename = 'run.prm'
    # replace strings in the 'base input file' with the parameter values, which are stored in a dictionary.
    subprocess.run(["cp",base_input_file,prm_filename])
    for key in parameters.keys():
        # use the sed command to replace each of the keys in the ditionary with its appropriate value.
        subprocess.run(["sed","-i","-e","s/"+key+"/"+'{:e}'.format(parameters[key])+"/g",prm_filename])
    
    # run aspect
    aspect_command = 'mpirun -n 20 ./aspect ' + prm_filename
    subprocess.run(aspect_command) # run aspect

def calculate_geoid(output_folder):
    # Do the geoid calculation
    mesh_file = output_folder + '/solution/mesh-00000.h5'
    file0 = output_folder + '/solution/solution-00000.h5'
    cells,x,y,rho,topo = load_data(mesh_file,file0) 
    ind_surf,x_surf,ind_botm,x_botm = get_boundary_arrays(x,y)
    x_obs,y_obs = get_observation_vector(x,y,use_nodes=False)
    delta_rho = get_density_anomaly(cells,x,y,rho)
    
    N_interior = interior_geoid(cells,x,y,delta_rho,x_obs,y_obs,method='line')
    N_surface = surface_geoid(ind_surf,x,rho,topo,x_obs)
    N_cmb = cmb_geoid(ind_botm,x,y,rho,topo,x_obs)
    N_total = N_surface + N_interior + N_cmb
    return N_total

def MCMC(starting_solution=None,n_steps=1000,save_start=500,save_skip=10,var=None):
    # This function should implement the MCMC procedure
    # 1. Define the perturbations (proposal distributions) for each parameter
    #    and the bounds on each parameter. Define the total number of steps and
    #    the number of steps between saving output.
    # if var is None, assume that this calculation uses a hierarchical hyperparameter.
    if var is None:
        accepted_var = 1.0
        allow_hierarchical = True
    else:
        accepted_var = var
        allow_hierarchical = False
    accepted_solution = deepcopy(starting_solution)
    # 2. For the initial guess, calculate the misfit and the likelihood.
    # ... fill in code here ...
    # 3. Begin the MCMC procedure
    for iter in range(n_steps):
        success = False
        n_tries = 0
        while( success is False ):
            n_tries += 1
            proposed_solution = deepcopy(accepted_solution)
            # n_options should be equal to the number of parameters if allow_hierarchical is False
            # otherwise, n_options = (number of parameters) + 1
            option = np.random.randi(n_options)
            # Choose one an opitions at random
            #     0. perturb the value of the 0th parameter
            # ...
            #     n-1. perturn the value of the (n-1)th parameter
            #     n. if allow_hierarchical, perturb the value of the variance (hyperparameter)
            
            # Check to ensure that the proposed solution satisfies any bounds placed on the parameters.
            # If checks pass, set success to True
            # ...
            # if n_tries > some threshold, we are stuck in an infinite loop. print an error message and exit.
        # Calculate the forward model for the proposed solution
        run_aspect(proposed_solution)
        # calculate the geoid from the aspect model.
        calculate_geoid(output_folder)
        # calculate the misfit
        proposed_misfit = None
        # calculate the likelihood
        proposed_likelihood = None
        # calculate the probability of acceptance using the Metropolis-Hastings Criterion
        prob_accept = None # note - this is the logarithm of the acceptance probability
        if prob_accept < np.random.rand():
            # accept the proposed solution by copying the proposed solution
            # to the accepted solution.
            pass
        if iter > save_start and not (iter % save_skip):
            # save the accepted solution to the archive
            # also save the accepted_var
            # also save the likelihood of the accepted solution
            # also save the misfit of the accepted solution
            pass                        
    return None # return the solution archive - this is the ensemble!

def main():
    # 1. guess an initial solution.
    parameters = dict()
    parameters['PREFACTOR0'] = 1.4250e-15
    parameters['PREFACTOR1'] = 1.4250e-15
    parameters['PREFACTOR2'] = 1.0657e-18
    # 2. call the MCMC function
    MCMC(parameters)
    # 3. plotting/analysis of the output.
    pass