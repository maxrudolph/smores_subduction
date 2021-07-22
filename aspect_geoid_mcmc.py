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

def setup_aspect_runs(run_dir='/dev/shm/geoidtmp/',base_input_file='boxslab_base.prm'):
    # make the run directory:
    try:
        subprocess.run(['mkdir',run_dir[:-1]])
    except:
        print('run directory '+run_dir+' already exists or cannot be created')        
    # copy the aspect executable to the run directory
    subprocess.run(['cp','aspect-master.fast',run_dir])
    subprocess.run(['cp',base_input_file,run_dir])
    subprocess.run(['cp','boxslab_topcrust.wb',run_dir])
    
    # return the directory in which aspect has been placed.
    return run_dir

def run_aspect(parameters,base_input_file = 'boxslab_base.prm',run_dir='./'): 
    
    #should remove previous output directory with same name
    #run as ls command before rm (be careful)
    
    prm_filename =  'run.prm'
    # replace strings in the 'base input file' with the parameter values, which are stored in a dictionary.
    subprocess.run(["cp",base_input_file,prm_filename],cwd=run_dir)
    for key in parameters.keys():
        # use the sed command to replace each of the keys in the ditionary with its appropriate value.
        subprocess.run(["sed","-i","-e","s/"+key+"/"+'{:e}'.format(parameters[key])+"/g",prm_filename],cwd=run_dir)
    
    # run aspect
    aspect_command = './aspect-master.fast' #+ prm_filename
    subprocess.run([aspect_command, prm_filename],cwd=run_dir) # run aspect

    #aspect_command = '../aspect.fast' 
    #subprocess.run(['mpirun', '-n', '20', aspect_command, prm_filename])
    
def calculate_geoid(output_folder,run_dir='./'):
    # Do the geoid calculation
    mesh_file = run_dir + output_folder + '/solution/mesh-00000.h5'
    #    print(mesh_file)

    file0 = run_dir + output_folder + '/solution/solution-00000.h5'
    cells,x,y,rho,topo = load_data(mesh_file,file0) 
    ind_surf,x_surf,ind_botm,x_botm = get_boundary_arrays(x,y)
    x_obs,y_obs = get_observation_vector(x,y,use_nodes=False)
    delta_rho = get_density_anomaly(cells,x,y,rho)
    
    N_interior = interior_geoid(cells,x,y,delta_rho,x_obs,y_obs,method='line')
    N_surface = surface_geoid(ind_surf,x,rho,topo,x_obs)
    N_cmb = cmb_geoid(ind_botm,x,y,rho,topo,x_obs)
    N_total = N_surface + N_interior + N_cmb
    return N_total

def MCMC(starting_solution=None, parameter_bounds=None, observed_geoid=None, n_steps=10,save_start=0,save_skip=1,var=None):
    # This function should implement the MCMC procedure
    # 1. Define the perturbations (proposal distributions) for each parameter
    #    and the bounds on each parameter. Define the total number of steps and
    #    the number of steps between saving output.
    # if var is None, assume that this calculation uses a hierarchical hyperparameter.
    if var is None:
        accepted_var = 1.0
        allow_hierarchical = True   #variance can change 
    else:
        accepted_var = var
        allow_hierarchical = False  #sigma prime = sigma
    accepted_solution = deepcopy(starting_solution)
    #initial perturbation / input starting_solution not used to produce observed_geoid
    # 2. For the initial guess, calculate the misfit and the likelihood.
    # ... fill in code here ...
    run_dir = setup_aspect_runs() # setup aspect to run in /dev/shm
    run_aspect(accepted_solution,run_dir=run_dir)
    accepted_geoid = calculate_geoid('boxslab_base',run_dir=run_dir)
    accepted_residual = accepted_geoid - observed_geoid
    accepted_magnitude = np.dot(accepted_residual, accepted_residual)
    print(accepted_magnitude)
    
    #create ensemble archive
    ensemble_residual = []
    solution_archive = []
    
    # 3. Begin the MCMC procedure
    for iter in range(n_steps):
        success = False
        n_tries = 0
        while( success is False and n_tries < 100):
            n_tries += 1
            proposed_solution = deepcopy(accepted_solution)
            # n_options should be equal to the number of parameters if allow_hierarchical is False
            # otherwise, n_options = (number of parameters) + 1
            if(allow_hierarchical == False):
                n_options = len(parameters)
            else:
                n_options = len(parameters) + 1
           
            
            # Choose one an opitions at random
            
            # pick integer between (0, n_options)
            
            
            
            vary_parameter = np.random.randint(0,n_options)
            key = list(parameters.keys())[vary_parameter]
            print(key)
            proposed_solution[key] = proposed_solution[key] + 1e-17*np.random.randn()
            print(proposed_solution[key])
            #need case for when allow_hierarchical is True
            
            if proposed_solution[key] > parameter_bounds[key][1] or \
                proposed_solution[key] < parameter_bounds[key][0]:
                success = False
            else:
                success = True
            #     0. perturb the value of the 0th parameter
            # ...
            #     n-1. perturn the value of the (n-1)th parameter
            #     n. if allow_hierarchical, perturb the value of the variance (hyperparameter)
            
            # Check to ensure that the proposed solution satisfies any bounds placed on the parameters.
            # If checks pass, set success to True
            # ...
            # if n_tries > some threshold, we are stuck in an infinite loop. print an error message and exit.
        # Calculate the forward model for the proposed solution
        run_aspect(proposed_solution, 'boxslab_base.prm',run_dir=run_dir)
        # calculate the geoid from the aspect model.
        proposed_geoid = calculate_geoid('boxslab_base',run_dir=run_dir)
        # calculate the misfit
        proposed_residual = proposed_geoid - observed_geoid
        proposed_magnitude = np.dot(proposed_residual, proposed_residual)
        print(proposed_magnitude)
        
        # calculate the likelihood
        proposed_var = 1.0  #?
        N = len(observed_geoid)
        #Cd_hat = np.identity(N)
        
        log_alpha = N/2*((np.log(accepted_var)) - np.log(proposed_var)) \
            - 1/(2*proposed_var)*proposed_magnitude + 1/(2*accepted_var)*accepted_magnitude
        print('log_alpha is:', log_alpha)

        proposed_likelihood = None
        # calculate the probability of acceptance using the Metropolis-Hastings Criterion
        if log_alpha > 0 or log_alpha > np.log(np.random.rand()):
            accepted_solution = deepcopy(proposed_solution)
            accepted_var = proposed_var
            accepted_magnitude = proposed_magnitude
            # accept the proposed solution by copying the proposed solution
            # to the accepted solution.
            pass
        if iter > save_start and not (iter % save_skip):
            ensemble_residual.append(accepted_magnitude)
            solution_archive.append(deepcopy(accepted_solution))
            
            # save the accepted solution to the archive
            # also save the accepted_var
            # also save the likelihood of the accepted solution
            # also save the misfit of the accepted solution
            pass                        
    return ensemble_residual, solution_archive # return the solution archive - this is the ensemble!


#def main():
    # 1. guess an initial solution.
parameters = dict()
parameters['PREFACTOR0'] = 2e-15#1.4250e-15 
parameters['PREFACTOR1'] = 1.4250e-15
parameters['PREFACTOR2'] = 1.0657e-18

parameter_bounds = dict()
#add third number for amplitude of pertubation
parameter_bounds['PREFACTOR0'] = [1.0e-15, 2.0e-15]
parameter_bounds['PREFACTOR1'] = [1.4e-15, 1.45e-15]
parameter_bounds['PREFACTOR2'] = [9e-19, 1.2e-18]

#run_aspect(parameters,'boxslab_base.prm')
observed_geoid = calculate_geoid('starter')
residual, solution_archive = MCMC(parameters, parameter_bounds, observed_geoid, var=1.0,n_steps=100,save_start=50)
    # 2. call the MCMC function
    #MCMC(parameters)
    # 3. plotting/analysis of the output.
#    pass

#calculate_geoid('boxslab_topcrust_deep')
#%% plotting
import matplotlib.pyplot as plt
plt.figure()
plt.plot(residual)
plt.show()

plt.figure()
for i in range(3):
    plt.subplot(1,3,i+1)
    key = list(parameters.keys())[i]
    plt.hist([p[key] for p in solution_archive])
plt.show()