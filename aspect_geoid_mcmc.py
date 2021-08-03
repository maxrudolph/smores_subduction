#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 09:24:24 2021

@author: max
"""
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import pickle
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
    #subprocess.run(['cp', 'aspect.fast', run_dir])
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
        subprocess.run(["sed","-i","-e","s/"+key+"/"+'{:e}'.format(parameters[key])+"/g",prm_filename], cwd = run_dir)

    # run aspect
    aspect_command = './aspect-master.fast' #+ prm_filename
    subprocess.run([aspect_command, prm_filename],cwd=run_dir) # run aspect

    #aspect_command = './aspect.fast' 
    #subprocess.run(['mpirun', '-n', '20', aspect_command, prm_filename],cwd=run_dir)
    
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

def MCMC(starting_solution=None, parameter_bounds=None, observed_geoid=None, n_steps=100,save_start=0,save_skip=1,var=None):
    # This function should implement the MCMC procedure
    # 1. Define the perturbations (proposal distributions) for each parameter
    #    and the bounds on each parameter. Define the total number of steps and
    #    the number of steps between saving output.
    # if var is None, assume that this calculation uses a hierarchical hyperparameter.
    var_min = 1e-6
    var_max = 1e10
    var_change = 0.1
        
    if var is None:
        accepted_var = 1
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
    #print(accepted_magnitude)
    
    #create ensemble archive
    ensemble_residual = []
    solution_archive = []
    var_archive = []

    # 3. Begin the MCMC procedure
    for iter in range(n_steps):
        success = False
        n_tries = 0
        while( success is False and n_tries < 100):
            n_tries += 1
            proposed_solution = deepcopy(accepted_solution)
            proposed_var = accepted_var
            # n_options should be equal to the number of parameters if allow_hierarchical is False
            # otherwise, n_options = (number of parameters) + 1
            if(allow_hierarchical == False):
                n_options = len(parameters)
            else:
                n_options = len(parameters) + 1
           
            
            # Choose one an opitions at random
            # pick integer between (0, n_options)
            
            
            vary_parameter = np.random.randint(0,n_options)
            if(vary_parameter < n_options - 1):    
                key = list(parameters.keys())[vary_parameter]
                #print(key)
                proposed_solution[key] = np.exp(np.log(proposed_solution[key]) + 0.1*np.random.randn())
                #print(proposed_solution[key])
                if proposed_solution[key] > parameter_bounds[key][1] or proposed_solution[key] < parameter_bounds[key][0]:
                    success = False
                else: 
                    success = True
                    pass
            else:
                proposed_var = np.exp(np.log(accepted_var) + var_change*np.random.randn())
                if proposed_var > var_max or proposed_var < var_min:
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
        run_aspect(proposed_solution, 'boxslab_base.prm', run_dir = run_dir)
        # calculate the geoid from the aspect model.
        proposed_geoid = calculate_geoid('boxslab_base', run_dir=run_dir)
        # calculate the misfit
        proposed_residual = proposed_geoid - observed_geoid
        proposed_magnitude = np.dot(proposed_residual, proposed_residual)
        #print(proposed_magnitude)
        
        N = len(observed_geoid)
        #Cd_hat = np.identity(N)
        
        log_alpha = N/2*((np.log(accepted_var)) - np.log(proposed_var)) \
            - 1/(2*proposed_var)*proposed_magnitude + 1/(2*accepted_var)*accepted_magnitude

        #print('log_alpha is:', log_alpha)

        proposed_likelihood = None
        # calculate the probability of acceptance using the Metropolis-Hastings Criterion
        if log_alpha > 0 or log_alpha > np.log(np.random.rand()):
            accepted_solution = deepcopy(proposed_solution)
            accepted_var = proposed_var
            accepted_magnitude = proposed_magnitude
            # accept the proposed solution by copying the proposed solution
            # to the accepted solution.
            pass
        
        ensemble_residual.append(accepted_magnitude) 
        var_archive.append(accepted_var)
        
        if iter > save_start and not (iter % save_skip):    
            solution_archive.append(deepcopy(accepted_solution))
            
            # save the accepted solution to the archive
            # also save the accepted_var
            # also save the likelihood of the accepted solution
            # also save the misfit of the accepted solution
            pass                        
    return ensemble_residual, solution_archive, var_archive# return the solution archive - this is the ensemble!


#def main():
    # 1. guess an initial solution.
parameters = dict()
parameters['PREFACTOR0'] = 2e-15#1.4250e-15 
parameters['PREFACTOR1'] = 1e-15 #1.4250e-15
parameters['PREFACTOR2'] = 3e-18#1.0657e-18

parameter_bounds = dict()
#add third number for amplitude of pertubation
parameter_bounds['PREFACTOR0'] = [1.425e-16, 1.425e-14]
parameter_bounds['PREFACTOR1'] = [1.425e-16, 1.425e-14]
parameter_bounds['PREFACTOR2'] = [1.0657e-19, 1.0657e-17]

starter_parameters = dict()
starter_parameters['PREFACTOR0'] = 1.4250e-15 
starter_parameters['PREFACTOR1'] = 1.4250e-15
starter_parameters['PREFACTOR2'] = 1.0657e-18


#create starter.prm from starter_parameters
run_aspect(parameters,'starter.prm')                
observed_geoid = calculate_geoid('starter')
residual, solution_archive, var_archive = MCMC(parameters, parameter_bounds, observed_geoid)
#%%


#save residual values, ensemble parameters, and variance archive to pickle files
with open('residuals.p', 'wb') as f1: 
    pickle.dump(residual, f1)
    
with open('parameters.p', 'wb') as f2: 
    pickle.dump(solution_archive, f2)
    
with open('variances.p', 'wb') as f3:
    pickle.dump(var_archive, f3)
    
with open('starter_parameters.p', 'wb') as f4:
    pickle.dump(starter_parameters, f4)

with open('parameter_bounds.p', 'wb') as f4:
    pickle.dump(parameter_bounds, f4)
# #print(residual)
# steps = len(residual)
# x = np.linspace(1, steps, steps)
# plt.plot(x, residual)
# plt.yscale('log')
# plt.title('residuals')
# plt.savefig('residuals.png')
# #plt.show()
# plt.close()

# steps2 = len(var_archive)
# x2 = np.linspace(1, steps2, steps2)
# plt.plot(x2, var_archive)
# plt.yscale('log')
# plt.title('variances')
# plt.savefig('variances.png')
# #plt.show()
# plt.close()


# plt.figure()
# for i in range(3):
#     plt.subplot(1,3,i+1)
#     key = list(parameters.keys())[i]
#     #plt.hist([p[key] for p in solution_archive])
#     plt.hist(np.log10([p[key] for p in solution_archive]), bins=100, range=np.log10(parameter_bounds[key]))
#     #add vertical line for starter parameter values
#     plt.axvline(np.log10(starter_parameters[key]), color='r', linewidth = 1)
    
# plt.title('parameter histograms')
# plt.savefig('parameters.png')
# #plt.show()
# plt.close()
