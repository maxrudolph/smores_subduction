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
import binascii
import os
import sys
from geoid_functions import *
from copy import deepcopy

#accept inputs if 3 numbers are given, otherwise use default
#Usage: python3 aspect_geoid_mcmc.py n_processors n_steps save_start save_skip resume_computation
if len(sys.argv) == 6: 
    n_processors = sys.argv[1]
    n_steps = int(sys.argv[2]) #need try/catch?
    save_start = int(sys.argv[3])
    save_skip = int(sys.argv[4])
    restore = str.lower(sys.argv[5])
    #accept either true or false (not case sensitive) for resume_computaion; SystemExit for other input
    if restore == 'true':
        resume_computation = True
    elif restore == 'false':
        resume_computation = False
    else: 
        raise SystemExit("Input resume_compuation as 'True' or 'False'")
#if no inputs are given, run with default 
elif len(sys.argv) == 1:
    n_processors = '20'
    n_steps = 1000
    save_start = 500
    save_skip = 1
    resume_computation = False
else: 
    raise SystemExit("Usage: python3 aspect_geoid_mcmc.py n_processors n_steps save_start save_skip resume_computation")
    

def setup_aspect_runs(run_dir='/dev/shm/geoidtmp/',base_input_file=['boxslab_base_start.prm','boxslab_base_resume.prm']):
    # make the run directory:
    try:
        run_dir = run_dir[:-1] + binascii.hexlify(os.urandom(8)).decode() + '/'
        subprocess.run(['mkdir',run_dir[:-1]])
    except:
        print('run directory '+run_dir+' already exists or cannot be created')        
    # copy the aspect executable to the run directory
    subprocess.run(['cp', 'aspect.fast', run_dir])
    for file in base_input_file:
        subprocess.run(['cp',file,run_dir])
    subprocess.run(['cp','boxslab_topcrust.wb',run_dir])
    
    # return the directory in which aspect has been placed.
    return run_dir

def cleanup(run_dir=None):
    if run_dir is not None:
        try:
            subprocess.run(['rm','-rf',run_dir])
        except:
            print('could not remove run directory',run_dir)
    
def run_aspect(parameters,base_input_file = 'boxslab_base.prm',run_dir='./', timeout = None,n_processors=n_processors): 
    
    #should remove previous output directory with same name
    #run as ls command before rm (be careful)
    
    prm_filename =  'run.prm'
    # replace strings in the 'base input file' with the parameter values, which are stored in a dictionary.
    subprocess.run(["cp",base_input_file,prm_filename],cwd=run_dir)
    for key in parameters.keys():
        # use the sed command to replace each of the keys in the ditionary with its appropriate value.
        subprocess.run(["sed","-i","-e","s/"+key+"/"+'{:e}'.format(parameters[key])+"/g",prm_filename], cwd = run_dir)

    # run aspect
    #aspect_command = './aspect-master.fast' #+ prm_filename
    #subprocess.run([aspect_command, prm_filename],cwd=run_dir) # run aspect

    aspect_command = './aspect.fast' 
    #timeout the aspect run after 100 seconds
    try:
        subprocess.run(['mpirun', '-n', n_processors, aspect_command, prm_filename],cwd=run_dir, timeout=timeout)
    except subprocess.TimeoutExpired:
        #add code to kill process
        print('\nProcess ran too long')
        return True
 
def calculate_geoid(output_folder,run_dir='./',step=0,mesh_step=0):
    # Do the geoid calculation
    mesh_file = run_dir + output_folder + '/solution/mesh-{:05d}.h5'.format(mesh_step)
    #    print(mesh_file)

    file0 = run_dir + output_folder + '/solution/solution-{:05d}.h5'.format(step)
    cells,x,y,rho,topo = load_data(mesh_file,file0) 
    ind_surf,x_surf,ind_botm,x_botm = get_boundary_arrays(x,y)
    x_obs,y_obs = get_observation_vector(x,y,use_nodes=False)
    delta_rho = get_density_anomaly(cells,x,y,rho)
    
    N_interior = interior_geoid(cells,x,y,delta_rho,x_obs,y_obs,method='line')
    N_surface = surface_geoid(ind_surf,x,rho,topo,x_obs)
    N_cmb = cmb_geoid(ind_botm,x,y,rho,topo,x_obs)
    N_total = N_surface + N_interior + N_cmb
    return N_total

def MCMC(starting_solution=None, parameter_bounds=None, observed_geoid=None, n_steps=10000,save_start=5000,save_skip=2,var=None, resume_computation = False,sample_prior=False,restart_interval=100):
    # This function should implement the MCMC procedure
    # 1. Define the perturbations (proposal distributions) for each parameter
    #    and the bounds on each parameter. Define the total number of steps and
    #    the number of steps between saving output.
    # if var is None, assume that this calculation uses a hierarchical hyperparameter.
    var_min = 1e-6
    var_max = 1e10
    var_change = 0.1
    step_size = 0.05
        
    
    #unpickle results if resume_computation is true
    if resume_computation == True:
        with open('results.p', 'rb') as f:
            saved_results = pickle.load(f)
        
        iter = saved_results['iteration']
        if iter > save_start:
            solution_archive = saved_results['parameters']
            geoid_archive = saved_results['geoids']
        else:
            solution_archive = []
            geoid_archive = []

        starting_solution = deepcopy(solution_archive[-1])
        ensemble_residual = saved_results['residuals'] 
        var_archive = saved_results['variances'] 
        accepted_var = var_archive[-1]
        
        if var is None:
            allow_hierarchical = True
        else:
            allow_hierarchical = False
        
                
    #otherwise begin with iteration 0 and empty archives
    else:
        iter = 0
        #create ensemble archive
        ensemble_residual = []
        solution_archive = []
        geoid_archive = []
        var_archive = []
        
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

    if sample_prior is True:
        run_dir = []
        accepted_geoid = deepcopy(observed_geoid)
        accepted_residual = 0.;
        accepted_magnitude = 0.;
    else:
        run_dir = setup_aspect_runs() # setup aspect to run in /dev/shm
        run_aspect(accepted_solution,base_input_file='boxslab_base_start.prm',run_dir=run_dir)
        accepted_geoid = calculate_geoid('boxslab_base',run_dir=run_dir,step=0,mesh_step=0)
        accepted_residual = accepted_geoid - observed_geoid
        accepted_magnitude = np.dot(accepted_residual, accepted_residual)
        #print(accepted_magnitude)

    

        # 3. Begin the MCMC procedure
    
    while iter < n_steps:
        if not iter%1000:
            print(iter)
        iter += 1
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
                proposed_solution[key] = np.exp(np.log(proposed_solution[key]) + step_size*np.random.randn() )
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
        if sample_prior is False:
            if iter % restart_interval == 0:
                timeout_check = run_aspect(proposed_solution, 'boxslab_base_start.prm', run_dir=run_dir, timeout=300) 
                # calculate the geoid from the aspect model.
                proposed_geoid = calculate_geoid('boxslab_base', run_dir=run_dir,step=0,mesh_step=0)
            else:
                timeout_check = run_aspect(proposed_solution, 'boxslab_base_resume.prm', run_dir=run_dir, timeout=100) 
                # calculate the geoid from the aspect model.
                proposed_geoid = calculate_geoid('boxslab_base', run_dir=run_dir,step=1,mesh_step=1)
            print("step number" + str(iter))
            #if aspect timed out, continue without recalculating the geoid
            if(timeout_check == True):
                iter -= 1 
                continue
                
            # calculate the misfit
            proposed_residual = proposed_geoid - observed_geoid
            proposed_magnitude = np.dot(proposed_residual, proposed_residual)
            #print(proposed_magnitude)
        else:
            proposed_geoid = deepcopy(observed_geoid)
            proposed_residual = 0.
            proposed_magnitude = 0.
        
        N = len(observed_geoid)
        if sample_prior:
            log_alpha = np.log10(1.0)
        else:
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
        
        ensemble_residual.append(accepted_magnitude) 
        var_archive.append(accepted_var)
        
        if iter > save_start and not (iter % save_skip):    
            solution_archive.append(deepcopy(accepted_solution))
            geoid_archive.append(deepcopy(proposed_geoid))
            # save the accepted solution to the archive
            # also save the accepted_var
            # also save the likelihood of the accepted solution
            # also save the misfit of the accepted solution
            pass    

        #save results to pickle file every 100 steps and for the last step 
        if (iter % 100 == 0) or (iter == n_steps - 1):
            #archive current accepted solution and variance 
            results['accepted_solution'] = deepcopy(accepted_solution)
            results['accepted_var'] = accepted_var
            results['iteration'] = iter
            results['residuals'] = ensemble_residual
            results['parameters'] = solution_archive
            results['geoids'] = geoid_archive
            results['variances'] = var_archive
            
            with open('results.p', 'wb') as f:
                pickle.dump(results, f)

    #cleanup(run_dir)
    return ensemble_residual, solution_archive, var_archive, geoid_archive# return the solution archive - this is the ensemble!


#def main():
    # 1. guess an initial solution.
parameters = dict()
parameters['PREFACTOR0'] = 2e-15#1.4250e-15 
parameters['PREFACTOR1'] = 1e-15 #1.4250e-15
parameters['PREFACTOR2'] = 3e-18#1.0657e-18
parameters['PREFACTOR3'] = 0.7e-20#0.5e-20
parameters['PREFACTOR4'] = 1.5e-15#1.4250e-15
parameters['PREFACTOR5'] = 0.8e-18#1.0657e-18


parameter_bounds = dict()
parameter_bounds['PREFACTOR0'] = [1.425e-16, 1.425e-14]
parameter_bounds['PREFACTOR1'] = [1.425e-16, 1.425e-14]
parameter_bounds['PREFACTOR2'] = [1.0657e-19, 1.0657e-17]
parameter_bounds['PREFACTOR3'] = [0.5e-21, 0.5e-19]
parameter_bounds['PREFACTOR4'] = [1.425e-16, 1.425e-14]
parameter_bounds['PREFACTOR5'] = [1.0657e-19, 1.0657e-17]

starter_parameters = dict()
starter_parameters['PREFACTOR0'] = 1.4250e-15
starter_parameters['PREFACTOR1'] = 1.4250e-15
starter_parameters['PREFACTOR2'] = 1.0657e-18
starter_parameters['PREFACTOR3'] = 0.5e-20
starter_parameters['PREFACTOR4'] = 1.4250e-15
starter_parameters['PREFACTOR5'] = 1.0657e-18

#initialize results dictionary
results = dict()
results['starter_parameters'] = starter_parameters
results['bounds'] = parameter_bounds
            
#create starter.prm from starter_parameters
#run_aspect(starter_parameters,'boxslab_base.prm')
#observed_geoid = calculate_geoid('boxslab_base')
sample_prior = False
if sample_prior is True:
    observed_geoid = np.zeros((101,))
else:
    run_aspect(starter_parameters,'boxslab_base_start.prm')
    observed_geoid = calculate_geoid('boxslab_base',step=0,mesh_step=0)

residual, solution_archive, var_archive, geoid_archive = MCMC(starter_parameters, parameter_bounds, observed_geoid, n_steps, save_start, save_skip, resume_computation = resume_computation,sample_prior=sample_prior)
