#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 20:54:47 2021

@author: elena
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle


#unpickle results of Bayesian inversion 
with open('residuals.p', 'rb') as f1: 
    residuals = pickle.load(f1)
    
with open('parameters.p', 'rb') as f2: 
    solution_archive = pickle.load(f2)
    
with open('variances.p', 'rb') as f3:
    variances = pickle.load(f3)
    
#unpickle parameters for starter model and 
with open('starter_parameters.p', 'rb') as f4:
    starter_parameters = pickle.load(f4)
    
with open('parameter_bounds.p', 'rb') as f5:
    parameter_bounds = pickle.load(f5)
    
steps = len(residuals)
x1 = np.linspace(1, steps, steps)
plt.plot(x1, residuals)
plt.yscale('log')
plt.title('total residual vs. step number')
#plt.savefig('residuals.png')
plt.show()
plt.close()


steps2 = len(variances)
x2 = np.linspace(1, steps2, steps2)
plt.plot(x2, variances)
plt.yscale('log')
plt.title('variances vs. step number')
#plt.savefig('variances.png')
plt.show()
plt.close()


parameter_count = len(starter_parameters)
fig, axs = plt.subplots(1, parameter_count, constrained_layout=True)
for i in range(parameter_count):
    key = list(starter_parameters.keys())[i]
    axs[i].hist(np.log10([p[key] for p in solution_archive]), bins=100, range=np.log10(parameter_bounds[key]))
    #add vertical line for starter parameter values
    axs[i].axvline(np.log10(starter_parameters[key]), color='r', linewidth = 1)
    axs[i].set_title(key)

plt.suptitle('parameter histograms')
#plt.savefig('parameters.png')
plt.show()
