#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 20:54:47 2021

@author: elena
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('results.p', 'rb') as f:
    results = pickle.load(f)
    
residuals = results['residuals'] 
solution_archive = results['parameters']
variances = results['variances'] 
starter_parameters = results['starter_parameters'] 
parameter_bounds = results['bounds'] 

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