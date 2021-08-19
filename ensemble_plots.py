#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 20:54:47 2021

@author: elena
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats as stats
#from aspect_geoid_mcmc import calculate_geoid
with open('results.p', 'rb') as f:
    results = pickle.load(f)
    
residuals = results['residuals']
solution_archive = results['parameters'][-5000:]
geoid_archive = results['geoids'][-5000:]
variances = results['variances'] 
starter_parameters = results['starter_parameters'] 
parameter_bounds = results['bounds'] 

steps = len(residuals)
x1 = np.linspace(1, steps, steps)
plt.plot(x1, residuals)
plt.yscale('log')
plt.title('total residual vs. step number')
plt.savefig('residuals.png')
plt.show()
plt.close()

steps2 = len(variances)
x2 = np.linspace(1, steps2, steps2)
plt.plot(x2, variances)
plt.yscale('log')
plt.title('variances vs. step number')
plt.savefig('variances.png')
plt.show()
plt.close()


# #Plot to show probability distribution for pertubation
# key = 'PREFACTOR0'
# sigma = 0.05
# mu = np.log10(2e-15)
# plt.axvline(mu, color = 'r', linewidth = 1)
# bounds = np.log10(parameter_bounds['PREFACTOR0'])
# x = np.linspace(bounds[0], bounds[1], 1000)
# y = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2 )
# plt.plot(x, stats.norm.pdf(x, np.log10(2e-15), 0.05))
# plt.xlabel('log(PREFACTOR0)')
# plt.ylabel('likelihood')
# plt.title('Probability Distribution for Change in PREFACTOR0')
# plt.show()

#plot ensemble parameter values
labels = ['A0', 'A1', 'A2', 'B0', 'B1', 'B2']
parameter_count = len(starter_parameters)
fig, axs = plt.subplots(nrows=2,ncols=3, constrained_layout=True, figsize=(8,5), sharey = True)
for j in range(parameter_count):
    key = list(starter_parameters.keys())[j]
    title = labels[j]
    if j<parameter_count/2:
        i = 0
    else:
        i=1
        j = j-parameter_count//2
    axs[i][j].hist(np.log10([p[key] for p in solution_archive]), bins=100, range=np.log10(parameter_bounds[key]))
    #add vertical line for starter parameter values
    axs[i][j].axvline(np.log10(starter_parameters[key]), color='r', linewidth = 1)
    axs[i][j].set_title(labels[j])
    axs[i][j].set_xlabel('log(prefactor value)')
    axs[i][0].set_ylabel('count')
plt.suptitle('Ensemble Solutions')
plt.savefig('parameters.png')
plt.show()

#pull individual parameter values
parameter_count = len(starter_parameters)
parameters_stepwise = dict()
for i in range(parameter_count):
    key = list(starter_parameters.keys())[i]
    current_parameters = []
    for p in solution_archive: 
        current_parameters.append(p[key])
    parameters_stepwise[key] = current_parameters
    
#log scale these later
prefactor0 = parameters_stepwise['PREFACTOR0']
prefactor1 = parameters_stepwise['PREFACTOR1']

#bounds = [np.log10(parameter_bounds['PREFACTOR0']), np.log10(parameter_bounds['PREFACTOR1'])]
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
ax1.hist(prefactor0, bins = 50) # ,  range = bounds[0])
#ax1.axvline(starter_parameters['PREFACTOR0'], color='r', linewidth = 1)
x_axis = ax1.axes.get_xaxis()
#x_axis.set_ticklabels([[]])

ax2.axis('off')

ax3.hist2d(prefactor0, prefactor1, bins=50) #, range = bounds)
ax3.set

ax4.hist(prefactor1, bins = 50, orientation=u'horizontal')
ax4.axhline(starter_parameters['PREFACTOR1'], color='r', linewidth = 1)
y_axis = ax4.axes.get_yaxis()
#y_axis.set_ticklabels([])
plt.show()

#plot ensemble geoids
geoid_archive = geoid_archive[-5000:]
steps3 = len(geoid_archive[0])
x3 = np.linspace(1, steps3, steps3)
for i in geoid_archive:
    plt.plot(x3, i, color= 'b')
#plt.plot(x3, observed_geoid, color = 'r', linewidth = 1)
plt.xlabel('distance along surface (m)')
plt.ylabel('Geoid anomaly (m)')
plt.title('Ensemble Geoids')
plt.show()

# parameters = dict()
# parameters['PREFACTOR0'] = 2e-15#1.4250e-15 
# parameters['PREFACTOR1'] = 1e-15 #1.4250e-15
# parameters['PREFACTOR2'] = 3e-18#1.0657e-18
# parameters['PREFACTOR3'] = 0.7e-20#0.5e-20
# parameters['PREFACTOR4'] = 1.5e-15#1.4250e-15
# parameters['PREFACTOR5'] = 0.8e-18#1.0657e-18

# parameter_count = len(starter_parameters)
# fig, axs = plt.subplots(nrows=2,ncols=3, constrained_layout=True, figsize=(8,5))
# for j in range(parameter_count):
#     key = list(starter_parameters.keys())[j]
#     if j<parameter_count/2:
#         i = 0
#     else:
#         i=1
#         j = j-parameter_count//2
#     #add vertical line for starter parameter values
#     axs[i][j].axvline(np.log10(starter_parameters[key]), color='r', linewidth = 1)
#     axs[i][j].axvline(np.log10(parameters[key]), color='b', linewidth = 1)
#     axs[i][j].set_title(key)
#     axs[i][j].set_xlabel('log(prefactor value)')
#     y_axis = axs[i][j].axes.get_yaxis()
#     y_axis.set_ticklabels([[]])
# plt.suptitle('Initial Guesses')
# plt.show()