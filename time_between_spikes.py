#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 16:34:19 2023

@author: birkbirkeland
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from EDPRmodel5C import *
from solve_EDPRmodel5C import solve_EDPRmodel
from scipy.signal import find_peaks
start_time = time.time()

t_dur = 100     # [s]
alpha = 2.0
I_stim =  4e-11 # [A]
stim_start = 2 # [s]
stim_end = 98  # [s]



""" ICS """
sol = solve_EDPRmodel(t_dur, alpha, I_stim, stim_start, stim_end, protocol='ICS')

Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, \
    Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, n, h, s, c, q, z = sol.y
t = sol.t

my_cell = EDPRmodel(309.14, Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, alpha, I_stim, Ca_si[0], Ca_di[0], n, h, s, c, q, z)

phi_si, phi_se, phi_di, phi_de, phi_ex, phi_sm, phi_dm = my_cell.membrane_potentials(I_stim=I_stim)


spikes, _ = find_peaks(phi_sm, height=0)

spike_times = list()
for spike in spikes:
    spike_times.append(t[spike])
    
time_between_spikes = [0]
for spike_nr in range(1, len(spike_times)):
    time_between_spikes.append(spike_times[spike_nr] - spike_times[spike_nr - 1])
    
spike_times_to_time_between_spikes = dict()
n = 0
for spike in spikes:
    spike_times_to_time_between_spikes[t[spike]] = time_between_spikes[n]
    n += 1
    
del spike_times_to_time_between_spikes[t[spikes[0]]]

f2 = plt.figure(1)
plt.plot(list(spike_times_to_time_between_spikes.keys()), list(spike_times_to_time_between_spikes.values()))
plt.title('Time between spikes')
plt.xlabel('Simulation time [s]')
plt.ylabel('Time between spikes[s]')
plt.legend(loc='upper right')



""" ECS """
sol = solve_EDPRmodel(t_dur, alpha, I_stim, stim_start, stim_end, protocol='ECS')

Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, \
    Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, n, h, s, c, q, z = sol.y
t = sol.t

my_cell = EDPRmodel(309.14, Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, alpha, I_stim, Ca_si[0], Ca_di[0], n, h, s, c, q, z)

phi_si, phi_se, phi_di, phi_de, phi_ex, phi_sm, phi_dm = my_cell.membrane_potentials(I_stim=I_stim)


spikes, _ = find_peaks(phi_sm, height=0)

spike_times = list()
for spike in spikes:
    spike_times.append(t[spike])
    
time_between_spikes = [0]
for spike_nr in range(1, len(spike_times)):
    time_between_spikes.append(spike_times[spike_nr] - spike_times[spike_nr - 1])
    
spike_times_to_time_between_spikes = dict()
n = 0
for spike in spikes:
    spike_times_to_time_between_spikes[t[spike]] = time_between_spikes[n]
    n += 1
    
del spike_times_to_time_between_spikes[t[spikes[0]]]

plt.plot(list(spike_times_to_time_between_spikes.keys()), list(spike_times_to_time_between_spikes.values()))
plt.title('Time between spikes')
plt.xlabel('Simulation time [s]')
plt.ylabel('Time between spikes[s]')
plt.legend(loc='upper right')


""" NaCl """
sol = solve_EDPRmodel(t_dur, alpha, I_stim, stim_start, stim_end, protocol='NaCl')

Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, \
    Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, n, h, s, c, q, z = sol.y
t = sol.t

my_cell = EDPRmodel(309.14, Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, alpha, I_stim, Ca_si[0], Ca_di[0], n, h, s, c, q, z)

phi_si, phi_se, phi_di, phi_de, phi_ex, phi_sm, phi_dm = my_cell.membrane_potentials(I_stim=I_stim)


spikes, _ = find_peaks(phi_sm, height=0)

spike_times = list()
for spike in spikes:
    spike_times.append(t[spike])
    
time_between_spikes = [0]
for spike_nr in range(1, len(spike_times)):
    time_between_spikes.append(spike_times[spike_nr] - spike_times[spike_nr - 1])
    
spike_times_to_time_between_spikes = dict()
n = 0
for spike in spikes:
    spike_times_to_time_between_spikes[t[spike]] = time_between_spikes[n]
    n += 1
    
del spike_times_to_time_between_spikes[t[spikes[0]]]


plt.plot(list(spike_times_to_time_between_spikes.keys()), list(spike_times_to_time_between_spikes.values()))
plt.title('Time between spikes')
plt.xlabel('Simulation time [s]')
plt.ylabel('Time between spikes[s]')
plt.legend(loc='upper right')


""" KCl """
sol = solve_EDPRmodel(t_dur, alpha, I_stim, stim_start, stim_end, protocol='KCl')

Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, \
    Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, n, h, s, c, q, z = sol.y
t = sol.t

my_cell = EDPRmodel(309.14, Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, alpha, I_stim, Ca_si[0], Ca_di[0], n, h, s, c, q, z)

phi_si, phi_se, phi_di, phi_de, phi_ex, phi_sm, phi_dm = my_cell.membrane_potentials(I_stim=I_stim)


spikes, _ = find_peaks(phi_sm, height=0)

spike_times = list()
for spike in spikes:
    spike_times.append(t[spike])
    
time_between_spikes = [0]
for spike_nr in range(1, len(spike_times)):
    time_between_spikes.append(spike_times[spike_nr] - spike_times[spike_nr - 1])
    
spike_times_to_time_between_spikes = dict()
n = 0
for spike in spikes:
    spike_times_to_time_between_spikes[t[spike]] = time_between_spikes[n]
    n += 1
    
del spike_times_to_time_between_spikes[t[spikes[0]]]


plt.plot(list(spike_times_to_time_between_spikes.keys()), list(spike_times_to_time_between_spikes.values()))
plt.title('Time between spikes')
plt.xlabel('Simulation time [s]')
plt.ylabel('Time between spikes[s]')
plt.legend(loc='upper right')



""" Na """
sol = solve_EDPRmodel(t_dur, alpha, I_stim, stim_start, stim_end, protocol='Na')

Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, \
    Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, n, h, s, c, q, z = sol.y
t = sol.t

my_cell = EDPRmodel(309.14, Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, alpha, I_stim, Ca_si[0], Ca_di[0], n, h, s, c, q, z)

phi_si, phi_se, phi_di, phi_de, phi_ex, phi_sm, phi_dm = my_cell.membrane_potentials(I_stim=I_stim)


spikes, _ = find_peaks(phi_sm, height=0)

spike_times = list()
for spike in spikes:
    spike_times.append(t[spike])
    
time_between_spikes = [0]
for spike_nr in range(1, len(spike_times)):
    time_between_spikes.append(spike_times[spike_nr] - spike_times[spike_nr - 1])
    
spike_times_to_time_between_spikes = dict()
n = 0
for spike in spikes:
    spike_times_to_time_between_spikes[t[spike]] = time_between_spikes[n]
    n += 1
    
del spike_times_to_time_between_spikes[t[spikes[0]]]


plt.plot(list(spike_times_to_time_between_spikes.keys()), list(spike_times_to_time_between_spikes.values()))
plt.title('Time between spikes')
plt.xlabel('Simulation time [s]')
plt.ylabel('Time between spikes[s]')
plt.legend(loc='upper right')


""" K """
sol = solve_EDPRmodel(t_dur, alpha, I_stim, stim_start, stim_end, protocol='K')

Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, \
    Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, n, h, s, c, q, z = sol.y
t = sol.t

my_cell = EDPRmodel(309.14, Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, alpha, I_stim, Ca_si[0], Ca_di[0], n, h, s, c, q, z)

phi_si, phi_se, phi_di, phi_de, phi_ex, phi_sm, phi_dm = my_cell.membrane_potentials(I_stim=I_stim)


spikes, _ = find_peaks(phi_sm, height=0)

spike_times = list()
for spike in spikes:
    spike_times.append(t[spike])
    
time_between_spikes = [0]
for spike_nr in range(1, len(spike_times)):
    time_between_spikes.append(spike_times[spike_nr] - spike_times[spike_nr - 1])
    
spike_times_to_time_between_spikes = dict()
n = 0
for spike in spikes:
    spike_times_to_time_between_spikes[t[spike]] = time_between_spikes[n]
    n += 1
    
del spike_times_to_time_between_spikes[t[spikes[0]]]

plt.plot(list(spike_times_to_time_between_spikes.keys()), list(spike_times_to_time_between_spikes.values()))
plt.title('Time between spikes')
plt.xlabel('Simulation time [s]')
plt.ylabel('Time between spikes[s]')
plt.legend(loc='upper right')



""" Cl """
sol = solve_EDPRmodel(t_dur, alpha, I_stim, stim_start, stim_end, protocol='Cl')

Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, \
    Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, n, h, s, c, q, z = sol.y
t = sol.t

my_cell = EDPRmodel(309.14, Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, alpha, I_stim, Ca_si[0], Ca_di[0], n, h, s, c, q, z)

phi_si, phi_se, phi_di, phi_de, phi_ex, phi_sm, phi_dm = my_cell.membrane_potentials(I_stim=I_stim)


spikes, _ = find_peaks(phi_sm, height=0)

spike_times = list()
for spike in spikes:
    spike_times.append(t[spike])
    
time_between_spikes = [0]
for spike_nr in range(1, len(spike_times)):
    time_between_spikes.append(spike_times[spike_nr] - spike_times[spike_nr - 1])
    
spike_times_to_time_between_spikes = dict()
n = 0
for spike in spikes:
    spike_times_to_time_between_spikes[t[spike]] = time_between_spikes[n]
    n += 1
    
del spike_times_to_time_between_spikes[t[spikes[0]]]

plt.plot(list(spike_times_to_time_between_spikes.keys()), list(spike_times_to_time_between_spikes.values()))
plt.title('Time between spikes')
plt.xlabel('Simulation time [s]')
plt.ylabel('Time between spikes[s]')
plt.legend(loc='upper right')


""" X """
sol = solve_EDPRmodel(t_dur, alpha, I_stim, stim_start, stim_end, protocol='X')

Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, \
    Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, n, h, s, c, q, z = sol.y
t = sol.t

my_cell = EDPRmodel(309.14, Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, alpha, I_stim, Ca_si[0], Ca_di[0], n, h, s, c, q, z)

phi_si, phi_se, phi_di, phi_de, phi_ex, phi_sm, phi_dm = my_cell.membrane_potentials(I_stim=I_stim)


spikes, _ = find_peaks(phi_sm, height=0)

spike_times = list()
for spike in spikes:
    spike_times.append(t[spike])
    
time_between_spikes = [0]
for spike_nr in range(1, len(spike_times)):
    time_between_spikes.append(spike_times[spike_nr] - spike_times[spike_nr - 1])
    
spike_times_to_time_between_spikes = dict()
n = 0
for spike in spikes:
    spike_times_to_time_between_spikes[t[spike]] = time_between_spikes[n]
    n += 1
    
del spike_times_to_time_between_spikes[t[spikes[0]]]


plt.plot(list(spike_times_to_time_between_spikes.keys()), list(spike_times_to_time_between_spikes.values()))
plt.title('Time between spikes')
plt.xlabel('Simulation time [s]')
plt.ylabel('Time between spikes[s]')
plt.legend(loc='upper right')

plt.show()