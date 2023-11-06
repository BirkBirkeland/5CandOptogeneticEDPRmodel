import numpy as np
import time
import matplotlib.pyplot as plt
from EDPRmodelOG5C import *
from solve_EDPRmodelOG5C import solve_EDPRmodel
from scipy.signal import find_peaks
start_time = time.time()

t_dur = 10    # [s]
alpha = 2.0
I_stim =  4e-11 # [A]
stim_start = 2 # [s]
stim_end = 8  # [s]
l = 1 # [ms^-1]

sol = solve_EDPRmodel(t_dur, alpha, I_stim, l, stim_start, stim_end, protocol='OG')

Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, \
    Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, n, h, s, c, q, z, O_1, O_2, C_1, C_2 = sol.y
t = sol.t

my_cell = EDPRmodel(309.14, Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, alpha, I_stim, l, O_1, O_2, C_1, C_2, Ca_si[0], Ca_di[0], n, h, s, c, q, z)

phi_si, phi_se, phi_di, phi_de, phi_ex, phi_sm, phi_dm = my_cell.membrane_potentials(I_stim=I_stim)



q_si = my_cell.total_charge([my_cell.Na_si[-1], my_cell.K_si[-1], my_cell.Cl_si[-1], my_cell.Ca_si[-1], my_cell.X_si[-1]], my_cell.V_si)
q_se = my_cell.total_charge([my_cell.Na_se[-1], my_cell.K_se[-1], my_cell.Cl_se[-1], my_cell.Ca_se[-1], my_cell.X_se[-1]], my_cell.V_se)        
q_di = my_cell.total_charge([my_cell.Na_di[-1], my_cell.K_di[-1], my_cell.Cl_di[-1], my_cell.Ca_di[-1], my_cell.X_di[-1]], my_cell.V_di)
q_de = my_cell.total_charge([my_cell.Na_de[-1], my_cell.K_de[-1], my_cell.Cl_de[-1], my_cell.Ca_de[-1], my_cell.X_de[-1]], my_cell.V_de)
q_ex = my_cell.total_charge([my_cell.Na_ex[-1], my_cell.K_ex[-1], my_cell.Cl_ex[-1], my_cell.Ca_ex[-1], my_cell.X_ex[-1]], my_cell.V_ex)
print("Final values")
print("----------------------------")
print("total charge at the end (C): ", q_si + q_se + q_di + q_de)
print("Q_si (C):", q_si)
print("Q_se (C):", q_se)
print("Q_di (C):", q_di)
print("Q_de (C):", q_de)
print("Q_ex (C):", q_ex)
print("----------------------------")
print('elapsed time: ', round(time.time() - start_time, 1), 'seconds')

f1 = plt.figure(1)

spikes, _ = find_peaks(phi_sm, height=0)
plt.plot(t[spikes], phi_sm[spikes]*1000, "x")

plt.plot(t, phi_sm*1000, '-', label='V_s')
plt.plot(t, phi_dm*1000, '-', label='V_d')

plt.title('Membrane potentials')
plt.xlabel('time [s]')
plt.ylabel('[mV]')
plt.legend(loc='upper right')

plt.show()

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

f2 = plt.figure(2)
plt.plot(list(spike_times_to_time_between_spikes.keys()), list(spike_times_to_time_between_spikes.values()))
plt.title('Time between spikes')
plt.xlabel('Simulation time [s]')
plt.ylabel('Time between spikes[s]')
plt.legend(loc='upper right')

plt.show()


