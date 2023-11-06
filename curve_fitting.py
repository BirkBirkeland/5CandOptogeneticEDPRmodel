#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:23:59 2023

@author: birkbirkeland
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

Na_si = 18.
Na_se = 140.
K_si = 99.
K_se = 4.3
Ca_si = 0.01
Ca_se = 1.1

F = 9.648e4    # [C * mol**-1]
R = 8.314      # [J * mol**-1 * K**-1] 
T = 309.14

def nernst_potential(Z, k_i, k_e):
    E = R*T / (Z*F) * np.log(k_e / k_i)
    return E

E_Na = nernst_potential(1, Na_si, Na_se)
E_K = nernst_potential(1, K_si, K_se)
E_Ca = nernst_potential(2, Ca_si*0.01, Ca_se)
E_ChR2 = 0
print(E_ChR2)
print(E_Na)
print(E_K)
print(E_Ca)

U_0  = 40e-3 # [V]
U_1 = 15e-3 # [V]


def G_ChR2(phi, g_Na, g_K,  g_Ca):
    G = (1-np.exp(-phi/U_0))/(phi/U_1)*(phi - E_ChR2)
    return G

def G_k_sum(phi, g_Na, g_K,  g_Ca):
    G = g_Na*(1-np.exp(-phi/U_0))/(phi/U_1)*(phi - E_Na) + g_K*(1-np.exp(-phi/U_0))/(phi/U_1)*(phi - E_K) \
    + g_Ca*(1-np.exp(-phi/U_0))/(phi/U_1)*(phi - E_Ca) 
    return G


xdata = np.linspace(-100e-3, 100e-3, 100) #e-3
ydata = G_ChR2(xdata, 0.5, 0.4, 0.1)

popt, pcov = curve_fit(G_k_sum, xdata, ydata)
#popt, pcov = curve_fit(G_k_sum, xdata, ydata, bounds=(0, [1., 1., 1]))
popt

g_Na, g_K, g_Ca = popt

def I(phi, E, g):
    I = (1-np.exp(-phi/U_0))/(phi/U_1)*(phi - E)*g
    return I

plt.figure()

xdata = np.linspace(-100e-3, 100e-3, 100) #e-3
y = I(xdata, E_Na, g_Na)
plt.plot(xdata, y, 'b-', label='Na')
plt.xlabel('x')
plt.ylabel('y')

xdata = np.linspace(-100e-3, 100e-3, 100) #e-3
y = I(xdata, E_K, g_K)
plt.plot(xdata, y, 'r-', label='K')
plt.xlabel('x')
plt.ylabel('y')

xdata = np.linspace(-100e-3, 100e-3, 100) #e-3
y = I(xdata, E_Ca, g_Ca)
plt.plot(xdata, y, 'g-', label='Ca')
plt.xlabel('x')
plt.ylabel('y')

plt.axhline(y=0, color='r', linestyle='--')
plt.legend()
plt.show()


plt.figure()

xdata = np.linspace(-100e-3, 100e-3, 100) #e-3
y = I(xdata, E_Na, g_Na) + I(xdata, E_K, g_K) + I(xdata, E_Ca, g_Ca)
plt.plot(xdata, y, 'b-', label='sum')
plt.xlabel('x')
plt.ylabel('y')


xdata = np.linspace(-100e-3, 100e-3, 100) #e-3
y = I(xdata, 0, 1)
plt.plot(xdata, y, 'g--', label='GhCr2')
plt.xlabel('x')
plt.ylabel('y')

plt.axhline(y=0, color='r', linestyle='--')
plt.legend()
plt.show()