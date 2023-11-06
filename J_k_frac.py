#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:42:39 2023

@author: birkbirkeland
"""

def J_k_frac(Na, K, Cl, k, D_k):
    """ Find what fraction of the current into the cell which is made up of ion species k.
    Inputs: Na, K, Cl: the fraction of
    For protocols A and B the values for Na, K, and Cl are the initial concentrations of 
    these ions in the ICS and the ECS, respectively.
    For protocols where one or more of the ions are not included in the current, these are set to zero."""
    
    D_Na = 1.33e-9 # Halnes et al. 2013
    D_K = 1.96e-9  # Halnes et al. 2013 
    D_Cl = 2.03e-9 # Halnes et al. 2013
    J_k_frac = (D_k*k)/(Na*D_Na + K*D_K + Cl*D_Cl)
    return J_k_frac

