import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from EDPRmodelOG5C import *
from somatic_injection_current5C import *
from scipy.integrate import solve_ivp
from J_k_frac import J_k_frac

def solve_EDPRmodel(t_dur, alpha, I_stim, l, stim_start, stim_end, protocol):
    """
    Solves the EDPR model using the solve_ivp function from scipy.

    Arguments:
        t_dur (float): duration of simulation [s]
        alpha (float): coupling strength
        I_stim (float): stimulus current [A]
        stim_start (float): time of stimulus onset [s]
        stim_end (float): time of stimulus offset [s]

    Returns:
        sol: solution from solve_ivp
    """

    T = 309.14 # temperature [K]

    # set initial conditions
    Na_si0 = 18.
    Na_se0 = 140.
    K_si0 = 99.
    K_se0 = 4.3
    Cl_si0 = 7.
    Cl_se0 = 134.
    Ca_si0 = 0.01
    Ca_se0 = 1.1

    Na_di0 = 18.
    Na_de0 = 140.
    K_di0 = 99.
    K_de0 = 4.3
    Cl_di0 = 7.
    Cl_de0 = 134.
    Ca_di0 = 0.01
    Ca_de0 = 1.1
    
    Na_ex0 = 140.
    K_ex0 = 4.3
    Cl_ex0 = 134.
    Ca_ex0 = 1.1

    res_i = -68e-3*3e-2*616e-12/(1437e-18*9.648e4)
    res_e = -68e-3*3e-2*616e-12/(718.5e-18*9.648e4)

    X_si0 = Na_si0 + K_si0 - Cl_si0 + 2*Ca_si0 - res_i
    X_se0 = Na_se0 + K_se0 - Cl_se0 + 2*Ca_se0 + res_e
    X_di0 = Na_di0 + K_di0 - Cl_di0 + 2*Ca_di0 - res_i
    X_de0 = Na_de0 + K_de0 - Cl_de0 + 2*Ca_de0 + res_e
    X_ex0 = Na_ex0 + K_ex0 - Cl_ex0 + 2*Ca_ex0 + res_e
    
    n0 = 0.0003
    h0 = 0.999
    s0 = 0.007
    c0 = 0.006
    q0 = 0.011
    z0 = 1.0
    
    O_10 = 0
    O_20 = 0
    C_10 = 0.5
    C_20 = 0.5

    frac_X=0
    
    if protocol == 'ICS':
        frac_Na = J_k_frac(Na=18., K=99., Cl=7., k=18., D_k=1.33e-9)
        frac_K = J_k_frac(Na=18., K=99., Cl=7., k=99., D_k=1.96e-9)
        frac_Cl = J_k_frac(Na=18., K=99., Cl=7., k=7., D_k=2.03e-9)
        l = 0
    elif protocol == 'ECS':
        frac_Na = J_k_frac(Na=140., K=4.3, Cl=134., k=140., D_k=1.33e-9)
        frac_K = J_k_frac(Na=140., K=4.3, Cl=134., k=4.3, D_k=1.96e-9)
        frac_Cl = J_k_frac(Na=140., K=4.3, Cl=134., k=134., D_k=2.03e-9)
        l = 0
    elif protocol == 'NaCl':
        frac_Na = J_k_frac(Na=1, K=0, Cl=1, k=1, D_k=1.33e-9)
        frac_K = J_k_frac(Na=1, K=0, Cl=1., k=0, D_k=1.96e-9)
        frac_Cl = J_k_frac(Na=1, K=0, Cl=1, k=1, D_k=2.03e-9)
        l = 0
    elif protocol == 'KCl':
        frac_Na = J_k_frac(Na=0, K=1, Cl=1, k=0, D_k=1.33e-9)
        frac_K = J_k_frac(Na=0, K=1, Cl=1., k=1, D_k=1.96e-9)
        frac_Cl = J_k_frac(Na=0, K=1, Cl=1, k=1, D_k=2.03e-9)
        l = 0
    elif protocol == 'Na':
        frac_Na = J_k_frac(Na=1, K=0, Cl=0, k=1, D_k=1.33e-9)
        frac_K = J_k_frac(Na=1, K=0, Cl=0., k=0, D_k=1.96e-9)
        frac_Cl = J_k_frac(Na=1, K=0, Cl=0, k=0, D_k=2.03e-9)
        l = 0
    elif protocol == 'K':
        frac_Na = J_k_frac(Na=0, K=1, Cl=0, k=0, D_k=1.33e-9)
        frac_K = J_k_frac(Na=0, K=1, Cl=0., k=1, D_k=1.96e-9)
        frac_Cl = J_k_frac(Na=0, K=1, Cl=0, k=0, D_k=2.03e-9)
        l = 0
    elif protocol == 'Cl':
        frac_Na = J_k_frac(Na=0, K=0, Cl=1, k=0, D_k=1.33e-9)
        frac_K = J_k_frac(Na=0, K=0, Cl=1., k=0, D_k=1.96e-9)
        frac_Cl = J_k_frac(Na=0, K=0, Cl=1, k=1, D_k=2.03e-9)
        l = 0
    elif protocol == 'X':
        frac_Na = 0
        frac_K = 0
        frac_Cl = 0
        frac_X = 1
        l = 0
    elif protocol == 'OG':
        frac_Na = 0
        frac_K = 0
        frac_Cl = 0
        I_stim = 0
    
    
    
    # define differential equations
    def dkdt(t,k):

        Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, n, h, s, c, q, z, O_1, O_2, C_1, C_2 = k

        if t > stim_start and t < stim_end:
            
            my_cell = EDPRmodel(T, Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, alpha, I_stim, l, O_1, O_2, C_1, C_2, Ca_si0, Ca_di0, n, h, s, c, q, z)
            
            dNadt_si, dNadt_se, dNadt_di, dNadt_de, dNadt_ex, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dKdt_ex, dCldt_si, dCldt_se, dCldt_di, dCldt_de, dCldt_ex, \
               dCadt_si, dCadt_se, dCadt_di, dCadt_de, dCadt_ex, dresdt_si, dresdt_se, dresdt_di, dresdt_de, dresdt_ex = my_cell.dkdt(I_stim=I_stim)
            
            dndt, dhdt, dsdt, dcdt, dqdt, dzdt = my_cell.dmdt(I_stim=I_stim)
            
            phi_si, phi_se, phi_di, phi_de, phi_ex, phi_sm, phi_dm  = my_cell.membrane_potentials(I_stim=I_stim)
            
            dO_1dt, dO_2dt, dC_1dt, dC_2dt = my_cell.dNdt(l)
            
            dNadt_si, dKdt_si, dCldt_si, dresdt_si = somatic_injection_current(my_cell, dNadt_si, dKdt_si, dCldt_si, dresdt_si, I_stim, frac_Na, frac_K, frac_Cl, frac_X)
            
        else:
            my_cell = EDPRmodel(T, Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, alpha, 0, 0, O_1, O_2, C_1, C_2, Ca_si0, Ca_di0, n, h, s, c, q, z)

            dNadt_si, dNadt_se, dNadt_di, dNadt_de, dNadt_ex, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dKdt_ex, dCldt_si, dCldt_se, dCldt_di, dCldt_de, dCldt_ex, \
                dCadt_si, dCadt_se, dCadt_di, dCadt_de, dCadt_ex, dresdt_si, dresdt_se, dresdt_di, dresdt_de, dresdt_ex = my_cell.dkdt(I_stim=0)
        
            dndt, dhdt, dsdt, dcdt, dqdt, dzdt = my_cell.dmdt(I_stim=0)
        
            phi_si, phi_se, phi_di, phi_de, phi_ex, phi_sm, phi_dm = my_cell.membrane_potentials(I_stim=0)
            
            dO_1dt, dO_2dt, dC_1dt, dC_2dt = my_cell.dNdt(l=0)
            
            
        return dNadt_si, dNadt_se, dNadt_di, dNadt_de, dNadt_ex, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dKdt_ex, \
            dCldt_si, dCldt_se, dCldt_di, dCldt_de, dCldt_ex, dCadt_si, dCadt_se, dCadt_di, dCadt_de, dCadt_ex, \
            dresdt_si, dresdt_se, dresdt_di, dresdt_de, dresdt_ex, dndt, dhdt, dsdt, dcdt, dqdt, dzdt, dO_1dt, dO_2dt, dC_1dt, dC_2dt


    init_cell = EDPRmodel(T, Na_si0, Na_se0, Na_di0, Na_de0, Na_ex0, K_si0, K_se0, K_di0, K_de0, K_ex0, Cl_si0, Cl_se0, Cl_di0, Cl_de0, Cl_ex0, Ca_si0, Ca_se0, Ca_di0, Ca_de0, Ca_ex0, X_si0, X_se0, X_di0, X_de0, X_ex0, alpha, 0, 0, O_10, O_20, C_10, C_20, Ca_si0, Ca_di0, n0, h0, s0, c0, q0, z0)
    phi_si, phi_se, phi_di, phi_de, phi_ex, phi_sm, phi_dm = init_cell.membrane_potentials(I_stim=0)
    E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d, E_Ca_s, E_Ca_d = init_cell.reversal_potentials()
    
    # print initial values    
    
    q_si = init_cell.total_charge([init_cell.Na_si, init_cell.K_si, init_cell.Cl_si, init_cell.Ca_si, init_cell.X_si], init_cell.V_si)
    q_se = init_cell.total_charge([init_cell.Na_se, init_cell.K_se, init_cell.Cl_se, init_cell.Ca_se, init_cell.X_se], init_cell.V_se)        
    q_di = init_cell.total_charge([init_cell.Na_di, init_cell.K_di, init_cell.Cl_di, init_cell.Ca_di, init_cell.X_di], init_cell.V_di)
    q_de = init_cell.total_charge([init_cell.Na_de, init_cell.K_de, init_cell.Cl_de, init_cell.Ca_de, init_cell.X_de], init_cell.V_de)
    q_ex = init_cell.total_charge([init_cell.Na_ex, init_cell.K_ex, init_cell.Cl_ex, init_cell.Ca_ex, init_cell.X_ex], init_cell.V_ex)
    
    print("----------------------------")
    print("Initial values")
    print("----------------------------")
    print("initial total charge(C): ", q_si + q_se + q_di + q_de)
    print("Q_si (C):", q_si)
    print("Q_se (C):", q_se)
    print("Q_di (C):", q_di)
    print("Q_de (C):", q_de)
    print("Q_ex (C):", q_ex)
    print("----------------------------")
    print("potentials [mV]")
    print('phi_ex: ', round(phi_ex*1000))
    print('phi_si: ', round(phi_si*1000))
    print('phi_se: ', round(phi_se*1000))
    print('phi_di: ', round(phi_di*1000))
    print('phi_de: ', round(phi_de*1000))
    print('phi_sm: ', round(phi_sm*1000))
    print('phi_dm: ', round(phi_dm*1000))
    print('E_Na_s: ', round(E_Na_s*1000))
    print('E_Na_d: ', round(E_Na_d*1000))
    print('E_K_s: ', round(E_K_s*1000))
    print('E_K_d: ', round(E_K_d*1000))
    print('E_Cl_s: ', round(E_Cl_s*1000))
    print('E_Cl_d: ', round(E_Cl_d*1000))
    print('E_Ca_s: ', round(E_Ca_s*1000))
    print('E_Ca_d: ', round(E_Ca_d*1000))
    print("----------------------------")
    
    # solve
    t_span = (0, t_dur)

    k0 = [Na_si0, Na_se0, Na_di0, Na_de0, Na_ex0, K_si0, K_se0, K_di0, K_de0, K_ex0, Cl_si0, Cl_se0, Cl_di0, Cl_de0, Cl_ex0, \
          Ca_si0, Ca_se0, Ca_di0, Ca_de0, Ca_ex0, X_si0, X_se0, X_di0, X_de0, X_ex0, n0, h0, s0, c0, q0, z0, O_10, O_20, C_10, C_20]

    sol = solve_ivp(dkdt, t_span, k0, max_step=1e-3, method='LSODA') #, method='LSODA'
    #Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, Ca_si, Ca_se, Ca_di, Ca_de, \
    #    X_si, X_se, X_di, X_de, n, h, s, c, q, z = sol.y
    Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, \
    Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, n, h, s, c, q, z, O_1, O_2, C_1, C_2 = sol.y

    return sol
