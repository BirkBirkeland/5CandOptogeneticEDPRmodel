import numpy as np
import warnings
warnings.filterwarnings("error")

class LeakyCell(): 
    """A two plus two plus one compartment neuron model with Na+, K+, and Cl- leak currents.

    Methods
    -------
    constructor(T, Na_si, Na_se, Na_di, Na_de, K_si, K_se, K_di, K_de, Cl_si, Cl_se, Cl_di, Cl_de, \
        Ca_si, Ca_se, Ca_di, Ca_de, X_si, X_se, X_di, X_de, alpha)
    j_Na_s(phi_sm, E_Na_s): compute the Na+ flux across the somatic membrane
    j_K_s(phi_sm, E_K_s): compute the K+ flux across the somatic membrane
    j_Cl_s(phi_sm, E_Cl_s): compute the Cl- flux across the somatic membrane
    j_Na_d(phi_dm, E_Na_d): compute the Na+ flux across the dendritic membrane
    j_K_d(phi_dm, E_K_d): compute the K+ flux across the dendritic membrane
    j_Cl_d(phi_dm, E_Cl_d): compute the Cl- flux across the dendritic membrane
    j_k_diff(D_k, tortuosity, k_s, k_d): compute the axial diffusion flux of ion k
    j_k_drift(D_k, Z_k, tortuosity, k_s, k_d, phi_s, phi_d): compute the axial drift flux of ion k
    conductivity_k(D_k, Z_k, tortuosity, k_s, k_d): compute axial conductivity of ion k
    total_charge(k, V): calculate the total charge within volume V
    nernst_potential(Z, k_i, k_e): calculate the reversal potential of ion k
    reversal_potentials(): calculate the reversal potentials of all ion species
    membrane_potentials(): calculate the membrane potentials
    dkdt(): calculate dk/dt for all ion species k
    """

    def __init__(self, T, Na_si, Na_se, Na_di, Na_de, Na_ex, K_si, K_se, K_di, K_de, K_ex, Cl_si, Cl_se, Cl_di, Cl_de, Cl_ex, Ca_si, Ca_se, Ca_di, Ca_de, Ca_ex, X_si, X_se, X_di, X_de, X_ex, alpha, I_stim, l, O_1, O_2, C_1, C_2):
        
        # absolute temperature [K]
        self.T = T

        # ion concentraions [mol * m**-3]
        self.Na_si = Na_si
        self.Na_se = Na_se
        self.Na_di = Na_di
        self.Na_de = Na_de
        self.Na_ex = Na_ex
        self.K_si = K_si
        self.K_se = K_se
        self.K_di = K_di
        self.K_de = K_de
        self.K_ex = K_ex
        self.Cl_si = Cl_si
        self.Cl_se = Cl_se 
        self.Cl_di = Cl_di 
        self.Cl_de = Cl_de
        self.Cl_ex = Cl_ex
        self.Ca_si = Ca_si
        self.Ca_se = Ca_se 
        self.Ca_di = Ca_di 
        self.Ca_de = Ca_de
        self.Ca_ex = Ca_ex
        self.free_Ca_si = 0.01*Ca_si
        self.free_Ca_di = 0.01*Ca_di
        self.X_si = X_si
        self.X_se = X_se
        self.X_di = X_di
        self.X_de = X_de
        self.X_ex = X_ex

        # membrane capacitance [F * m**-2]
        self.C_sm = 3e-2 # Pinsky and Rinzel 1994
        self.C_dm = 3e-2 # Pinsky and Rinzel 1994
       
        # volumes and areas
        self.alpha = alpha
        self.A_s = 616e-12             # [m**2]
        self.A_d = 616e-12             # [m**2]
        self.A_i = self.alpha*self.A_s # [m**2]
        self.A_e = self.A_i/2.         # [m**2]
        self.V_si = 1437e-18           # [m**3]
        self.V_di = 1437e-18           # [m**3]
        self.V_se = 718.5e-18          # [m**3]
        self.V_de = 718.5e-18          # [m**3]
        self.dx = 667e-6               # [m]
        
        
        self.A_ex = self.A_e #616e-12             # [m**2]
        self.V_ex = 718.5e-18          # [m**3]
        

        # diffusion constants [m**2 s**-1]
        self.D_Na = 1.33e-9 # Halnes et al. 2013
        self.D_K = 1.96e-9  # Halnes et al. 2013 
        self.D_Cl = 2.03e-9 # Halnes et al. 2013
        self.D_Ca = 0.71e-9 # Halnes et al. 2016

        # tortuosities
        self.lamda_i = 3.2 # Halnes et al. 2013
        self.lamda_e = 1.6 # Halnes et al. 2013

        # valencies
        self.Z_Na = 1.
        self.Z_K = 1.
        self.Z_Cl = -1.
        self.Z_Ca = 2.
        self.Z_X = -1.

        # constants
        self.F = 9.648e4    # [C * mol**-1]
        self.R = 8.314      # [J * mol**-1 * K**-1] 

        # conductances [S * m**-2]
        self.g_Na_leak = 0.247 # Wei et al. 2014
        self.g_K_leak = 0.5    # Wei et al. 2014
        self.g_Cl_leak = 1.0   # Wei et al. 2014
        
        
        # Optogenetic
        self.G_ChR2 = 4.0 #S/m^2 dette er fra williams
        self.y = 0.1 #ratio of conductances of O_2/O_1, should be a gamma, this is from williams
        self.g_O1 = self.G_ChR2*(1-self.y)
        self.g_O2 = self.G_ChR2*self.y
        
        self.E_ChR2 = 0 # reversal potential for ChR2
        
        
        self.U_0  = 40e-3 # [V]
        self.U_1 = 15e-3 # [V]

        self.g_Na_ChR2 = 0.51669809
        self.g_K_ChR2 = 0.42476143
        self.g_Ca_ChR2 = 0.05854048

        self.O_1 = O_1
        self.O_2 = O_2
        self.C_1 = C_1
        self.C_2 = C_2
        

    def j_Na_s(self, phi_sm, E_Na_s):
        j = self.g_Na_leak*(phi_sm - E_Na_s) / (self.F*self.Z_Na)
        return j 

    def j_K_s(self, phi_sm, E_K_s):
        j = self.g_K_leak*(phi_sm - E_K_s) / (self.F*self.Z_K)
        return j

    def j_Cl_s(self, phi_sm, E_Cl_s):
        j = self.g_Cl_leak*(phi_sm - E_Cl_s) / (self.F*self.Z_Cl)
        return j

    def j_Na_d(self, phi_dm, E_Na_d):
        j = self.g_Na_leak*(phi_dm - E_Na_d) / (self.F*self.Z_Na) 
        return j

    def j_K_d(self, phi_dm, E_K_d):
        j = self.g_K_leak*(phi_dm - E_K_d) / (self.F*self.Z_K) 
        return j

    def j_Cl_d(self, phi_dm, E_Cl_d):
        j = self.g_Cl_leak*(phi_dm - E_Cl_d) / (self.F*self.Z_Cl) 
        return j

    def j_k_diff(self, D_k, tortuosity, k_s, k_d):
        j = - D_k * (k_d - k_s) / (tortuosity**2 * self.dx)
        return j

    def j_k_drift(self, D_k, Z_k, tortuosity, k_s, k_d, phi_s, phi_d):
        j = - D_k * self.F * Z_k * (k_d + k_s) * (phi_d - phi_s) / (2 * tortuosity**2 * self.R * self.T * self.dx)
        return j

    def conductivity_k(self, D_k, Z_k, tortuosity, k_s, k_d): 
        sigma = self.F**2 * D_k * Z_k**2 * (k_d + k_s) / (2 * self.R * self.T * tortuosity**2)
        return sigma

    def total_charge(self, k, V):
        Z_k = [self.Z_Na, self.Z_K, self.Z_Cl, self.Z_Ca, self.Z_X]
        q = 0.0
        for i in range(0, 5):
            q += Z_k[i]*k[i]
        q = q*self.F*V
        return q

    def nernst_potential(self, Z, k_i, k_e):
        E = self.R*self.T / (Z*self.F) * np.log(k_e / k_i)
        return E
    
    
    """" """
    def e(self, l):
        if l == 0:
            e_12  = 0
            e_21 = 0
        else:
            e_12 = 0.011 + 0.005*np.log(l/0.024) # [ms^-1] rate constant O_1 ->O_2
            e_21 = 0.008 + 0.004*np.log(l/0.024) # [ms^-1] rate constant O_2 ->O_1
        return e_12, e_21
        
    def G_ChR2(self, phi):
        G_ChR2 = (self.g_O1*self.O_1 + self.g_O2*self.O_2)*(1 - np.exp(-phi/self.U_0))/(phi/U_1)
        return G_ChR2
        

        
    def k(self, l):
        k_a1 = 0.5*l # [ms^-1] activation rate for C_1 -> 0_1 
        k_a2 = 0.12*l # [ms^-1] activation rate for C_2 -> 0_2 
        k_d1 = 0.1 # [ms^-1] closing rate for 0_1 -> C_1
        k_d2 = 0.05 # [ms^-1] closing rate for 0_2 -> C_2
        k_r = 1/3000
        return k_a1, k_a2, k_d1, k_d2, k_r
    
    def dNdt(self, l=0):
        e_12, e_21 = self.e(l)
        k_a1, k_a2, k_d1, k_d2, k_r = self.k(l)

        dO_1dt = k_a1*self.C_1 - self.O_1*(k_d1 + e_12) + e_21*self.O_2 # (1)
        dO_2dt =  k_a2*self.C_2 + e_12*self.O_1 + self.O_2*(k_d2 + e_21) # (2)
        dC_2dt =  k_d2*self.O_2 - self.C_2*(k_a2 + k_r)# (3)
        
        dC_1dt = - dO_1dt - dO_2dt - dC_2dt
        
        return dO_1dt, dO_2dt, dC_1dt, dC_2dt

    """"""
    
    def reversal_potentials(self):
        E_Na_s = self.nernst_potential(self.Z_Na, self.Na_si, self.Na_se)
        E_Na_d = self.nernst_potential(self.Z_Na, self.Na_di, self.Na_de)
        E_K_s = self.nernst_potential(self.Z_K, self.K_si, self.K_se)
        E_K_d = self.nernst_potential(self.Z_K, self.K_di, self.K_de)
        E_Cl_s = self.nernst_potential(self.Z_Cl, self.Cl_si, self.Cl_se)
        E_Cl_d = self.nernst_potential(self.Z_Cl, self.Cl_di, self.Cl_de)
        E_Ca_s = self.nernst_potential(self.Z_Ca, self.free_Ca_si, self.Ca_se)
        E_Ca_d = self.nernst_potential(self.Z_Ca, self.free_Ca_di, self.Ca_de)
        return E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d, E_Ca_s, E_Ca_d

    def membrane_potentials(self, I_stim=0):
        I_i_diff = self.F * (self.Z_Na*self.j_k_diff(self.D_Na, self.lamda_i, self.Na_si, self.Na_di) \
            + self.Z_K*self.j_k_diff(self.D_K, self.lamda_i, self.K_si, self.K_di) \
            + self.Z_Cl*self.j_k_diff(self.D_Cl, self.lamda_i, self.Cl_si, self.Cl_di) \
            + self.Z_Ca*self.j_k_diff(self.D_Ca, self.lamda_i, self.free_Ca_si, self.free_Ca_di))
        I_e_diff = self.F * (self.Z_Na*self.j_k_diff(self.D_Na, self.lamda_e, self.Na_se, self.Na_de) \
            + self.Z_K*self.j_k_diff(self.D_K, self.lamda_e, self.K_se, self.K_de) \
            + self.Z_Cl*self.j_k_diff(self.D_Cl, self.lamda_e, self.Cl_se, self.Cl_de) \
            + self.Z_Ca*self.j_k_diff(self.D_Ca, self.lamda_e, self.Ca_se, self.Ca_de))

        sigma_i = self.conductivity_k(self.D_Na, self.Z_Na, self.lamda_i, self.Na_si, self.Na_di) \
            + self.conductivity_k(self.D_K, self.Z_K, self.lamda_i, self.K_si, self.K_di) \
            + self.conductivity_k(self.D_Cl, self.Z_Cl, self.lamda_i, self.Cl_si, self.Cl_di) \
            + self.conductivity_k(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_Ca_si, self.free_Ca_di)
        sigma_e = self.conductivity_k(self.D_Na, self.Z_Na, self.lamda_e, self.Na_se, self.Na_de) \
            + self.conductivity_k(self.D_K, self.Z_K, self.lamda_e, self.K_se, self.K_de) \
            + self.conductivity_k(self.D_Cl, self.Z_Cl, self.lamda_e, self.Cl_se, self.Cl_de) \
            + self.conductivity_k(self.D_Ca, self.Z_Ca, self.lamda_e, self.Ca_se, self.Ca_de)

        q_di = self.total_charge([self.Na_di, self.K_di, self.Cl_di, self.Ca_di, self.X_di], self.V_di)
        q_si = self.total_charge([self.Na_si, self.K_si, self.Cl_si, self.Ca_si, self.X_si], self.V_si)
      
        phi_ex = 0.
        
        phi_de = ((self.dx*I_stim*self.lamda_e**2)/(self.F*self.A_ex) - (self.Z_Na*self.D_Na*(self.Na_de-self.Na_ex) \
                  + self.Z_K*self.D_K*(self.K_de-self.K_ex) + self.Z_Cl*self.D_Cl*(self.Cl_de-self.Cl_ex) \
                  + self.Z_Ca*self.D_Ca*(self.Ca_de-self.Ca_ex))) /  ((self.F/(2*self.R*self.T)) \
                        * (self.D_Na*(self.Na_de+self.Na_ex)*self.Z_Na**2 \
                        + self.D_K*(self.K_de+self.K_ex)*self.Z_K**2 \
                        + self.D_Cl*(self.Cl_de+self.Cl_ex)*self.Z_Cl**2 \
                        + self.D_Ca*(self.Ca_de+self.Ca_ex)*self.Z_Ca**2))
        
        phi_di = q_di / (self.C_dm * self.A_d) + phi_de
        

        phi_se = (self.dx/(self.A_e*sigma_e + self.A_i*sigma_i))*(-self.A_e*I_e_diff  \
                 - self.A_i*I_i_diff \
                 + self.A_e*sigma_e*phi_de/self.dx + self.A_i*sigma_i*phi_di/self.dx \
                 - (self.A_i*sigma_i*q_si)/(self.dx*self.C_sm*self.A_s) + I_stim)
        
        phi_si = q_si / (self.C_sm * self.A_s) + phi_se
        phi_sm = phi_si - phi_se
        phi_dm = phi_di - phi_de

        return phi_si, phi_se, phi_di, phi_de, phi_ex, phi_sm, phi_dm
    
    """"""
        
    def I_ChR2(self, phi):
        
        E_Na_s = self.nernst_potential(self.Z_Na, self.Na_si, self.Na_se)
        E_K_s = self.nernst_potential(self.Z_K, self.K_si, self.K_se)
        E_Ca_s = self.nernst_potential(self.Z_Ca, self.free_Ca_si, self.Ca_se)
        
        G = (self.g_O1*self.O_1 + self.g_O2*self.O_2)*(1 - np.exp(-phi/self.U_0))/(phi/self.U_1)
        #I_ChR2 = G*(phi - self.E_ChR2)
        I_ChR2_Na = G*self.g_Na_ChR2*(phi - E_Na_s) #I_ChR2*g_Na_ChR2
        I_ChR2_K = G*self.g_K_ChR2*(phi - E_K_s) #I_ChR2*g_K_ChR2
        I_ChR2_Ca = G*self.g_Ca_ChR2*(phi - E_Ca_s) #I_ChR2*g_Ca_ChR2 #Skal denne deles på 2 ?? pga z_Ca??
        return I_ChR2_Na, I_ChR2_K, I_ChR2_Ca
    
    def j_ChR2(self, phi):
        #phi_si, phi_se, phi_di, phi_de, phi_ex, phi_sm, phi_dm  = self.membrane_potentials(0)
        #G = (self.g_O1*self.O_1 + self.g_2*self.O_2)*(1 - np.exp(-phi/self.U_0))/(phi/self.U_1)
        
        I_ChR2_Na, I_ChR2_K, I_ChR2_Ca = self.I_ChR2(phi)

        j_ChR2_Na = I_ChR2_Na/(self.F*self.Z_Na) #I_ChR2*g_Na_ChR2
        j_ChR2_K = I_ChR2_K/(self.F*self.Z_K) #I_ChR2*g_K_ChR2
        j_ChR2_Ca = I_ChR2_Ca/(self.F*self.Z_Ca) #I_ChR2*g_Ca_ChR2
        return j_ChR2_Na, j_ChR2_K, j_ChR2_Ca
    
    """"""

    def dkdt(self, I_stim):
       
        phi_si, phi_se, phi_di, phi_de, phi_ex, phi_sm, phi_dm  = self.membrane_potentials(I_stim)
        E_Na_s, E_Na_d, E_K_s, E_K_d, E_Cl_s, E_Cl_d, E_Ca_s, E_Ca_d = self.reversal_potentials()
        
        j_ChR2_Na, j_ChR2_K, j_ChR2_Ca = self.j_ChR2(phi_sm)

        j_Na_sm = self.j_Na_s(phi_sm, E_Na_s)
        j_K_sm = self.j_K_s(phi_sm, E_K_s)
        j_Cl_sm = self.j_Cl_s(phi_sm, E_Cl_s)

        j_Na_dm = self.j_Na_d(phi_dm, E_Na_d)
        j_K_dm = self.j_K_d(phi_dm, E_K_d)    
        j_Cl_dm = self.j_Cl_d(phi_dm, E_Cl_d)

        j_Na_i = self.j_k_diff(self.D_Na, self.lamda_i, self.Na_si, self.Na_di) \
            + self.j_k_drift(self.D_Na, self.Z_Na, self.lamda_i, self.Na_si, self.Na_di, phi_si, phi_di) 
        j_K_i = self.j_k_diff(self.D_K, self.lamda_i, self.K_si, self.K_di) \
            + self.j_k_drift(self.D_K, self.Z_K, self.lamda_i, self.K_si, self.K_di, phi_si, phi_di)
        j_Cl_i = self.j_k_diff(self.D_Cl, self.lamda_i, self.Cl_si, self.Cl_di) \
            + self.j_k_drift(self.D_Cl, self.Z_Cl, self.lamda_i, self.Cl_si, self.Cl_di, phi_si, phi_di)
        j_Ca_i = self.j_k_diff(self.D_Ca, self.lamda_i, self.free_Ca_si, self.free_Ca_di) \
            + self.j_k_drift(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_Ca_si, self.free_Ca_di, phi_si, phi_di)

        j_Na_e = self.j_k_diff(self.D_Na, self.lamda_e, self.Na_se, self.Na_de) \
            + self.j_k_drift(self.D_Na, self.Z_Na, self.lamda_e, self.Na_se, self.Na_de, phi_se, phi_de)
        j_K_e = self.j_k_diff(self.D_K, self.lamda_e, self.K_se, self.K_de) \
            + self.j_k_drift(self.D_K, self.Z_K, self.lamda_e, self.K_se, self.K_de, phi_se, phi_de)
        j_Cl_e = self.j_k_diff(self.D_Cl, self.lamda_e, self.Cl_se, self.Cl_de) \
            + self.j_k_drift(self.D_Cl, self.Z_Cl, self.lamda_e, self.Cl_se, self.Cl_de, phi_se, phi_de)
        j_Ca_e = self.j_k_diff(self.D_Ca, self.lamda_e, self.Ca_se, self.Ca_de) \
            + self.j_k_drift(self.D_Ca, self.Z_Ca, self.lamda_e, self.Ca_se, self.Ca_de, phi_se, phi_de)
            
        j_Na_ex = self.j_k_diff(self.D_Na, self.lamda_e, self.Na_de, self.Na_ex) \
            + self.j_k_drift(self.D_Na, self.Z_Na, self.lamda_e, self.Na_de, self.Na_ex, phi_de, phi_ex)
        j_K_ex = self.j_k_diff(self.D_K, self.lamda_e, self.K_de, self.K_ex) \
            + self.j_k_drift(self.D_K, self.Z_K, self.lamda_e, self.K_de, self.K_ex, phi_de, phi_ex)
        j_Cl_ex = self.j_k_diff(self.D_Cl, self.lamda_e, self.Cl_de, self.Cl_ex) \
            + self.j_k_drift(self.D_Cl, self.Z_Cl, self.lamda_e, self.Cl_de, self.Cl_ex, phi_de, phi_ex)
        j_Ca_ex = self.j_k_diff(self.D_Ca, self.lamda_e, self.Ca_de, self.Ca_ex) \
            + self.j_k_drift(self.D_Ca, self.Z_Ca, self.lamda_e, self.Ca_de, self.Ca_ex, phi_de, phi_ex)

        dNadt_si = -j_Na_sm*(self.A_s / self.V_si) - j_Na_i*(self.A_i / self.V_si) - j_ChR2_Na*(self.A_s / self.V_si)
        dNadt_di = -j_Na_dm*(self.A_d / self.V_di) + j_Na_i*(self.A_i / self.V_di)
        dNadt_se = j_Na_sm*(self.A_s / self.V_se) - j_Na_e*(self.A_e / self.V_se) + j_ChR2_Na*(self.A_s / self.V_se)
        dNadt_de = j_Na_dm*(self.A_d / self.V_de) + j_Na_e*(self.A_e / self.V_de) - j_Na_ex*(self.A_ex / self.V_de) 
        
        
        dKdt_si = -j_K_sm*(self.A_s / self.V_si) - j_K_i*(self.A_i / self.V_si) - j_ChR2_K*(self.A_s / self.V_si)
        dKdt_di = -j_K_dm*(self.A_d / self.V_di) + j_K_i*(self.A_i / self.V_di)
        dKdt_se = j_K_sm*(self.A_s / self.V_se) - j_K_e*(self.A_e / self.V_se) + j_ChR2_K*(self.A_s / self.V_se)
        dKdt_de = j_K_dm*(self.A_d / self.V_de) + j_K_e*(self.A_e / self.V_de) - j_K_ex*(self.A_ex / self.V_de)

        
        dCldt_si = -j_Cl_sm*(self.A_s / self.V_si) - j_Cl_i*(self.A_i / self.V_si)
        dCldt_di = -j_Cl_dm*(self.A_d / self.V_di) + j_Cl_i*(self.A_i / self.V_di)
        dCldt_se = j_Cl_sm*(self.A_s / self.V_se) - j_Cl_e*(self.A_e / self.V_se)
        dCldt_de = j_Cl_dm*(self.A_d / self.V_de) + j_Cl_e*(self.A_e / self.V_de) - j_Cl_ex*(self.A_ex / self.V_de)

        dCadt_si = - j_Ca_i*(self.A_i / self.V_si) - j_ChR2_Ca*(self.A_s / self.V_si)
        dCadt_di = j_Ca_i*(self.A_i / self.V_di)
        dCadt_se = - j_Ca_e*(self.A_e / self.V_se) + j_ChR2_Ca*(self.A_s / self.V_se)
        dCadt_de = j_Ca_e*(self.A_e / self.V_de) - j_Ca_ex*(self.A_ex / self.V_de)

        dXdt_si = 0
        dXdt_di = 0
        dXdt_se = 0
        dXdt_de = 0
        
        dNadt_ex = 0
        dKdt_ex = 0
        dCadt_ex = 0
        dCldt_ex = 0
        dXdt_ex = 0
        
        return dNadt_si, dNadt_se, dNadt_di, dNadt_de, dNadt_ex, dKdt_si, dKdt_se, dKdt_di, dKdt_de, dKdt_ex, dCldt_si, dCldt_se, dCldt_di, dCldt_de, dCldt_ex, dCadt_si, dCadt_se, dCadt_di, dCadt_de, dCadt_ex, dXdt_si, dXdt_se, dXdt_di, dXdt_de, dXdt_ex
