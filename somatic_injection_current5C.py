


def somatic_injection_current(neuron, dNadt_si, dKdt_si, dCldt_si, dXdt_si, I_stim, frac_Na, frac_K, frac_Cl, frac_X=0):
    dNadt_si += frac_Na * I_stim/ (neuron.V_si * neuron.F)
    dKdt_si += frac_K * I_stim / (neuron.V_si * neuron.F)
    dCldt_si -= frac_Cl * I_stim / (neuron.V_si * neuron.F)
    dXdt_si -= frac_X * I_stim / (neuron.V_si * neuron.F)
    return dNadt_si, dKdt_si, dCldt_si, dXdt_si