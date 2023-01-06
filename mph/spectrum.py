import numpy as np
from numba import njit

@njit
def compute_absorption(evecs, evals, omega, vibmax, dim, index_1p, fcmat, gamma, window=None):

    energy  = evals/omega
    gamma /= omega

    #omega = 1.0

    min_energy = min(energy)
    max_energy = max(energy)
    energy_range = abs(max_energy - min_energy)

    # Calculate the oscillator strength
    f_osc = np.zeros(dim, dtype=np.complex128)
    for s in range(dim):
        for v in range(vibmax):
            i = index_1p[v]
            if i == -1: continue

            f_osc[s] += evecs[i, s] * fcmat[0, v]
        f_osc[s] *= np.conj(f_osc[s])

    # Calculate the absorption spectrum
    nsteps = int(np.floor((energy_range + 8.0*gamma)/(10/omega))) # 10 cm-1 resolution

    # Restrict to a user-defined spectral window
    nsteps = min(nsteps, 1000)
    if window is None:
        spec_range = min(max_energy, min_energy + 10000/omega) - min_energy + 8.0*gamma
    else:
        spec_range = window/omega + 8.0*gamma
    step = spec_range / nsteps

    # Compute absorbance
    photon_energy = np.zeros(nsteps)
    absorbance = np.zeros(nsteps, dtype=np.complex128)
    for point in range(nsteps):
        if point == 0:
            photon_energy[point] = min_energy - 4.0 * gamma
        else:
            photon_energy[point] = photon_energy[point - 1] + step
        for s in range(dim):
            transition_energy = energy[s]
            # gaussian lineshape function
            lineshape = np.exp(-(photon_energy[point] - transition_energy)**2/(2*gamma**2))/np.sqrt(2*np.pi*gamma**2)
            absorbance[point] += f_osc[s] * lineshape

    return photon_energy * omega, absorbance


