import numpy as np
from matplotlib import pyplot as plt

def plot_spectrum(H, nroot=1):

    fig = plt.figure(figsize=(8.0,6.0))

    #plot the spectrum with matplotlib
    plt.subplot(2,1,1)
    plt.plot( H.photon_energy/H.omega, H.absorbance, c='r' )
    plt.xlabel(r'Energy (h$\omega$)')
    plt.ylabel('Absorption (a.u.)')
    plt.title('Absorption Spectrum')

    #plot the dispersion with matplotlib
    plt.subplot(2,1,2)
    for i in range(nroot):
        plt.plot( H.k/np.pi, H.evals[:, i]/H.omega )
        plt.scatter( H.k/np.pi, H.evals[:, i]/H.omega, s=5 )
    plt.xlabel(r'k ($\pi$)')
    plt.ylabel(r'Energy (h$\omega$)')
    plt.title('Exciton Dispersion')
    plt.tight_layout()
    plt.show()
    plt.close()

    return

def print_1p_index(H):

    for v in range(H.vibmax):
        print(v, H.index_1p[v])
    return

def print_2p_index(H):

    for v in range(H.vibmax):
        for s in H.s:
            for vv in range(1, H.vibmax):
                print(v, s, vv, H.index_2p[v, s, vv])
    return
