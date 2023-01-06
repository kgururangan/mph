import numpy as np
from matplotlib import pyplot as plt
from mph.mph import MPH
from mph.utilities import plot_spectrum, print_1p_index, print_2p_index

if __name__ == "__main__":

    nmol = 20
    vibmax = 5
    Ef = 14000
    omega = 1400
    lambda_f = 2.0
    J = np.zeros(nmol)
    J[1] = 700
    r2max = 10

    H = MPH(nmol, vibmax, omega, Ef, J, lambda_f, r2max=r2max)

    H.kernel()

    for i in range(len(H.k)):
        print(H.evals[i, 0] / H.omega)

    H.get_abs_spectrum(gamma=250)

    plot_spectrum(H)
