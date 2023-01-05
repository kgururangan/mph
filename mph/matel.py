import numpy as np
from numba import njit

@njit
def build_1p1p(H, k, nmol, vibmax, Ef, omega, J, index_1p, fcmat):

    for v1 in range(vibmax):
        i = index_1p[v1]

        H[i, i] += v1*omega + Ef

        for v2 in range(vibmax):
            j = index_1p[v2]

            Jk = 2.0 * J[1] * np.cos(k)
            H[i, j] += Jk * fcmat[0, v1] * fcmat[0, v2]

    return H

@njit
def build_1p2p(H, k, nmol, vibmax, Ef, omega, J, srange, index_1p, index_2p, fcmat):

    for v1 in range(vibmax):
        i = index_1p[v1]

        for v2 in range(vibmax):
            for s in srange:
                for vv2 in range(vibmax):

                    j = index_2p[v2, s, vv2]
                    if j == -1: continue

                    H[i, j] += J[abs(s)] * fcmat[0, v2] * fcmat[vv2, v1] * np.exp(1j*k*s)
                    H[j, i] = np.conj(H[i, j])

    return H

@njit
def build_2p2p(H, k, nmol, vibmax, Ef, omega, J, srange, index_2p, fcmat):

    for v1 in range(vibmax):
        for s1 in srange:
            for vv1 in range(vibmax):

                i = index_2p[v1, s1, vv1]
                if i == -1: continue
    
                # Diagonal energy
                H[i, i] += (v1 + vv1) * omega + Ef

                for v2 in range(vibmax):
                    for s2 in srange:
                        for vv2 in range(vibmax):

                            j = index_2p[v2, s2, vv2]
                            if j == -1: continue

                            # Linker-type coupling
                            if vv1 == vv2 and s1 + s2 != 0:
                                ds = s1 - s2
                                # Bring distance inside aggregate range, if necessary
                                if ds < srange[0]: ds += nmol
                                if ds >= srange[-1]: ds -= nmol
                                H[i, j] += J[abs(ds)] * fcmat[0, v1] * fcmat[0, v2] * np.exp(-1j*k*ds)

                            # Exchange-type coupling
                            if s1 + s2 == 0:
                                H[i, j] += J[abs(ds)] * fcmat[vv2, v1] * fcmat[vv1, v2] * np.exp(-1j*k*s1)

    return H
