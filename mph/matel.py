"""This module computes the matrix elements of the Frenkel-Holstein Hamiltonian
in the many-particle delocalized Bloch basis. 

The following definition of the Hamiltonian is used
    H = Z + V, where Z = \sum_{n} A_n^+A_n + \sum_n b_n^+b_n and 
    V = \sum_{mn}' J(m-n) A_n+A_m + J(n-m) A_m^+A_n. 
The delocalized Bloch states used are
    (1p) |ku> = N**(-1/2) \sum_p exp(ikp) |pu>
    (2p) |ku,sv> = N**(-1/2) \sum_p exp(ikp) |pu,(p+s)v>
    (ct) |ku,sv> = N**(-1/2) \sum_p exp(ikp) |p+u+,(p+s)-v-> (|cation,anion>)
"""

import numpy as np
from numba import njit

@njit
def build_1p1p(H, k, nmol, vibmax, Ef, omega, J, index_1p, fcmat):
    """ Matrix element between the 1p Bloch states
        <ku|z|kv> = delta(u,v)*[Ef + hw*v]
        <ku|v|kv> = L*(u|0)*(0|v), where L = \sum_{s=0}^N J(s)*exp(iks) + J(-s)*exp(-iks)
    """

    for v1 in range(vibmax):
        i = index_1p[v1]

        H[i, i] += v1*omega + Ef

        for v2 in range(vibmax):
            j = index_1p[v2]

            L = 0.0
            for s in range(nmol):
                L += 2.0 * J[s] * np.cos(k*s)
            H[i, j] += L * fcmat[0, v1] * fcmat[0, v2]

    return H

@njit
def build_1p2p(H, k, nmol, vibmax, Ef, omega, J, srange, index_1p, index_2p, fcmat):
    """ Matrix element between the 1p and 2p Bloch states
        <ku|z|kv,sv'> = 0
        <ku|v|kv,sv'> = J(s)*(u|v')*(0|v)*exp(iks)
    """

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
    """ Matrix element between the 2p Bloch states
        <ku,sv|z|ku',s'v'> = delta(u,u')*delta(v,v')*delta(s,s')*[Ef + hw(u'+v')]
        <ku,sv|v|kv',s'v'> = delta(s1+s2)*[J(s)*(u|v')*(v|u')*exp(iks)] (Linker coupling)
                             + [1 - delta(s+s')][delta(v,v')*J(s-s')*(u|0)*(0|u')*exp(-ik(s-s')) (Exchange coupling)
    """

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

                                # Excitation transfer is s1-s2, wrap around for periodic boundary conditions
                                if s1 - s2 > srange[-1]: ds = s1 - s2 + nmol
                                elif s1 - s2 < srange[0]: ds = s1 - s2 - nmol
                                else: ds = s1 - s2

                                if ds == 0: continue # protect against case where s1 = s2; excitation transfer distance should be > 0
                                H[i, j] += J[abs(ds)] * fcmat[0, v1] * fcmat[0, v2] * np.exp(-1j*k*ds)

                            # Exchange-type coupling
                            if s1 + s2 == 0:
                                H[i, j] += J[abs(s1)] * fcmat[vv2, v1] * fcmat[vv1, v2] * np.exp(-1j*k*s1)

    return H

@njit
def build_1pct(H, k, nmol, vibmax, omega, te, th, index_1p, index_ct, fcmat_gc, fcmat_ga, fcmat_cf, fcmat_af):

    for v1 in range(vibmax):
        i = index_1p[v1]

        for vc in range(vibmax):
            for s in range(-1, 2):
                for va in range(vibmax):
                    j = index_ct[vc, s, va]
                    if j == -1: continue

                    H[i, j] += te * fcmat_ga[0, va] * fcmat_cf[vc, v1]
                    H[i, j] += th * fcmat_gc[0, vc] * fcmat_af[va, v1] * np.exp(1j*k*s)

                    H[j, i] = np.conj(H[i, j])
    return H

@njit 
def build_2pct(H, k, nmol, vibmax, omega, te, th, srange, index_2p, index_ct, fcmat_gc, fcmat_ga, fcmat_cf, fcmat_af):

    for v1 in range(vibmax):
        for s1 in srange:
            for vv1 in range(vibmax):
                i = index_2p[v1, s1, vv1]
                if i == -1: continue

                for vc in range(vibmax):
                    for s2 in range(-1, 2):
                        for va in range(vibmax):
                            j = index_ct[vc, s2, va]
                            if j == -1: continue

                            # Enforcing periodic boundary conditions?
                            #if -s2 > nmol // 2: s3 = -s2 - nmol
                            #elif -s2 < nmol // 2: s3 = -s2 + nmol
                            #else: s3 = -s2

                            s3 = -s2
                            if s1 == s2:
                                H[i, j] += te * fcmat_ga[vv1, va] * fcmat_cf[vc, v1]
                            if s1 == s3:
                                H[i, j] += th * fcmat_gc[vv1, vc] * fcmat_af[va, v1] * np.exp(1j*k*s3)

                            H[j, i] = np.conj(H[i, j])
    return H

@njit
def build_ctct(H, k, nmol, vibmax, ECT, ECTInf, omega, te, th, index_ct, fcmat_gc, fcmat_ga):

    for vc1 in range(vibmax):
        for s1 in range(-1, 2):
            for va1 in range(vibmax):
                i = index_ct[vc1, s1, va1]
                if i == -1: continue

                H[i, i] += (ECTInf*(abs(s1)-1) + ECT)/abs(s1) + (vc1 + va1) * omega

                for vc2 in range(vibmax):
                    for s2 in range(-1, 2):
                        for va2 in range(vibmax):
                            j = index_ct[vc2, s2, va2]
                            if j == -1: continue

                            s = s2 - s1

                            # Enforcing periodic boundary conditions?
                            #if s > nmol // 2: s = s - nmol
                            #elif s < nmol // 2: s = s + nmol

                            if abs(s) != 1: continue

                            if vc1 == vc2:
                                H[i, j] += te * fcmat_ga[0, va1] * fcmat_ga[0, va2]
                            if va1 == va2:
                                H[i, j] += th * fcmat_gc[0, vc1] * fcmat_gc[0, vc2] * np.exp(1j*k*(s2-s1))
    return H

