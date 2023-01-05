import numpy as np
import mph.index as index
import mph.condon as condon
import mph.matel as matel
import mph.spectrum as spectrum

class MPH:

    def __init__(self, nmol, vibmax, omega, Ef, J, lambda_f, r2max=-1):
        self.nmol = nmol           # Number of molecules
        self.vibmax = vibmax       # Maximum number of vibrational quanta
        self.Ef = Ef               # On-site Frenkel exciton energy
        self.omega = omega         # Harmonic vibrational frequency
        self.J = J                 # Intermolecular Coulombic coupling (array of length nmol)
        self.lambda_f = lambda_f   # Neutral Frenkel exciton-phonon coupling
        self.r2max = r2max         # Maximum cutoff length for 2-particle states

        # By default, construct all possible 2p states
        if self.r2max == -1:
            self.r2max = self.nmol // 2

        self.s = np.asarray(list(range(-self.r2max + 1, self.r2max + 1)))
        self.k = np.asarray([2.0*np.pi/self.nmol * n for n in range(-self.nmol//2 + 1, self.nmol//2 + 1)])
        self.index_1p, self.dim_1p = self.get_index_1p()
        self.index_2p, self.dim_2p = self.get_index_2p()
        self.dim = self.dim_1p + self.dim_2p

        self.fcmat = self.get_fc_matrix()

    def get_index_1p(self):
        return index.index_1p(self.vibmax)

    def get_index_2p(self):
        return index.index_2p(self.vibmax, self.s, self.dim_1p)

    def get_fc_matrix(self):
        return condon.fcmatrix(self.vibmax, self.lambda_f)

    def kernel(self):

        self.evals = np.zeros((len(self.k), self.dim), dtype=np.complex64)
        self.evecs = np.zeros((len(self.k), self.dim, self.dim), dtype=np.complex64)

        for ik, k in enumerate(self.k):
            H = np.zeros((self.dim, self.dim), dtype=np.complex64)
        
            H = matel.build_1p1p(H, k, self.nmol, self.vibmax, self.Ef, self.omega, self.J, self.index_1p, self.fcmat)
            if self.r2max > 1:
                H = matel.build_1p2p(H, k, self.nmol, self.vibmax, self.Ef, self.omega, self.J, self.s, self.index_1p, self.index_2p, self.fcmat)
                H = matel.build_2p2p(H, k, self.nmol, self.vibmax, self.Ef, self.omega, self.J, self.s, self.index_2p, self.fcmat)

            e, v = np.linalg.eigh(H)
            isort = np.argsort(e)

            self.evals[ik, :] = e[isort]
            self.evecs[ik, :, :] = v[:, isort]

    def get_abs_spectrum(self, gamma):
        
        for i, val in enumerate(self.k):
            if val == 0:
                i0 = i
                break

        self.photon_energy, self.absorbance = spectrum.compute_absorption(self.evecs[i0, :, :], 
                                                                          np.real(self.evals[i0, :]), 
                                                                          self.omega, 
                                                                          self.vibmax, 
                                                                          self.dim, 
                                                                          self.index_1p, 
                                                                          self.fcmat,
                                                                          gamma)

        




