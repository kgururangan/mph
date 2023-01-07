import numpy as np
import mph.index as index
import mph.condon as condon
import mph.matel as matel
import mph.spectrum as spectrum

class MPH:

    def __init__(self, nmol, vibmax, omega, Ef, ECT, ECTInf, J, te, th, lambda_f, lambda_c, lambda_a, r2max=-1, rctmax=0):
        self.nmol = nmol           # Number of molecules
        self.vibmax = vibmax       # Maximum number of vibrational quanta
        self.Ef = Ef               # On-site Frenkel exciton energy
        self.ECT = ECT             # Energy of nearest-neighbor separated CT state
        self.ECTInf = ECTInf       # Energy of CT state at infinite separation
        self.omega = omega         # Harmonic vibrational frequency
        self.J = J                 # Intermolecular Coulombic coupling (array of length nmol)
        self.te = te               # Electron CT hopping integral
        self.th = th               # Hole CT hopping integral
        self.lambda_f = lambda_f   # Neutral Frenkel exciton-phonon coupling
        self.lambda_c = lambda_c   # Cationic CT exciton-phonon coupling
        self.lambda_a = lambda_a   # Anionic CT exciton-phonon coupling

        # Compute the s-site separated CT energy used in computation: ECT(s)_comp = (ECTInf(s - 1) + ECT(s))/s
        # ECT is an s-site separated CT energy (for simplicity, we are using NN only, so ECT is a scalar)
        # ECTInf is the energy of a CT state at infinite separation
        #self.ECT = np.zeros(self.nmol)
        #for s in range(nmol):
        #    if s != 1: continue
        #    self.ECT[s] = (ECTInf * (s - 1) + ECT)/s

        # k is the range of k points, corresponding to k = 2*pi*i*n/N, n = -N/2,...,N/2
        self.k = np.asarray([2.0*np.pi/self.nmol * n for n in range(-self.nmol//2, self.nmol//2 + 1)])

        self.r2max = r2max         # Maximum cutoff length for 2-particle states (set to 0 to turn off 2p states)
        # By default, construct all possible 2p states |pu,(p+s)v>, where s = +/- 1, 2, 3, ..., nmol/2
        if self.r2max == -1:
            self.r2max = self.nmol // 2
        self.s = np.asarray(list(range(-self.r2max, self.r2max + 1)))

        self.rctmax = rctmax         # Maximum cutoff length for CT states (default is 0 to turn off CT states; set to 1 for NN coupling)
        self.sct = np.asarray(list(range(-self.rctmax, self.rctmax + 1)))

        self.index_1p, self.dim_1p = self.get_index_1p()
        self.index_2p, self.dim_2p = self.get_index_2p()
        self.index_ct, self.dim_ct = self.get_index_ct()

        self.dim = self.dim_1p + self.dim_2p + self.dim_ct

        self.fcmat = self.get_fc_matrix()

    def get_index_1p(self):
        return index.index_1p(self.vibmax)

    def get_index_2p(self):
        return index.index_2p(self.vibmax, self.s, self.dim_1p)

    def get_index_ct(self):
        return index.index_ct(self.vibmax, self.sct, self.dim_1p + self.dim_2p)

    def get_fc_matrix(self):
        return condon.fcmatrix(self.vibmax, self.lambda_f, self.lambda_c, self.lambda_a)

    def kernel(self):

        self.evals = np.zeros((len(self.k), self.dim), dtype=np.complex64)
        self.evecs = np.zeros((len(self.k), self.dim, self.dim), dtype=np.complex64)

        for ik, k in enumerate(self.k):
            H = np.zeros((self.dim, self.dim), dtype=np.complex64)
        
            H = matel.build_1p1p(H, k, self.nmol, self.vibmax, self.Ef, self.omega, self.J, self.index_1p, self.fcmat['gf'])

            # Include 2p states
            if self.r2max > 0:
                H = matel.build_1p2p(H, k, self.nmol, self.vibmax, 
                                     self.Ef, self.omega, self.J, 
                                     self.s, self.index_1p, self.index_2p, 
                                     self.fcmat['gf'])
                H = matel.build_2p2p(H, k, self.nmol, self.vibmax, 
                                     self.Ef, self.omega, self.J, 
                                     self.s, self.index_2p, 
                                     self.fcmat['gf'])

            # Include CT states
            if self.rctmax > 0:
                H = matel.build_1pct(H, k, self.nmol, self.vibmax, 
                                     self.omega, self.te, self.th,
                                     self.sct, self.index_1p, self.index_ct, 
                                     self.fcmat["gc"], self.fcmat["ga"], self.fcmat["cf"], self.fcmat["af"])
                # Include 2p states in addition to CT states
                if self.r2max > 0:
                    H = matel.build_2pct(H, k, self.nmol, self.vibmax, 
                                         self.omega, self.te, self.th, 
                                         self.s, self.sct, self.index_2p, self.index_ct, 
                                         self.fcmat["gc"], self.fcmat["ga"], self.fcmat["cf"], self.fcmat["af"])
                H = matel.build_ctct(H, k, self.nmol, self.vibmax, 
                                     self.ECT, self.ECTInf, self.omega, self.te, self.th, 
                                     self.sct, self.index_ct, 
                                     self.fcmat["gc"], self.fcmat["ga"])

            e, v = np.linalg.eigh(H)
            isort = np.argsort(e)

            self.evals[ik, :] = e[isort]
            self.evecs[ik, :, :] = v[:, isort]

    def get_abs_spectrum(self, gamma, window=None):
        
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
                                                                          self.fcmat['gf'],
                                                                          gamma,
                                                                          window)

        




