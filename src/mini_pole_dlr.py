import numpy as np
from con_map import *
from esprit import *

class MiniPoleDLR:
    '''
    A Python program implementing the MPM-DLR algorithm.
    '''
    def __init__(self, Al_dlr, xl_dlr, beta, n0, err = None, err_type = "abs", symmetry = False, nmax = None, Lfactor = 0.4):
        '''
        Al_dlr (numpy.ndarray): DLR coefficients, either of shape (r,) or (r, n_orb, n_orb).
        xl_dlr (numpy.ndarray): DLR grid for the real frequency, an array of shape (r,).
        beta (float): Inverse temperature of the system (1/kT).
        n0 (int): Number of initial points to discard, typically in the range (0, 10).
        err (float): Error tolerance for calculations.
        err_type (str): Specifies the type of error, "abs" for absolute error or "rel" for relative error.
        symmetry (bool): Whether to impose up-down symmetry (True or False).
        nmax (int): Cutoff for the Matsubara frequency when symmetry is False.
        Lfactor (float): Ratio of L/N in the ESPRIT algorithm.
        '''
        #make sure Al_dlr is of size (r, n_orb, n_orb)
        if Al_dlr.ndim == 1:
            Al_dlr = Al_dlr.reshape(-1, 1, 1)
        assert Al_dlr.ndim == 3
        assert Al_dlr.shape[0] == xl_dlr.size and Al_dlr.shape[1] == Al_dlr.shape[2]
        n_orb = Al_dlr.shape[1]
        
        # Construct holomorphic mapping
        w_n0 = (2 * n0 + 1) * np.pi / beta
        if symmetry is False:
            nmax = beta if nmax is None else nmax
            w_nmax = (2 * nmax + 1) * np.pi / beta
            w_m = 0.5 * (w_n0 + w_nmax)
            dw_h = 0.5 * (w_nmax - w_n0)
            self.con_map = ConMapGeneric(w_m, dw_h)
        else:
            self.con_map = ConMapGapless(w_n0)
        
        # Calculate contour integral
        self.xl_p = self.con_map.w(xl_dlr)
        V = np.vander(self.xl_p, N = int(self.xl_p.size / Lfactor), increasing=True).T
        self.Al_p =  Al_dlr.reshape(-1, n_orb ** 2) / self.con_map.dz(self.xl_p).reshape(-1, 1)
        self.h_k = V @ self.Al_p
        
        # Extract pole information:
        # 1) Pole weights are stored in a numpy array of shape (M,) for single-orbital systems,
        #    or in an array of shape (M, n_orb, n_orb) for multi-orbital systems.
        # 2) Pole locations are stored in a numpy array of shape (M,).
        self.p = ESPRIT(self.h_k, err=err, err_type=err_type, Lfactor=Lfactor)
        idx = np.abs(self.p.gamma) < 1
        Al = self.p.omega[idx] * self.con_map.dz(self.p.gamma[idx]).reshape(-1, 1)
        if n_orb == 1:
            self.pole_weight = Al[:, 0]
        else:
            self.pole_weight = Al.reshape(-1, n_orb, n_orb)
        self.pole_location = self.con_map.z(self.p.gamma[idx])
