import numpy as np
from .con_map import ConMapRfFnt, ConMapRfInf
from .esprit import ESPRIT

class MiniPoleRfDPR:
    def __init__(self, Al_dpr, xl_dpr, interval_type = "infinite", w_min = -10, w_max = 10, wp_max = 1, err = None, err_type = "abs", cutoff_err = None, cutoff_err_type = "abs", M = None, k_max = 999, Lfactor = 0.4, alpha = 1.0, minimal_k = False):
        '''
        Al_dpr (numpy.ndarray): Complex pole weights, either of shape (r,) or (r, n_orb, n_orb).
        xl_dpr (numpy.ndarray): Complex pole locations, an array of shape (r,).
        interval_type (str): Specifies the type of real-frequency interval; either "infinite" or "finite".
        w_min (float): Lower bound of the finite real-frequency interval.
        w_max (float): Upper bound of the finite real-frequency interval.
        wp_max (float): Parameter used in the Möbius transform for the infinite real-frequency interval.
        err (float): Error tolerance used during the approximation process.
        err_type (str): Type of error to use; either "abs" for absolute error or "rel" for relative error.
        cutoff_err (float): Cutoff value for h_k.
        cutoff_err_type (str): Specifies whether the cutoff is based on absolute ("abs") or relative ("rel") error.
        M (int): Number of poles in the final result.
        k_max (int): Maximum number of contour integrals.
        Lfactor (float): Ratio L / N used in ESPRIT.
        alpha (float): Scaling parameter inside the unit disk to accelerate convergence.
        minimal_k (bool): Whether to use a minimal number of h_k based on the size of `xl_dpr`.
        '''
        # make sure Al_dpr is of size (r, n_orb, n_orb)
        if Al_dpr.ndim == 1:
            Al_dpr = Al_dpr.reshape(-1, 1, 1)
        assert Al_dpr.ndim == 3
        assert Al_dpr.shape[0] == xl_dpr.size and Al_dpr.shape[1] == Al_dpr.shape[2]
        n_orb = Al_dpr.shape[1]
        assert interval_type in ["finite", "infinite"]
        assert alpha <= 1.0
        
        self.n_orb = n_orb
        self.Al_dpr = Al_dpr
        self.xl_dpr = xl_dpr
        self.interval_type = interval_type
        self.err = err
        self.err_type = err_type
        self.cutoff_err = cutoff_err
        self.cutoff_err_type = cutoff_err_type
        self.M = M
        self.k_max = k_max
        self.Lfactor = Lfactor
        self.alpha = alpha
        self.minimal_k = minimal_k
        
        if interval_type == "finite":
            assert w_min < w_max and w_min > -np.inf and w_max < np.inf
            self.w_min = w_min
            self.w_max = w_max
            #construct the conformal mapping
            w_m  = 0.5 * (w_max + w_min)
            dw_h = 0.5 * (w_max - w_min)
            self.con_map = ConMapRfFnt(w_m, dw_h)
        else:
            assert wp_max > 0
            self.wp_max = wp_max
            #construct the conformal mapping
            self.con_map = ConMapRfInf(wp_max)
        
        # Calculate contour integral
        self.xl_p = self.con_map.w(xl_dpr) * alpha
        if self.minimal_k is False:
            V = np.vander(self.xl_p, N = k_max, increasing=True).T
        else:
            V = np.vander(self.xl_p, N = min(int((self.xl_p.size + 1) / Lfactor), k_max), increasing=True).T
        self.Al_p =  Al_dpr.reshape(-1, n_orb ** 2) / self.con_map.dz(self.xl_p / alpha).reshape(-1, 1)
        self.h_k = V @ self.Al_p
        if cutoff_err is not None:
            cutoff = cutoff_err if cutoff_err_type == "abs" else np.abs(self.h_k[0]) * cutoff_err
            idx = np.where(np.abs(self.h_k) < cutoff)[0][0]
            self.h_k = self.h_k[:idx]
        
        #apply the Prony's approximation to recover poles
        self.find_poles()
    
    def find_poles(self):
        '''
        Recover poles from contour integrals h_k.
        '''
        # Extract pole information:
        # 1) Pole weights are stored in a numpy array of shape (M,) for single-orbital systems,
        #    or in an array of shape (M, n_orb, n_orb) for multi-orbital systems.
        # 2) Pole locations are stored in a numpy array of shape (M,).
        self.p = ESPRIT(self.h_k, err=self.err, err_type=self.err_type, M=self.M, Lfactor=self.Lfactor)
        idx = np.abs(self.p.gamma) < self.alpha
        Al = self.p.omega[idx] * self.con_map.dz(self.p.gamma[idx] / self.alpha).reshape(-1, 1)
        if self.n_orb == 1:
            weight = Al[:, 0]
        else:
            weight = Al.reshape(-1, self.n_orb, self.n_orb)
        location = self.con_map.z(self.p.gamma[idx] / self.alpha)
        #rearrange poles so that \xi_1.real <= \xi_2.real <= ... <= \xi_M.real
        idx2 = np.argsort(location.real)
        self.pole_weight   = weight[idx2]
        self.pole_location = location[idx2]
    
    def change_M(self, M):
        self.M = M
        self.find_poles()
