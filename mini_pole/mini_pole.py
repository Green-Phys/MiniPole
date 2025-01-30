import numpy as np
import scipy.integrate as integrate
from .esprit import ESPRIT
from .con_map import ConMapGeneric, ConMapGapless
from .green_func import GreenFunc

class MiniPole:
    def __init__(self, G_w, w, n0 = "auto", n0_shift = 0, err = None, err_type = "abs", M = None, symmetry = False, G_symmetric = False, compute_const = False, plane = None, include_n0 = False, k_max = 999, ratio_max = 10):
        '''
        A Python program for obtaining the matrix-valued minimal pole representation.

        Parameters
        ----------
        G_w : ndarray
            An (n_w, n_orb, n_orb) or (n_w,) array containing the Matsubara data.
        w : ndarray
            An (n_w,) array containing the corresponding real-valued Matsubara grid.
        n0 : int or str, default="auto"
            If "auto", n0 is automatically selected with an additional shift specified by n0_shift.
            If a non-negative integer is provided, n0 is fixed at that value.
        n0_shift : int, default=0
            The shift applied to the automatically determined n0.
        err : float
            Error tolerance for calculations.
        err_type : str, default="abs"
            Specifies the type of error: "abs" for absolute error or "rel" for relative error.
        M : int, optional
            The number of poles in the final result. If not specified, the precision from the first ESPRIT is used to extract poles in the second ESPRIT.
        symmetry : bool, default=False
            Determines whether to preserve up-down symmetry.
        G_symmetric : bool, default=False
            If True, the Matsubara data will be symmetrized such that G_{ij}(z) = G_{ji}(z).
        compute_const : bool, default=False
            Determines whether to compute the constant term in G(z) = sum_l Al / (z - xl) + const.
            If False, the constant term is fixed at 0.
        plane : str, optional
            Specifies whether to use the original z-plane or the mapped w-plane to compute pole weights.
        include_n0 : bool, default=True
            Determines whether to include the first n0 input points when weights are calculated in the z-plane.
        k_max : int, default=999
            The maximum number of contour integrals.
        ratio_max : float, default=10
            The maximum ratio of oscillation when automatically choosing n0.
        
        Returns
        -------
        Minimal pole representation of the given data.
        Pole weights are stored in `self.pole_weight', a numpy array of shape (M, n_orb, n_orb).
        Shared pole locations are stored in `self.pole_location', a numpy array of shape (M,).
        '''
        if G_w.ndim == 1:
            G_w = G_w.reshape(-1, 1, 1)
        assert G_w.ndim == 3
        assert G_w.shape[0] == w.size and G_w.shape[1] == G_w.shape[2]
        assert w[0] >= 0.0
        assert np.linalg.norm(np.diff(np.diff(w / np.abs(w).max())), ord=np.inf) < 1.e-6
        
        self.n_w = w.size
        self.n_orb = G_w.shape[1]
        if G_symmetric is True:
            self.G_w = 0.5 * (G_w + np.transpose(G_w, axes=(0, 2, 1)))
        else:
            self.G_w = G_w
        self.w = w
        self.G_symmetric = G_symmetric
        self.err = err
        self.err_type = err_type
        self.M = M
        self.symmetry = symmetry
        if symmetry is True and compute_const == True:
            raise Exception("Set symmetry to be False to calculate the overall constant!")
        self.compute_const = compute_const
        if plane is not None:
            self.plane = plane
        elif self.symmetry is False:
            self.plane = "z"
        else:
            self.plane = "w"
        assert self.plane in ["z", "w"]
        self.include_n0 = include_n0
        self.k_max = k_max
        self.ratio_max = ratio_max
        
        #perform the first ESPRIT approximation to approximate Matsubara data
        G_w_vector = self.G_w.reshape(-1, self.n_orb ** 2)
        self.p_o = [ESPRIT(G_w_vector[:, i], self.w[0], self.w[-1], err=self.err, err_type=self.err_type, Lfactor=0.4) for i in range(self.n_orb ** 2)]
        self.G_approx = [lambda x, idx=i: self.p_o[idx].get_value(x) for i in range(self.n_orb ** 2)]
        idx_sigma = np.argmax([self.p_o[i].sigma for i in range(self.n_orb ** 2)])
        self.S = self.p_o[idx_sigma].S
        self.sigma   = self.p_o[idx_sigma].sigma
        if n0 == "auto":
            assert isinstance(n0_shift, int) and n0_shift >= 0
            p_o2 = [ESPRIT(G_w_vector[:, i], self.w[0], self.w[-1], err=self.err, err_type=self.err_type, Lfactor=0.5) for i in range(self.n_orb ** 2)]
            w_cont = np.linspace(self.w[0], self.w[-1], 10 * self.w.size - 9)
            G_L1 = [self.p_o[i].get_value(w_cont)[:-1].reshape(self.w.size - 1, 10) for i in range(self.n_orb ** 2)]
            G_L2 = [    p_o2[i].get_value(w_cont)[:-1].reshape(self.w.size - 1, 10) for i in range(self.n_orb ** 2)]
            self.err_max = max(max([self.p_o[i].err_max for i in range(self.n_orb ** 2)]), max([p_o2[i].err_max for i in range(self.n_orb ** 2)]))
            G_L_diff = [np.abs(G_L2[i] - G_L1[i]).max(axis=1) for i in range(self.n_orb ** 2)]
            ctrl_interval = [np.logical_and(G_L_diff[i][:-1] <= self.err_max, G_L_diff[i][0:-1] / G_L_diff[i][1:] < ratio_max) for i in range(self.n_orb ** 2)]
            self.n0 = max([np.argmax(ctrl_interval[i]) for i in range(self.n_orb ** 2)]) + n0_shift
        else:
            assert isinstance(n0, int) and n0 >= 0
            self.err_max = max([self.p_o[i].err_max for i in range(self.n_orb ** 2)])
            self.n0 = n0
        
        if self.symmetry is False:
            #get the corresponding conformal mapping
            w_m = 0.5 * (self.w[self.n0] + self.w[-1])
            dw_h = 0.5 * (self.w[-1] - self.w[self.n0])
            self.con_map = ConMapGeneric(w_m, dw_h)
            #calculate contour integrals
            self.cal_hk_generic(self.G_approx, k_max)
        else:
            #use complex poles to approximate Matsubara data in [1j * w[-1], +inf)
            p = MiniPole(G_w, w, n0=n0, n0_shift=n0_shift, err=err, err_type=err_type, G_symmetric=G_symmetric, compute_const=compute_const, include_n0=False, k_max=k_max, ratio_max=ratio_max)
            self.G_approx_tail = [lambda x, Al=p.pole_weight.reshape(-1, self.n_orb ** 2)[:, i], xl=p.pole_location: self.cal_G_scalar(1j * x, Al, xl) for i in range(self.n_orb ** 2)]
            self.const = p.const
            #get the corresponding conformal mapping
            self.con_map = ConMapGapless(self.w[self.n0])
            #calculate contour integrals
            if G_symmetric is True:
                self.cal_hk_gapless_symmetric(self.G_approx, self.G_approx_tail, k_max)
            else:
                self.cal_hk_gapless(self.G_approx, self.G_approx_tail, k_max)
        
        #apply the second ESPRIT approximation to recover poles
        self.find_poles()
    
    def cal_hk_generic(self, G_approx, k_max = 999):
        '''
        Calculate the contour integrals.
        '''
        cutoff = self.err_max
        err = 0.01 * cutoff
        
        self.h_k = np.zeros((k_max, len(G_approx)), dtype=np.complex128)
        for k in range(self.h_k.shape[0]):
            for i in range(self.h_k.shape[1]):
                self.h_k[k, i] = self.cal_hk_generic_indiv(G_approx[i], k, err)
            if k >= 1:
                cutoff_matrix = np.logical_and(np.abs(self.h_k[k]) < cutoff, np.abs(self.h_k[k - 1]) < cutoff)
                if np.all(cutoff_matrix):
                    break
        self.h_k = self.h_k[:(k + 1)]
    
    def cal_hk_generic_indiv(self, G_approx, k, err):
        if k % 2 == 0:
            return (1.0j / np.pi) * integrate.quad(lambda x: G_approx(self.con_map.w_m + self.con_map.dw_h * np.sin(x)), -0.5 * np.pi, 0.5 * np.pi, weight="sin", wvar=k + 1, complex_func=True, epsabs=err, epsrel=err, limit=10000)[0]
        else:
            return (1.0  / np.pi) * integrate.quad(lambda x: G_approx(self.con_map.w_m + self.con_map.dw_h * np.sin(x)), -0.5 * np.pi, 0.5 * np.pi, weight="cos", wvar=k + 1, complex_func=True, epsabs=err, epsrel=err, limit=10000)[0]
    
    def cal_hk_gapless_symmetric(self, G_approx_head, G_approx_tail, k_max = 999):
        '''
        Calculate the contour integrals.
        '''
        cutoff = self.err_max
        err = 0.01 * cutoff
        
        theta0 = np.arcsin(self.con_map.w_min / self.w[-1])
        self.h_k = np.zeros((k_max, len(G_approx_head)), dtype=np.float64)
        for k in range(self.h_k.shape[0]):
            for i in range(self.h_k.shape[1]):
                    self.h_k[k, i] = self.cal_hk_gapless_symmetric_indiv(G_approx_head[i], k, err, theta0 + 1.e-12, 0.5 * np.pi) + \
                                     self.cal_hk_gapless_symmetric_indiv(G_approx_tail[i], k, err, 1.e-6, theta0 - 1.e-12)
            if k >= 1:
                cutoff_matrix = np.logical_and(np.abs(self.h_k[k]) < cutoff, np.abs(self.h_k[k - 1]) < cutoff)
                if np.all(cutoff_matrix):
                    break
        self.h_k = self.h_k[:(k + 1)]
    
    def cal_hk_gapless_symmetric_indiv(self, G_approx, k, err, theta_min, theta_max):
        if k % 2 == 0:
            return (-2.0 / np.pi) * integrate.quad(lambda x: G_approx(self.con_map.w_min / np.sin(x)).imag, theta_min, theta_max, weight="sin", wvar=k + 1, epsabs=err, epsrel=err, limit=10000)[0]
        else:
            return (+2.0 / np.pi) * integrate.quad(lambda x: G_approx(self.con_map.w_min / np.sin(x)).real, theta_min, theta_max, weight="cos", wvar=k + 1, epsabs=err, epsrel=err, limit=10000)[0]
    
    def cal_hk_gapless(self, G_approx_head, G_approx_tail, k_max = 999):
        '''
        Calculate the contour integrals.
        '''
        cutoff = self.err_max
        err = 0.01 * cutoff
        
        theta0 = np.arcsin(self.con_map.w_min / self.w[-1])
        self.h_k = np.zeros((k_max, len(G_approx_head)), dtype=np.complex128)
        for k in range(self.h_k.shape[0]):
            for i in range(self.n_orb):
                for j in range(i, self.n_orb):
                    if i == j:
                        idx = i * self.n_orb + j
                        self.h_k[k, idx] = self.cal_hk_gapless_symmetric_indiv(G_approx_head[idx], k, err, theta0 + 1.e-12, 0.5 * np.pi) + \
                                           self.cal_hk_gapless_symmetric_indiv(G_approx_tail[idx], k, err, 1.e-6, theta0 - 1.e-12)
                    else:
                        idx1 = i * self.n_orb + j
                        idx2 = j * self.n_orb + i
                        h1 = self.cal_hk_gapless_indiv(G_approx_head[idx1], k, err, theta0 + 1.e-12, 0.5 * np.pi) + \
                             self.cal_hk_gapless_indiv(G_approx_tail[idx1], k, err, 1.e-6, theta0 - 1.e-12)
                        h2 = self.cal_hk_gapless_indiv(G_approx_head[idx2], k, err, theta0 + 1.e-12, 0.5 * np.pi) + \
                             self.cal_hk_gapless_indiv(G_approx_tail[idx2], k, err, 1.e-6, theta0 - 1.e-12)
                        if k % 2 == 0:
                            self.h_k[k, idx1] = 1.0j / np.pi * (h1 - np.conjugate(h2))
                            self.h_k[k, idx2] = 1.0j / np.pi * (h2 - np.conjugate(h1))
                        else:
                            self.h_k[k, idx1] = 1.0 / np.pi * (h1 + np.conjugate(h2))
                            self.h_k[k, idx2] = 1.0 / np.pi * (h2 + np.conjugate(h1))
            if k >= 1:
                cutoff_matrix = np.logical_and(np.abs(self.h_k[k]) < cutoff, np.abs(self.h_k[k - 1]) < cutoff)
                if np.all(cutoff_matrix):
                    break
        self.h_k = self.h_k[:(k + 1)]
    
    def cal_hk_gapless_indiv(self, G_approx, k, err, theta_min, theta_max):
        if k % 2 == 0:
            return integrate.quad(lambda x: G_approx(self.con_map.w_min / np.sin(x)), theta_min, theta_max, weight="sin", wvar=k + 1, complex_func=True, epsabs=err, epsrel=err, limit=10000)[0]
        else:
            return integrate.quad(lambda x: G_approx(self.con_map.w_min / np.sin(x)), theta_min, theta_max, weight="cos", wvar=k + 1, complex_func=True, epsabs=err, epsrel=err, limit=10000)[0]
    
    def find_poles(self):
        '''
        Recover poles from contour integrals h_k.
        '''
        #apply the second ESPRIT
        if self.M is None:
            self.p_f = ESPRIT(self.h_k, err=0.5 * self.err_max, Lfactor=0.5)
        else:
            self.p_f = ESPRIT(self.h_k, M=self.M, Lfactor=0.5)
        
        #make sure all mapped poles are inside the unit disk
        idx0 = np.abs(self.p_f.gamma) < 1.0
        #tranform poles from w-plane to z-plane
        location = self.con_map.z(self.p_f.gamma[idx0])
        weight = self.p_f.omega[idx0] * self.con_map.dz(self.p_f.gamma[idx0]).reshape(-1, 1)
        
        if self.compute_const is False:
            self.const = 0.0
        else:
            G_w_approx = self.cal_G_vector(1j * self.w[self.n0:], weight, location)
            const = (self.G_w[self.n0:] - G_w_approx.reshape(-1, self.n_orb, self.n_orb)).mean(axis=0)
            self.const = const if np.abs(const).max() > 100.0 * self.err_max else 0.0
        
        if self.plane == "z":
            w_tmp   = self.w   if self.include_n0 else self.w[self.n0:]
            G_w_tmp = self.G_w if self.include_n0 else self.G_w[self.n0:]
            if self.symmetry is False:
                w = w_tmp
                G_w = G_w_tmp
            else:
                w = np.hstack((-w_tmp[::-1], w_tmp))
                G_w = np.concatenate((np.conjugate(np.transpose(G_w_tmp, axes=(0, 2, 1)))[::-1], G_w_tmp), axis=0)
            A = np.zeros((w.size, location.size), dtype=np.complex128)
            for i in range(location.size):
                A[:, i] = 1.0 / (1j * w - location[i])
            weight, residuals, rank, s = np.linalg.lstsq(A, (G_w - self.const).reshape(-1, self.n_orb ** 2), rcond=-1)
            self.lstsq_quality = (residuals, rank, s)
        
        #discard poles with negligible weights
        idx1 = np.abs(weight).max(axis=1) > self.err_max
        weight   = weight[idx1]
        location = location[idx1]
        
        #rearrange poles so that \xi_1.real <= \xi_2.real <= ... <= \xi_M.real
        idx2 = np.argsort(location.real)
        self.pole_weight   = weight[idx2].reshape(-1, self.n_orb, self.n_orb)
        self.pole_location = location[idx2]
    
    @staticmethod
    def cal_G_scalar(z, Al, xl):
        G_z = 0.0
        for i in range(xl.size):
            G_z += Al[i] / (z - xl[i])
        return G_z
    
    @staticmethod
    def cal_G_vector(z, Al, xl):
        G_z = 0.0
        for i in range(xl.size):
            G_z += Al[[i]] / (z.reshape(-1, 1) - xl[i])
        return G_z
    
    def plot_spectrum(self, orb_list = None, w_min = -10, w_max = 10, epsilon = 0.01):
        import matplotlib.pyplot as plt
        
        w = np.linspace(w_min, w_max, 10000)
        if orb_list is None:
            orb_list = [(i, j) for i in range(self.n_orb) for j in range(self.n_orb)]
        #dynamically generate colors, line styles, and markers based on the number of curves
        num_curves = len(orb_list)
        line_styles = ['-', '-.', ':', '--'] * (num_curves // 4 + 1)
        plt.figure()
        for idx, orb in enumerate(orb_list):
            i, j = orb
            gf = GreenFunc('F', 1.0, "discrete", A_i=self.pole_weight[:, i, j], x_i=self.pole_location)
            A_r = gf.get_spectral(w, epsilon=epsilon)
            plt.plot(w, A_r, linestyle=line_styles[idx], label="element (" + str(i) + ", " + str(j) + ")")
        if self.h_k.shape[1] <= 16:
            plt.legend()
        plt.xlabel(r"$\omega$")
        plt.ylabel(r"$A(\omega)$")
        plt.show()
    
    def check_valid(self):
        self.check_svd()
        self.check_first_approx()
        self.check_h_k()
        self.check_final_approx()
        self.check_pts()
    
    def check_svd(self):
        import matplotlib.pyplot as plt
        #check svd of the input data
        plt.figure()
        plt.semilogy(self.S, ".")
        plt.semilogy([0, self.S.size - 1], [self.sigma, self.sigma], color="gray", linestyle="--", label="singular value")
        plt.semilogy([0, self.S.size - 1], [self.err_max, self.err_max], color="k", label="precision")
        plt.legend()
        plt.xlabel(r"$n$")
        plt.ylabel(r"$\sigma_n$")
        plt.title("SVD of the input data")
        plt.show()
    
    def check_first_approx(self):
        import matplotlib.pyplot as plt
        num_curves = self.n_orb ** 2
        line_styles = ['-', '-.', ':', '--'] * (num_curves // 4 + 1)
        #check the first approximation
        plt.figure()
        for i in range(self.n_orb ** 2):
            row, col = i // self.n_orb, i % self.n_orb
            G_w1 = self.G_approx[i](self.w) if self.symmetry is False else self.G_approx[i](self.w) + (self.const + np.zeros((self.n_orb, self.n_orb))).reshape(-1)[i]
            plt.semilogy(self.w, np.abs(np.squeeze(G_w1) - self.G_w[:, row, col]), linestyle=line_styles[i], label="element (" + str(row) + ", " + str(col) + ")")
        plt.semilogy([self.w[0], self.w[-1]], [self.err_max, self.err_max], color="k", label="precision")
        if self.h_k.shape[1] <= 16:
            plt.legend()
        plt.xlabel(r"$\omega_n$")
        plt.ylabel(r"$|\hat{G}(i\omega_n) - G(i\omega_n)|$")
        plt.title("First approximation")
        plt.show()
    
    def check_h_k(self):
        import matplotlib.pyplot as plt
        #check h_k
        #part 1
        plt.figure()
        for i in range(self.n_orb ** 2):
            row, col = i // self.n_orb, i % self.n_orb
            plt.semilogy(np.abs(self.h_k[:, i]), '.', label="element (" + str(row) + ", " + str(col) + ")")
        plt.semilogy([0, self.h_k.shape[0] - 1], [self.err_max, self.err_max], color="k", label="precision")
        if self.h_k.shape[1] <= 16:
            plt.legend()
        plt.xlabel(r"$k$")
        plt.ylabel(r"$h_k$")
        plt.title("Contour integrals: value")
        plt.show()
        #part 2
        plt.figure()
        plt.semilogy(self.p_f.S, ".")
        if self.M is not None:
            plt.semilogy([0, self.p_f.S.size - 1], [self.p_f.S[self.M], self.p_f.S[self.M]], color="gray", linestyle="--", label="M poles")
        plt.semilogy([0, self.p_f.S.size - 1], [0.5 * self.err_max, 0.5 * self.err_max], color="k", label="0.5 precision")
        plt.legend()
        plt.xlabel(r"$n$")
        plt.ylabel(r"$\sigma_n$")
        plt.title("Contour integrals: SVD")
        plt.show()
        #part 3
        plt.figure()
        h_k_approx = self.p_f.get_value(np.linspace(0, 1, self.h_k.shape[0]))
        for i in range(self.n_orb ** 2):
            row, col = i // self.n_orb, i % self.n_orb
            plt.semilogy(np.abs(h_k_approx[:, i] - self.h_k[:, i]), '.', label="element (" + str(row) + ", " + str(col) + ")")
        if self.M is not None:
            plt.semilogy([0, self.h_k.shape[0] - 1], [self.p_f.S[self.M], self.p_f.S[self.M]], color="gray", linestyle="--", label="M poles")
        else:
            plt.semilogy([0, self.h_k.shape[0] - 1], [self.err_max, self.err_max], color="k", label="precision")
        if self.h_k.shape[1] <= 16:
            plt.legend()
        plt.xlabel(r"$k$")
        plt.ylabel(r"$|\hat{h}_k - h_k|$")
        plt.title("Contour integrals: approximation")
        plt.show()
    
    def check_final_approx(self):
        import matplotlib.pyplot as plt
        num_curves = self.n_orb ** 2
        line_styles = ['-', '-.', ':', '--'] * (num_curves // 4 + 1)
        #check the final approximation
        plt.figure()
        G_w2 = self.cal_G_vector(1j * self.w, self.pole_weight.reshape(-1, self.n_orb ** 2), self.pole_location).reshape(-1, self.n_orb, self.n_orb) + self.const
        for i in range(self.n_orb ** 2):
            row, col = i // self.n_orb, i % self.n_orb
            plt.semilogy(self.w, np.abs(G_w2[:, row, col] - self.G_w[:, row, col]), linestyle=line_styles[i], label="element (" + str(row) + ", " + str(col) + ")")
        if self.M is not None:
            plt.semilogy([self.w[0], self.w[-1]], [self.p_f.S[self.M], self.p_f.S[self.M]], color="gray", linestyle="--", label="M poles")
        else:
            plt.semilogy([self.w[0], self.w[-1]], [self.err_max, self.err_max], color="k", label="precision")
        if self.h_k.shape[1] <= 16:
            plt.legend()
        plt.xlabel(r"$\omega_n$")
        plt.ylabel(r"$|\hat{G}(i\omega_n) - G(i\omega_n)|$")
        plt.title("Final approximation")
        plt.show()
    
    def check_pts(self):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        colors = [(1, 1, 1), (0, 0, 1)] #(R, G, B) tuples for white and blue
        n_bins = 100 #Discretize the interpolation into bins
        cmap_name = "WtBu"
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins) #Create the colormap
        
        #check pole locations
        pts = self.pole_location
        scatter = plt.scatter(pts.real, pts.imag, c=np.linalg.norm(self.pole_weight.reshape(-1, self.n_orb ** 2), axis=1), vmin=0, vmax=1, cmap=cmap)
        cbar = plt.colorbar(scatter)
        cbar.set_label('weight')
        x_max = np.abs(self.pole_location.real).max() * 1.2
        y_max = max(np.abs(self.pole_location.imag).max() * 1.2, 1.0)
        plt.xlim([-x_max, x_max])
        plt.ylim([-y_max, y_max])
        plt.xlabel(r"Real($z$)")
        plt.ylabel(r"Imag($z$)")
        plt.show()
        
        #check mapped pole locations
        theta = np.arange(1001) * 2.0 * np.pi / 1000
        pts = self.con_map.w(self.pole_location)
        plt.plot(np.cos(theta), np.sin(theta), color="tab:orange")
        scatter = plt.scatter(pts.real, pts.imag, c=np.linalg.norm(self.pole_weight.reshape(-1, self.n_orb ** 2), axis=1), vmin=0, vmax=1, cmap=cmap)
        cbar = plt.colorbar(scatter)
        cbar.set_label('weight')
        plt.xlim([-1.05, 1.05])
        plt.ylim([-1.05, 1.05])
        plt.xlabel(r"Real($w$)")
        plt.ylabel(r"Imag($w$)")
        plt.show()
