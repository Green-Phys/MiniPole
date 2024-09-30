import numpy as np
import scipy.integrate as integrate
from esprit import *
from con_map import *
from green_func import *

class MiniPoleMatrix:
    '''
    A python program for obtaining the minimal pole representation, suitable for any case.
    '''
    def __init__(self, G_w, w, n0 = "auto", n0_shift = 0, err = None, err_type = "abs", M = None, symmetry = False, G_symmetric = True, compute_const = False, pole_real = False, plane = None, include_n0 = False, elementwise = True, Lfactor = 0.4, k_max = 999, x_range = [-np.inf, np.inf], y_range = [-np.inf, np.inf]):
        '''
        G_w is an (n_w, n_orb, n_orb) array containing the Matsubara data, w (real-valued) is the corresponding sampling grid;
        If G_symmetric is True, the Matsubara data will be symmetrized so that G_{ij}(z) = G_{ji}(z);
        If the error tolerance err is given, the continuation will be carried out in this tolerance;
        else the tolerance will be chosen to be the last singular value in the exponentially decaying range;
        It is suggested to always provide err (>=1.e-12);
        M is the number of poles in the final result. If it is not given, the precision in the first ESPRIT will be used to extract poles in the second ESPRIT;
        symmetry determines whether to preserve the up-down symmetry;
        compute_const determines whether to compute the constant in G(z) = sum_l Al / (z - xl) + const. If it is False, const is fixed at 0;
        pole_real determines whether to restrict the poles exactly on the real axis when symmetry is True;
        plane decides whehter to use the original plane (z plane) or mapped plane (w plane) to compute pole weights;
        Lfactor determines the L parameter in ESPRIT (L = Lfactor * N, where N is the number of sampling points);
        k_max is the maximum number of contour integrals;
        Only poles located within the rectangle  x_range[0] < x < x_rang[1], y_range[0] < y < y_range[1] are retained.
        '''
        if G_w.ndim == 1:
            G_w = G_w.reshape(-1, 1, 1)
        assert G_w.ndim == 3
        assert G_w.shape[0] == w.size and G_w.shape[1] == G_w.shape[2]
        assert w[0] >= 0.0
        assert np.linalg.norm(np.diff(np.diff(w)), ord=np.inf) < 1.e-10
        
        self.n_w = w.size
        self.n_orb = G_w.shape[1]
        if symmetry is True:
            assert G_symmetric is True #update later to deal with unsymmetric case!!!
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
        self.compute_const = compute_const
        self.pole_real = pole_real
        if plane is not None:
            self.plane = plane
        elif self.symmetry is False:
            self.plane = "z"
        else:
            self.plane = "w"
        assert self.plane in ["z", "w"]
        self.include_n0 = include_n0
        self.elementwise = elementwise
        self.Lfactor = Lfactor
        self.k_max = k_max
        self.x_range = x_range
        self.y_range = y_range
        
        if self.symmetry is False:
            #perform the first ESPRIT approximation to approximate Matsubara data
            if self.elementwise is False:
                self.p_o = ESPRIT(self.G_w, self.w[0], self.w[-1], err=self.err, err_type=self.err_type, Lfactor=self.Lfactor)
                self.G_approx = [lambda x, idx=i: self.p_o.get_value_indiv(x, idx) for i in range(self.n_orb ** 2)]
                self.S = self.p_o.S
                self.sigma   = self.p_o.sigma
                self.err_max = self.p_o.err_max
                self.err_ave = self.p_o.err_ave
                if n0 == "auto":
                    assert self.Lfactor != 0.5
                    assert isinstance(n0_shift, int) and n0_shift >= 0
                    p_o2 = ESPRIT(self.G_w, self.w[0], self.w[-1], err=self.err, err_type=self.err_type, Lfactor=0.5)
                    w_cont = np.linspace(self.w[0], self.w[-1], 10 * self.w.size - 9)
                    G_L1 = self.p_o.get_value(w_cont)[:-1].reshape(self.w.size - 1, 10, self.n_orb ** 2)
                    G_L2 =     p_o2.get_value(w_cont)[:-1].reshape(self.w.size - 1, 10, self.n_orb ** 2)
                    ctrl_interval = np.all(np.abs(G_L2 - G_L1) <= self.p_o.err_max, axis=(1, 2))
                    self.n0 = np.argmax(ctrl_interval) + n0_shift #maybe change it later...
                else:
                    assert isinstance(n0, int) and n0 >= 0
                    self.n0 = n0
            else:
                G_w_vector = self.G_w.reshape(-1, self.n_orb ** 2)
                self.p_o = [ESPRIT(G_w_vector[:, i], self.w[0], self.w[-1], err=self.err, err_type=self.err_type, Lfactor=self.Lfactor) for i in range(self.n_orb ** 2)]
                self.G_approx = [lambda x, idx=i: self.p_o[idx].get_value(x) for i in range(self.n_orb ** 2)]
                idx_sigma = np.argmax([self.p_o[i].sigma   for i in range(self.n_orb ** 2)])
                self.S = self.p_o[idx_sigma].S
                self.sigma   = self.p_o[idx_sigma].sigma
                self.err_max = max([self.p_o[i].err_max for i in range(self.n_orb ** 2)])
                self.err_ave = max([self.p_o[i].err_ave for i in range(self.n_orb ** 2)])
                if n0 == "auto":
                    assert self.Lfactor != 0.5
                    assert isinstance(n0_shift, int) and n0_shift >= 0
                    p_o2 = [ESPRIT(G_w_vector[:, i], self.w[0], self.w[-1], err=self.err, err_type=self.err_type, Lfactor=0.5) for i in range(self.n_orb ** 2)]
                    w_cont = np.linspace(self.w[0], self.w[-1], 10 * self.w.size - 9)
                    G_L1 = [self.p_o[i].get_value(w_cont)[:-1].reshape(self.w.size - 1, 10) for i in range(self.n_orb ** 2)]
                    G_L2 = [    p_o2[i].get_value(w_cont)[:-1].reshape(self.w.size - 1, 10) for i in range(self.n_orb ** 2)]
                    ctrl_interval = [np.all(np.abs(G_L2[i] - G_L1[i]) <= self.err_max, axis=1) for i in range(self.n_orb ** 2)]
                    self.n0 = max([np.argmax(ctrl_interval[i]) for i in range(self.n_orb ** 2)]) + n0_shift #maybe change it later...
                else:
                    assert isinstance(n0, int) and n0 >= 0
                    self.n0 = n0
            #get the corresponding conformal mapping
            w_m = 0.5 * (self.w[self.n0] + self.w[-1])
            dw_h = 0.5 * (self.w[-1] - self.w[self.n0])
            self.con_map = ConMapGeneric(w_m, dw_h)
            #calculate contour integrals
            self.cal_hk_generic(self.G_approx, k_max)
        else:
            #use complex poles to approximate Matsubara data in [1j * w[0], +inf)
            p = MiniPoleMatrix(G_w, w, n0=n0, n0_shift=n0_shift, err=err, err_type=err_type, G_symmetric=G_symmetric, compute_const=compute_const, include_n0=False, elementwise=elementwise, Lfactor=Lfactor, k_max=k_max)
            self.S = p.S
            self.G_approx = [lambda x, Al=p.pole_weight.reshape(-1, self.n_orb ** 2)[:, i], xl=p.pole_location: self.cal_G_scalar(1j * x, Al, xl) for i in range(self.n_orb ** 2)]
            self.const = p.const
            self.sigma = p.sigma
            self.n0 = p.n0
            G_w_approx = self.cal_G_vector(1j * self.w[self.n0:], p.pole_weight.reshape(-1, self.n_orb ** 2), p.pole_location).reshape(-1, self.n_orb, self.n_orb)
            self.err_max = np.abs(G_w_approx + self.const - self.G_w[self.n0:]).max()
            self.err_ave = np.abs(G_w_approx + self.const - self.G_w[self.n0:]).mean()
            #get the corresponding conformal mapping
            self.con_map = ConMapGapless(self.w[self.n0])
            #calculate contour integrals
            self.cal_hk_gapless(self.G_approx, k_max)
        
        #apply the second ESPRIT approximation to recover poles
        self.find_poles()
        self.cut_pole(self.x_range[0], self.x_range[1], self.y_range[0], self.y_range[1])
    
    def cal_hk_generic(self, G_approx, k_max = 999):
        '''
        Calculate the contour integrals.
        '''
        cutoff = self.err_max
        err = 0.01 * cutoff
        
        self.h_k = np.zeros((k_max, len(G_approx)), dtype=np.complex_)
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
    
    def cal_hk_gapless(self, G_approx, k_max = 999):
        '''
        Calculate the contour integrals.
        '''
        cutoff = self.err_max
        err = 0.01 * cutoff
        
        self.h_k = np.zeros((k_max, len(G_approx)), dtype=np.float_)
        for k in range(self.h_k.shape[0]):
            for i in range(self.h_k.shape[1]):
                    self.h_k[k, i] = self.cal_hk_gapless_indiv(G_approx[i], k, err)
            if k >= 1:
                cutoff_matrix = np.logical_and(np.abs(self.h_k[k]) < cutoff, np.abs(self.h_k[k - 1]) < cutoff)
                if np.all(cutoff_matrix):
                    break
        self.h_k = self.h_k[:(k + 1)]
    
    def cal_hk_gapless_indiv(self, G_approx, k, err):
        theta0 = 1.e-6
        if k % 2 == 0:
            return (-2.0 / np.pi) * integrate.quad(lambda x: G_approx(self.con_map.w_min / np.sin(x)).imag, theta0, 0.5 * np.pi, weight="sin", wvar=k + 1, epsabs=err, epsrel=err, limit=10000)[0]
        else:
            return (+2.0 / np.pi) * integrate.quad(lambda x: G_approx(self.con_map.w_min / np.sin(x)).real, theta0, 0.5 * np.pi, weight="cos", wvar=k + 1, epsabs=err, epsrel=err, limit=10000)[0]
    
    def find_poles(self):
        '''
        Recover poles from contour integrals h_k.
        '''
        #apply the second ESPRIT
        if self.M is None:
            self.p_f = ESPRIT(self.h_k, err=self.err_max, Lfactor=self.Lfactor)
        else:
            self.p_f = ESPRIT(self.h_k, M=self.M, Lfactor=self.Lfactor)
        
        if self.symmetry and self.pole_real:
            self.p_f.gamma = self.p_f.gamma[np.abs(self.p_f.gamma.imag) < 1.e-3].real
            self.p_f.find_omega()
        
        #tranform poles from w-plane to z-plane
        location = self.con_map.z(self.p_f.gamma)
        weight = self.p_f.omega * self.con_map.dz(self.p_f.gamma).reshape(-1, 1)
        
        if self.symmetry is False:
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
            A = np.zeros((w.size, location.size), dtype=np.complex_)
            for i in range(location.size):
                A[:, i] = 1.0 / (1j * w - location[i])
            weight, residuals, rank, s = np.linalg.lstsq(A, (G_w - self.const).reshape(-1, self.n_orb ** 2), rcond=-1)
            self.lstsq_quality = (residuals, rank, s)
        
        #rearrange poles so that \xi_1.real <= \xi_2.real <= ... <= \xi_M.real
        idx = np.argsort(location.real)
        self.pole_weight   = weight[idx].reshape(-1, self.n_orb, self.n_orb)
        self.pole_location = location[idx]
    
    def cut_pole(self, x_min, x_max, y_min, y_max):
        '''
        Only keep poles located within (x_min, x_max) and (y_min, y_max).
        '''
        idx_x = np.logical_and(self.pole_location.real > x_min, self.pole_location.real < x_max)
        idx_y = np.logical_and(self.pole_location.imag > y_min, self.pole_location.imag < y_max)
        self.pole_weight   = self.pole_weight[np.logical_and(idx_x, idx_y)]
        self.pole_location = self.pole_location[np.logical_and(idx_x, idx_y)]
    
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
        import matplotlib.pyplot as plt
        #dynamically generate colors, line styles, and markers based on the number of curves
        num_curves = self.n_orb ** 2
        line_styles = ['-', '-.', ':', '--'] * (num_curves // 4 + 1)
        
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
        plt.semilogy([0, self.p_f.S.size - 1], [self.err_max, self.err_max], color="k", label="precision")
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
