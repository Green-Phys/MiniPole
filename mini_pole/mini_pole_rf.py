import numpy as np
import scipy.integrate as integrate
from .esprit import ESPRIT
from .con_map import ConMapRfFnt, ConMapRfInf

class MiniPoleRf:
    def __init__(self, G_rf, func_type = "real", interval_type = "infinite", w_min = -10, w_max = 10, wp_max = 1, sing_vals = None, err = None, M = None, compute_const = False, k_max = 999, Lfactor = 0.4):
        '''
        G_rf (list): A list of length n_orb² containing analytic expressions of the real-frequency Green's functions.
        func_type (str): Specifies the type of functions in G_rf; either "real" for real-valued or "complex" for complex-valued.
        interval_type (str): Specifies the type of real-frequency interval; either "infinite" or "finite".
        w_min (float): Lower bound of the finite real-frequency interval.
        w_max (float): Upper bound of the finite real-frequency interval.
        wp_max (float): Parameter used in the Möbius transform for the infinite real-frequency interval.
        sing_vals (list): List of singular values of G_rf.
        err (float): Error tolerance used during the approximation process.
        M (int): Number of poles in the final result.
        compute_constant (bool): Whether to compute the constant term in the approximation.
        k_max (int): Maximum number of contour integrals.
        Lfactor (float): Ratio L / N used in ESPRIT.
        '''
        self.n_orb = round(np.sqrt(len(G_rf)))
        assert len(G_rf) == self.n_orb ** 2
        assert func_type in ["real", "complex"] #currently works for real-valued functions and retarded functions (not for generic complex-valued functions)
        assert interval_type in ["finite", "infinite"]
        if interval_type == "finite":
            assert w_min < w_max and w_min > -np.inf and w_max < np.inf
            self.w_min = w_min
            self.w_max = w_max
        else:
            assert wp_max > 0
            self.wp_max = wp_max
        assert err is not None
        
        self.G_rf = G_rf
        self.func_type = func_type
        self.interval_type = interval_type
        self.sing_vals = sing_vals
        self.err = err
        self.M = M
        self.compute_const = compute_const
        self.k_max = k_max
        self.Lfactor = Lfactor
        
        if self.interval_type == "finite":
            #construct the conformal mapping
            w_m  = 0.5 * (w_max + w_min)
            dw_h = 0.5 * (w_max - w_min)
            self.con_map = ConMapRfFnt(w_m, dw_h)
            #calculate contour integrals
            self.cal_hk_fnt(self.G_rf, self.k_max, self.sing_vals)
        else:
            #construct the conformal mapping
            self.con_map = ConMapRfInf(wp_max)
            #calculate contour integrals
            self.cal_hk_inf(self.G_rf, self.k_max, self.sing_vals)
        
        #apply the Prony's approximation to recover poles
        self.find_poles()
        
        #calculate the constant term
        self.cal_const()
    
    def cal_hk_fnt(self, G_approx, k_max = 999, sing_vals = None):
        '''
        Calculate the contour integrals.
        '''
        cutoff = self.err
        tol = 0.1 * cutoff
        
        if sing_vals is None:
            iter_theta = [[0, np.pi]]
        else:
            sing_theta = np.concatenate(([0], np.sort(np.angle(self.con_map.w(sing_vals))), [np.pi]))
            iter_theta = [[sing_theta[i], sing_theta[i+1]] for i in range(len(sing_theta) - 1)]
        
        self.h_k = np.zeros((k_max, len(G_approx)), dtype=np.complex_)
        for k in range(self.h_k.shape[0]):
            for i in range(self.h_k.shape[1]):
                for theta_min, theta_max in iter_theta:
                    self.h_k[k, i] += self.cal_hk_fnt_indiv(G_approx[i], k, tol, theta_min, theta_max)
            if k >= 1:
                cutoff_matrix = np.logical_and(np.abs(self.h_k[k]) < cutoff, np.abs(self.h_k[k - 1]) < cutoff)
                if np.all(cutoff_matrix):
                    break
        self.h_k = self.h_k[:(k + 1)]
    
    def cal_hk_fnt_indiv(self, G_approx, k, tol, theta_min, theta_max):
        return (1.0 / np.pi) * integrate.quad(lambda x: G_approx(self.con_map.w_m.real + self.con_map.dw_h * np.cos(x)), theta_min, theta_max, weight="cos", wvar=k + 1, complex_func=(self.func_type=="complex"), epsabs=tol, epsrel=tol, limit=10000)[0]
    
    def cal_hk_inf(self, G_approx, k_max = 999, sing_vals = None):
        '''
        Calculate the contour integrals.
        '''
        cutoff = self.err
        tol = 0.1 * cutoff
        
        if sing_vals is None:
            #maybe to be updated later
            iter_theta = [[1.e-12, 2.0 * np.pi - 1.e-12]]
        else:
            sing_theta = np.concatenate(([1.e-12], np.sort(np.angle(self.con_map.w(sing_vals))), [2.0 * np.pi - 1.e-12]))
            iter_theta = [[sing_theta[i], sing_theta[i+1]] for i in range(len(sing_theta) - 1)]
        
        self.h_k = np.zeros((k_max, len(G_approx)), dtype=np.complex_)
        for k in range(self.h_k.shape[0]):
            for i in range(self.h_k.shape[1]):
                for theta_min, theta_max in iter_theta:
                    self.h_k[k, i] += self.cal_hk_inf_indiv(G_approx[i], k, tol, theta_min, theta_max)
            if k >= 1:
                cutoff_matrix = np.logical_and(np.abs(self.h_k[k]) < cutoff, np.abs(self.h_k[k - 1]) < cutoff)
                if np.all(cutoff_matrix):
                    break
        self.h_k = self.h_k[:(k + 1)]
    
    def cal_hk_inf_indiv(self, G_approx, k, tol, theta_min, theta_max):
        return (0.5  / np.pi) * integrate.quad(lambda x: G_approx(self.con_map.w_max / np.tan(0.5 * x)), theta_min, theta_max, weight="cos", wvar=k + 1, complex_func=(self.func_type=="complex"), epsabs=tol, epsrel=tol, limit=10000)[0] + \
               (0.5j / np.pi) * integrate.quad(lambda x: G_approx(self.con_map.w_max / np.tan(0.5 * x)), theta_min, theta_max, weight="sin", wvar=k + 1, complex_func=(self.func_type=="complex"), epsabs=tol, epsrel=tol, limit=10000)[0]
    
    def find_poles(self):
        '''
        Recover poles from contour integrals h_k.
        '''
        #apply ESPRIT
        if self.M is None:
            self.p_f = ESPRIT(self.h_k, err=self.err, Lfactor=self.Lfactor)
        else:
            self.p_f = ESPRIT(self.h_k, M=self.M, Lfactor=self.Lfactor)
        
        #make sure all mapped poles are inside the unit disk
        idx0 = np.abs(self.p_f.gamma) < 1.0
        #tranform poles from w-plane to z-plane
        location = self.con_map.z(self.p_f.gamma[idx0])
        weight   = self.p_f.omega[idx0] * self.con_map.dz(self.p_f.gamma[idx0]).reshape(-1, 1)
        
        #discard poles with negligible weights
        idx1 = np.abs(weight).max(axis=1) > self.err
        weight   = weight[idx1]
        location = location[idx1]
        
        #rearrange poles so that \xi_1.real <= \xi_2.real <= ... <= \xi_M.real
        idx2 = np.argsort(location.real)
        self.pole_weight   = weight[idx2].reshape(-1, self.n_orb, self.n_orb)
        self.pole_location = location[idx2]
    
    def cal_const(self):
        #calculate the constant term
        if self.compute_const is False:
            self.const = 0.0
        else:
            if self.interval_type == "finite":
                w = np.linspace(self.w_min, self.w_max, 1000000)
            else:
                w = np.linspace(-3 * self.wp_max, 3 * self.wp_max, 1000000)
            G_w_input  = np.array([self.G_rf[i](w) for i in range(self.n_orb ** 2)]).transpose()
            G_w_approx = self.cal_G_vector(w, self.pole_weight.reshape(-1, self.n_orb ** 2), self.pole_location)
            if self.func_type == "real":
                if self.interval_type == "finite":
                    self.const = (G_w_input - G_w_approx.real).mean(axis=0).reshape(-1, self.n_orb, self.n_orb)
                else:
                    self.const = (G_w_input - 2.0 * G_w_approx.real).mean(axis=0).reshape(-1, self.n_orb, self.n_orb)
            else:
                self.const = (G_w_input - G_w_approx).mean(axis=0).reshape(-1, self.n_orb, self.n_orb)
    
    def change_M(self, M):
        self.M = M
        self.find_poles()
    
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
    
    def plot_spectrum(self, orb_list = None, w_min = None, w_max = None):
        import matplotlib.pyplot as plt
        
        if orb_list is None:
            orb_list = [(i, j) for i in range(self.n_orb) for j in range(self.n_orb)]
        if w_min is None or w_max is None:
            if self.interval_type == "finite":
                w = np.linspace(self.w_min, self.w_max, 2000)
            else:
                w = np.linspace(-3 * self.wp_max, 3 * self.wp_max, 2000)
        else:
            w = np.linspace(w_min, w_max, 2000)
        #dynamically generate colors, line styles, and markers based on the number of curves
        num_curves = len(orb_list)
        line_styles = ['-', '-.', ':', '--'] * (num_curves // 4 + 1)
        plt.figure()
        for idx, orb in enumerate(orb_list):
            i, j = orb
            A_r = self.cal_G_scalar(w, self.pole_weight[:, i, j], self.pole_location).real
            plt.plot(w, A_r, linestyle=line_styles[idx], label="element (" + str(i) + ", " + str(j) + ")")
        if self.h_k.shape[1] <= 16:
            plt.legend()
        plt.xlabel(r"$\omega$")
        plt.ylabel(r"Re[$G(\omega)$]")
        plt.show()
        
        plt.figure()
        for idx, orb in enumerate(orb_list):
            i, j = orb
            A_r = self.cal_G_scalar(w, self.pole_weight[:, i, j], self.pole_location).imag
            plt.plot(w, A_r, linestyle=line_styles[idx], label="element (" + str(i) + ", " + str(j) + ")")
        if self.h_k.shape[1] <= 16:
            plt.legend()
        plt.xlabel(r"$\omega$")
        plt.ylabel(r"Im[$G(\omega)$]")
        plt.show()
    
    def check_valid(self):
        import matplotlib.pyplot as plt
        #dynamically generate colors, line styles, and markers based on the number of curves
        num_curves = self.n_orb ** 2
        line_styles = ['-', '-.', ':', '--'] * (num_curves // 4 + 1)
        
        #check h_k
        #part 1
        plt.figure()
        for i in range(self.n_orb ** 2):
            row, col = i // self.n_orb, i % self.n_orb
            plt.semilogy(np.abs(self.h_k[:, i]), '.', label="element (" + str(row) + ", " + str(col) + ")")
        plt.semilogy([0, self.h_k.shape[0] - 1], [self.err, self.err], color="k", label="precision")
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
        plt.semilogy([0, self.p_f.S.size - 1], [self.err, self.err], color="k", label="precision")
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
            plt.semilogy([0, self.h_k.shape[0] - 1], [self.err, self.err], color="k", label="precision")
        if self.h_k.shape[1] <= 16:
            plt.legend()
        plt.xlabel(r"$k$")
        plt.ylabel(r"$|\hat{h}_k - h_k|$")
        plt.title("Contour integrals: approximation")
        plt.show()
        
        #check the final approximation
        plt.figure()
        if self.interval_type == "finite":
            w = np.linspace(self.w_min, self.w_max, 2000)
        else:
            w = np.linspace(-3 * self.wp_max, 3 * self.wp_max, 2000)
        G_w_input  = np.array([self.G_rf[i](w) for i in range(self.n_orb ** 2)]).transpose().reshape(-1, self.n_orb, self.n_orb)
        if self.func_type == "real":
            if self.interval_type == "finite":
                G_w_approx = self.cal_G_vector(w, self.pole_weight.reshape(-1, self.n_orb ** 2), self.pole_location).reshape(-1, self.n_orb, self.n_orb).real + self.const
            else:
                G_w_approx = 2.0 * self.cal_G_vector(w, self.pole_weight.reshape(-1, self.n_orb ** 2), self.pole_location).reshape(-1, self.n_orb, self.n_orb).real + self.const
        else:
            G_w_approx = self.cal_G_vector(w, self.pole_weight.reshape(-1, self.n_orb ** 2), self.pole_location).reshape(-1, self.n_orb, self.n_orb) + self.const
        for i in range(self.n_orb ** 2):
            row, col = i // self.n_orb, i % self.n_orb
            plt.semilogy(w, np.abs(G_w_approx[:, row, col] - G_w_input[:, row, col]), linestyle=line_styles[i], label="element (" + str(row) + ", " + str(col) + ")")
        if self.M is not None:
            plt.semilogy([w[0], w[-1]], [self.p_f.S[self.M], self.p_f.S[self.M]], color="gray", linestyle="--", label="M poles")
        else:
            plt.semilogy([w[0], w[-1]], [self.err, self.err], color="k", label="precision")
        if self.h_k.shape[1] <= 16:
            plt.legend()
        plt.xlabel(r"$\omega$")
        plt.ylabel(r"$|\hat{G}(\omega) - G(\omega)|$")
        plt.title("Final approximation")
        plt.show()
        
        from matplotlib.colors import LinearSegmentedColormap
        colors = [(1, 1, 1), (0, 0, 1)] #(R, G, B) tuples for white and blue
        n_bins = 100 #Discretize the interpolation into bins
        cmap_name = "WtBu"
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins) #Create the colormap
        
        #check pole locations
        pts = self.pole_location
        scatter = plt.scatter(pts.real, pts.imag, c=np.linalg.norm(self.pole_weight.reshape(-1, self.n_orb ** 2), axis=1), vmin=0, cmap=cmap)
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
        scatter = plt.scatter(pts.real, pts.imag, c=np.linalg.norm(self.pole_weight.reshape(-1, self.n_orb ** 2), axis=1), vmin=0, cmap=cmap)
        cbar = plt.colorbar(scatter)
        cbar.set_label('weight')
        plt.xlim([-1.05, 1.05])
        plt.ylim([-1.05, 1.05])
        plt.xlabel(r"Real($w$)")
        plt.ylabel(r"Imag($w$)")
        plt.show()
