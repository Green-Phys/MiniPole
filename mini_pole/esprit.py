import numpy as np
from kneed import KneeLocator
import warnings

class ESPRIT:
    '''
    Matrix version of the ESPRIT method for approximating functions with complex exponentials.
    '''
    def __init__(self, h_k, x_min = 0, x_max = 1, err = None, err_type = "abs", M = None, Lfactor = 0.4, tol = 1.e-15, ctrl_ratio = 10):
        '''
        Initialize with function values sampled on a uniform grid from x_min to x_max.
        '''
        h_k = h_k.reshape(h_k.shape[0], -1)
        self.N = h_k.shape[0]
        self.dim = h_k.shape[1]
        if Lfactor < 1.0 / 3.0 or Lfactor > 0.5:
            warnings.warn("It is suggested to set 1 / 3 <= Lfactor <= 1 / 2.")
        self.L = int(Lfactor * (self.N - 1))
        assert (self.N - self.L) >= (self.L + 1)
        assert x_min < x_max
        assert err_type in ["abs", "rel"]
        
        if np.max(np.abs(h_k.imag)) < tol:
            self.type = "real"
            self.h_k = np.array(h_k.real)
        elif np.max(np.abs(h_k.real)) < tol:
            self.type = "imag"
            self.h_k = np.array(1j * h_k.imag)
        else:
            self.type = "cplx"
            self.h_k = np.array(h_k)
        self.x_min = x_min
        self.x_max = x_max
        self.x_k = np.linspace(self.x_min, self.x_max, self.N)
        self.err = err
        self.err_type = err_type
        self.M = M
        self.tol = tol
        
        #note to set data type to be complex even if the input is real! Otherwise the result might be unstable!
        self.H = np.zeros((self.dim * (self.N - self.L), self.L + 1), dtype=np.complex128)
        for l in range(self.N - self.L):
            self.H[(self.dim * l):(self.dim * (l + 1)), :] = self.h_k[l:(l + self.L + 1)].T
        
        #for some specific versions of numpy, there is a very low chance that SVD does not converge
        while True:
            try:
                _, self.S, self.W = np.linalg.svd(self.H, full_matrices=False)
                break
            except:
                #reconstruct the Hankel matrix
                self.L -= 1
                self.H = np.zeros((self.dim * (self.N - self.L), self.L + 1), dtype=np.complex128)
                for l in range(self.N - self.L):
                    self.H[(self.dim * l):(self.dim * (l + 1)), :] = self.h_k[l:(l + self.L + 1)].T
        
        if self.M is None:
            self.find_M_with_err() if self.err is not None else self.find_M_with_exp_decay()
            if self.S[self.M] / self.S[0] < 1.e-14:
                self.err = 1.e-14
                self.err_type = "rel"
                self.find_M_with_err()
        else:
            self.M = min(self.M, self.S.size - 1)
        while True:
            self.sigma = self.S[self.M]
            self.W_0 = self.W[:self.M, :-1]
            self.W_1 = self.W[:self.M, 1:]
            self.F_M = np.linalg.pinv(self.W_0.T) @ self.W_1.T
            
            self.gamma = np.linalg.eigvals(self.F_M)
            self.find_omega()
            self.cal_err()
            
            if self.err_max < max(ctrl_ratio * self.sigma, 1.e-14 * self.S[0]):
                break
            else:
                self.M -= 1
            if self.M == 0:
                raise Exception("Could not find controlled approximation!")
    
    def find_M_with_err(self):
        '''
        Find the rank M for the given error tolerance.
        '''
        cutoff = self.err if self.err_type == "abs" else self.S[0] * self.err
        for idx in range(self.S.size):
            if self.S[idx] < cutoff:
                break
        if self.S[idx] >= cutoff:
            print("err is set to be too small!")
        self.M = idx
    
    def find_M_with_exp_decay(self):
        '''
        Find the maximum index for the exponentially decaying region.
        '''
        kneedle = KneeLocator(np.arange(self.S.size), np.log(self.S), S=1, curve='convex', direction='decreasing')
        self.dlogS = np.abs(np.diff(np.log(self.S[:(kneedle.knee + 1)]), n=1))
        self.M = np.where(self.dlogS > self.dlogS.max() / 3)[0][-1] + 1
    
    def find_omega(self):
        '''
        Find weights of corresponding nodes gamma.
        '''
        V = np.zeros((self.h_k.shape[0], self.M), dtype=np.complex128)
        for i in range(V.shape[0]):
            V[i, :] = self.gamma ** i
        #using least-squares solution is more stable than using pseudo-inverse
        #setting rcond=None (default) sometimes leads to incorrect result for high-precision input
        self.omega, residuals, rank, s = np.linalg.lstsq(V, self.h_k, rcond=-1)
        self.lstsq_quality = (residuals, rank, s)
    
    def cal_err(self):
        h_k_approx = self.get_value(self.x_k)
        self.err_max = np.abs(h_k_approx - self.h_k).max(axis=0).max()
        self.err_ave = np.abs(h_k_approx - self.h_k).mean(axis=0).max()
    
    def get_value_indiv(self, x, col):
        '''
        Get the approximated function value at point x for column col.
        '''
        assert col >= 0 and col < self.dim
        x0 = (x - self.x_min) / (self.x_max - self.x_min)
        if np.any(x0 < -1.e-12) or np.any(x0 > 1.0 + 1.e-12):
            warnings.warn("This approximation only has error control for x in [x_min, x_max]!")
        
        if np.isscalar(x0):
            V = self.gamma ** ((self.h_k.shape[0] - 1) * x0)
            value = np.dot(V, self.omega[:, col])
        else:
            V = np.zeros((x0.size, self.gamma.size), dtype=np.complex128)
            for i in range(V.shape[0]):
                V[i, :] = self.gamma ** ((self.h_k.shape[0] - 1) * x0[i])
            value = np.dot(V, self.omega[:, col])
        return value if self.type == "cplx" else value.real if self.type == "real" else 1j * value.imag
    
    def get_value(self, x):
        '''
        Get the approximated function value at point x.
        '''
        x0 = (x - self.x_min) / (self.x_max - self.x_min)
        if np.any(x0 < -1.e-12) or np.any(x0 > 1.0 + 1.e-12):
            warnings.warn("This approximation only has error control for x in [x_min, x_max]!")
        
        if np.isscalar(x0):
            V = self.gamma ** ((self.h_k.shape[0] - 1) * x0)
            value = np.dot(V, self.omega)
        else:
            V = np.zeros((x0.size, self.gamma.size), dtype=np.complex128)
            for i in range(V.shape[0]):
                V[i, :] = self.gamma ** ((self.h_k.shape[0] - 1) * x0[i])
            value = np.dot(V, self.omega)
        return value if self.type == "cplx" else value.real if self.type == "real" else 1j * value.imag
