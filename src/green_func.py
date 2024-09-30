import numpy as np
import scipy.integrate as integrate

class GreenFunc:
    '''
    Green's function class which works for both discrete and continuous cases.
    '''
    def __init__(self, statistics, beta, flag = "discrete", A_i = None, x_i = None, A_x = None, x_min = -10, x_max = 10, tol = 0.01):
        '''
        Meaning of each parameter:
        1. statistics stands for the statistics of the system, which is either "F" (for fermions) or "B" (for bosons);
        2. beta is the inverse temperature in Hartree atomic units;
        3. flag is "discrete" for discrete cases (with delta peaks) or "continuous" for continuous cases (with broadened peaks);
        4. For discrete cases, an array of pole weights A_i and pole locations x_i should be provided,
           while for continuous cases, the function form A_x should be provided, along with non-zero range. 
           If A_x is non-zero everywhere, x_min should be -np.inf and x_max should be np.inf.
        5. tol is the value to distinguish delta peaks and broadened peaks for discrete cases, according to the imaginary part of pole locations.
        '''
        assert statistics.upper() in ["F", "B"]
        assert flag.lower() in ["discrete", "continuous"]
        
        self.statistics = statistics.upper()
        self.beta = beta
        self.flag = flag.lower()
        if self.flag == "discrete":
            assert A_i is not None and x_i is not None
            assert A_i.size == x_i.size
            #pole locations and weights can both be complex-valued
            self.A_i = A_i
            if np.max(np.abs(x_i.imag)) < 1.e-15:
                self.x_i = x_i.real
            else:
                self.x_i = x_i
            self.tol = tol
        else:
            #A_x(x) is a continuous function of x
            assert A_x is not None
            self.A_x = A_x
            self.x_min = x_min
            self.x_max = x_max
    
    def get_matsubara(self, N_up = 1000, lower = False):
        '''
        Calculate the Matsubara data. 
        N_up is the number of non-negative frequencies. 
        lower controls whether data on negative frequencies is calculated or not.
        '''
        if self.statistics == "F":
            if lower is False:
                n_sample = np.arange(N_up)
            else:
                n_sample = np.arange(-N_up, N_up)
            self.w = (2 * n_sample + 1) * np.pi / self.beta
        else:
            if lower is False:
                n_sample = np.arange(N_up)
            else:
                n_sample = np.arange(-(N_up - 1), N_up)
            self.w = 2 * n_sample * np.pi / self.beta
        
        self.N_tot = self.w.size
        self.w_min = self.w[0]
        self.w_max = self.w[-1]
        self.G_w   = self.get_G(1j * self.w)
    
    def cal_G(self, z, err=1.e-13):
        '''
        Calculate G(z) at point z. err is the accuracy of the integration we want to achieve.
        If err is chosen to be too small, there will be some warnings.
        '''
        if self.flag == "discrete":
            G_z = 0.0
            for i in range(self.x_i.size):
                G_z += self.A_i[i] / (z - self.x_i[i])
        else:
            G_z = integrate.quad(lambda x: self.A_x(x) * (1.0 / (z - x)).real, self.x_min, self.x_max, epsabs=err, epsrel=err, limit=10000)[0] \
           + 1j * integrate.quad(lambda x: self.A_x(x) * (1.0 / (z - x)).imag, self.x_min, self.x_max, epsabs=err, epsrel=err, limit=10000)[0]

        return G_z
    
    def get_G(self, z, err=1.e-13):
        '''
        Vectorized version of G(z) which can deal with an array.
        '''
        return np.vectorize(lambda x: self.cal_G(x, err))(z)

    def get_spectral(self, x, epsilon = 0.01):
        '''
        epsilon is the broadening parameter for delta peaks.
        '''
        if self.flag == "discrete":
            A_w = 0.0
            for i in range(self.x_i.size):
                if np.abs(self.x_i[i].imag) < self.tol:
                    A_w += (1.0 / np.pi) * self.A_i[i].real * np.abs(epsilon) / ((x - self.x_i[i].real)**2.0 + epsilon**2.0)
                else:
                    A_w += -1.0 / np.pi * (self.A_i[i] / (x - self.x_i[i])).imag
        else:
            A_w = np.vectorize(self.A_x)(x)
        
        return A_w
