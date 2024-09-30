import numpy as np
from scipy.special import ellipk

'''
Function tools for testing the performance of Prony Analytic Continuation.
'''
def semi_circle(w, w0 = 0.0, r = 1.0):
    '''
    A semicircle function with the center being w0 and the radius being r.
    '''
    assert r > 0.0
    if np.abs(w - w0) >= r:
        return 0.0
    else:
        return (2.0 / (np.pi * r**2.0)) * np.sqrt(r**2.0 - (w - w0)**2.0)

def rectangle(w, w0 = 0.0, dw_h = 0.5):
    '''
    A rectangle function with the center being w0 and the half-width being dw_h.
    '''
    assert dw_h > 0.0
    if np.abs(w - w0) >= dw_h:
        return 0.0
    else:
        return 1.0 / (2.0 * dw_h)

def triangle(w, w0 = 0.0, dw_h = 1.0):
    '''
    A triangle function with the center being w0 and the half-length of the bottom edge being dw_h.
    '''
    assert dw_h > 0.0
    if np.abs(w - w0) >= dw_h:
        return 0.0
    else:
        h = 1.0 / dw_h
        return h * (1.0 - np.abs(w - w0) / dw_h)

def gaussian(w, mu = 0.0, sigma = 1.0):
    '''
    A Gaussian function with the expected value mu and the standard deviation sigma.
    '''
    return 1.0 / (np.sqrt(2.0 * np.pi) * sigma) * np.exp(-0.5 * ((w - mu) / sigma)**2.0)

def lorentzian(w, w0 = 0.0, half_width = 1.0):
    '''
    A Lorentzian function with the center being w0 and the half-width at half-maximum being half-width.
    '''
    return 1.0 / np.pi * half_width / ((w - w0)**2.0 + half_width**2.0)

def bethe_lattice_dos_indiv(x, t = 1.0):
    '''
    Intermediate function of bethe_lattice_dos(w, t) which only works for a single point.
    '''
    if np.abs(x) > 2 * t:
        return 0.0
    else:
        return 1.0 / (2.0 * np.pi * t**2.0) * np.sqrt(4 * t**2.0 - x**2.0)

def bethe_lattice_dos(w, t = 1.0):
    '''
    Density of states for the tight-binding model in Bethe lattice with interaction t.
    '''
    return np.vectorize(lambda x: bethe_lattice_dos_indiv(x, t))(w)

def bethe_lattice_G(w_n, t = 1.0):
    '''
    Matsubara Green's function for the tight-binding model in Bethe lattice with interaction t. 
    w_n is the imaginary part of the Matsubara Green's function.
    '''
    assert np.linalg.norm(w_n.imag) == 0.0
    assert np.all(w_n > 0.0)    
    return 1.0j / (2.0 * t**2.0) * (w_n - np.sqrt(w_n**2.0 + 4.0 * t**2.0))

def square_lattice_dos_indiv(x, t = 1.0):
    '''
    Intermediate function of square_lattice_dos(w, t) which only works for a single point.
    '''
    if np.abs(x) > 4 * t:
        return 0.0
    else:
        return 1.0 / (2.0 * np.pi**2 * t) * ellipk(1 - (x / (4 * t))**2)

def square_lattice_dos(w, t = 1.0):
    '''
    Density of states for the tight-binding model in square lattice with nearest-neighbor interaction t.
    '''
    return np.vectorize(lambda x: square_lattice_dos_indiv(x, t))(w) 

def square_lattice_G(w_n, t = 1.0):
    '''
    Matsubara Green's function for the tight-binding model in square lattice with nearest-neighbor interaction t. 
    w_n is the imaginary part of the Matsubara Green's function.
    '''
    assert np.linalg.norm(w_n.imag) == 0.0
    return -1.0j / (2.0 * np.pi * t * np.sqrt(1 + (w_n / (4.0 * t))**2.0)) * ellipk(1.0 / (1.0 + (w_n / (4.0 * t))**2.0))

def triangle_lattice_dos_indiv(x, t = 1.0, tp = 0.5):
    '''
    Intermediate function of triangle_lattice_dos(w, t, tp) which only works for a single point.
    '''
    assert tp != 0.0
    u = t / tp
    E = x / t
    assert u >= 0.0
    
    w_min = -4.0 - 2.0 / u
    w_max = (u + 2.0 / u if u <= 2.0 else 4.0 - 2.0 / u)
    
    if E < w_min or E > w_max:
        return 0.0
    
    r = u * np.sqrt(u**2.0 - E * u + 2.0)
    p = 4.0 * r
    q = (r - u**2.0)**2.0 * (r**2.0 - 4.0 * u**2.0 + 2.0 * r * u**2.0 + u ** 4.0) / (4.0 * u**4.0)
    
    if q < 0.0:
        z0 = p - q
        z1 = p
    elif q < p:
        z0 = p
        z1 = p - q
    else:
        z0 = q
        z1 = q - p
    
    return 1.0 / (np.pi**2.0 * tp * np.sqrt(z0)) * ellipk(z1 / z0)

def triangle_lattice_dos(w, t = 1.0, tp = 0.5):
    '''
    Density of states for the tight-binding model in triangular lattice with anisotropic nearest-neighbor interactions t and tp.
    '''
    return np.vectorize(lambda x: triangle_lattice_dos_indiv(x, t, tp))(w)
