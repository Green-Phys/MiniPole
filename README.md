[![Downloads](https://pepy.tech/badge/mini-pole)](https://pepy.tech/project/mini-pole)
[![GitHub license](https://img.shields.io/github/license/Green-Phys/MiniPole?cacheSeconds=3600&color=informational&label=License)](./LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15121302.svg)](https://zenodo.org/doi/10.5281/zenodo.15121302)
[![PyPI version](https://img.shields.io/pypi/v/mini-pole.svg?logo=python&logoColor=white)](https://pypi.org/project/mini-pole/)

# Minimal Pole Method (MPM)

This repository provides a Python implementation of the **matrix-valued Minimal Pole Method (MPM)** for both Matsubara and real-frequency data.

## 🔬 For the Analytic Continuation Community

The method is described in [Phys. Rev. B 110, 235131 (2024)](https://doi.org/10.1103/PhysRevB.110.235131), which extends the scalar-valued approach introduced in [Phys. Rev. B 110, 035154 (2024)](https://doi.org/10.1103/PhysRevB.110.035154).

The input Matsubara data is $G(i\omega_n)$, sampled on a *non-negative* uniform grid $\lbrace i\omega_0, i\omega_1, \cdots, i\omega_{n_\omega - 1} \rbrace$, where  
- $\omega_n = \frac{(2n+1)\pi}{\beta}$ for fermions
- $\omega_n = \frac{2n\pi}{\beta}$ for bosons 
- $n_\omega$ is the total number of sampling points

**Relevant classes**: `MiniPole`, `MiniPoleDLR`

## 🌱 For the HEOM Community

For applications involving real-frequency data used in Hierarchical Equations of Motion (HEOM), further details are provided in [J. Chem. Phys. 162, 214111 (2025)](https://doi.org/10.1063/5.0273763).

**Relevant classes**: `MiniPoleRf`, `MiniPoleRfDPR`

## 1. Installation

### Dependencies
- `numpy`
- `scipy`
- `matplotlib`
- `kneed`

### Installation Commands
1. Install the latest (unreleased) version from source
   ```bash
   python3 setup.py install

2. Install the latest released version via pip
   ```bash
   pip install mini_pole

## 2. Usage
### i) The standard MPM is performed using the following command:

**p = MiniPole(G_w, w, n0 = "auto", n0_shift = 0, err = None, err_type = "abs", M = None, symmetry = False, G_symmetric = False, compute_const = False, plane = None, include_n0 = True, k_max = 999, ratio_max = 10)**
        
    Parameters
    ----------
    1. G_w : ndarray
        An (n_w, n_orb, n_orb) or (n_w,) array containing the Matsubara data.
    2. w : ndarray
        An (n_w,) array containing the corresponding real-valued Matsubara grid.
    3. n0 : int or str, default="auto"
        If "auto", n0 is automatically selected with an additional shift specified by n0_shift.
        If a non-negative integer is provided, n0 is fixed at that value.
    4. n0_shift : int, default=0
        The shift applied to the automatically determined n0.
    5. err : float
        Error tolerance for calculations.
    6. err_type : str, default="abs"
        Specifies the type of error: "abs" for absolute error or "rel" for relative error.
    7. M : int, optional
        The number of poles in the final result. If not specified, the precision from the first ESPRIT is used to extract poles in the second ESPRIT.
    8. symmetry : bool, default=False
        Determines whether to preserve up-down symmetry.
    9. G_symmetric : bool, default=False
        If True, the Matsubara data will be symmetrized such that G_{ij}(z) = G_{ji}(z).
    10. compute_const : bool, default=False
        Determines whether to compute the constant term in G(z) = sum_l Al / (z - xl) + const.
        If False, the constant term is fixed at 0.
    11. plane : str, optional
        Specifies whether to use the original z-plane or the mapped w-plane to compute pole weights.
    12. include_n0 : bool, default=False
        Determines whether to include the first n0 input points when weights are calculated in the z-plane.
    13. k_max : int, default=999
        The maximum number of contour integrals.
    14. ratio_max : float, default=10
        The maximum ratio of oscillation when automatically choosing n0.
    
    Returns
    -------
    Minimal pole representation of the given data.
    Pole weights are stored in `p.pole_weight`, a numpy array of shape (M, n_orb, n_orb).
    Shared pole locations are stored in `p.pole_location`, a numpy array of shape (M,).

### ii) The MPM-DLR algorithm is performed using the following command:

**p = MiniPoleDLR(Al_dlr, xl_dlr, beta, n0, nmax = None, err = None, err_type = "abs", M = None, symmetry = False, k_max=200, Lfactor = 0.4)**

    Parameters
    ----------
    1. Al_dlr (numpy.ndarray): DLR coefficients, either of shape (r,) or (r, n_orb, n_orb).
    2. xl_dlr (numpy.ndarray): DLR grid for the real frequency, an array of shape (r,).
    3. beta (float): Inverse temperature of the system (1/kT).
    4. n0 (int): Number of initial points to discard, typically in the range (0, 10).
    5. nmax (int): Cutoff for the Matsubara frequency when symmetry is False.
    6. err (float): Error tolerance for calculations.
    7. err_type (str): Specifies the type of error, "abs" for absolute error or "rel" for relative error.
    8. M (int): Specifies the number of poles to be recovered.
    9. symmetry (bool): Whether to impose up-down symmetry (True or False).
    10. k_max (int): Number of moments to be calculated.
    11. Lfactor (float): Ratio of L/N in the ESPRIT algorithm.
    
    Returns
    -------
    Minimal pole representation of the given data.
    Pole weights are stored in `p.pole_weight`, a numpy array of shape (M, n_orb, n_orb).
    Shared pole locations are stored in `p.pole_location`, a numpy array of shape (M,).

### iii) The standard MPM for real-frequency fitting is performed using the following command:

**p = MiniPoleRf(G_rf, func_type = "real", interval_type = "infinite", w_min = -10, w_max = 10, wp_max = 1, sing_vals = None, err = None, M = None, compute_const = False, k_max = 999, Lfactor = 0.4)**

    Parameters
    ----------
    1. G_rf : list
       A list of length n_orb² containing analytic expressions of the real-frequency Green's functions.
    2. func_type : str
        Specifies the type of functions in G_rf; either "real" for real-valued or "complex" for complex-valued.
    3. interval_type : str
        Specifies the type of real-frequency interval; either "infinite" or "finite".
    4. w_min : float
        Lower bound of the finite real-frequency interval.
    5. w_max : float
        Upper bound of the finite real-frequency interval.
    6. wp_max : float
        Parameter used in the Möbius transform for the infinite real-frequency interval.
    7. sing_vals : list
        List of singular values of G_rf.
    8. err : float
        Error tolerance used during the approximation process.
    9. M : int
        Number of poles in the final result.
    10. compute_constant : bool
        Whether to compute the constant term in the approximation.
    11. k_max : int
        Maximum number of contour integrals.
    12. Lfactor : float
        Ratio L / N used in ESPRIT.
    
    Returns
    -------
    Minimal pole representation of the real-frequency Green's functions.
    Pole weights are stored in `p.pole_weight`, a numpy array of shape (M, n_orb, n_orb).  
    Shared pole locations are stored in `p.pole_location`, a numpy array of shape (M,).

### iv) The MPM algorithm for real-frequency fitting using a discrete pole representation (e.g., from AAA results) can be executed with the following command:

**p = MiniPoleRfDPR(Al_dpr, xl_dpr, interval_type = "infinite", w_min = -10, w_max = 10, wp_max = 1, err = None, err_type = "abs", cutoff_err = None, cutoff_err_type = "abs", M = None, k_max = 999, Lfactor = 0.4, alpha = 1.0, minimal_k = False)**

    Parameters
    ----------
    1. Al_dpr : numpy.ndarray
        Complex pole weights, either of shape (r,) or (r, n_orb, n_orb).
    2. xl_dpr : numpy.ndarray
        Complex pole locations, an array of shape (r,).
    3. interval_type : str
        Specifies the type of real-frequency interval; either "infinite" or "finite".
    4. w_min : float
        Lower bound of the finite real-frequency interval.
    5. w_max : float
        Upper bound of the finite real-frequency interval.
    6. wp_max : float
        Parameter used in the Möbius transform for the infinite real-frequency interval.
    7. err : float
        Error tolerance used during the approximation process.
    8. err_type : str
        Type of error to use; either "abs" for absolute error or "rel" for relative error.
    9. cutoff_err : float
        Cutoff value for h_k.
    10. cutoff_err_type : str
        Specifies whether the cutoff is based on absolute ("abs") or relative ("rel") error.
    11. M : int
        Number of poles in the final result.
    12. k_max : int
        Maximum number of contour integrals.
    13. Lfactor : float
        Ratio L / N used in ESPRIT.
    14. alpha : float
        Scaling parameter inside the unit disk to accelerate convergence.
    15. minimal_k : bool
        Whether to use a minimal number of h_k based on the size of `xl_dpr`.
    
    Returns
    -------
    Minimal pole representation of the given data.
    Pole weights are stored in `p.pole_weight`, a numpy array of shape (M, n_orb, n_orb).  
    Shared pole locations are stored in `p.pole_location`, a numpy array of shape (M,).

## 3. Examples

The scripts in the *examples* folder demonstrate the usage of MPM, MPM-DLR and MPM-RF.

### i) MPM Algorithm

The *examples/MPM* folder includes a Jupyter notebook that demonstrates how to use `MiniPole` to recover synthetic spectral functions. You can modify the lambda expression in the `GreenFunc` class to recover a different spectrum, but please remember to update the lower and upper bounds (x_min and x_max) of the spectrum accordingly. Additional details will be provided in the future.

### ii) MPM-DLR Algorithm

The *examples/MPM_DLR* folder contains scripts to recover the band structure of Si, as shown in the middle panel of Fig. 9 in [Phys. Rev. B 110, 235131 (2024)](https://doi.org/10.1103/PhysRevB.110.235131).

#### Steps:

a) Download the input data file [Si_dlr.h5](https://drive.google.com/file/d/1_bNvbgOHewiujHYEcf-CCpGxlZP9cRw_/view?usp=drive_link) to the *examples/MPM_DLR/* directory.

b) Obtain the recovered poles by running **python3 cal_band_dlr.py --obs=`<option>`**, where **`<option>`** can be "S" (self-energy), "Gii" (scalar-valued Green's function), or "G" (matrix-valued Green's function).

c) Plot the band structure by running **python3 plt_band_dlr.py --obs=`<option>`**.

#### Note:

a) Reference runtime on a single core of a laptop (using the M1 Max Apple chip as an example): 13 seconds for "Gii" and 160 seconds for both "G" and "S".

b) Parallel computation is supported in **cal_band_dlr.py** to speed up the process on multiple cores. Use the following command: **mpirun -n `<num_cores>` python3 cal_band_dlr.py --obs=`<option>`**, where **`<num_cores>`** is the number of cores and **`<option>`** is "S," "Gii," or "G".

c) Full Parameters for **cal_band_dlr.py**:

   - `--obs` (str): Observation type used in the script. Default is `"S"`.
   - `--n0` (int): Parameter $n_0$ as described in [Phys. Rev. B 110, 235131 (2024)](https://doi.org/10.1103/PhysRevB.110.235131).
   - `--err` (float): Error tolerance for computations. Default is `1.e-10`.
   - `--symmetry` (bool): Specifies whether to preserve up-down symmetry in calculations.

d) Full Parameters for **plt_band_dlr.py**:

   - `--obs` (str): Observation type used in the script. Default is `"S"`.
   - `--w_min` (float): Lower bound of the real frequency in eV. Default is `-12`.
   - `--w_max` (float): Upper bound of the real frequency in eV. Default is `12`.
   - `--n_w` (int): Number of frequencies between `w_min` and `w_max`. Default is `200`.
   - `--eta` (float): Broadening parameter. Default is `0.005`.

### iii) MPM Algorithm for real-frequency fitting 

The *examples/MPM_RF* folder contains Jupyter notebooks that demonstrate how to use `MiniPoleRf` to obtain poles for both typical spectral functions and a sub-Ohmic bath.
