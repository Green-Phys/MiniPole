# Minimal Pole Method (MPM)
The Python code provided implements the matrix-valued version of the Minimal Pole Method (MPM) as described in [Phys. Rev. B 110, 235131 (2024)](https://doi.org/10.1103/PhysRevB.110.235131), extending the scalar-valued method introduced in [Phys. Rev. B 110, 035154 (2024)](https://doi.org/10.1103/PhysRevB.110.035154).

The input of the simulation is the Matsubara data $G(i \omega_n)$ sampled on a uniform grid $\lbrace i\omega_{0}, i\omega_{1}, \cdots, i\omega_{n_{\omega}-1} \rbrace$, where  $\omega_n=\frac{(2n+1)\pi}{\beta}$ for fermions and $\frac{2n\pi}{\beta}$ for bosons, and $n_{\omega}$ is the total number of sampling points.

## 1. Installation

### Dependencies
- `numpy`
- `scipy`
- `matplotlib`

### Installation Commands
1. Via `setup.py`:
   ```bash
   python3 setup.py install

2. Or via `pip`:
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
    Pole weights are stored in p.pole_weight, a numpy array of shape (M, n_orb, n_orb).
    Shared pole locations are stored in p.pole_location, a numpy array of shape (M,).

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
    Pole weights are stored in p.pole_weight, a numpy array of shape (M, n_orb, n_orb).
    Shared pole locations are stored in p.pole_location, a numpy array of shape (M,).

## 3. Examples

The scripts in the *examples* folder demonstrate the usage of MPM and MPM-DLR.

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
