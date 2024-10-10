# MiniPole
The Python code provided implements the matrix-valued version of the Minimal Pole Method (MPM)

The input of the simulation is the Matsubara data $G(i \omega_n)$ sampled on a uniform grid $\lbrace i\omega_{0}, i\omega_{1}, \cdots, i\omega_{n_{\omega}-1} \rbrace$, where  $\omega_n=\frac{(2n+1)\pi}{\beta}$ for fermions and $\frac{2n\pi}{\beta}$ for bosons, and $n_{\omega}$ is the total number of sampling points.

## The standard MPM is performed using the following command:

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
    12. include_n0 : bool, default=True
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

## The MPM-DLR algorithm is performed using the following command:

**p = MiniPoleDLR(Al_dlr, xl_dlr, beta, n0, err = None, err_type = "abs", symmetry = False, nmax = None, Lfactor = 0.4)**

    Parameters
    ----------
    1. Al_dlr (numpy.ndarray): DLR coefficients, either of shape (r,) or (r, n_orb, n_orb).
    2. xl_dlr (numpy.ndarray): DLR grid for the real frequency, an array of shape (r,).
    3. beta (float): Inverse temperature of the system (1/kT).
    4. n0 (int): Number of initial points to discard, typically in the range (0, 10).
    5. err (float): Error tolerance for calculations.
    6. err_type (str): Specifies the type of error, "abs" for absolute error or "rel" for relative error.
    7. symmetry (bool): Whether to impose up-down symmetry (True or False).
    8. nmax (int): Cutoff for the Matsubara frequency when symmetry is False.
    9. Lfactor (float): Ratio of L/N in the ESPRIT algorithm.

    Returns
    -------
    Minimal pole representation of the given data.
    Pole weights are stored in p.pole_weight, a numpy array of shape (M, n_orb, n_orb).
    Shared pole locations are stored in p.pole_location, a numpy array of shape (M,).
