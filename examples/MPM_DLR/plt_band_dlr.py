#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse

# Function tools
def cal_G(z, Al, xl):
    G_z = 0.0
    for i in range(xl.size):
        G_z += Al[i] / (z - xl[i])
    return G_z

def cal_G_vector(z, Al, xl):
    G_z = 0.0
    for i in range(xl.size):
        G_z += Al[[i]] / (z.reshape(-1, 1) - xl[i])
    return G_z

#------input parameters-------
parser = argparse.ArgumentParser(description="Parameters for plotting the band structure of silicon")
parser.add_argument("--obs"     , type=str  , default="S")
parser.add_argument("--w_min"   , type=float, default=-12)
parser.add_argument("--w_max"   , type=float, default=12)
parser.add_argument("--n_w"     , type=int  , default=200)
parser.add_argument("--eta"     , type=float, default=0.005)
args = parser.parse_args()
#-----------------------------
obs = args.obs; assert obs in ["Gii", "G", "S"]
w_min = args.w_min
w_max = args.w_max
n_w = args.n_w
eta = args.eta; assert eta >= 0.0

# Obtain the grid for the real frequency
HartreeToEv = 27.2114
y = np.linspace(w_min, w_max, n_w)
freqs = y / HartreeToEv

# Read necessary data
data = h5py.File("Si_dlr.h5", "r")
Fk = data["Fk"][:][0]
mu = data["mu"][()]
path = data["hs_path/path"][:]
sp_points = data["hs_path/sp_points"][:]
sp_labels = data["hs_path/sp_labels"][:]
sp_labels = [x.decode('utf-8') for x in sp_labels]
data.close()

if obs == "Gii":
    # Read complex pole information
    data = h5py.File("pole_Gii.h5", "r")
    Al_Gii = data["Al"][:]
    xl_Gii = data["xl"][:]
    np_Gii = data["np"][:]
    data.close()
    # Obtain band structure
    nk  = Al_Gii.shape[1]
    nao = Al_Gii.shape[2]
    DOS_G_dlr_diag = np.zeros((freqs.size, nk, nao), dtype=np.float64)
    for k in range(nk):
        for n in range(nao):
            DOS_G_dlr_diag[:, k, n] = -1.0 / np.pi * cal_G(freqs + 1j * eta, Al_Gii[:np_Gii[k, n], k, n], xl_Gii[:np_Gii[k, n], k, n]).imag
    band = DOS_G_dlr_diag.sum(axis=2)
elif obs == "G":
    # Read complex pole information
    data = h5py.File("pole_G.h5", "r")
    Al_G = data["Al"][:]; Al_G = np.einsum("ijjk->ijk", Al_G)
    xl_G = data["xl"][:]
    np_G = data["np"][:]
    data.close()
    # Obtain band structure
    nao = Al_G.shape[1]
    nk  = Al_G.shape[2]
    DOS_G_dlr = np.zeros((freqs.size, nk, nao), dtype=np.float64)
    for k in range(nk):
        DOS_G_dlr[:, k, :] = -1.0 / np.pi * cal_G_vector(freqs + 1j * eta, Al_G[:np_G[k], :, k], xl_G[:np_G[k], k]).imag
    band = DOS_G_dlr.sum(axis=2)
else:
    # Read complex pole information
    data = h5py.File("pole_S.h5", "r")
    Al_S = data["Al"][:]
    xl_S = data["xl"][:]
    np_S = data["np"][:]
    data.close()
    # Obtain retarded self-energy
    nao = Al_S.shape[1]
    nk  = Al_S.shape[3]
    S_ret_dlr = np.zeros((freqs.size, nk, nao, nao), dtype=np.complex128)
    for k in range(nk):
        S_ret_dlr[:, k, :, :] = cal_G_vector(freqs + 1j * eta,  Al_S[:np_S[k], :, :, k].reshape(-1, nao**2), xl_S[:np_S[k], k]).reshape(-1, nao, nao)
    # Perform Dyson equation above the real axis
    G0_ret_inv = ((freqs + 1j * eta).reshape(-1, 1, 1, 1) * np.eye(nao).reshape(1, 1, nao, nao) + mu * np.eye(nao).reshape(1, 1, nao, nao) - Fk.reshape(1, nk, nao, nao))
    G_ret = np.zeros_like(S_ret_dlr)
    for i in range(G_ret.shape[0]):
        for j in range(G_ret.shape[1]):
            G_ret[i, j] = np.linalg.pinv(G0_ret_inv[i, j] - S_ret_dlr[i, j])
    # Obtain band structure
    band = (-1.0 / np.pi * np.einsum("ijkk->ijk", G_ret[:, :, :, :]).imag).sum(axis=2)

# Plot band structure
plt.figure(figsize=(10, 6))
plt.imshow(band, aspect='auto', origin='lower', extent=[path[0], path[-1], y[0], y[-1]], cmap="RdBu", vmin=0, vmax=1 / eta)
plt.xticks(sp_points, sp_labels)
plt.tick_params(labelsize=20)
plt.ylabel(r'$\omega$(eV)', fontsize=20)
title = "G_{ii}" if obs == "Gii" else "G" if obs == "G" else "S"
plt.title(r"from $" + title + "$", fontsize=20)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=20)
plt.savefig("Si_dlr_" + obs + ".pdf")
plt.show()
