#!/usr/bin/env python3
import numpy as np
import h5py
import argparse
from mpi4py import MPI
import time
from mini_pole import MiniPoleDLR

# Initialize the MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#------input parameters-------
parser = argparse.ArgumentParser(description="Parameters for calculating the band structure of silicon")
parser.add_argument("--obs"     , type=str  , default="S")
parser.add_argument("--n0"      , type=int  , default=None)
parser.add_argument("--err"     , type=float, default=1.e-10)
parser.add_argument('--symmetry', type=lambda x: (str(x).lower() == 'true'), default=None)
parser.add_argument("--np_max"  , type=int  , default=50)
args = parser.parse_args()
#-----------------------------
obs = args.obs; assert obs in ["Gii", "G", "S"]
n0  = args.n0 if args.n0 is not None else 3 if obs == "S" else 4
err = args.err
symmetry = args.symmetry if args.symmetry is not None else False if obs == "S" else True
np_max = args.np_max

if rank == 0:
    t_start = time.time()
    if obs == "S":
        print("Observable: matrix-valued self-energy", flush=True)
    elif obs == "G":
        print("Observable: matrix-valued Green's function", flush=True)
    else:
        print("Observable: scalar-valued Green's function", flush=True)
    print("n0 =", n0, flush=True)
    print("error tolerance =", err, flush=True)
    print("up-down symmetry is", symmetry, flush=True)

# Read input file
data = h5py.File("Si_dlr.h5", "r")
if obs == "S":
    Al_dlr = data["S/Al"][:][:, 0, :, :, :]
    xl_dlr = data["S/xl"][:]
else:
    Al_dlr = data["G/Al"][:][:, 0, :, :, :]
    xl_dlr = data["G/xl"][:]
    if obs == "Gii":
        Al_dlr = np.einsum("ijkk->ijk", Al_dlr)
beta = data["beta"][()]
data.close()

# System sizes
nk  = Al_dlr.shape[1]
nao = Al_dlr.shape[2]

# Total number of iterations
n_iter = nk if obs == "S" or obs == "G" else nk * nao

# Calculate the start and end indices for each rank
iter_list = np.arange(rank, n_iter, size)
if obs == "S" or obs == "G":
    Al_i = np.zeros((np_max, nao, nao, iter_list.size), dtype=np.complex128)
    xl_i = np.zeros((np_max, iter_list.size), dtype=np.complex128)
    np_i = np.zeros((iter_list.size,), dtype=np.int_)
else:
    Al_i = np.zeros((np_max, iter_list.size), dtype=np.complex128)
    xl_i = np.zeros((np_max, iter_list.size), dtype=np.complex128)
    np_i = np.zeros((iter_list.size,), dtype=np.int_)

# Parallelize the for loop
for i in range(iter_list.size):
    if obs == "S" or obs == "G":
        k = iter_list[i]
        p = MiniPoleDLR(Al_dlr[:, k, :, :], xl_dlr, beta=beta, n0=n0, err=err, symmetry=symmetry)
        Al_i[:p.pole_location.size, :, :, i] = p.pole_weight
        xl_i[:p.pole_location.size, i] = p.pole_location
        np_i[i] = p.pole_location.size
    else:
        k, n = iter_list[i] // nao, iter_list[i] % nao
        p = MiniPoleDLR(Al_dlr[:, k, n], xl_dlr, beta=beta, n0=n0, err=err, symmetry=symmetry)
        Al_i[:p.pole_location.size, i] = p.pole_weight
        xl_i[:p.pole_location.size, i] = p.pole_location
        np_i[i] = p.pole_location.size

# Gather data
iter_tot = comm.gather(iter_list, root=0)
Al = comm.gather(Al_i, root=0)
xl = comm.gather(xl_i, root=0)
np_tot = comm.gather(np_i, root=0)

if rank == 0:
    # Order and reshape data
    iter_tot = np.concatenate(iter_tot, axis=0)
    idx = np.argsort(iter_tot)
    if obs == "S" or obs == "G":
        np_tot = np.concatenate(np_tot, axis=0)[idx]
        np_real_max = np.max(np_tot)
        Al = np.concatenate(Al, axis=-1)[:, :, :, idx].reshape(np_max, nao, nao, nk)[:np_real_max, :, :, :]
        xl = np.concatenate(xl, axis=-1)[:, idx].reshape(np_max, nk)[:np_real_max, :]
    else:
        np_tot = np.concatenate(np_tot, axis=0)[idx].reshape(nk, nao)
        np_real_max = np.max(np_tot)
        Al = np.concatenate(Al, axis=-1)[:, idx].reshape(np_max, nk, nao)[:np_real_max, :, :]
        xl = np.concatenate(xl, axis=-1)[:, idx].reshape(np_max, nk, nao)[:np_real_max, :, :]
    
    # Write output file
    data = h5py.File("pole_" + obs + ".h5", "w")
    data["Al"] = Al
    data["xl"] = xl
    data["np"] = np_tot
    data.close()
    
    t_end = time.time()
    print("Finished in", t_end - t_start, "s")
