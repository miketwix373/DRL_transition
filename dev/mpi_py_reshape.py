from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 1:
    # Assuming the matrix size is known or has been communicated beforehand
    nx, ny = 10, 10

    # Prepare a buffer to receive the data
    matrix = np.empty(nx * ny, dtype='float64')

    # Receive the data
    comm.Recv(matrix, source=0)

    # Reshape and transpose the matrix to match Fortran's column-major order
    matrix = matrix.reshape((nx, ny)).T

    # Now matrix is ready to use
    print(matrix)

