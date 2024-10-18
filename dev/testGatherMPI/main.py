from mpi4py import MPI
import numpy

# Spawing a process running an executable
version = MPI.Get_version()
print("MPI version for mpi4py = ")
print(version)
# sub_comm is an MPI intercommunicator
sub_comm = MPI.COMM_SELF.Spawn('./child', args=[], maxprocs=10)
# common_comm is an intracommunicator accross the python process and the spawned process.
# All kind sof collective communication (Bcast...) are now possible between the python process and the c process
common_comm=sub_comm.Merge(False)
print('parent in common_comm ', common_comm.Get_rank(), ' of  ', common_comm.Get_size())
data = numpy.zeros((10*6),dtype=numpy.int32)
print(data)
soma = numpy.array(0,dtype=numpy.int32)
common_comm.Recv([soma, MPI.INT], source=1, tag=2)
common_comm.Recv([data, MPI.INT], source=1, tag=3)
print(soma)
print(data)
data = data.reshape((6,10)).T
print(data)

print("Python over")
# free the (merged) intra communicator
common_comm.Free()
# disconnect the inter communicator is required to finalize the spawned process.
sub_comm.Disconnect()
