import sys
import time
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import scipy.linalg as la
#from types import FunctionType
#from ctypes import *
#import ctypes
from mpi4py import MPI

import loops
from loops import pomus64, pomus32
#from loops import Sloop, cproject, cproject64, cproject_to64
from ffusion.common import Timer
from ffusion.stop import *
from ffusion.engine import Engine

EPSILON = np.finfo(np.float64).eps
MAXILON = 10**(9)

### MPI calls ###

def dtype_to_mpi(t):
    if hasattr(MPI, '_typedict'):
        mpi_type = MPI._typedict[np.dtype(t).char]
    elif hasattr(MPI, '__TypeDict__'):
        mpi_type = MPI.__TypeDict__[np.dtype(t).char]
    else:
        raise ValueError('cannot convert type')
        return mpi_type

bufint_cpu = lambda arr: arr

class MpiEngine(Engine):
    def __init__(self, n_proc=1, stop='p10', epsilon=6, dtype=np.float32):
        self.stop = stop
        self.epsilon = epsilon
        
        self.profile = False
        self._locals = {}
        self.dtype = dtype
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.n_proc = n_proc
        
        #import ctypes
        #mkl_rt = ctypes.CDLL('libmkl_rt.so')
        #mkl_get_max_threads = mkl_rt.mkl_get_max_threads
        #def mkl_set_num_threads(cores):
        #    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))
        
        #mkl_set_num_threads(1)
        #print("N thhreads", mkl_get_max_threads())
    
    def mpi_send(self, x_gpu, d):
        #print(type(x_gpu), x_gpu.dtype, dtype_to_mpi(x_gpu.dtype))
        #return comm.Send(bufint_cpu(x_gpu), dest=d)
        return self.comm.Send([bufint_cpu(x_gpu), dtype_to_mpi(x_gpu.dtype)], dest=d)
    
    def mpi_recv(self, x_gpu, d):
        #print(type(x_gpu), x_gpu.dtype, dtype_to_mpi(x_gpu.dtype))
        #return comm.Recv(bufint_cpu(x_gpu), source=d)
        return self.comm.Recv([bufint_cpu(x_gpu), dtype_to_mpi(x_gpu.dtype)], source=d)

    def methods(self):
        return {func: getattr(self, func) for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__")}