import sys
import time
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import scipy.linalg as la
import pycuda.autoinit

import pycuda.cumath as cumath
import pycuda.gpuarray as gpuarray
import skcuda.linalg as linalg
import skcuda.misc
from pycuda import driver
#from pycuda.tools import make_default_context

from cuda_cffi import cusparse

from types import FunctionType

#from loops import pomus64, pomus32
#from loops import Sloop, cproject, cproject64, cproject_to64
from ffusion.common import Timer
from ffusion.stop import *

from mpi4py import MPI
from ffusion.gengine import Gengine
from pycuda.compiler import SourceModule
import gpucode

EPSILON = np.finfo(np.float64).eps
EPSILON32 = np.finfo(np.float32).eps
MAXILON = 10**(9)

def to_gpu(X):
    X = np.array(X, dtype=X.dtype, order='F')
    return gpuarray.to_gpu(X)

def sync_only():
    sync_gpu(driver.Event())
    
def sync_gpu(event):
    event.record()
    event.synchronize()


### MPI calls ###

def dtype_to_mpi(t):
    if hasattr(MPI, '_typedict'):
        mpi_type = MPI._typedict[np.dtype(t).char]
    elif hasattr(MPI, '__TypeDict__'):
        mpi_type = MPI.__TypeDict__[np.dtype(t).char]
    else:
        raise ValueError('cannot convert type')
        return mpi_type

bufint_gpu = lambda arr: arr.gpudata.as_buffer(arr.nbytes)
bufint_cpu = lambda arr: arr

class MgpuEngine(Gengine):
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
        
        #mkl_set_num_threads(6)
        #print("N thhreads", mkl_get_max_threads())
        self.sparse = False
        #ctx = make_default_context()
        #ngpus = driver.Device.count()
        #gpuid = self.rank % ngpus
        
        self.ctx  = driver.Device(self.rank).make_context()
        
        linalg.init()
        cusparse.init()
        
        self.mod = SourceModule(gpucode.code)

    def mpi_send(self, x_gpu, d):
        #return comm.Send(bufint_cpu(x_gpu), dest=d)
        self.comm.Send([bufint_gpu(x_gpu), dtype_to_mpi(x_gpu.dtype)], dest=d)
    
    def mpi_recv(self, x_gpu, d):
        #return comm.Recv(bufint_cpu(x_gpu), source=d)
        self.comm.Recv([bufint_gpu(x_gpu), dtype_to_mpi(x_gpu.dtype)], source=d)

    def methods(self):
        return {func: getattr(self, func) for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__")}