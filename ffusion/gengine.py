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
from ctypes import *
import ctypes

#from loops import pomus64, pomus32
#from loops import Sloop, cproject, cproject64, cproject_to64
from ffusion.common import Timer
from ffusion.stop import *

from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule
import gpucode
from gpukernels import *

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


class Gengine():
    def __init__(self, n_proc=1, stop='p10', epsilon=6, dtype=np.float32):
        self.stop = stop
        self.epsilon = epsilon
        self.profile = False
        self.memory = {}
        self._locals = {}
        self.dtype = np.float32
        self.sparse = False
        self.n_proc = 1
        self.rank = 0
        #ctx = make_default_context()
        #ngpus = driver.Device.count()
        #gpuid = self.rank % ngpus

        self.ctx  = driver.Device(0).make_context()
        
        linalg.init()
        cusparse.init()
        
        self.mod = SourceModule(gpucode.code)
    
    def __exit__(self):
        self.ctx.detach()
    
    def clear(self, X):
        kern_clear(X)

    def check_stop(self, history):
        return score_history(history, stop=self.stop, epsilon=self.epsilon)

    def bigdot(self, X, Y, out=None, mask=None):
        if out:
            raise Exception("Not implemented in-place sparse multiplication")
        else:
            if type(X) == cusparse.CSR:
                Z = X.mm(Y)
            else:
                Z = linalg.dot(X, Y)
            sync_only()
            return Z

    def dot(self, X, Y, out=None, transa='N', transb='N'):
        if out:
            Z = linalg.dot(X, Y, out=out, transa=transa, transb=transb)
        else:
            Z = linalg.dot(X, Y, transa=transa, transb=transb)
        sync_only()
        return Z
    
    def add(self, X, Y, out=None):
        if out:
            kern_add(X, Y, out)
            sync_only()
        else:
            return skcuda.misc.add(X, Y)
    
    def sub(self, X, Y, out=None):
        return skcuda.misc.subtract(X, Y)
    
    def multiply(self, X, Y, out=None):
        if out:
            kern_mul(X, Y, out)
            return out
        else:
            return skcuda.misc.multiply(X, Y)
    
    def divide(self, X, Y, out=None):
        if out:
            kern_div(X, Y, out, EPSILON32)
            return out
        else:
            out = self.zeros_like(X)
            kern_div(X, Y, out, EPSILON32)
            return out

    def trace(self, X):
        return np.trace(X.get())
    
    def ginverse(self, A):
        A = A.transpose()
        #print("INV", A.shape, type(A), A.flags.c_contiguous, A.flags.f_contiguous)
        #A = gpuarray.to_gpu(np.array(A.get(), dtype=A.dtype, order='C'))
        #print("INV", A.shape, type(A), A.flags.c_contiguous, A.flags.f_contiguous)
        
        out = linalg.pinv(A)
        out = out.transpose()
        #o = out.get()
        #print("Sum", np.sum(np.subtract(o, o.T)))
        #print("INV2", out.shape, type(out), out.flags.c_contiguous, out.flags.f_contiguous)
        #out = gpuarray.to_gpu(np.array(out.get(), dtype=out.dtype, order='F'))
        #A = A.get()
        #A = np.nan_to_num(A)
        #out = self.togpu(la.pinv(A), dtype=A.dtype)
        sync_only()
        return out

    
    def inverse(self, A):
        A = A.get()
        A = np.nan_to_num(A)
        out = self.togpu(la.pinv(A), dtype=A.dtype)
        sync_only()
        return out


    def vsum(self, A):
        return np.sum(A, axis=0).reshape(1,-1)
    
    def project(self, A):
        if A.dtype == np.float64:
            cproject64(A, A)
        else:
            cproject(A, A)
        return A
    
    def project_to(self, X, Y, i):
        cproject_to64(i, X, Y)
    
    def square(self, A):
        return self.multiply(A, A)
    
    def sqrt(self, X):
        Z = pycuda.cumath.sqrt(X)
        return Z
    
    def norm1(self, X):
        return skcuda.misc.sum(X)
    
    def cod_u(self, U, KK14, NK13, AK16):
        Uloop(U, KK14, NK13, AK16)
        k = U.shape[1]
        
    def cod_v(self, V, LL20, ML18, AL22):
        m = V.shape[0]
        k = V.shape[1]
        Vloop(m, k, V, LL20, ML18, AL22, EPSILON)
    
    def cod_s(self, S, KK24, LL25, KL23, AK27, AL29):
        Sloop(S, KK24, LL25, KL23, AK27, AL29)

    def pomus(self, X, Y, Z):
        kern_pomus(X, Y, Z)
        """
        X = X.get()
        if X.dtype == np.float32:
            B, C = pomus32(X)
        else:
            B, C = pomus64(X)
        B = self.togpu(B)
        C = self.togpu(C)
        """
        sync_only()
        return Y, Z

    def zeros_like(self, X):
        n, m = X.shape
        return gpuarray.zeros((n, m), dtype=self.dtype, order='F')

    def zeros(self, shape):
        return gpuarray.zeros(shape, dtype=self.dtype, order='F')

    def sparse_init(self, X):
        if type(X) == csr_matrix:
            self.sparse = True
            return cusparse.CSR.to_CSR(X)
        else:
            return self.togpu(X)

    def togpu(self, X, dtype=None):
        if dtype is None:
            dtype = X.dtype
        X = np.array(X, dtype=dtype, order='F')
        return gpuarray.to_gpu(X)

    def fromgpu(self, X):
        return X.get()

    def mpi_recv(self, x_gpu, d):
        raise Exception("Calling MPI from non-MPI engine")
        
    def mpi_send(self, x_gpu, d):
        raise Exception("Calling MPI from non-MPI engine")

    def slice_assign(self, A, B, i, j):
        i = np.int32(i)
        j = np.int32(j)
        n = np.int32(j-i)
        ldb = np.int32(B.shape[0])
        m = np.int32(B.shape[1])
        
        grix = int(np.ceil(np.float(m)/16))
        griy = int(np.ceil(np.float(n)/16))
        
        func = self.mod.get_function("slice_assign")
        func(A, B, ldb, n, m, i, j, block=(16,16,1), grid=(grix, griy))
        sync_only()
        return B
    
    def slice_assign_back(self, A, B, i, j):
        i = np.int32(i)
        j = np.int32(j)
        n = np.int32(j-i)
        ldb = np.int32(B.shape[0])
        m = np.int32(B.shape[1])
        
        grix = int(np.ceil(np.float(m)/16))
        griy = int(np.ceil(np.float(n)/16))
        
        func = self.mod.get_function("slice_assign_back")
        func(A, B, ldb, n, m, i, j, block=(16,16,1), grid=(grix, griy))
        sync_only()
        return A

    def methods(self):
        return {func: getattr(self, func) for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__")}