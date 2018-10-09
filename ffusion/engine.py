import sys
import time
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import scipy.linalg as la
#from types import FunctionType
#from ctypes import *
#import ctypes

import loops
from loops import pomus64, pomus32
#from loops import Sloop, cproject, cproject64, cproject_to64
from ffusion.common import Timer
from ffusion.stop import *

EPSILON = np.finfo(np.float64).eps
MAXILON = 10**(9)

class Engine():
    def __init__(self, n_proc=1, stop='p10', epsilon=6, dtype=np.float32):
        self.stop = stop
        self.epsilon = epsilon
        
        self.profile = False
        self._locals = {}
        self.dtype = dtype
        self.rank = 0
        self.n_proc = 1
        
        #import ctypes
        #mkl_rt = ctypes.CDLL('libmkl_rt.so')
        #mkl_get_max_threads = mkl_rt.mkl_get_max_threads
        #def mkl_set_num_threads(cores):
        #    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))
        
        #mkl_set_num_threads(1)
        #print("N thhreads", mkl_get_max_threads())
    
    
    def __exit__(self):
        pass
    
    def clear(self):
        pass

    def check_stop(self, history):
        return score_history(history, stop=self.stop, epsilon=self.epsilon)

    def bigdot(self, X, Y, out=None, mask=None):
        if not out is None:
            if type(X) == csr_matrix or type(X) == csc_matrix:
                out[:,:] = X.dot(Y)
            else:
                return np.dot(X, Y, out=out)
        else:
            if type(X) == csr_matrix or type(X) == csc_matrix:
                return X.dot(Y)
            else:
                return np.dot(X, Y)

    def dot(self, X, Y, out=None, transa='N', transb='N'):
        if transa == 'T':
            X = X.T
        if transb == 'T':
            Y = Y.T
        if not out is None:
            return np.dot(X, Y, out=out)
        else:
            return np.dot(X, Y)
    
    def add(self, X, Y, out=None):
        if not out is None:
            if type(out) == np.ndarray:
                return np.add(X, Y, out=out)
            else:
                if out[1] in out[0]:
                    return np.add(X, Y, out=out[0][out[1]])
                else:
                    raise Exception("Out has no %d key" % out[1], out[0])
        else:
            return np.add(X, Y)
    
    def sub(self, X, Y, out=None):
        return np.subtract(X, Y)
    
    def multiply(self, X, Y, out=None):
        if type(X) == np.ndarray:
            D = np.multiply(X, Y)
        else:
            D = X.multiply(Y)
        return D
    
    def divide(self, X, Y, out=None):
        EPS = EPSILON
        if self.dtype == np.float32:
            EPS = np.finfo(np.float32).eps
        
        if np.isscalar(Y):
            if Y < EPS:
                Y = EPS
        else:
            Y[np.where(Y < EPS)] = EPS
        
        return np.divide(X, Y)

    def trace(self, X):
        return np.trace(X)
    
    def inverse(self, A):
        A = np.nan_to_num(A)
        return la.pinv(A)

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
        return np.sqrt(X)
    
    def norm1(self, X):
        return np.sum(X)
    
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
        if X.dtype == np.float32:
            pomus32(X, Y, Z)
        else:
            pomus64(X, Y, Z)
        return Y, Z

    #def pomus_old(self, X):
    #    t = (X > 0).astype(int)
    #    B = np.multiply(X, t).astype(np.float32)
    #    t = np.subtract(t, 1)
    #    C = np.multiply(X, t).astype(np.float32)
    #    return B, C

    def zeros_like(self, X):
        return np.zeros_like(X)

    def zeros(self, shape):
        return np.zeros(shape=shape, dtype=self.dtype)

    def sparse_init(self, X):
        return X
        
    def togpu(self, X):
        return X
    
    def fromgpu(self, X):
        return X
    
    def mpi_recv(self, x_gpu, d):
        raise Exception("Calling MPI from non-MPI engine")
        
    def mpi_send(self, x_gpu, d):
        raise Exception("Calling MPI from non-MPI engine")

    def slice_assign(self, A, B, i, j):
        B[i:j,:] = A[:,:]
        return B
    
    def slice_assign_back(self, A, B, i, j):
        A[:,:] = B[i:j,:]
        return A

    def methods(self):
        return {func: getattr(self, func) for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__")}