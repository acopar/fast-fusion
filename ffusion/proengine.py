import sys
import time
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import scipy.linalg as la
from types import FunctionType
from ctypes import *
import ctypes

from ffusion.common import Timer
from ffusion.engine import Engine
from ffusion.stop import *

class EngineProfiler(Engine):
    def __init__(self, engine, stop='p10', epsilon=6):
        self.engine = engine
        self.x = 0
        self.stop = stop
        self.epsilon = epsilon
        self.operations = 0
        self.soperations = 0
        self.profile = True
        self.timer = Timer()
        self._locals = {}
        
        for key in self.engine.__dict__:
            if key not in self.__dict__:
                self.__dict__[key] = self.engine.__dict__[key]
        #mkl = cdll.LoadLibrary("libmkl_rt.so")
        #mkl.mkl_set_num_threads(ctypes.byref(ctypes.c_int(48)))
        #print("Threads", mkl.mkl_get_max_threads())
    
    def __exit__(self):
        self.engine.__exit__()
    
    def clear(self, X):
        return self.engine.clear(X)
    
    def clean_profile(self):
        self.operations = 0
        self.soperations = 0

    def check_stop(self, history):
        self.timer.split(sys._getframe().f_code.co_name)
        return score_history(history, stop=self.stop, epsilon=self.epsilon)

    def bigdot(self, X, Y, out=None):
        self.timer.split(sys._getframe().f_code.co_name)
        if type(X) == csr_matrix or type(X) == csc_matrix:
            self.operations += X.nnz * Y.shape[1]
        else:
            self.operations += X.shape[0] * X.shape[1] * Y.shape[1]
        #return super(EngineProfiler, self).bigdot(X, Y, out=out)
        return self.engine.bigdot(X, Y, out=out)

    def dot(self, X, Y, out=None, transa='N', transb='N'):
        self.timer.split(sys._getframe().f_code.co_name)
        if len(Y.shape) == 1:
            self.operations += X.shape[0] * X.shape[1]
        else:
            self.operations += X.shape[0] * X.shape[1] * Y.shape[1]
        return self.engine.dot(X, Y, out=out, transa=transa, transb=transb)
        #return super(EngineProfiler, self).dot(X, Y, out=out)
    
    def add(self, X, Y, out=None):
        self.timer.split(sys._getframe().f_code.co_name)
        if len(X.shape) == 0:
            self.soperations += 1
        elif len(X.shape) == 1:
            self.soperations += X.shape[0]
        else:
            self.soperations += X.shape[0] * X.shape[1]
        return self.engine.add(X, Y, out=out)
        #return super(EngineProfiler, self).add(X, Y, out=out)
    
    def sub(self, X, Y, out=None):
        self.timer.split(sys._getframe().f_code.co_name)
        if len(X.shape) == 0:
            self.soperations += 1
        elif len(X.shape) == 1:
            self.soperations += X.shape[0]
        else:
            self.soperations += X.shape[0] * X.shape[1]
        return self.engine.sub(X, Y, out=out)
        #return super(EngineProfiler, self).sub(X, Y)
    
    def multiply(self, X, Y, out=None):
        self.timer.split(sys._getframe().f_code.co_name)
        if len(X.shape) == 0:
            self.soperations += 1
        elif len(X.shape) == 1:
            self.soperations += X.shape[0]
        else:
            self.soperations += X.shape[0] * X.shape[1]
        return self.engine.multiply(X, Y, out=out)
        #return super(EngineProfiler, self).multiply(X, Y)
    
    def divide(self, X, Y):
        self.timer.split(sys._getframe().f_code.co_name)
        if len(X.shape) == 0:
            self.soperations += 1
        elif len(X.shape) == 1:
            self.soperations += X.shape[0]
        else:
            self.soperations += X.shape[0] * X.shape[1]
        return self.engine.divide(X, Y)
        #return super(EngineProfiler, self).divide(X, Y)

    def trace(self, X):
        self.timer.split(sys._getframe().f_code.co_name)
        return self.engine.trace(X)
        #return super(EngineProfiler, self).trace(X)
    
    def inverse(self, A):
        self.timer.split(sys._getframe().f_code.co_name)
        return self.engine.inverse(A)
        #return super(EngineProfiler, self).inverse(A)

    def vsum(self, A):
        self.timer.split(sys._getframe().f_code.co_name)
        self.soperations += A.shape[0] * A.shape[1]
        return self.engine.vsum(A)
        #return super(EngineProfiler, self).vsum(A)
    
    def project(self, A):
        self.timer.split(sys._getframe().f_code.co_name)
        return self.engine.project(A)
        #return super(EngineProfiler, self).project(A)
    
    def project_to(self, X, Y, i):
        self.timer.split(sys._getframe().f_code.co_name)
        super(EngineProfiler, self).project_to64(X, Y, i)
    
    def square(self, A):
        return self.multiply(A, A)
    
    def sqrt(self, X):
        self.timer.split(sys._getframe().f_code.co_name)
        self.soperations += X.shape[0] * X.shape[1]
        return self.engine.sqrt(X)
        #return super(EngineProfiler, self).sqrt(X)
    
    def norm1(self, X):
        self.timer.split(sys._getframe().f_code.co_name)
        return self.engine.norm1(X)
        #return super(EngineProfiler, self).norm1(X)
    
    def cod_u(self, U, KK14, NK13, AK16):
        self.timer.split(sys._getframe().f_code.co_name)
        return super(EngineProfiler, self).cod_u(U, KK14, NK13, AK16)
        
    def cod_v(self, V, LL20, ML18, AL22):
        self.timer.split(sys._getframe().f_code.co_name)
        return super(EngineProfiler, self).cod_v(V, LL20, ML18, AL22)
    
    def cod_s(self, S, KK24, LL25, KL23, AK27, AL29):
        self.timer.split(sys._getframe().f_code.co_name)
        k = S.shape[0]
        l = S.shape[1]
        self.operations += k * l * k + k * l * (l + 3)
        return super(EngineProfiler, self).cod_s(S, KK24, LL25, KL23, AK27, AL29)
    
    def pomus(self, X, Y, Z):
        self.timer.split(sys._getframe().f_code.co_name)
        return self.engine.pomus(X, Y, Z)
        #return super(EngineProfiler, self).pomus(X)

    def zeros_like(self, X):
        self.timer.split(sys._getframe().f_code.co_name)
        return self.engine.zeros_like(X)
        #return super(EngineProfiler, self).zeros_like(X)
        
    def zeros(self, shape):
        return self.engine.zeros(shape)
        #return super(EngineProfiler, self).zeros(shape)

    def sparse_init(self, X):
        self.timer.split(sys._getframe().f_code.co_name)
        return self.engine.sparse_init(X)

    def togpu(self, X):
        self.timer.split(sys._getframe().f_code.co_name)
        return self.engine.togpu(X)

    def fromgpu(self, X):
        self.timer.split(sys._getframe().f_code.co_name)
        return self.engine.fromgpu(X)

    def mpi_send(self, x_gpu, d):
        self.timer.split(sys._getframe().f_code.co_name)
        self.engine.mpi_send(x_gpu, d)
    
    def mpi_recv(self, x_gpu, d):
        self.timer.split(sys._getframe().f_code.co_name)
        self.engine.mpi_recv(x_gpu, d)

    def slice_assign(self, A, B, i, j):
        self.timer.split(sys._getframe().f_code.co_name)
        return self.engine.slice_assign(A, B, i, j)

    def slice_assign_back(self, A, B, i, j):
        self.timer.split(sys._getframe().f_code.co_name)
        return self.engine.slice_assign_back(A, B, i, j)

    def methods(self):
        return {func: getattr(self, func) for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__")}