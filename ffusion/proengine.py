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

EPSILON = np.finfo(np.float64).eps
MAXILON = 10**(9)

class EngineProfiler(Engine):
    def __init__(self, stop='p10', epsilon=6):
        self.x = 0
        self.stop = stop
        self.epsilon = epsilon
        self.operations = 0
        self.soperations = 0
        self.profile = True
        self.timer = Timer()
        
        #mkl = cdll.LoadLibrary("libmkl_rt.so")
        #mkl.mkl_set_num_threads(ctypes.byref(ctypes.c_int(48)))
        #print("Threads", mkl.mkl_get_max_threads())
    
    def clean(self):
        self.operations = 0
        self.soperations = 0

    def check_stop(self, history):
        self.timer.split(sys._getframe().f_code.co_name)
        return score_history(history, stop=self.stop, epsilon=self.epsilon)

    def bigdot(self, X, Y):
        self.timer.split(sys._getframe().f_code.co_name)
        if type(X) == csr_matrix or type(X) == csc_matrix:
            self.operations += X.nnz * Y.shape[1]
        else:
            self.operations += X.shape[0] * X.shape[1] * Y.shape[1]
        return super(EngineProfiler, self).bigdot(X, Y)

    def dot(self, X, Y):
        self.timer.split(sys._getframe().f_code.co_name)
        if len(Y.shape) == 1:
            self.operations += X.shape[0] * X.shape[1]
        else:
            self.operations += X.shape[0] * X.shape[1] * Y.shape[1]
        return super(EngineProfiler, self).dot(X, Y)
    
    def add(self, X, Y):
        self.timer.split(sys._getframe().f_code.co_name)
        if len(X.shape) == 0:
            self.soperations += 1
        elif len(X.shape) == 1:
            self.soperations += X.shape[0]
        else:
            self.soperations += X.shape[0] * X.shape[1]
        return super(EngineProfiler, self).add(X, Y)
    
    def sub(self, X, Y):
        self.timer.split(sys._getframe().f_code.co_name)
        if len(X.shape) == 0:
            self.soperations += 1
        elif len(X.shape) == 1:
            self.soperations += X.shape[0]
        else:
            self.soperations += X.shape[0] * X.shape[1]
        return super(EngineProfiler, self).sub(X, Y)
    
    def multiply(self, X, Y):
        self.timer.split(sys._getframe().f_code.co_name)
        if len(X.shape) == 0:
            self.soperations += 1
        elif len(X.shape) == 1:
            self.soperations += X.shape[0]
        else:
            self.soperations += X.shape[0] * X.shape[1]
        return super(EngineProfiler, self).multiply(X, Y)
    
    def divide(self, X, Y):
        self.timer.split(sys._getframe().f_code.co_name)
        if len(X.shape) == 0:
            self.soperations += 1
        elif len(X.shape) == 1:
            self.soperations += X.shape[0]
        else:
            self.soperations += X.shape[0] * X.shape[1]
        return super(EngineProfiler, self).divide(X, Y)

    def trace(self, X):
        self.timer.split(sys._getframe().f_code.co_name)
        return super(EngineProfiler, self).trace(X)
    
    def inverse(self, A):
        self.timer.split(sys._getframe().f_code.co_name)
        return super(EngineProfiler, self).inverse(A)

    def vsum(self, A):
        self.timer.split(sys._getframe().f_code.co_name)
        self.soperations += A.shape[0] * A.shape[1]
        return super(EngineProfiler, self).vsum(A)
    
    def project(self, A):
        self.timer.split(sys._getframe().f_code.co_name)
        return super(EngineProfiler, self).project(A)
    
    def project_to(self, X, Y, i):
        self.timer.split(sys._getframe().f_code.co_name)
        super(EngineProfiler, self).project_to64(X, Y, i)
    
    def square(self, A):
        return self.multiply(A, A)
    
    def sqrt(self, X):
        self.timer.split(sys._getframe().f_code.co_name)
        self.soperations += X.shape[0] * X.shape[1]
        return super(EngineProfiler, self).sqrt(X)
    
    def norm1(self, X):
        self.timer.split(sys._getframe().f_code.co_name)
        return super(EngineProfiler, self).norm1(X)
    
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
    
    def pomus(self, X):
        self.timer.split(sys._getframe().f_code.co_name)
        return super(EngineProfiler, self).pomus(X)

    def methods(self):
        return {func: getattr(self, func) for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__")}