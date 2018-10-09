import numpy as np
cimport numpy as np
np.import_array()

def pomus64(np.ndarray[np.float64_t, ndim=2]A,
    np.ndarray[np.float64_t, ndim=2]B,
    np.ndarray[np.float64_t, ndim=2]C):
    
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i,j] > 0:
                B[i,j] = A[i,j]
            else:
                B[i,j] = 0
            
            if A[i,j] < 0:
                C[i,j] = -A[i,j]
            else:
                C[i,j] = 0
    
    return B, C
    
    
def pomus32(np.ndarray[np.float32_t, ndim=2]A,
    np.ndarray[np.float32_t, ndim=2]B,
    np.ndarray[np.float32_t, ndim=2]C):
    
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i,j] > 0:
                B[i,j] = A[i,j]
            else:
                B[i,j] = 0
            
            if A[i,j] < 0:
                C[i,j] = -A[i,j]
            else:
                C[i,j] = 0
    
    return B, C
    