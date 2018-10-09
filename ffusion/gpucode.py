code = """
__global__ void slice_assign(float *A, float *B, int ldb, int n, int m, int startrange, int endrange){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int indey = threadIdx.y + blockIdx.y * blockDim.y;
    if ( index >= m ){
        return;
    }
    
    if ( indey >= n ){
        return;
    }
    
    B[index*ldb + indey + startrange] = A[index*n + indey];
    
    return;
}


__global__ void slice_assign_back(float *A, float *B, int ldb, int n, int m, int startrange, int endrange){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int indey = threadIdx.y + blockIdx.y * blockDim.y;
    if ( index >= m ){
        return;
    }
    
    if ( indey >= n ){
        return;
    }
    
    A[index*n + indey] = B[index*ldb + indey + startrange];
        
    return;
}
"""