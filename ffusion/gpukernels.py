from pycuda.elementwise import ElementwiseKernel

kern_clear = ElementwiseKernel(
    "float *x",
    "x[i] = 0.0;",
    "kern_clear")


kern_mul = ElementwiseKernel(
    "float *x, float *y, float *z",
    "z[i] = x[i] * y[i];",
    "kern_mul")

kern_add = ElementwiseKernel(
    "float *x, float *y, float *z",
    "z[i] = x[i] + y[i];",
    "kern_add")


kern_div = ElementwiseKernel(
    "float *x, float *y, float *z, float eps",
    "z[i] = x[i] / (y[i] + eps);",
    "kern_div")
    
code_pomus = """
if (x[i] > 0){
    y[i] = x[i];
} else {
    y[i] = 0.0;
}

if (x[i] < 0){
    z[i] = -x[i];
} else {
    z[i] = 0.0;
}
"""

kern_pomus = ElementwiseKernel(
    "float *x, float *y, float *z",
    code_pomus,
    "kern_pomus")