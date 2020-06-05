/*
 * $ nvcc -g -arch=sm_30 array_CUDA_dynamic.cu -o acd
 *
 * error: expression must have a constant value
 *   int (*cube)[m][n] = arr;
 *               ^
 */


#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>



__global__ void array_dynamic(int l, int m, int n, int *arr) {

    int (*cube)[m][n] = arr;
    int i, j, k;
    
    for (i=0; i < l; ++i)
        for (j=0; j < m; ++j)
            for (k=0; k < n; ++k)
                cube[i][j][k] = i*j*k;

}

int main() {
    int lmn = 3*5*4;
    int *ar = (int *) malloc(lmn*sizeof(int));
    
    array_dynamic<<<1,1>>>(3, 5, 4, ar);
    
    
    // cudaDeviceSynchronize();
  return 0;
}
