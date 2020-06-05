/*
 * Compile:
 *
 * nvcc -g -arch=sm_30 nvcc_type_sizes.cu -o nvcc_type_sizes
 *
 */

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>


void cudaAssert(const cudaError err, const char *file, const int line)
{ 
    if( cudaSuccess != err) {                                                
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        
                file, line, cudaGetErrorString(err) );
        exit(1);
    } 
}

__global__ void print_sizes() {

  printf("sizeof(int) = \t\t%ld bytes = %ld bits\n", sizeof(int), 
	 8*sizeof(int)); 
  printf("sizeof(long) = \t\t%ld bytes = %ld bits\n", sizeof(long), 
	 8*sizeof(long)); 
  printf("sizeof(long long) = \t%ld bytes = %ld bits\n", sizeof(long long), 
	 8*sizeof(long long)); 
  printf("sizeof(float) = \t%ld bytes = %ld bits\n", sizeof(float), 
	 8*sizeof(float)); 
  printf("sizeof(double) = \t%ld bytes = %ld bits\n", sizeof(double), 
	 8*sizeof(double)); 
  printf("sizeof(long double) = \t%ld bytes = %ld bits\n", sizeof(long double), 
	 8*sizeof(long double)); 

}

int main() {
    print_sizes<<<1,1>>>();
    cudaDeviceSynchronize();
  return 0;
}
