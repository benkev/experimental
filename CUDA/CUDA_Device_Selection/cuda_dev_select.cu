//
// Compile:
// $ nvcc cuda_dev_select.cu -o cuda_dev_select
//
//

#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>


int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d, %s, compute capability %d.%d\n",
               device, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    // printf("\n\n");
    printf("deviceCount = %d\n", deviceCount);
    
}
