These programs list lengths in bytes of some built-in types in
CPU (gcc compiler) and
CUDA GPU (nvcc compiler).

Compilation:

gcc -g gcc_type_sizes.c -o gcc_type_sizes

nvcc -g -arch=sm_30 nvcc_type_sizes.cu -o nvcc_type_sizes



Note that although ./nvcc_type_sizes prints
"sizeof(long double) = 16 bytes = 128 bits", 

the nvcc compiler warns:

nvcc -g -arch=sm_30 nvcc_type_sizes.cu -o nvcc_type_sizes

nvcc_type_sizes.cu(34): warning: 'long double' is treated as 'double' in device
                                  code

nvcc_type_sizes.cu(35): warning: 'long double' is treated as 'double' in device
                                  code


