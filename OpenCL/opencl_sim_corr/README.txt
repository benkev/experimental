OpenCL pseudorandom number generators (PRNG) on GPU and CPU and demo programs.

The algorithms are obtained from the webpage  by Prof. Sebastiano Vigna
"Xoshiro / xoroshiro generators and the PRNG shootout" at
http://xoshiro.di.unimi.it/




Files with 'bits64' in their names: generators of 64-bit unsigned integers.
Files with 'unif64' in their names: generators of double precision
    floating-point numbers uniformly distributed over the [0..1] interval.
Files with 'norm64' in their names: generators of double precision
    floating-point numbers with normal (Gaussian) distribution having the mean
    at 0 and the standard deviation equal 1.


rand_lib.c: functions to initialize the states of PRNG


Compilation of demo programs:

$ gcc -std=c99 rand_lib.c gpu_rand_unif64.c -o gpu_rand_unif64 -l OpenCL

$ gcc -std=c99 cpu_rand_unif64.c -o cpu_rand_unif64

$ gcc -std=c99 rand_lib.c gpu_rand_bits64.c -o gpu_rand_bits64 -l OpenCL

$ gcc -std=c99 cpu_rand_bits64.c -o cpu_rand_bits64


==========================================================

Some "bencmarks".

GPU: Nvidia GeForce GTX 670

  In CUDA, ~/experimental/CUDA/cuda_sim_corr/
  $ ./ssc  1e6 160 64 4 2.0 1.75
  CPU_time_used: 257.940s;  as MM:SS = 04:17.940

  In OpenCL:
  ~/experimental/OpenCL/opencl_sim_corr/
  $ ipython2 --pylab
  %run ocl_calc_vvcorr.py 1e6 160 64 4 2.0 1.75
  Elapsed time 6 min 36 sec.

GPU: AMD Radeon RX 580

  In OpenCL:
  ~/experimental/OpenCL/opencl_sim_corr/
  $ ipython2 --pylab
  %run ocl_calc_vvcorr.py 1e6 160 64 4 2.0 1.75
  Elapsed time 5 min 24 sec.
