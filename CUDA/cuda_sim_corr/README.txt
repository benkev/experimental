This code is intended for testing the van Vleck correction method on various
number sequences, including those very weakly correlated.
In save_sim_cuda.cu the list of tested correlations is
thrho[] = {0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999};

The sampling correlation, r, calculated for two number  sequences of the
length N, has the standard deviation

        std = (1 - r^2)/sqrt(N),

which can be treated as the error in correlation computation. For example, if
r = 0.01, and N = 10000, the error is (1 - 0.0001)/100 = 0.01.

/*
 * save_sim_cuda.cu
 *
 * Calculate correlation of very long random sequences and make the
 * van Vleck correction of the correlation. The computation of ~10^10
 * sequences takes ~3-4 minutes thanks to the use of Nvidia CUDA GPU.
 *
 * Compilation/linking:
 *
 * $ nvcc -g -arch=sm_30  vanvleck.cu calc_corr.cu save_sim_cuda.cu -lm -lgsl \
 *        -lgslcblas -o ssc
 *
 * Example:
 *
 * $ ./ssc  1e6  160  64  4  2.0 1.75
 *
 * -- use 10^6 samples in each thread times 160 blocks times 64 threads
 * per block, which makes 10,240,000,000 samples on total.
 *
 */


Notes.
The C/C++ files contents:

calc_corr.cu: contains the CUDA kernels, 
    __global__ setup_randgen(), - initialization of the random number
                                  generators,
    __global__ void calc_corr(), and
    __device__ double adc(), the simulator of an N-bit symmetric
                             analog-to-digital converter.
                             adc() is called by calc_corr() kernel only.

vanvleck.cu: provides functions
    double vvfun()
    double lininterp_vv() 
    void log10space()
    int load_vvslopes()
    
    


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

    
    
    
    
    
    
    
    
