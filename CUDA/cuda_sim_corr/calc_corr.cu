/*
 * calc_corr.cu
 *
 * Calculate correlation of very long random sequences on Nvidia CUDA GPU.
 * The computation of ~10^10 sequences take ~3 minutes.
 *
 */

#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "vanvleck.cuh"

/*
 * This function initializes the CUDA random number generator.
 */
__global__ void setup_randgen(calc_corr_rnd rnd[], int nseq) {

    long iseq = blockDim.x*blockIdx.x+threadIdx.x;

    if (iseq >= nseq) return;

    curand_init(rnd[iseq].seed, iseq, 0, &rnd[iseq].rndst);

}

/*
 * Calculate correlation of very long random sequences on Nvidia CUDA GPU.
 * The computations of ~10^10 sequences take ~3 minutes.
 */

__global__ void calc_corr(calc_corr_input ci, calc_corr_rnd rnd[], 
                          calc_corr_result cr[], int nseq) {

    long iseq = blockDim.x*blockIdx.x+threadIdx.x;

    if (iseq >= nseq) return;

    // register double s, xn, yn, x, y, qx, qy, r, qr, acx, acy;
    // register double a = ci.a, b = ci.b, sx = ci.sx, sy = ci.sy;
    double s, xn, yn, x, y, qx, qy, r, qr, acx, acy;
    double a = ci.a, b = ci.b, sx = ci.sx, sy = ci.sy;
    long i;

    r = qr = acx = acy = 0;
    
    for (i = 0; i < ci.nrand; i++) {

        s =  curand_normal_double(&rnd[iseq].rndst);    // signal
        xn = curand_normal_double(&rnd[iseq].rndst);    // noise
        yn = curand_normal_double(&rnd[iseq].rndst);    // noise

        x = sx*(a*s + b*xn);
        y = sy*(a*s + b*yn);

        qx = adc(x, ci.nbit);
        qy = adc(y, ci.nbit);

        r   += x*y;
        acx += x*x;
        acy += y*y;

        qr += qx*qy;

    }   /* for (i = 0; i < nrand; i++) */

    cr[iseq].r   = r;
    cr[iseq].qr  = qr;
    cr[iseq].acx = acx;
    cr[iseq].acy = acy;

}



__device__ double adc(double x, int nbit) {
    /* 
     * ADC, analog-to-digital converter
     * Inputs:
     *   x: analog quantity
     *   nbit: precision in bits.
     * Returns x after quantization.
     *
     * The code below is symmetric. 
     * Negative intervals are open on the left and closed on the right.
     * Positive intervals are closed on the left and open on the right.
     * The center (-0.5, 0.5) is open on both sides.
     * For example, if nbits = 3:
     *    (-Inf,-2.5] --> -3
     *    (-2.5,-1.5] --> -2 
     *    (-1.5,-0.5] --> -1
     *    (-0.5, 0.5) -->  0
     *    [ 0.5, 1.5) -->  1 
     *    [ 1.5, 2.5) -->  2 
     *    [ 2.5,+Inf) -->  3 
	 */
    int j, nlev;
    double aj, y;  /* Output */

    nlev = (1 << (nbit-1)) - 1;  /* = 2^(nbit-1)-1: Number of quant. levels */ 

    if (x > -0.5) {
        y = nlev;
        for (j = 0; j < nlev; j++) {
            aj = (double) j;
            if (x < (aj + 0.5)) {
                y = aj;
                break;
            }
        }
    }

    else {
        y = -nlev;
        for (j = -1; j > -nlev; j--) {
            aj = (double) j;
            if (x > (aj - 0.5)) {
                y = aj;
                break;
            }
        }
    } 
  
    return y; 
}

