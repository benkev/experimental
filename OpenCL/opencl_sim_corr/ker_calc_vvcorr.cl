/*
 * ker_calc_vvcorr.cl
 *
 * Calculate correlation of very long random sequences on GPU.
 * The computation of ~10^10 sequences takes ~5-7 minutes.
 *
 * The used random number generator xoshiro256** is based on 64-bit floating
 * point (double) arithmetic.
 *
 */

//#include "vanvleck_structs.h"

double adc(double x, int nbit);
ulong rand64_bits(__global ulong state[]);
double rand64_uniform(__global ulong state[]);
double rand64_normal(__global ulong state[]);
static inline ulong rotl(const ulong x, int k);



__constant double inv_ULONG_MAX = 1./((double) ULONG_MAX);

/*
 * Calculate correlation of very long random sequences on Nvidia CUDA GPU.
 *
 * The declarations for structures calc_corr_input and calc_corr_result are
 * generated from numpy.ndtype-s by pyopencl.tools.match_dtype_to_c_struct() and
 * prepended to this kernel by string concatenation before its compilation with
 * pyopenclcl.Program(ctx, ker).build(). 
 */

__kernel void calc_corr(__global ulong *rndst,
                        __global calc_corr_input *ci,
                        __global calc_corr_result *cr) {

    size_t iseq = get_global_id(0);

    int nseq = ci->nseq;

    // if (iseq >= nseq) return;

    double s, xn, yn, x, y, qx=0., qy=0., r, qr, acx, acy;
    double a = ci->a, b = ci->b, sx = ci->sx, sy = ci->sy;
    int nbit = ci->nbit;
    long i;

    //printf("nbit=%d\n", nbit);
    // printf("a=%f, b=%f, sx=%f, sy=%f, nseq=%d, nbit=%d\n", 
    //        a, b, sx, sy, nseq, nbit);

    // __global ulong *ps = &rndst[iseq];  //(rndst + iseq);

    // for (i = 0; i < 10; i++) {
    //     printf("rndst[%1d][0:4] = %20lu, %20lu, %20lu, %20lu\n", i,
    //            *ps++, *ps++, *ps++, *ps++);
    // }
    // printf("rndst[%1d][0:4] = %20lu, %20lu, %20lu, %20lu\n", iseq,
    //        *ps++, *ps++, *ps++, *ps++);
    // printf("rndst[%1d][0:4] = %20lu, %20lu, %20lu, %20lu\n", iseq,
    //        rndst[nseq*iseq+0], rndst[nseq*iseq+1], rndst[nseq*iseq+2],
    //        rndst[nseq*iseq+3]);
    // printf("iseq=%ld, rndst[0][0]=%lu, rndst[0][1]=%lu, rndst[0][2]=%lu,\n",
    //           iseq, rndst[0][0], rndst[0][1], rndst[0][2]);
    // printf("iseq=%ld, seed=%lu, a=%g, b=%g, sx=%g, sy=%g\n\n", iseq,
    //        ci->seed, a, b, sx, sy);

    r = qr = acx = acy = 0;
    
    for (i = 0; i < ci->nrand; i++) {

        s =  rand64_normal(&rndst[iseq]);    // signal
        xn = rand64_normal(&rndst[iseq]);    // noise
        yn = rand64_normal(&rndst[iseq]);    // noise

        x = sx*(a*s + b*xn);
        y = sy*(a*s + b*yn);

        qx = adc(x, nbit);
        qy = adc(y, nbit);

        r   += x*y;
        acx += x*x;
        acy += y*y;

        qr += qx*qy;

    // if (iseq == 0) {
    //     printf("i=%d, r=%f, acx=%f, acy=%f, s=%f, xn=%f, yn=%f, x=%f, y=%f, " \
    //            "qx=%f, qy=%f, qr=%f\n",                              \
    //            i, r, acx, acy, s, xn, yn, x, y, qx, qy, qr);
    //     //printf("\n");
    // }

    }   /* for (i = 0; i < nrand; i++) */

    cr[iseq].r = r;
    cr[iseq].qr = qr;
    cr[iseq].acx = acx;
    cr[iseq].acy = acy;

    // if (iseq == 0) {
    //     
    //     printf("iseq=%ld, a=%f, b=%f, sx=%f, sy=%f\n\n", iseq, a, b, sx, sy);
    //     printf("iseq=%d, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", \
    //            iseq, r, qr, acx, acy, s, xn, yn, x, y, qx, qy, qr);
    //     //printf("\n");
    // }
    
}



double adc(double x, int nbit) {
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

    /* 
     * nlev = 2^(nbit-1)-1: Number of quantization levels 
     */
    nlev = (((uint)1) << (nbit-1)) - 1;

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




/*
 * Rotate left the 64-bit word x by k bits 
 */
static inline ulong rotl(const ulong x, int k) {
	return (x << k) | (x >> (64 - k));
}



/* 
 * This is xoshiro256** 1.0, our all-purpose, rock-solid generator. It has
 * excellent (sub-ns) speed, a state (256 bits) that is large enough for
 * any parallel application, and it passes all tests we are aware of.
 *
 *  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)
 */
ulong rand64_bits(__global ulong state[]) {

    __global ulong *s = state; /* Just for brevity */
    
	const ulong result_starstar = rotl(s[1] * 5, 7) * 9;
	const ulong t = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;

	s[3] = rotl(s[3], 45);

	return result_starstar;
}



/*
 * rand64_uniform: returns floating-point double pseudorandom number
 * uniformly distributed over the 0 .. 1 interval.  
 *
 * This is xoshiro256** 1.0, our all-purpose, rock-solid generator. It has
 * excellent (sub-ns) speed, a state (256 bits) that is large enough for
 * any parallel application, and it passes all tests we are aware of.
 *
 *  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)
 */
double rand64_uniform(__global ulong state[]) {

    __global ulong *s = state; /* Just for brevity */
    
	const ulong result_starstar = rotl(s[1] * 5, 7) * 9;
	const ulong t = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;

	s[3] = rotl(s[3], 45);

	return (double) result_starstar * (inv_ULONG_MAX);
}



/*
 * rand64_normal: returns floating-point double pseudorandom number
 * from standard normal distribution, i.e. with mu=0, sigma=1.   
 *
 * Unfortunately, of two generated normal randoms, only one is returned; 
 * the other one (z1) is lost.
 *
 *  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)
 */
double rand64_normal(__global ulong state[]) {

    const double two_pi = 2.0*3.14159265358979323846;
    double u1, u2, z0, z1;

    u1 = rand64_uniform(state);
    u2 = rand64_uniform(state);
    
    /*
     * The native_ functions apparently do not to work on AMD GPU
     */
    z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
    // z0 = sqrt(-2.0 * log(u1)) * native_cos(two_pi * u2);
    // z1 = sqrt(-2.0 * log(u1)) * native_sin(two_pi * u2);    /* Is lost */
    
    return z0;
    
}





