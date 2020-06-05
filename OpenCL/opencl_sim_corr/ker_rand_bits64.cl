/*
 * ker_rand_bits64.cl
 *
 *
 */
//#pragma OPENCL EXTENSION cl_amd_printf

//#include "/home/benkev/experimental/OpenCL/opencl_sim_corr/vanvleck_structs.h"
#include "vanvleck_structs.h"

#define inv_ULONG_MAX  1./((double) ULONG_MAX)

double adc(double x, int nbit);
ulong rand64_bits(__global rand64State *state);
double rand64_uniform(__global rand64State *state);
double rand64_normal(__global rand64State *state);


__kernel void genrand(__global rand64State *rndst, uint nrnd, \
                      __global ulong *rndu) {

    size_t iseq = get_global_id(0);

    __global ulong *rndu_local = rndu + iseq*nrnd;
    
    //printf("iseq = %ld\n", iseq);
    //printf("iseq = %ld\n", iseq);
    if (iseq == 0) printf("ULONG_MAX = %'lu, inv_ULONG_MAX = %g\n", \
                          ULONG_MAX, inv_ULONG_MAX);
    //     printf("iseq = %d, rndst = 0x%lx\n", iseq, rndst);
    //printf("iseq = %ld\n", iseq); //, rndu = 0x%lx\n", iseq, rndu);
    //     printf("iseq = %d, rndu_local = 0x%lx\n", iseq, rndu_local);
    //     printf("Kernel rndst:\n");
    // for (int i=0; i<4; i++)
    //     printf("%20ld %20ld %20ld %20ld\n", rndst[0].s[i], rndst[1].s[i], \
    //            rndst[2].s[i],  rndst[3].s[i]);
    //}
    
    for (int irnd=0; irnd<nrnd; irnd++)
        *rndu_local++ = rand64_bits(&rndst[iseq]);

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
ulong rand64_bits(__global rand64State *state) {

    __global ulong *s = state->s; /* Just for brevity */
    
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
 * rand64u: retutns floating-point double pseudorandom number
 * uniformly distributed over the 0 .. 1 interval.  
 *
 * This is xoshiro256** 1.0, our all-purpose, rock-solid generator. It has
 * excellent (sub-ns) speed, a state (256 bits) that is large enough for
 * any parallel application, and it passes all tests we are aware of.
 *
 *  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)
 */
ulong rand64u(__global rand64State *state) {

    __global ulong *s = state->s; /* Just for brevity */
    
	const ulong result_starstar = rotl(s[1] * 5, 7) * 9;
	const ulong t = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;

	s[3] = rotl(s[3], 45);

	return (double) result_starstar * inv_ULONG_MAX;
}


