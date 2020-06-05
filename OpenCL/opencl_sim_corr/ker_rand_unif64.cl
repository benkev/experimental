/*
 * ker_rand_unif64.cl
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
double randu64(__global rand64State *state);

/*
 * Inputs:
 *   rndst[Nproc]: array of Nproc 256-bit PRNG states, where Nproc is the
 *                 number of parallel processes, Nproc = Ngroup * Nwitem.
 *   nrnd: number of the random numbers generated in each of the Nproc parallel
 *         processes.
 * Output:
 *   rndu[Nproc,nrnd]: generated random numbers.
 *
 */
__kernel void genrand(__global rand64State *rndst, uint nrnd, \
                      __global double *rndu) {

    size_t iproc = get_global_id(0);  /* Unique process number */

    __global double *rndu_local = rndu + iproc*nrnd;

    for (int irnd=0; irnd<nrnd; irnd++)
        *rndu_local++ = randu64(&rndst[iproc]);

    // size_t gid = get_group_id(0);
    // size_t lid = get_local_id(0);

    // printf("Global ID %ld, Group %ld, Item %ld\n", iproc, gid, lid);
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
 * randu64: retutns floating-point double pseudorandom number
 * uniformly distributed over the 0 .. 1 interval.  
 *
 * This is xoshiro256** 1.0, our all-purpose, rock-solid generator. It has
 * excellent (sub-ns) speed, a state (256 bits) that is large enough for
 * any parallel application, and it passes all tests we are aware of.
 *
 *  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)
 */
double randu64(__global rand64State *state) {

    __global ulong *s = state->s; /* Just for brevity */
    
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


