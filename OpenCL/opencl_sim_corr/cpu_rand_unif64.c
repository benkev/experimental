/*
 * cpu_rand_unif64.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <locale.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define ULONG_MAX	0xffffffffffffffffUL
#define inv_ULONG_MAX  1./((double) ULONG_MAX)


#include "vanvleck_structs.h"
#include "simulate_vvcorr.h"

cl_double randu64(rand64State *state);

/* typedef unsigned long long ulong; */
/* typedef unsigned int uint; */

/* typedef struct rand64State { */
/*     uint64_t s[4]; */
/* } rand64State; */

/* uint64_t splitmix64(uint64_t *state); */
/* void rand64_init(uint64_t seed, uint nsequence, rand64State *state); */
/* void rand64_initn(uint64_t seed, uint nstates, rand64State state[nstates]);*/
/* uint64_t rand64_bits(rand64State *state); */
/* void jump_ahead(rand64State *state); */
/* double rand64_uniform(rand64State *state); */
/* double rand64_normal(rand64State *state); */


cl_ulong const Nwitem = 128, Nwgroup = 10;
cl_uint const nrnd=10000;
/* cl_ulong const Nwitem = 256, Nwgroup = 400; */
/* cl_uint const nrnd=10000; */


int main() {
    
    setlocale(LC_NUMERIC, "");

    //uint64_t xstate = 12345;
    //uint64_t seed = 9876;
    printf("ULONG_MAX = %'lu, inv_ULONG_MAX = %g\n", ULONG_MAX, inv_ULONG_MAX);

    uint64_t seed = 90752;

    cl_ulong nrnd_total = Nwitem * Nwgroup * nrnd;
    /* cl_ulong *rndb = (cl_ulong *) malloc(nrnd_total*sizeof(cl_ulong)); */
    /* if (rndb == NULL) { */
    /*     printf("Error! rndb[nrnd_total] not allocated."); */
    /*     exit(0); */
    /* } */
    cl_double *rndu = (cl_double *) malloc(nrnd_total*sizeof(cl_double));
    if (rndu == NULL) {
        printf("Error! rndu[nrnd_total] not allocated.");
        exit(0);
    }

    /*
     * Initialize random states
     */
    /* Number of parallel random number sequences */
    cl_ulong Nproc = Nwitem*Nwgroup;
    rand64State *rndst = (rand64State *) malloc(Nproc*sizeof(rand64State));
    if (rndst == NULL) {
        printf("Error! rndst[Nproc] not allocated.");
        exit(0);
    }
    rand64_initn(seed, Nproc, rndst);
    
    cl_ulong memst = Nproc*sizeof(rand64State);
    cl_ulong memrnd = nrnd_total*sizeof(cl_ulong);
    printf("Work items: %'ld\n", Nwitem);
    printf("Work groups: %'ld\n", Nwgroup);
    printf("Processes (treads) total : %'ld\n", Nproc);
    printf("Numbers in one thread nrnd: %'ld\n", nrnd);
    printf("Total numbers nrnd_total: %'ld\n", nrnd_total);
    printf("Memory for %'ld PRNG states: %'ld B = %'.1f KiB = %'.1f MiB\n", \
           Nproc, memst, memst/1024., memst/1024./1024.);
    printf("Memory for %'ld 64-bit random numbers: " \
           "%'ld B = %'.1f KiB = %'.1f MiB = %'.2f GiB\n",
           nrnd_total, memrnd, memrnd/1024., memrnd/1024./1024.,    \
           memrnd/1024./1024./1024.);

    printf("Application rndst:\n");
    for (int i=0; i<4; i++)
        printf("%20ld %20ld %20ld %20ld\n", rndst[0].s[i], rndst[1].s[i], \
               rndst[2].s[i], rndst[3].s[i]);

    /*
     * Generate random numbers
     */
    /* cl_ulong *rndb_local; */
    
    /* for (size_t iseq=0; iseq<Nproc; iseq++) { */
    /*     rndb_local = rndb + iseq*nrnd; */
    /*     for (cl_ulong irnd=0; irnd<nrnd; irnd++) */
    /*         *rndb_local++ = rand64_bits(&rndst[iseq]); */
    /* } */

    /* printf("Application rndb:\n"); */
    /* for (int i=265; i<301; i++) */
    /*     printf("%20lld %20lld %20lld %20lld \n", rndb[i], rndb[nrnd+i], \ */
    /*            rndb[2*nrnd+i], rndb[3*nrnd+i]); */

    /*
     * rndu_local points into rndu[nrnd_total] treating it as rndu[Nproc][nrnd].
     */
    cl_double *rndu_local;
    cl_double rnd, rndmin = 1e20, rndmax = 1e-20;
    cl_double *rndu_avg = (cl_double *) calloc(Nproc, sizeof(cl_double));
    
    for (size_t iseq=0; iseq<Nproc; iseq++) {
        rndu_avg[iseq] = 0.;
        rndu_local = rndu + iseq*nrnd;
        for (cl_ulong irnd=0; irnd<nrnd; irnd++) {
            rnd = randu64(&rndst[iseq]);
            rndu_avg[iseq] += rnd;
            if (rnd < rndmin) rndmin = rnd;
            if (rnd > rndmax) rndmax = rnd;
            *rndu_local++ = rnd;
        }
        rndu_avg[iseq] /= (cl_double) nrnd; /* Save average over nrand */
    }
    
    printf("Application rndu:\n");
    for (int i=265; i<301; i++)
        printf("%20.16f %20.16f %20.16f %20.16f \n", rndu[i], rndu[nrnd+i], \
               rndu[2*nrnd+i], rndu[3*nrnd+i]);
    
    printf("Application rndu averages:\n");
    printf("%20.16f %20.16f %20.16f %20.16f \n", rndu_avg[0], rndu_avg[1], \
               rndu_avg[2], rndu_avg[3]);

    printf("\n");
    printf("Application rndmin = %20.16f rndmax = %20.16f\n", rndmin, rndmax);



    /* rand64State rndst;      /\* xoshiro256** state, 4 64-bit words *\/ */
    /* rand64State rndst1;     /\* xoshiro256** state, 4 64-bit words *\/ */
    /* rand64State *rndst2 = (rand64State *) malloc(10*sizeof(rand64State)); */
    
    /* rand64_init(seed, 0, &rndst); */
    /* rand64_init(seed, 6, &rndst1); */
    
    /* rand64_initn(seed, 10, rndst2); */
    
    /* printf("rndst:  "); */
    /* for (int i=0; i<4; i++) { */
    /*     printf("%21lu \n", rndst.s[i]); */
    /* } */
    /* printf("\n"); */
    /* printf("rndst1: "); */
    /* for (int i=0; i<4; i++) { */
    /*     printf("%21lu \n", rndst1.s[i]); */
    /* } */
    /* printf("\n\n"); */
    
    /* printf("rndst2[0]:  "); */
    /* for (int i=0; i<4; i++) { */
    /*     printf("%21lu \n", rndst2[0].s[i]); */
    /* } */
    /* printf("\n"); */
    /* printf("rndst2[2]: "); */
    /* for (int i=0; i<4; i++) { */
    /*     printf("%21lu \n", rndst2[6].s[i]); */
    /* } */
    /* printf("\n\n"); */

    /*
     * O'Neil examples of 5 duplicates is generated by the initial states
     * http://www.pcg-random.org/posts/implausible-output-from-xoshiro256.html
     */
    /* 0xf1ece002a3004704  repeats 5 times for: */
    /* rndst.s[0] = 0x216b13fa05d2c01e; */
    /* rndst.s[1] = 0x0165f953d45afc83; */
    /* rndst.s[2] = 0x7557a4909bdae724; */
    /* rndst.s[3] = 0x10718ed2c884dc75; */

    /* 0xbf18dec850198308 repeats 5 times for: */
    /* rndst.s[0] = 0x1a3e1765f0739e39; */
    /* rndst.s[1] = 0xf5d7f1bb420e645d; */
    /* rndst.s[2] = 0x16a5269645b3a76f; */
    /* rndst.s[3] = 0xbb126a5fd07a5f1e; */

    /* printf("rndst:  "); */
    /* for (int i=0; i<4; i++) { */
    /*     printf("%016llx \n", rndst.s[i]); */
    /* } */
    /* printf("\n"); */
    
    /* printf("xoshiro256**:\n"); */
    /* for (int i=0; i<160; i++) { */
    /*     printf("- %016llx %016llx \n", rand64_bits(&rndst), \ */
    /*            rand64_bits(&rndst1)); */
    /*     printf("+ %016llx %016llx \n", rand64_bits(&rndst2[0]), \ */
    /*            rand64_bits(&rndst2[6])); */
    /* } */
    /* printf("\n\n"); */
    
    return 0;
}



/*
 * splitmix64 is a PRNG with 64 bits of state and the period 2^64.
 * Written in 2015 by Sebastiano Vigna (vigna@acm.org) 
 */
cl_ulong splitmix64(cl_ulong *xstate) {
	cl_ulong z = (*xstate += 0x9e3779b97f4a7c15);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
	return z ^ (z >> 31);
}



/*
 * The state must be seeded so that it is not everywhere zero. If you have
 * a 64-bit seed, we suggest to seed a splitmix64 generator and use its
 * output to fill s.
 *
 * The text below has been plagiarized from the cuRAND documentation :)
 *
 * The rand64_init() function sets up an initial state allocated by the caller
 * using the given seed and sequence number. Different seeds are guaranteed to
 * produce different starting states and different sequences. The same seed
 * always produces the same state and the same sequence. The state set up will
 * be the state after (2^128) * nsequence.
 *
 * Sequences generated with different seeds usually do not have statistically
 * correlated values, but some choices of seeds may give statistically
 * correlated sequences. Sequences generated with the same seed and different
 * sequence numbers will not have statistically correlated values.
 *
 * For the highest quality parallel pseudorandom number generation, each
 * experiment should be assigned a unique seed. Within an experiment, each
 * thread of computation should be assigned a unique sequence number. If an
 * experiment spans multiple kernel launches, it is recommended that threads
 * between kernel launches be given the same seed, and sequence numbers be
 * assigned in a monotonically increasing way. If the same configuration of
 * threads is launched, random state can be preserved in global memory between
 * launches to avoid state setup time.
 *
 */
void rand64_init(cl_ulong seed, cl_uint nsequence, rand64State *state) {
    
    cl_ulong xstate = seed;
    int iseq;

    state->s[0] = seed;
    state->s[1] = splitmix64(&xstate);
    state->s[2] = splitmix64(&xstate);
    state->s[3] = splitmix64(&xstate);

    for (iseq=0; iseq<nsequence; iseq++)
        jump_ahead(state);
}


/* 
 * The rand64_initn() function sets up initial states using the given seed in an
 * array state[nstates] allocated by the caller.
 * It is equivalent to nstates calls to rand64_init() with the sequence numbers
 * progressing from 0 to nstates-1.  
 */
void rand64_initn(cl_ulong seed, cl_uint nstates, rand64State state[nstates]) {

    cl_ulong xstate = seed;
    int ist;
    
    state[0].s[0] = seed;
    state[0].s[1] = splitmix64(&xstate);
    state[0].s[2] = splitmix64(&xstate);
    state[0].s[3] = splitmix64(&xstate);
        
    if (nstates <= 1) return; /* ======================================= >>> */
    
    for (ist=1; ist<nstates; ist++) {
        state[ist] = state[ist-1];
        jump_ahead(&state[ist]);    
    }
}


/* 
 * Rotate left the 64-bit word x by k bits 
 */
static inline cl_ulong rotl(const cl_ulong x, int k) {
	return (x << k) | (x >> (64 - k));
}



/* 
 * This is xoshiro256** 1.0, our all-purpose, rock-solid generator. It has
 * excellent (sub-ns) speed, a state (256 bits) that is large enough for
 * any parallel application, and it passes all tests we are aware of.
 *
 * Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)
 */
cl_ulong rand64_bits(rand64State *state) {

    cl_ulong *s = state->s; /* Just for brevity */
    
	const cl_ulong result_starstar = rotl(s[1] * 5, 7) * 9;
	const cl_ulong t = s[1] << 17;

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
cl_double randu64(rand64State *state) {

    cl_ulong *s = state->s; /* Just for brevity */
    
	const cl_ulong result_starstar = rotl(s[1] * 5, 7) * 9;
	const cl_ulong t = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;

	s[3] = rotl(s[3], 45);

	return ((double) result_starstar) * inv_ULONG_MAX;
}




/* 
 * This is the jump function for the xoshiro256** 1.0 generator. 
 * It is equivalent to 2^128 calls to next(); 
 * it can be used to generate 2^128 non-overlapping subsequences 
 * for parallel computations.
 *
 *  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)
 */
void jump_ahead(rand64State *state) {
	static const cl_ulong JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, \
                                     0xa9582618e03fc9aa, 0x39abdc4529b1661c };
	cl_ulong s0 = 0;
	cl_ulong s1 = 0;
	cl_ulong s2 = 0;
	cl_ulong s3 = 0;
	for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
		for(int b = 0; b < 64; b++) {
			if (JUMP[i] & UINT64_C(1) << b) {
				s0 ^= state->s[0];
				s1 ^= state->s[1];
				s2 ^= state->s[2];
				s3 ^= state->s[3];
			}
			rand64_bits(state);	
		}
		
	state->s[0] = s0;
	state->s[1] = s1;
	state->s[2] = s2;
	state->s[3] = s3;
}






















