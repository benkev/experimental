/*
 * cpu_rand_unif32.c
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <locale.h>

#define UINT_MAX	0xffffffff
#define inv_UINT_MAX  1./((double) ULONG_MAX)

typedef unsigned int uint;
typedef unsigned long ulong;

typedef union intdouble {
    double d;
    uint32_t i[2];
} intdouble;


typedef union intfloat {
    float f;
    uint32_t i;
} intfloat;



uint32_t splitmix32(uint32_t *xstate);
void rand32_init(uint32_t seed, int nsequence, uint32_t *state);
void rand32_initn(uint32_t seed, int nstates, uint32_t state[nstates]);
uint32_t rand32_bits(uint32_t *state);
void jump_ahead(uint32_t *state);
double rand64_uniform(uint32_t *state);
float rand32_uniform(uint32_t *state);
float rand32_normal(uint32_t *state);


/* ulong const Nwitem = 1, Nwgroup = 1; */
/* uint const nrnd=100; */

ulong const Nwitem = 64, Nwgroup = 10;
uint const nrnd=10000;

/* ulong const Nwitem = 256, Nwgroup = 128; */
/* uint const nrnd=10000; */


int main() {
    
    setlocale(LC_NUMERIC, "");

    uint32_t seed = 90752;

    ulong i, nrnd_total = Nwitem * Nwgroup * nrnd;
    //float *rndu = (float *) malloc(nrnd_total*sizeof(float));
    double *rndu = (double *) malloc(nrnd_total*sizeof(double));
    if (rndu == NULL) {
        printf("Error! rndu[nrnd_total] not allocated.");
        exit(0);
    }

    /*
     * Initialize random states
     */
    /* Number of parallel random number sequences */
    ulong Nproc = Nwitem*Nwgroup;
    uint32_t *rndst = (uint32_t *) malloc(Nproc*4*sizeof(uint32_t));
    if (rndst == NULL) {
        printf("Error! rndst[Nproc] not allocated.");
        exit(0);
    }
    
    rand32_initn(seed, Nproc, rndst);
    
    ulong memst = Nproc*sizeof(uint32_t);
    ulong memrnd = nrnd_total*sizeof(ulong);
    printf("Work items: %'ld\n", Nwitem);
    printf("Work groups: %'ld\n", Nwgroup);
    printf("Processes (treads) total : %'ld\n", Nproc);
    printf("Numbers in one thread nrnd: %'ld\n", nrnd);
    printf("Total numbers nrnd_total: %'ld\n", nrnd_total);
    printf("Memory for %'ld PRNG states: %'ld B = %'.1f KiB = %'.1f MiB\n", \
           Nproc, memst, memst/1024., memst/1024./1024.);
    printf("Memory for %'ld 32-bit random numbers: " \
           "%'ld B = %'.1f KiB = %'.1f MiB = %'.2f GiB\n",
           nrnd_total, memrnd, memrnd/1024., memrnd/1024./1024.,    \
           memrnd/1024./1024./1024.);

    printf("Application rndst:\n");
    for (int i=0; i<4; i++)
        printf("%8x %8x %8x %8x\n", rndst[4*i+0], rndst[4*i+1], \
               rndst[4*i+2], rndst[4*i+3]);

    /*
     * rndu_local points into rndu[nrnd_total] treating it as rndu[Nproc][nrnd].
     */
    /* float *rndu_local; */
    /* float rnd, rndmin = 1e20, rndmax = 1e-20; */
    /* float *rndu_avg = (float *) calloc(Nproc, sizeof(float)); */
    
    double *rndu_local;
    double rnd, rndmin = 1e20, rndmax = 1e-20;
    double *rndu_avg = (double *) calloc(Nproc, sizeof(double));
    
    for (size_t iseq=0; iseq<Nproc; iseq++) {
        rndu_avg[iseq] = 0.;
        rndu_local = rndu + iseq*nrnd;
        for (ulong irnd=0; irnd<nrnd; irnd++) {

            // rnd = rand32_uniform(&rndst[4*iseq]);
            rnd = rand64_uniform(&rndst[4*iseq]);

            rndu_avg[iseq] += rnd;
            if (rnd < rndmin) rndmin = rnd;
            if (rnd > rndmax) rndmax = rnd;
            *rndu_local++ = rnd;
        }
        rndu_avg[iseq] /= (double) nrnd; /* Save average over nrand */
    }
    
    // return 0;
    
    printf("Application rndu:\n");
    //    for (int i=265; i<301; i++) {
    for (int i=0; i<50; i++) {
        printf(" %14.7e  %14.7e  %14.7e  %14.7e hex:", rndu[i], rndu[nrnd+i], \
               rndu[2*nrnd+i], rndu[3*nrnd+i]);
        printf(" %10a  %10a  %10a  %10a \n", rndu[i], rndu[nrnd+i], \
               rndu[2*nrnd+i], rndu[3*nrnd+i]);
    }
    
    printf("Application rndu averages:\n");
    printf(" %14.7e  %14.7e  %14.7e  %14.7e \n", rndu_avg[0], rndu_avg[1], \
               rndu_avg[2], rndu_avg[3]);

    printf("\n");
    printf("Application rndmin =  %14.7e rndmax =  %14.7e\n", rndmin, rndmax);

    /*
     * Save the generated numbers in a text file
     */
    FILE* fh = fopen("cpu_rand_unif32.txt", "w");

    fprintf(fh, "%d\n", nrnd_total);
    for (i=0; i<nrnd_total; i++)
        // fprintf(fh, "%15.8e\n", rndu[i]);
        fprintf(fh, "%22.16e\n", rndu[i]);
    fclose(fh);
    
    return 0;
}



/*
 * splitmix32 is a PRNG with 32 bits of state and the period 2^32.
 * From 
 * https://stackoverflow.com/questions/17035441/ \
 *        looking-for-decent-quality-prng-with-only-32-bits-of-state
 */
uint32_t splitmix32(uint32_t *xstate) {
    uint32_t z = (*xstate += 0x9e3779b9);
    z ^= z >> 15; // 16 for murmur3 (???)
    z *= 0x85ebca6b;
    z ^= z >> 13;
    z *= 0xc2b2ae3d; // 0xc2b2ae35 for murmur3  (???)
    return z ^= z >> 16;
}

/*
 * The state must be seeded so that it is not everywhere zero. If you have
 * a 32-bit seed, we suggest to seed a splitmix32 generator and use its
 * output to fill s.
 *
 * The text below has been plagiarized from the cuRAND documentation :)
 *
 * The rand32_init() function sets up an initial state allocated by the caller
 * using the given seed and sequence number. Different seeds are guaranteed to
 * produce different starting states and different sequences. The same seed
 * always produces the same state and the same sequence. The state set up will
 * be the state after (2^64) * nsequence.
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
void rand32_init(uint32_t seed, int nsequence, uint32_t *state) {
    
    uint32_t xstate = seed;
    // uint32_t *s = state;
    int iseq;

    state[0] = seed;
    state[1] = splitmix32(&xstate);
    state[2] = splitmix32(&xstate);
    state[3] = splitmix32(&xstate);

    for (iseq=0; iseq<nsequence; iseq++)
        jump_ahead(state);
}



/* 
 * The rand32_initn() function sets up initial states using the given seed in an
 * array state[nstates] allocated by the caller.
 * It is equivalent to nstates calls to rand32_init() with the sequence numbers
 * progressing from 0 to nstates-1.  
 */
void rand32_initn(uint32_t seed, int nstates, uint32_t state[nstates]) {

    uint32_t xstate = seed;
    // uint32_t *s = state;
    int ist;
    
    state[0] = seed;
    state[1] = splitmix32(&xstate);
    state[2] = splitmix32(&xstate);
    state[3] = splitmix32(&xstate);
        
    if (nstates <= 1) return; /* ======================================= >>> */
    
    for (ist=1; ist<nstates; ist++) {
        state[4*ist] = state[4*(ist-1)];
        jump_ahead(&state[4*ist]);    
    }
}


/* 
 * Rotate left the 32-bit word x by k bits 
 */
static inline uint32_t rotl(const uint32_t x, int k) {
	return (x << k) | (x >> (32 - k));
}



/* This is xoshiro128** 1.0, our 32-bit all-purpose, rock-solid generator. It
   has excellent (sub-ns) speed, a state size (128 bits) that is large
   enough for mild parallelism, and it passes all tests we are aware of.

   For generating just single-precision (i.e., 32-bit) floating-point
   numbers, xoshiro128+ is even faster.

   The state must be seeded so that it is not everywhere zero.

   Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org) */

uint32_t rand32_bits(uint32_t *state) {

    const uint32_t result_starstar = rotl(state[0] * 5, 7) * 9;
	const uint32_t t = state[1] << 9;

	state[2] ^= state[0];
	state[3] ^= state[1];
	state[1] ^= state[2];
	state[0] ^= state[3];

	state[2] ^= t;

	state[3] = rotl(state[3], 11);

	return result_starstar;
}




/*
 * rand32_uniform: retutns floating-point pseudorandom number
 * uniformly distributed over the 0 .. 1 interval.  
 *
 * This is xoshiro128** 1.0, our 32-bit all-purpose, rock-solid generator. It
 *  has excellent (sub-ns) speed, a state size (128 bits) that is large
 *  enough for mild parallelism, and it passes all tests we are aware of.
 *
 *  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)
 */
float rand32_uniform(uint32_t *state) {

    intfloat pun;
    const uint32_t result_starstar = rotl(state[0] * 5, 7) * 9;
	const uint32_t t = state[1] << 9;

	state[2] ^= state[0];
	state[3] ^= state[1];
	state[1] ^= state[2];
	state[0] ^= state[3];

	state[2] ^= t;

	state[3] = rotl(state[3], 11);

    pun.i = result_starstar;
    
    /* Set the 8 bits of exponent to 127 */
    pun.i = 0x007fffffU & pun.i; 
    pun.i = 0x3F800000U | pun.i; 
    
    /* printf("pun.i = %08x pun.f = %g\n\n", pun.i, pun.f); */
    
    return pun.f - 1.0f;

    //	return ((double) result_starstar) * inv_ULONG_MAX;   
}




/*
 * rand64_uniform: retutns floating-point double pseudorandom number
 * uniformly distributed over the 0 .. 1 interval.  
 *
 * This is xoshiro128** 1.0, our 32-bit all-purpose, rock-solid generator. It
 *  has excellent (sub-ns) speed, a state size (128 bits) that is large
 *  enough for mild parallelism, and it passes all tests we are aware of.
 *
 *  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)
 */
double rand64_uniform(uint32_t *state) {

    intdouble pun;

    uint32_t result_starstar = rotl(state[0] * 5, 7) * 9;
	uint32_t t = state[1] << 9;

	state[2] ^= state[0];
	state[3] ^= state[1];
	state[1] ^= state[2];
	state[0] ^= state[3];

	state[2] ^= t;

	state[3] = rotl(state[3], 11);

    pun.i[0] = result_starstar;
    
    result_starstar = rotl(state[0] * 5, 7) * 9;
	t = state[1] << 9;

	state[2] ^= state[0];
	state[3] ^= state[1];
	state[1] ^= state[2];
	state[0] ^= state[3];

	state[2] ^= t;

	state[3] = rotl(state[3], 11);

    pun.i[1] = result_starstar;
    
    /* Set the 11 bits of exponent to 127 */
    pun.i[1] = 0x000fffffU & pun.i[1]; 
    pun.i[1] = 0x3FF00000U | pun.i[1]; 
    
    /* printf("pun.i = %08x pun.f = %g\n\n", pun.i, pun.f); */
    
    return pun.d - 1.0;

    //	return ((double) result_starstar) * inv_ULONG_MAX;   
}




/* This is the jump function for the generator. It is equivalent
   to 2^64 calls to next(); it can be used to generate 2^64
   non-overlapping subsequences for parallel computations.

 * Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org) */

void jump_ahead(uint32_t *state) {
    
	static const uint32_t JUMP[] = { 0x8764000b, 0xf542d2d3,
                                     0x6fa035c3, 0x77f2db5b };
	uint32_t s0 = 0;
	uint32_t s1 = 0;
	uint32_t s2 = 0;
	uint32_t s3 = 0;
    
	for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
		for(int b = 0; b < 32; b++) {
			if (JUMP[i] & UINT32_C(1) << b) {
				s0 ^= state[0];
				s1 ^= state[1];
				s2 ^= state[2];
				s3 ^= state[3];
			}
            rand32_bits(state);
		}
		
	state[0] = s0;
	state[1] = s1;
	state[2] = s2;
	state[3] = s3;
}


















