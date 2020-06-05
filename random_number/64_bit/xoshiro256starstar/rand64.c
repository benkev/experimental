#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
/* #include <iostream> */
/* using namespace std; */

/* typedef unsigned long long ulong; */
/* typedef unsigned int uint; */

typedef struct rand64State {
    uint64_t s[4];
} rand64State;


uint64_t splitmix64(uint64_t *state);
void rand64_init(uint64_t seed, uint nsequence, rand64State *state);
void rand64_init(uint64_t seed, uint nstates, rand64State *state[]);
uint64_t rand64_bits(rand64State *state);
void jump_ahead(rand64State *state);
double rand64_uniform(rand64State *state);
double rand64_normal(rand64State *state);

int main() {
    
    uint64_t xstate = 12345;
    uint64_t seed = 9876;
    rand64State rndst;      /* xoshiro256** state, 4 64-bit words */
    rand64State rndst1;      /* xoshiro256** state, 4 64-bit words */

    /* printf("splitmix64 output:\n"); */
    /* for (int i=0; i<10; i++) { */
    /*     printf("%3d %21lu \n", i, splitmix64(&xstate)); */
    /* } */
    /* printf("\n\n"); */

    
    rand64_init(seed, 0, &rndst);
    rand64_init(seed, 2, &rndst1);
    
    printf("rndst:  ");
    for (int i=0; i<4; i++) {
        printf("%21lu \n", rndst.s[i]);
    }
    printf("\n");
    printf("rndst1: ");
    for (int i=0; i<4; i++) {
        printf("%21lu \n", rndst1.s[i]);
    }
    printf("\n\n");

    /* 
     * O'Neil example of 5 duplicates is generated by:
     */
    /* rndst.s[0] = 0x216b13fa05d2c01e; */
    /* rndst.s[1] = 0x0165f953d45afc83; */
    /* rndst.s[2] = 0x7557a4909bdae724; */
    /* rndst.s[3] = 0x10718ed2c884dc75; */

    
    printf("xoshiro256**:\n");
    for (int i=0; i<160; i++) {
        /* printf("State: %21lu ", state); */
        /* printf("%3d %21lu \n", i, rand64_bits(&rndst)); */
        printf("0x%016llx 0x%016llx \n", rand64_bits(&rndst), \
               rand64_bits(&rndst1));
    }
    printf("\n\n");
    
    return 0;
}



/*
 * splitmix64 is a PRNG with 64 bits of state and the period 2^64.
 * Written in 2015 by Sebastiano Vigna (vigna@acm.org) 
 */
uint64_t splitmix64(uint64_t *xstate) {
	uint64_t z = (*xstate += 0x9e3779b97f4a7c15);
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
void rand64_init(uint64_t seed, uint nsequence, rand64State *state) {
    
    uint64_t xstate = seed;

    state->s[0] = seed;
    state->s[1] = splitmix64(&xstate);
    state->s[2] = splitmix64(&xstate);
    state->s[3] = splitmix64(&xstate);

    for (int i=0; i<nsequence; i++)
        jump_ahead(state);
}


void rand64_init(uint64_t seed, uint nstates, rand64State *state[]) {
    
    uint64_t xstate = seed;

    state[0]->s[0] = seed;
    state[0]->s[1] = splitmix64(&xstate);
    state[0]->s[2] = splitmix64(&xstate);
    state[0]->s[3] = splitmix64(&xstate);
    jump_ahead(state[0]);
        
    if (nstates <= 1) return; /* ======================================= >>> */
    
    for (int ist=1; ist<nstates; ist++) {
        state[ist]->s[0] = splitmix64(&xstate);
        state[ist]->s[1] = splitmix64(&xstate);
        state[ist]->s[2] = splitmix64(&xstate);
        state[ist]->s[3] = splitmix64(&xstate);
        jump_ahead(state[ist]);
    }
}


/*
 * Rotate left the 64-bit word x by k bits 
 */
static inline uint64_t rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}



/* 
 * This is xoshiro256** 1.0, our all-purpose, rock-solid generator. It has
 * excellent (sub-ns) speed, a state (256 bits) that is large enough for
 * any parallel application, and it passes all tests we are aware of.
 *
 *  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)
 */
uint64_t rand64_bits(rand64State *state) {

    uint64_t *s = state->s; /* Just for brevity */
    
	const uint64_t result_starstar = rotl(s[1] * 5, 7) * 9;
	const uint64_t t = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;

	s[3] = rotl(s[3], 45);

	return result_starstar;
}



/* 
 * This is the jump function for the generator. It is equivalent
 * to 2^128 calls to next(); it can be used to generate 2^128
 * non-overlapping subsequences for parallel computations.
 *
 *  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)
 */
void jump_ahead(rand64State *state) {
	static const uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, \
                                     0xa9582618e03fc9aa, 0x39abdc4529b1661c };
	uint64_t s0 = 0;
	uint64_t s1 = 0;
	uint64_t s2 = 0;
	uint64_t s3 = 0;
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




















