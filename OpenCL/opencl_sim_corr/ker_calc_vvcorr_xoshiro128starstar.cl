/*
 * calc_vvcorr.cl
 *
 * Calculate correlation of very long random sequences on a GPU.
 * 
 *
 */

typedef union intdouble {
    double d;
    uint i[2];
} intdouble;

typedef union intfloat {
    float f;
    uint i;
} intfloat;


double adc(double x, int nbit);
ulong rand64l_bits(__global ulong state[]);
double rand64l_uniform(__global ulong state[]);
double rand64l_normal(__global ulong state[]);
static inline ulong rot64l(const ulong x, int k);

uint splitmix32(uint *xstate);
//void rand32_init(uint seed, int nsequence, uint *state);
//void rand32_initn(uint seed, int nstates, uint state[nstates]);
uint rand32_bits(__global uint *state);
void jump_ahead(uint *state);
double rand64_uniform(__global uint *state);
float rand32_uniform(__global uint *state);
//float rand32_normal(__global uint *state);
double rand64_normal(__global uint *state);



__constant double inv_ULONG_MAX = 1./((double) ULONG_MAX);

/*
 * Calculate correlation of very long random sequences on a GPU.
 *
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
        
    }   /* for (i = 0; i < nrand; i++) */

    cr[iseq].r = r;
    cr[iseq].qr = qr;
    cr[iseq].acx = acx;
    cr[iseq].acy = acy;  
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
 * splitmix32 is a PRNG with 32 bits of state and the period 2^32.
 * From 
 * https://stackoverflow.com/questions/17035441/ \
 *        looking-for-decent-quality-prng-with-only-32-bits-of-state
 */
uint splitmix32(uint *xstate) {
    uint z = (*xstate += 0x9e3779b9);
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
// void rand32_init(uint seed, int nsequence, uint *state) {
    
//     uint xstate = seed;
//     // uint *s = state;
//     int iseq;

//     state[0] = seed;
//     state[1] = splitmix32(&xstate);
//     state[2] = splitmix32(&xstate);
//     state[3] = splitmix32(&xstate);

//     for (iseq=0; iseq<nsequence; iseq++)
//         jump_ahead(state);
// }



// /* 
//  * The rand32_initn() function sets up initial states using the given seed in an
//  * array state[nstates] allocated by the caller.
//  * It is equivalent to nstates calls to rand32_init() with the sequence numbers
//  * progressing from 0 to nstates-1.  
//  */
// void rand32_initn(uint seed, int nstates, uint state[nstates]) {

//     uint xstate = seed;
//     // uint *s = state;
//     int ist;
    
//     state[0] = seed;
//     state[1] = splitmix32(&xstate);
//     state[2] = splitmix32(&xstate);
//     state[3] = splitmix32(&xstate);
        
//     if (nstates <= 1) return; /* ======================================= >>> */
    
//     for (ist=1; ist<nstates; ist++) {
//         state[4*ist] = state[4*(ist-1)];
//         jump_ahead(&state[4*ist]);    
//     }
// }


/* 
 * Rotate left the 32-bit word x by k bits 
 */
static inline uint rotl(const uint x, int k) {
	return (x << k) | (x >> (32 - k));
}



/* This is xoshiro128** 1.0, our 32-bit all-purpose, rock-solid generator. It
   has excellent (sub-ns) speed, a state size (128 bits) that is large
   enough for mild parallelism, and it passes all tests we are aware of.

   For generating just single-precision (i.e., 32-bit) floating-point
   numbers, xoshiro128+ is even faster.

   The state must be seeded so that it is not everywhere zero.

   Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org) */

uint rand32_bits(uint *state) {

    const uint result_starstar = rotl(state[0] * 5, 7) * 9;
	const uint t = state[1] << 9;

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
float rand32_uniform(uint *state) {

    intfloat pun;
    const uint result_starstar = rotl(state[0] * 5, 7) * 9;
	const uint t = state[1] << 9;

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
double rand64_uniform(uint *state) {

    intdouble pun;

    uint result_starstar = rotl(state[0] * 5, 7) * 9;
	uint t = state[1] << 9;

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

void jump_ahead(uint *state) {
    
	static const uint JUMP[] = { 0x8764000b, 0xf542d2d3,
                                     0x6fa035c3, 0x77f2db5b };
	uint s0 = 0;
	uint s1 = 0;
	uint s2 = 0;
	uint s3 = 0;
    
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





//============================ 64-bit ARITHMETIC ============================

/*
 * Rotate left the 64-bit word x by k bits 
 */
static inline ulong rot64l(const ulong x, int k) {
	return (x << k) | (x >> (64 - k));
}



/* 
 * This is xoshiro256** 1.0, our all-purpose, rock-solid generator. It has
 * excellent (sub-ns) speed, a state (256 bits) that is large enough for
 * any parallel application, and it passes all tests we are aware of.
 *
 *  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)
 */
ulong rand64l_bits(__global ulong state[]) {

    __global ulong *s = state; /* Just for brevity */
    
	const ulong result_starstar = rot64l(s[1] * 5, 7) * 9;
	const ulong t = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;

	s[3] = rot64l(s[3], 45);

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
double rand64l_uniform(__global ulong state[]) {

    __global ulong *s = state; /* Just for brevity */
    
	const ulong result_starstar = rot64l(s[1] * 5, 7) * 9;
	const ulong t = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;

	s[3] = rot64l(s[3], 45);

	return (double) result_starstar * (inv_ULONG_MAX);
}



/*
 * rand64l_normal: returns floating-point double pseudorandom number
 * from standard normal distribution, i.e. with mu=0, sigma=1.   
 *
 * Unfortunately, of two generated normal randoms, only one is returned; 
 * the other one (z1) is lost.
 *
 *  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)
 */
double rand64l_normal(__global ulong state[]) {

    const double two_pi = 2.0*3.14159265358979323846;
    double u1, u2, z0, z1;

    u1 = rand64l_uniform(state);
    u2 = rand64l_uniform(state);
    
    /*
     * The native_ functions apparently do not to work on AMD GPU
     */
    z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
    // z0 = sqrt(-2.0 * log(u1)) * native_cos(two_pi * u2);
    // z1 = sqrt(-2.0 * log(u1)) * native_sin(two_pi * u2);    /* Is lost */
    
    return z0;
    
}





