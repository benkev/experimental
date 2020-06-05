/*
 * Compile:
 *
 * gcc -w -std=gnu99  simulate_vvcorr.c rand_lib.c -o simulate_vvcorr -l OpenCL
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "vanvleck_structs.h"
#include "simulate_vvcorr.h"

int const Nwitem = 64, Nwgroup = 36;


int main() {

    cl_uint i;
    FILE *fp;
    char *source_str=NULL;
    size_t source_size;
    fp = fopen("ker_calc_vvcorr.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel file.\n"); exit(1);
    }
    /*
     * Find the kernel file size source_size, allocate memory for it, 
     * and read it in the source_str[source_size] string.
     */
    fseek(fp, 0L, SEEK_END);
    source_size = ftell(fp);
    rewind(fp); /* Reset the file pointer to 'fread' from the beginning */
    source_str = (char*) malloc((source_size + 1)*sizeof(char));
    fread( source_str, sizeof(char), source_size, fp);
    source_str[source_size] = '\0';
    fclose( fp );
    
    /*
     * Initialize random states
     */

    /* Number of parallel random number sequences */
    int nrndst = Nwitem*Nwgroup;
    cl_uint seed = 90752;
    rand64State *rndst = (rand64State *) malloc(nrndst*sizeof(rand64State));
    rand64_initn(seed, nrndst, rndst);

    /*
     * Get platform and device information
     */
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    
    //cl_int ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

    printf("ret_num_platforms=%u, ret=%u\n", ret_num_platforms, ret);
    
    // ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, 
    //        NULL, &ret_num_devices);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, 
                         &device_id, &ret_num_devices);

    printf("ret_num_devices=%u, ret=%u\n", ret_num_devices, ret);
    
    /* Create an OpenCL context: default properties, for 1 device, no callback,
     * no user_data */
    cl_context context =                                            \
        clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    
    /* Create a command queue for one device */
    cl_command_queue cmd_queue = \
        clCreateCommandQueue(context, device_id, 0, &ret);

    /* Create memory buffer ON THE DEVICE for the PRNG states  */
    cl_mem rndst_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                          nrndst*sizeof(rand64State), NULL, &ret);
    
    /* Copy the PRNG states to their respective memory buffer on device */
    ret = clEnqueueWriteBuffer(cmd_queue, rndst_mem, CL_TRUE, 0,
            nrndst*sizeof(rand64State), rndst, 0, NULL, NULL);

    /* Create a program from the kernel source */
    cl_program program = clCreateProgramWithSource(context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &ret);

    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(rndst_mem);
    ret = clReleaseCommandQueue(cmd_queue);
    ret = clReleaseContext(context);
    
    return 0;
}




/* /\* */
/*  * splitmix64 is a PRNG with 64 bits of state and the period 2^64. */
/*  * Written in 2015 by Sebastiano Vigna (vigna@acm.org)  */
/*  *\/ */
/* cl_ulong splitmix64(cl_ulong *xstate) { */
/* 	cl_ulong z = (*xstate += 0x9e3779b97f4a7c15); */
/* 	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9; */
/* 	z = (z ^ (z >> 27)) * 0x94d049bb133111eb; */
/* 	return z ^ (z >> 31); */
/* } */



/* /\* */
/*  * The state must be seeded so that it is not everywhere zero. If you have */
/*  * a 64-bit seed, we suggest to seed a splitmix64 generator and use its */
/*  * output to fill s. */
/*  * */
/*  * The text below has been plagiarized from the cuRAND documentation :) */
/*  * */
/*  * The rand64_init() function sets up an initial state allocated by the caller */
/*  * using the given seed and sequence number. Different seeds are guaranteed to */
/*  * produce different starting states and different sequences. The same seed */
/*  * always produces the same state and the same sequence. The state set up will */
/*  * be the state after (2^128) * nsequence. */
/*  * */
/*  * Sequences generated with different seeds usually do not have statistically */
/*  * correlated values, but some choices of seeds may give statistically */
/*  * correlated sequences. Sequences generated with the same seed and different */
/*  * sequence numbers will not have statistically correlated values. */
/*  * */
/*  * For the highest quality parallel pseudorandom number generation, each */
/*  * experiment should be assigned a unique seed. Within an experiment, each */
/*  * thread of computation should be assigned a unique sequence number. If an */
/*  * experiment spans multiple kernel launches, it is recommended that threads */
/*  * between kernel launches be given the same seed, and sequence numbers be */
/*  * assigned in a monotonically increasing way. If the same configuration of */
/*  * threads is launched, random state can be preserved in global memory between */
/*  * launches to avoid state setup time. */
/*  * */
/*  *\/ */
/* void rand64_init(cl_ulong seed, cl_uint nsequence, rand64State *state) { */
    
/*     cl_ulong xstate = seed; */
/*     int iseq; */

/*     state->s[0] = seed; */
/*     state->s[1] = splitmix64(&xstate); */
/*     state->s[2] = splitmix64(&xstate); */
/*     state->s[3] = splitmix64(&xstate); */

/*     for (iseq=0; iseq<nsequence; iseq++) */
/*         jump_ahead(state); */
/* } */


/* /\*  */
/*  * The rand64_initn() function sets up initial states using the given seed in an */
/*  * array state[nstates] allocated by the caller. */
/*  * It is equivalent to nstates calls to rand64_init() with the sequence numbers */
/*  * progressing from 0 to nstates-1.   */
/*  *\/ */
/* void rand64_initn(cl_ulong seed, cl_uint nstates, rand64State state[nstates]) { */

/*     cl_ulong xstate = seed; */
/*     int ist; */
    
/*     state[0].s[0] = seed; */
/*     state[0].s[1] = splitmix64(&xstate); */
/*     state[0].s[2] = splitmix64(&xstate); */
/*     state[0].s[3] = splitmix64(&xstate); */
        
/*     if (nstates <= 1) return; /\* ======================================= >>> *\/ */
    
/*     for (ist=1; ist<nstates; ist++) { */
/*         state[ist] = state[ist-1]; */
/*         jump_ahead(&state[ist]);     */
/*     } */
/* } */


/* /\*  */
/*  * Rotate left the 64-bit word x by k bits  */
/*  *\/ */
/* static inline cl_ulong rotl(const cl_ulong x, int k) { */
/* 	return (x << k) | (x >> (64 - k)); */
/* } */



/* /\*  */
/*  * This is xoshiro256** 1.0, our all-purpose, rock-solid generator. It has */
/*  * excellent (sub-ns) speed, a state (256 bits) that is large enough for */
/*  * any parallel application, and it passes all tests we are aware of. */
/*  * */
/*  * Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org) */
/*  *\/ */
/* cl_ulong rand64_bits(rand64State *state) { */

/*     cl_ulong *s = state->s; /\* Just for brevity *\/ */
    
/* 	const cl_ulong result_starstar = rotl(s[1] * 5, 7) * 9; */
/* 	const cl_ulong t = s[1] << 17; */

/* 	s[2] ^= s[0]; */
/* 	s[3] ^= s[1]; */
/* 	s[1] ^= s[2]; */
/* 	s[0] ^= s[3]; */

/* 	s[2] ^= t; */

/* 	s[3] = rotl(s[3], 45); */

/* 	return result_starstar; */
/* } */





/* /\*  */
/*  * This is the jump function for the xoshiro256** 1.0. generator.  */
/*  * It is equivalent to 2^128 calls to rand64_bits(); it can be used to  */
/*  * generate 2^128 non-overlapping subsequences for parallel computations. */
/*  * */
/*  *  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org) */
/*  *\/ */
/* void jump_ahead(rand64State *state) { */
/* 	static const cl_ulong JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, \ */
/*                                      0xa9582618e03fc9aa, 0x39abdc4529b1661c }; */
/* 	cl_ulong s0 = 0; */
/* 	cl_ulong s1 = 0; */
/* 	cl_ulong s2 = 0; */
/* 	cl_ulong s3 = 0; */
/* 	for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++) */
/* 		for(int b = 0; b < 64; b++) { */
/* 			if (JUMP[i] & UINT64_C(1) << b) { */
/* 				s0 ^= state->s[0]; */
/* 				s1 ^= state->s[1]; */
/* 				s2 ^= state->s[2]; */
/* 				s3 ^= state->s[3]; */
/* 			} */
/* 			rand64_bits(state);	 */
/* 		} */
		
/* 	state->s[0] = s0; */
/* 	state->s[1] = s1; */
/* 	state->s[2] = s2; */
/* 	state->s[3] = s3; */
/* } */













