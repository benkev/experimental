#include <stdint.h>

/* typedef unsigned int uint; */
/* typedef unsigned long long ulong; */


cl_ulong splitmix64(cl_ulong *state);
void rand64_init(cl_ulong seed, cl_uint nsequence, rand64State *state);
void rand64_initn(cl_ulong seed, cl_uint nstates, rand64State state[nstates]);
cl_ulong rand64_bits(rand64State *state);
void jump_ahead(rand64State *state);
cl_double rand64_uniform(rand64State *state);
cl_double rand64_normal(rand64State *state);
 
