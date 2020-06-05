import numpy as np
#import pycuda.tools
import pycuda.autoinit
from pycuda import characterize
import pycuda.driver as cuda
import pycuda.compiler
from pycuda.compiler import SourceModule
#from pycuda import gpuarray as ga
import sys

source_code = """
#include <curand_kernel.h>

extern "C"
{
  /*
   * Initialize states one for all the threads
   */
__global__ void init_rng(int nstates, curandState *s,
                         unsigned long long seed,
                         unsigned long long offset) {
                         
        int id = blockIdx.x*blockDim.x + threadIdx.x;
        //int id = threadIdx.x;
        //int id = blockIdx.x;

        if (id >= nstates)
                return;

        curand_init(seed, id, offset, &s[id]);
}
  /*
   * Generate gridDim.x sequences 
   */
__global__ void genrand(float *a, int nrnd, int nstates, curandState *state) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j, offs = nrnd*i;
  
  if (i >= nstates)
    return;

  for (j = 0; j < nrnd; j++) {
    a[offs+j] = curand_uniform(&state[i]);
  }
}

} // extern "C"
"""
module = SourceModule(source_code, no_extern_c=True)
init_rng = module.get_function('init_rng')
genrand = module.get_function('genrand')

seed = 4321
nstates = 1024
nrnd = 100  # Random numbers in each sequence generated on blocks

## rng_states = cuda.mem_alloc(nstates*characterize.sizeof(
##     'curandStateXORWOW', '#include <curand_kernel.h>'))
#sizeof_state = characterize.sizeof('curandStateMtgp32',
sizeof_state = characterize.sizeof('curandStateXORWOW', \
                                   '#include <curand_kernel.h>')
rng_states = cuda.mem_alloc(nstates*sizeof_state)

init_rng(np.int32(nstates), rng_states, np.uint64(seed),      \
         np.uint64(0), block=(32,1,1), grid=(nstates//32+1,1))

#sys.exit(0)

ar = np.zeros((nstates,nrnd), dtype=np.float32)

genrand(cuda.Out(ar), np.int32(nrnd), np.int32(nstates), rng_states, \
        block=(32,1,1), grid=(nstates//32+10,1))

arr = ar.reshape((nstates*nrnd))
