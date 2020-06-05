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
                         
        //int id = blockIdx.x*blockDim.x + threadIdx.x;
        //int id = threadIdx.x;
        //int id = blockDim.x*blockIdx.x + threadIdx.x;
        int id = threadIdx.x*gridDim.x + blockIdx.x;

        if (id >= nstates)
          return;

        curand_init(seed, id, offset, &s[id]);
}
  /*
   * Generate gridDim.x sequences 
   */
__global__ void genrand(float *a, int nrnd, int nblk, curandState *state) {

  //int istate = blockDim.x*blockIdx.x + threadIdx.x;
  int istate = threadIdx.x*gridDim.x + blockIdx.x;
  int j, offs = istate*nrnd;
  
    if (blockIdx.x >= nblk)
      return;

  for (j = 0; j < nrnd; j++) {
    a[offs+j] = curand_uniform(&state[istate]);
  }
}

} // extern "C"
"""
module = SourceModule(source_code, no_extern_c=True)
init_rng = module.get_function('init_rng')
genrand = module.get_function('genrand')

seed = 4321
nblk = 16
nrnd = 100  # Random numbers in each sequence generated on blocks
nthreads = 64
nstates = nblk*nthreads


## rng_states = cuda.mem_alloc(nstates*characterize.sizeof(
##     'curandStateXORWOW', '#include <curand_kernel.h>'))
#sizeof_state = characterize.sizeof('curandStateMtgp32',
sizeof_state = characterize.sizeof('curandStateXORWOW', \
                                   '#include <curand_kernel.h>')
rng_states = cuda.mem_alloc(nstates*sizeof_state)

init_rng(np.int32(nstates), rng_states, np.uint64(seed),      \
         np.uint64(0), block=(nthreads,1,1), grid=(nblk,1))

#sys.exit(0)

ar = np.zeros((nblk,nthreads,nrnd), dtype=np.float32)

genrand(cuda.Out(ar), np.int32(nrnd), np.int32(nblk), rng_states, \
        block=(nthreads,1,1), grid=(nblk,1))

arr = ar.reshape((nthreads*nblk*nrnd))

