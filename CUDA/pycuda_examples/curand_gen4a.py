import numpy as np
from numpy import pi, array, float32, int32, int64, uint64, where, zeros
#from numpy import *
#import pycuda.tools
import pycuda.autoinit
from pycuda import characterize
import pycuda.driver as cu
import pycuda.compiler
from pycuda.compiler import SourceModule
#from pycuda import gpuarray as ga
import sys
from obsdatafit import readamp1, readcphs1
import time

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
    a[offs+j] = curand_normal(&state[istate]);
  }
}

__global__ void gen_proposal(curandState *st, float *pout,
                             float *ptent, int *ptentn,
//                             float *pmin, float *pmax,
                             int nprm, //int nitr, int iprm, int itr,
//                             float *pstp, int *pdescr, 
                             int nstates) {

  int ist = threadIdx.x*gridDim.x + blockIdx.x;
  //int ip = ((blockDim.x*iprm + threadIdx.x)*gridDim.x + blockIdx.x)*nitr + itr;
  int k;
  float r, pcur, stdev;
  if (ist >= nstates) return;

  do {
    r = curand_uniform(&st[ist]);
    k = (int) (nprm*r); 
  } while (k >= nprm);
  ptentn[ist] = k;
  
  //pcur = pout[ip]; /* pout[iprm,threadIdx,blockIdx,itr] */
  //stdev = pstp[blockDim.x*iprm + threadIdx.x];
  //ptent[ip] = pcur + stdev*curand_normal(st[ist]);

}

//--------------------------------------------------------------------

__device__ void check_p(int iprm, float *pmin, float *pmax, float prm, 
			int *pdescr, int nprm, int *flag) {
    /*
     * Check if parameters satisfy the prior conditions.
     * Here the parameter conditions are independent of one another,
     * therefore only the proposed parameter, prm[iprm], is tested.
     */
    *flag = 1;
    if (pdescr[iprm] == 2) 
      prm = atan2(sin(prm),cos(prm));
    if (prm < pmin[iprm] || prm > pmax[iprm])  
      *flag = -1;
}

} // extern "C"
"""
module = SourceModule(source_code, no_extern_c=True)
init_rng = module.get_function('init_rng')
genrand = module.get_function('genrand')

nseq = 16   # Number of processing cores (CUDA blocks)
nbeta = 64  # Number of temperatures
nsb = nseq*nbeta

#seed = 4321
seed = uint64(np.trunc(1e6*time.time()%(10*nsb)))
nblk = nseq
nrnd = 100  # Random numbers in each sequence generated on blocks
nthreads = nbeta
nstates = nseq*nbeta


## rng_states = cu.mem_alloc(nstates*characterize.sizeof(
##     'curandStateXORWOW', '#include <curand_kernel.h>'))
#sizeof_state = characterize.sizeof('curandStateMtgp32',
sizeof_state = characterize.sizeof('curandStateXORWOW', \
                                   '#include <curand_kernel.h>')
rng_states = cu.mem_alloc(nstates*sizeof_state)

init_rng(np.int32(nstates), rng_states, np.uint64(seed),      \
         np.uint64(0), block=(nthreads,1,1), grid=(nblk,1))

#sys.exit(0)

## ar = np.zeros((nblk,nthreads,nrnd), dtype=np.float32)

## genrand(cu.Out(ar), np.int32(nrnd), np.int32(nblk), rng_states, \
##         block=(nthreads,1,1), grid=(nblk,1))

## arr = ar.reshape((nthreads*nblk*nrnd))

ulam, vlam, amp, phase = readamp1('sgra01_uvdata.txt');
cphs, cuv, gha, tcode  = readcphs1('sgra01_clph.txt')

bnd = array(((0.01, 5.0), (20., 60.), (0., 1.), (0., 1.), (0., 1.), \
             (0.1, 2.), (0.1, 1.), (0., 1.), (-pi, pi)), dtype=float32)
pmint = bnd[:,0]; pmaxt = bnd[:,1]

pdescr = array((0, 1, 1, 1, 0, 0, 0, 0, 1), dtype=int32) # uxring model
ptotal = array((2.4, 0., 0., 0., 1., 0., 0., 0., 0.), dtype=float32)
npall = len(ptotal)

nitr = int32(1000)
nburn = int32(1000) # Number of initial iters ignored due to the transience 
nadj = 100
beta1 = 1.
betan = 0.0001   # ~exp(chi2/2)
#nseq = int32(16)     # Number of processing cores (CUDA blocks)
nvis = len(ulam)
ncph = len(cuv[:,0])

# Copy variated parameter boundaries from pmint to pnin, from pmaxt to pmax
ivar = where(pdescr != 0)[0]
nprm = int32(len(ivar)) # Find the number of parameters as number of pdescr != 0
pmin = pmint[ivar]
pmax = pmaxt[ivar]
### Copy variated parameters from  ptotal into prm ???
#prm = ptotal[ivar]
iang = where(pdescr == 2)[0]  # Angular parameters requiring adjustment

#
# Global variables and arrays
#
itr =  int32(0)   # Iteration number is same for all sequences and betas
ipm =  int32(0)   # Moving prm. number is same for all sequences & betas
pout = zeros((nprm,nbeta,nseq*nitr), dtype=float32)
tout = zeros((nbeta,nseq*nitr), dtype=float32)
chi2 = zeros((nseq*nitr), dtype=float32)
beta = zeros((nbeta), dtype=float32)
rate_acpt = zeros((nprm,nbeta), dtype=float32)
rate_exch = zeros((nbeta), dtype=float32)
n_acpt = zeros((nprm,nbeta,nseq), dtype=int32)
n_exch = zeros((nbeta,nseq), dtype=int32)

n_cnt = zeros((nprm,nbeta,nseq), dtype=int32) # number of Metropolis trials
n_hist = zeros((nadj,nprm,nbeta,nseq), dtype=int32) # accept/reject history
pstp = zeros((nprm,nbeta,nseq), dtype=float32) 
ptentn = zeros((nbeta,nseq), dtype=int32)
ptent =  zeros((nbeta,nseq), dtype=float32)


# Variances
## vamp = eamp**2
## vcphs = ecphs**2

# Initialize parameter steps  
for i in xrange(nbeta):
    pstp[:,i] = (pmax - pmin)/50.

# Initialize temperatures 
bstp = (betan/beta1)**(1.0/float(nbeta-1))
for i in xrange(nbeta):
    beta[i] = beta1*bstp**i

mod = SourceModule(source_code, no_extern_c=True)
gen_proposal = mod.get_function('gen_proposal')

itr = int64(0)
iprm = int64(0)

#sys.exit(0)
gen_proposal(rng_states, cu.In(pout), cu.Out(ptent), cu.Out(ptentn),  \
#             cu.In(pmin), cu.In(pmax), \
             nprm, #nitr, iprm, itr, \
#             cu.In(pstp), cu.In(pdescr), \
             np.int32(nstates), \
             block=(nbeta,1,1), grid=(nseq,1))

