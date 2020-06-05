import numpy as np
from numpy import *
from pylab import *
import pycuda.autoinit
from pycuda import characterize
import pycuda.driver as cu
import pycuda.compiler
from pycuda.compiler import SourceModule
import obsdatafit
from obsdatafit import readamp1, readcphs1
import time

source_code = """
#include <curand_kernel.h>
#include <stdio.h>

extern "C"
{
__device__ void check_p(int iprm, float *pmin, float *pmax, float prm, 
			int *pdescr, int nprm, int *flag);

  /*
   * Initialize states one for each block (streaming multiprocessor)
   */
__global__ void init_rng(int nstates, curandState *s,
                         unsigned long long seed,
                         unsigned long long offset) {
                         
        //int id =   ibeta    *  nseq    +   iseq;
        //int   id = threadIdx.x*gridDim.x + blockIdx.x;
        int id = blockDim.x*blockIdx.x + threadIdx.x;
        
        if (id >= nstates)
                return;

        curand_init(seed, id, offset, &s[id]);
}


__global__ void gen_proposal(float *pout, int *ptentn, float *ptent,
			       float *pmin, float *pmax, int nprm, 
			       int itr, int iprm, int nitr,
			       float *pstp, int *pdescr, 
			       curandState *rndstate) {
  /*
   * Generate MCMC proposal parameters for current iteration and parameter 
   * number.
   *
   * This kernel is called by the host nprm*niter times and runs 
   * simultaineously on nseq CUDA blocks, in nbeta threads in each block.
   * The gen_proposal() kernel is supposed to be invoked in a double loop
   * with the (itr, iprm) parameters, itr parameter running from 0 to nitr-1, 
   * and the iprm parameter running from 0 to nprm-1.  
   * 
   * Each gen_proposal() invocation generates nseq*nbeta proposal parameters,
   * saved in ptent[nbeta,nseq]. The randomly generated parameter number
   * for each temperature [0..nbeta-1] is saved in the ptentn[nbeta,nseq] 
   * integer array. If the proposed parameter does not belong to the prior,
   * its position in ptentn[ibeta,iseq] is marked with -1. 
   *   
   *  
   * pout[nprm,nbeta,nseq*nitr]: Markov chains of parameters, for all the 
   *       parameters, temperatures, and iterations in all sequences
   * ptent[nbeta,nseq]: tentative (proposal) parameters. 
   *
   */

  int nseq =  gridDim.x,  iseq =  blockIdx.x;
  int nbeta = blockDim.x, ibeta = threadIdx.x;
  int k, flag;
  float r;
  //int ip = threadIdx.x*gridDim.x + blockIdx.x;
  //int ip = ibeta*nseq + iseq;
  int ip = blockDim.x*blockIdx.x + threadIdx.x;
  
  //do { /* choose a random parameter number k */
    r = curand_uniform(&rndstate[ip]);
    k = (int) (nprm*r);
  //} while (k >= nprm);
  
  ptentn[ip] = k; /* Randomly chosen proposal parameter  number */
  ptent[ip] = r; //pout[(iprm*nbeta + ibeta)*nseq*nitr + itr];
  //ptent[ip] = pout[(iprm*nbeta + ibeta)*nseq*nitr + itr];
  //ptent[ip] = ptent[iseq] + pstp[k*nbeta+ibeta]*curand_normal(&rndstate[iseq]);

  //check_p(iprm, pmin, pmax, ptent[ip], pdescr, nprm, &flag);

 // if (flag < 0) ptentn[ip] = -1; /* Mark as outside of the prior */
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
    // For this, define prm as float *prm in the header:
    // if (pdescr[iprm] == 2) 
    //   prm[iprm] = atan2(sin(prm[iprm]),cos(prm[iprm]));
    // if (prm[iprm] < pmin[iprm] || prm[iprm] > pmax[iprm])  
    //   *flag = -1;

}


}
"""


def init_random(nseq, nbeta):
    """
    Initialize nseq*bbeta independent Mersenne twister random generators
    """
    nsb = nseq*nbeta
    ## sizeof_1state = characterize.sizeof('curandStateMtgp32',
    sizeof_1state = characterize.sizeof('curandStateXORWOW',
                                        '#include <curand_kernel.h>')
    #rng_states = cu.mem_alloc(uint64(nseq)*sizeof_1state)
    rng_states = cu.mem_alloc(nsb*sizeof_1state)
    module = SourceModule(source_code, no_extern_c=True)
    init_rng = module.get_function('init_rng')
    #seed0 = uint64(np.trunc(1e6*time.time()%(10*nsb)))
    #seeds = seed0 + randint(0, nsb, nsb) # nsb uint64 samples in [0..nsb]
    seed0 = 4321;
    init_rng(np.int32(nsb), rng_states, np.uint64(seed0),   \
             np.uint64(0), block=(nbeta,1,1), grid=(nseq,1))
    ##init_rng(np.int32(nsb), rng_states, cu.In(seeds),   \
    ##         np.uint64(0), block=(nbeta,1,1), grid=(nseq,1))
    return rng_states, sizeof_1state*nsb


ulam, vlam, amp, phase = readamp1('sgra01_uvdata.txt');
cphs, cuv, gha, tcode  = readcphs1('sgra01_clph.txt')

bnd = array(((0.01, 5.0), (20., 60.), (0., 1.), (0., 1.), (0., 1.), \
             (0.1, 2.), (0.1, 1.), (0., 1.), (-pi, pi)), dtype=float32)
pmint = bnd[:,0]; pmaxt = bnd[:,1]

pdescr = array((0, 1, 1, 1, 0, 0, 0, 0, 1), dtype=int32) # uxring model
ptotal = array((2.4, 0., 0., 0., 1., 0., 0., 0., 0.), dtype=float32)
npall = len(ptotal)

## def mcmcfit(ulam, vlam, amp, eamp, cuv, cphs, ecphs, \
##             pdescr, ptotal, pmint, pmaxt, \
##             nitr, nburn, nadj, nbeta, beta1, betan):
##     """
##     pdescr, ptotal, pmint, pmaxt have nptotal elements.
    
##     Returns: pout, tout, chi2, rate_acpt, rate_exch
##     """
    
nitr = int32(1000)
nburn = int32(1000) # Number of initial iters ignored due to the transience 
nadj = 100
nbeta = 64  # Number of temperatures
beta1 = 1.
betan = 0.0001   # ~exp(chi2/2)
#nseq = int32(16)     # Number of processing cores (CUDA blocks)
nseq = 16    # Number of processing cores (CUDA blocks)
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
n_cnt = zeros((nprm,nbeta), dtype=int32) # number of Metropolis trials
n_hist = zeros((nadj,nprm,nbeta), dtype=int32) # accept/reject history
pstp = zeros((nprm,nbeta), dtype=float32) 
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

# Initialize 'ncores' independent Mersenne twister random generators 
rndstates, sizeofallstates = init_random(nseq, nbeta)

#
# Main loop
#
mod = SourceModule(source_code, no_extern_c=True)
gen_proposal = mod.get_function('gen_proposal')

itr = int64(0)
iprm = int64(0)

nsb = nseq*nbeta
gen_proposal(cu.In(pout), cu.Out(ptentn), cu.Out(ptent),
             cu.In(pmin), cu.In(pmax), nprm, itr, iprm, nitr,
             cu.In(pstp), cu.In(pdescr),
             rndstates, block=(nbeta,1,1), grid=(nseq,1))
sys.exit(0)

#===================================================================

gen_proposal(cu.In(pout), cu.Out(ptentn), cu.Out(ptent), cu.In(pmin),
             cu.In(pmax), nprm, itr, iprm, nitr, cu.In(pstp), cu.In(pdescr),
             rndstates, block=(nbeta,1,1), grid=(nseq,1))

## return ptent, ptentn

## # Burn-in
## for iburn in xrange(nburn*nprm):
##     single_prm_mcmc()
##     adjust_step()
##     calc_chi2()
##     replica_exchange()

## # Main phase
## for itr in xrange(nitr*nprm):
##     single_prm_mcmc()
##     calc_chi2()
##     replica_exchange()

## for iseq in xrange(nseq):
##     rate_acpt += n_acpt[:,:,iseq]
##     rate_exch += n_exch[:,iseq]

## rate_acpt /= nitr*nseq 
## rate_exch /= nitr*nseq 

## return pout, tout, chi2, rate_acpt, rate_exch












