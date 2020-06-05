#include <cuda.h>
//#include <curand_kernel.h>
//#include <stdio.h>
#include "mcmcjob.h"

//
// For convenience and brevity, define the variable and coefficient names
//
#define X (mc->pcur[ipt])
#define Y (mc->pcur[ipt+1])


__device__ int model(CCalcModel *mc, int id, int ipt, int imd, int ipass) {

  /* Arguments:
   *   id: index into mc->dat. 
   *   ipt: index of parameters' set: pcur[ipt] ~ pcur[ibeta,iseq,:].
   *        pcur[ipt] is the first model parameter, and 
   *        pcur[ipt+mc->nptot-1] is the last parameter.
   *        is
   *   imd: "through index" into datm[ibeta,isec,id]. Only used 
   *        to save the result before return.
   *   ipass: pass number, starting from 0. The model() function can be
   *          called multiple times, or in many passes, so pass is to know
   *          which time model() is called now.  
   *
   * The result must be saved in
   *   mc->datm[imd]
   */

   /* Arguments:
   *   id: index into mc->dat
   *   ipt: index of parameters' set: pcur[ipt] ~ pcur[ibeta,iseq,:]
   *   imd: "through index" into datm[ibeta,isec,id]
   *   ipass: pass numberstarting from 0.
   *
   * Result:
   *   mc->datm[imd]
   */

   
   if (id == 0) /* (1 - x)^2 + 100(y - x^2)^2 */
       mc->datm[imd] = pow(1.f - X, 2) + 100.f*pow(Y - pow(X, 2), 2);
   
       // mc->datm[imd] = pow(1.f - mc->pcur[ipt], 2) + 
       //     100.f*pow(mc->pcur[ipt+1] - pow(mc->pcur[ipt], 2), 2);

   
   return 1;
}

//
// Always undefine the names after use!  
//
#undef X
#undef Y

