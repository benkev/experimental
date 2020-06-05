//#include <curand_kernel.h>
//#include <stdio.h>

__device__ int model(CCalcModel *mc, int id, int ipt, int imd, int ipass) {

  /* Arguments:
   *   id: index into mc->dat
   *   ipt: index of parameters' set: pcur[ipt] ~ pcur[ibeta,iseq,:]
   *   imd: "through index" into datm[ibeta,isec,id]
   *   ipass: pass number. At pass 0 the visibility amplitudes and phases
   *          are calculated. At pass 1 the closure phases from already
   *          available phases are calculated.
   * Result:
   *   mc->datm[imd]
   */

   
   if (id == 0) /* (1 - x)^2 + 100(y - x^2)^2 */
       mc->datm[imd] = pow(1.f - mc->pcur[ipt], 2) + 
           100.f*pow(mc->pcur[ipt+1] - pow(mc->pcur[ipt], 2), 2);

   
   return 1;
}


 
