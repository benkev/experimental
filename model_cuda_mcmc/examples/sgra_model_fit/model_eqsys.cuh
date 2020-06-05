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

   
   if (id == 0) /* x^2 + y^2 */
       mc->datm[imd] = pow(mc->pcur[ipt++],mc->coor[0]) + 
                         pow(mc->pcur[ipt],mc->coor[1]);
   else           /* -0.25*x + y */
       mc->datm[imd] = -mc->coor[2]*mc->pcur[ipt++] + mc->pcur[ipt];

   
   // int ist = threadIdx.x*gridDim.x + blockIdx.x;

   //   if (ist == 0) {
   //    printf("DEVICE id=%d, pcur=%f, datm=%f, coor[0]=%f \n",
   //           id, mc->pcur[ipt], mc->datm[imd], mc->ndat, mc->coor[0]);
   //}

   return 1;
}


 
