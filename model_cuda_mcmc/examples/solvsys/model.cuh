//
// model.cuh
//
// This is for the solvsys.py script.
//
// This CUDA kernel calculates one of the two left-hand-sides of the
// nonlinear equation system, dependent on the parameter id:
// if id == 0:       x^2 + y^2;
// if id == 1: -0.25*x +   y.
//
// The result - a single number - is stored in datm[imd].
//
// For convenience and brevity, define the variable and coefficient names
// (BE CAREFUL not to mix with the names in src/gpu_mcmc.cu !!!
// To avoid the mess, always undefine the names after use!):
//
#define X (mc->pcur[ipt++])
#define Y (mc->pcur[ipt])
#define P1 (mc->coor[0])
#define P2 (mc->coor[1])
#define K (mc->coor[2])

__device__ int model(CCalcModel *mc, int id, int ipt, int imd, int ipass) {

  /* Arguments:
   *   id: index into mc->dat
   *   ipt: index of parameters' set: pcur[ipt] ~ pcur[ibeta,iseq,:]
   *   imd: "through index" into datm[ibeta,isec,id]
   *   ipass: pass number, starting from 0.
   *
   * Result:
   *   mc->datm[imd]
   */

   
   if (id == 0)           /* x^2 + y^2 */
       mc->datm[imd] = pow(X,P1) + pow(Y,P2);
   else /* i.e. id == 1:    -0.25*x + y */
       mc->datm[imd] = -K*X + Y;

   return 1;
}

#undef X
#undef Y
#undef P1
#undef P2
#undef K

 
