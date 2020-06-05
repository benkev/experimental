/*
 * Functions used for the multilevel Van Vleck correction
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include "vanvleck.h"
//#define pi M_PI
//#define TRUE  1
//#define FALSE 0
//#define inv_RAND_MAX  1./((double)RAND_MAX + 1.)
//#define RANDU()  rand()*inv_RAND_MAX  /* Uniformly distributed in [0..1] */
////#define NS 301    /* Old Number of rh ticks in stdev dimension */
//#define NS 315    /* Number of rh ticks in stdev dimension */
//#define NR 101    /* Number of rh ticks in rho dimension */
//#define NQS 178   /* Number of qsig rows: qsig[NQS][NQC] */
//#define NQC 5     /* Number of qsig columns: qsig[NQS][NQC] */
/* New sizes: */
// #define NS  995    /* Number of rh ticks in stdev dimension */
// #define NQS 498    /* Number of qsig rows: qsig[NQS][NQC] */


double randn(void)
{
  double v1, v2, r, fac, aamp, vv1;
  v1 = r = 0.0;
  while(r>1.0 || r==0.0){
    v1 = 2.0*RANDU() - 1.0; 
    v2 = 2.0*RANDU() - 1.0;
    r = v1*v1 + v2*v2;
    //printf("r=%g\n", r);
  }
  fac = sqrt(-2.0*log(r)/r);
  vv1 = v1*fac;
  return(vv1);
} 



double adc(double x, int nbit) {
    /* 
     * ADC, analog-to-digital converter
     * Inputs:
     *   x: analog quantity
     *   nbit: precision in bits.
     * Returns x after quantization.
     *
     * The code below is symmetric. 
     * Negative intervals are open on the left and closed on the right.
     * Positive intervals are closed on the left and open on the right.
     * The center (-0.5, 0.5) is open on both sides.
     * For example, if nbits = 3:
     *    (-Inf,-2.5] --> -3
     *    (-2.5,-1.5] --> -2 
     *    (-1.5,-0.5] --> -1
     *    (-0.5, 0.5) -->  0
     *    [ 0.5, 1.5) -->  1 
     *    [ 1.5, 2.5) -->  2 
     *    [ 2.5,+Inf) -->  3 
	 */
    int j, nlev;
    double aj, y;  /* Output */

    nlev = (1 << (nbit-1)) - 1;  /* = 2^(nbit-1)-1: Number of quant. levels */ 

    if (x > -0.5) {
        y = nlev;
        for (j = 0; j < nlev; j++) {
            aj = (double) j;
            if (x < (aj + 0.5)) {
                y = aj;
                break;
            }
        }
    }

    else {
        y = -nlev;
        for (j = -1; j > -nlev; j--) {
            aj = (double) j;
            if (x > (aj - 0.5)) {
                y = aj;
                break;
            }
        }
    } 
  
    return y; 
}



double vvfun(double xi, double s1, double s2, int nbit) {
    /* 
     * Integrand of the Van Vleck correction function 
     *
     *\begin{equation} 
     * <\hat{x}\hat{y}> = \frac{1}{2\pi}\int\limits_0^\rho
     *     \frac{1}{\sqrt{1-\zeta^2}}
     *     \sum\limits_{i=-n}^{n-1} \sum\limits_{k=-n}^{n-1}  
     *     \exp\left\{ -\frac{1}{2(1-\zeta^2)} 
     *     \left[ \frac{(i+\tfrac{1}{2})^2}{\sigma_1^2} 
     *   + \frac{(k+\tfrac{1}{2})^2}{\sigma_2^2} 
     *   - 2\frac{ (i+\tfrac{1}{2}) (k+\tfrac{1}{2})\zeta}{\sigma_1\sigma_2} 
     *     \right] \right\} \mathrm{d} \zeta
     *\end{equation}
     *
     */
    int nlev, i, k;
    double ap, aq, g, f = 0.;

    nlev = (1 << (nbit-1)) - 1;  /* = 2^(nbit-1)-1: Number of quant. levels */ 

    for (i=-nlev; i<nlev; i++) {
        ap = (double) i + 0.5;
        for (k=-nlev; k<nlev; k++) {
            aq = (double) k + 0.5;
            g = (0.5/(pi*sqrt(1. - pow(xi,2))))*				 \
                exp(-0.5*(pow((ap/s1),2) + pow((aq/s2),2) \
                          - 2.*ap*aq*xi/(s1*s2))          \
                    /(1 - pow(xi,2)) );
            f = f + g;
        }
    }
    return f;
}


int inds(double sig[], double s) {
  /*
   * Return isig index into sig[NS]:  sig[isig] <= s <= sig[isig+1]
   */
  double dsig;
  int isig;

  dsig = sig[1] - sig[0];   /* Step is the same over the sig[NS] ruler */
  isig = floor((s - sig[0])/dsig);
  if ((isig >= NS-1) || (isig < 0)) {   // sig[0] <= s1 <= sig[NS-1]
    printf("In inds(): s=%g outside sig[]\n", s);
    return -1.0;
  }
  return isig;
}



int load_rho_table(double *sig, double *rho, double (*rh)[NS][NR]) {
  FILE *fh;
  int len;

  fh = fopen("sig_ruler.bin", "rb");
  if (fh == NULL) {
    printf("Cannot open file 'sig_ruler.bin'\n");
    return 1;
  }
  len = fread(sig, NS*sizeof(double), 1, fh);
  fclose(fh);

  //==================

  fh = fopen("rho_ruler.bin", "rb");
  if (fh == NULL) {
    printf("Cannot open file 'rho_ruler.bin'\n");
    return 1;
  }
  len = fread(rho, NR*sizeof(double), 1, fh);
  fclose(fh);

  //==================

  fh = fopen("xy_table.bin", "rb");
  if (fh == NULL) {
    printf("Cannot open file 'xy_table.bin'\n");
    return 1;
  }
  len = fread(rh, NS*NS*NR*sizeof(double), 1, fh);
  fclose(fh);

  return 0;

}  /* load_rho_table() */


int load_qsig_table(double (*qsig)[NQC]) {
  FILE *fh;
  int len;

  fh = fopen("qsig.bin", "rb");
  if (fh == NULL) {
    printf("Cannot open file 'qsig.bin'\n");
    return 1;
  }
  len = fread(qsig, NQS*NQC*sizeof(double), 1, fh);
  fclose(fh);
  return 0;
}  /* load_qsig_table() */


int load_vvslopes(double (*vvslp)[12][12]) {
  FILE *fh;

  fh = fopen("vvslopes.bin", "rb");
  if (fh == NULL) {
    printf("Cannot open file 'vvslopes.bin'\n");
    return 1;
  }
  fread(vvslp, 3*12*12*sizeof(double), 1, fh);
  fclose(fh);
  return 0;
}  /* load_vvslopes() */




double bilin(double *sig, double (*rh)[NS][NR], double s1, double s2, int ir) {

  /* 
   * Evaluate rh[:,:,ir] at (s1,s2) using bilinear interpolation 
   */
  double dsig, res;
  int is1, is2;
  /*
   * Find (is1,is2) coordinates
   */
  dsig = sig[1] - sig[0];   /* Step in s1 and s2 dimensions */
  is1 = floor((s1 - sig[0])/dsig);
  if ((is1 >= NS-1) || (is1 < 0)) {   // sig[0] <= s1 <= sig[NS-1]
    printf("In bilin(): s1=%g outside sig\n", s1);
    return -1.0;
  }
  is2 = floor((s2 - sig[0])/dsig);
  if ((is2 >= NS-1) || (is2 < 0)) {   // sig[0] <= s2 <= sig[NS-1]
    printf("In bilin(): s2=%g outside sig\n", s2);
    return -1.0;
  }
    
  res = (1./pow(dsig,2))*      \
      (rh[is1]  [is2]  [ir]*(s1-sig[is1+1])*(s2-sig[is2+1])	   \
     - rh[is1+1][is2]  [ir]*(s1-sig[is1])*(s2-sig[is2+1])          \
     - rh[is1]  [is2+1][ir]*(s1-sig[is1+1])*(s2-sig[is2])          \
     + rh[is1+1][is2+1][ir]*(s1-sig[is1])*(s2-sig[is2]));
  
  return res;
  
}



double corr_estim(double sig[], double rho[], double (*rh)[NS][NR], \
		  double s1, double s2, double xy) {
  /*
   * Estimate the true correlation for the product of quantized 
   * signals xy = <xy>  via interpolation over the rh table
   */
  int xyNeg, ix0, ix1, ixm;
  double xy0, xy1, p, pm, corr;

  /*
   * Symmetry wrt the coordinate origin:
   */
  if (xy < 0.) {
    xyNeg = TRUE;
    xy = -xy;
  }
  else
    xyNeg = FALSE;
   
   /*
   * Find ix in [0..nx-2] such that xy is in (rh[s1,s2,ix] .. rh[s1,s2,ix+1])
   */   
  ix0 = 0;
  ix1 = NR - 1;
  xy0 = xy1 = NAN;

  if (xy == 0.0) return 0.0;   /* Useless to search */

  p = bilin(sig, rh, s1, s2, ix1);  /* Approx rh at (is1,is2,nx-1) */

  if (xy > p) {
    printf("In corr_estim(): input <xy>=%g is too large for\n", xy);
    printf("the given values of s1=%g and s2=%g\n", s1, s2);
    printf("Assumed correlation 0.9999999\n");
    if (xyNeg) 
      return -0.9999999;
    else
      return  0.9999999;
  }

  /*
   * Bisection search
   */
  while(ix1 - ix0 > 1) {
    ixm = (ix0 + ix1)/2;
    pm = bilin(sig, rh, s1, s2, ixm);  /* Approx rh at (is1,is2,ixm) */

    if (xy > pm) {
      ix0 = ixm;
      xy0 = pm;
    }
    else if (xy < pm) {
      ix1 = ixm;
      xy1 = pm;
    }
    else {        /* xy == pm */
      ix0 = ixm;  /* xy is found exactly at ix0 */
      xy0 = pm;
      break;
    }
  }
     

  if (isnan(xy0)) xy0  = bilin(sig, rh, s1, s2, ix0);
  if (isnan(xy1)) xy1  = bilin(sig, rh, s1, s2, ix0+1);

  /*
   * Linear interpolation
   */
  corr = rho[ix0] + ((rho[ix1]-rho[ix0])/(xy1-xy0))*(xy - xy0);

  if (isnan(corr)) {
    printf("In corr_estim(): corr == NAN!");
    return 0.0;
  }
  /*
   * Symmetry wrt the coordinate origin:
   */
  if (xyNeg) corr = -corr;

  return corr; 
}

 

 
double qstd(double sigma, int nbit) {
  /*
   * Theoretical prediction for the standard deviation of a normally
   * distributed signal quantized into nbit integers.
   * Output: predicted sigma of quantized_signal
   * Inputs:
   *   sigma: STD, i.e. sqrt(variance) of analog, non-quantized signal
   *   nbit:  precision of the quantization
  */
  int nlev, k;
  double cnorm, s1sqrt2, arg; 

  nlev = (1 << (nbit-1)) - 1;  /* = 2^(nbit-1)-1: Number of quant. levels */ 
  s1sqrt2 = sigma*sqrt(2.);
  cnorm = pow((double)nlev,2);
  for (k=0; k<nlev; k++) {
    arg = (float)(k+0.5)/s1sqrt2;
    cnorm = cnorm - (double)(2*k+1)*erf(arg);
  }
  return sqrt(cnorm);
}


double std_estim(double (*qsig)[NQC], int nbit, double squant) {
  /*
   * Estimate the 'analog' standard deviation of a signal. 
   * via linear interpolation over the qsig table.
   * Parameters:
   *   qsig[NQS][NQS]: table, column 0 has the values of analog std,
   *                   columns 1 to 4 contain corresponding std's
   *                   of signals quantized with 2 to 5 bit precision.
   *   nbit: number of bits of quantization
   *   squant: input quantized std.
   * Returns: sanalog, estimate of analog std obtained by the std table lookup
   *          and linear interpolation.
   */
  int i0, i1, im, ncol = nbit - 1;
  double sanalog, s0, s1, sm, coef;
  
   /*
   * Find i such that squant is in (rh[s1,s2,i] .. rh[s1,s2,i+1])
   */   
  i0 = 0;
  i1 = NQS - 1;

  if (squant == 0.0) return 0.0;   /* Useless to search */

  if (nbit < 2 || nbit > 5) {
    printf("In std_estim(): input nbit=%d is beyond the range [2..5].\n", \
	   nbit);
    return  NAN;
  }
  if (squant > qsig[i1][ncol] || squant < qsig[i0][ncol]) {
    printf("In std_estim(): input squant=%g is beyond the qsig table.\n", \
	   squant);
    return  NAN;
  }
  
  while(i1 - i0 > 1) {
    im = (i0 + i1)/2;
    sm = qsig[im][ncol];  

    /* printf("qsig[%d]=%g,  qsig[%d]=%g,  qsig[%d]=%g\n", \ */
    /* 	   i0, qsig[i0][ncol], im, qsig[im][ncol], i1, qsig[i1][ncol]); */

    if (squant > sm) {
      i0 = im;
      s0 = sm;
    }
    else if (squant < sm) {
      i1 = im;
      s1 = sm;
    }
    else {      /* squant == sm */
      i0 = im;  /* squant is found exactly at i0 */
      s0 = sm;
      break;
    }
  }

  i1 = i0 + 1;    /* Found squant between i0 and i0+1. Set i1 to i0+1  */ 

  /*
   * Linear interpolation of qsig[:][0], the analog std. 
   */
  coef = (qsig[i1][0] - qsig[i0][0])/(s1 - s0);
  sanalog = qsig[i0][0] + coef*(squant - s0);

  return sanalog;
}


double mean(int n, double x[n]) {
    double s = 0, an = (double)n;
    int i;
    for (i = 0; i < n; i++) {
        s += x[i];
    }
    return s/an;
}


double std(int n, double x[n]) {
    double s2 = 0, s = 0, an = (double)n;
    int i;
    for (i = 0; i < n; i++) {
        s2 += pow(x[i],2);
        s += x[i];
    }
    return sqrt(s2/an - pow(s/an,2));
}

/*
 * Return np numbers spaced evenly on a log scale in array sp.
 * The sequence starts at 10^start and ends with 10^stop.
 */

void log10space(double start, double stop, long np, double sp[np]) {
  double delx = (stop - start)/(np - 1); 
  double x = start;
  long i;

  for (i = 0; i < np; i++) {
    sp[i] = pow(10.0, x);
    x = x + delx;
  }
} 
 
/*
 * Return np numbers spaced evenly on a log scale in array sp.
 * The sequence starts at base^start and ends with base^stop.
 */

void logspace(double base, double start, double stop, long np, double sp[np]) {
  double delx = (stop - start)/(np - 1); 
  double x = start;
  long i;

  for (i = 0; i < np; i++) {
    sp[i] = pow(base, x);
    x = x + delx;
  }
} 




double gauss(void)
{
    double v1,v2,r,fac,aamp,vv1;
    v1=r=0.0;
    while(r>1.0 || r==0.0){
        v1=2.0*(rand()/2147483648.0)-1.0;v2=2.0*(rand()/2147483648.0)-1.0;
        r=v1*v1+v2*v2;
    }
    fac=sqrt(-2.0*log(r)/r);vv1=v1*fac;
    aamp=vv1;
    return(aamp);
} 



size_t inline locate(size_t n, double tb[n], double x) {
    /*
     * Locate position of the x value in the ordered table tb[n]
     * using bisection search.
     * Ii case of the overflow danger, change for im = ilo + ((iup - ilo)/2);
     * Returns lower index of interval containing x.
     * If x is less than tb[0], returns -1.
     * If x is greater than tb[n-1], returns n-1 (index of the last element).
     */
    size_t ilo, im, iup;
    //double pm;

    ilo = -1;
    iup = n;

    while(iup - ilo > 1) {

        im = (ilo + iup)/2; 

        if (x > tb[im]) ilo = im;
        else            iup = im;
    }

    return ilo;
}


double inline lininterp(size_t n, double xx[n], double yy[n], double x) {
    /*
     * Linear interpolation of the function yy = f(xx) specified in the 
     * form of the ordered table.
     * For given x, ilo and iup are searched such that 
     *    xx[ilo] <= x <= xx[iup],  iup = ilo+1. 
     * The returned value y is found between yy[ilo] and yy[iup]. 
     * It is proportional to the position of x between xx[ilo] and xx[iup].
     */
    size_t ilo, iup;
    double xlo, xup, ylo, yup, y;

    ilo = locate(n, xx, x);
    if (ilo == -1) {
        printf("ERROR in lininterp(): x < xx[0]\n");
        exit(0);
    }
    if (ilo == n-1) {
        printf("ERROR in lininterp(): x > xx[n-1]\n");
        exit(0);
    }
    iup = ilo + 1;
    xlo = xx[ilo];
    xup = xx[iup];
    ylo = yy[ilo];
    yup = yy[iup];

    y = ylo + ((yup - ylo)/(xup - xlo))*(x - xlo);
    return y;
}



double inline lininterp_vv(size_t n, double qcov[n], double rho[n], double q) {
    /*
     * Linear interpolation of the function rho = f(qcov) specified in the 
     * form of the ordered table. The function relates quantized covariance
     * qcov and analog correlation rho, i.e it is the "Van Vleck curve".
     * Both tables, qcov and rho, must be monotonously growing functions,
     * qcov values starting from -1 and ending at +1 (not necessarily 
     * inclusive, the interval -1..+1 may be open). Formally, qcov in (-1,1).
     * For given covariance q, indices ilo and iup are searched such that 
     *    qcov[ilo] <= q <= qcov[iup],  iup = ilo+1. 
     * The returned analog correlation value r is found between 
     * rho[ilo] and rho[iup]. It is proportional to the position of q 
     * between qcov[ilo] and qcov[iup].
     */
    size_t ilo, iup;
    double qlo, qup, rlo, rup, r;

    ilo = locate(n, qcov, q);
    if (ilo == -1) {
        return -1.;  /* For a too low covariance, rho = -1 is assumed. */
    }
    if (ilo == n-1) {
        return +1.;  /* For a too high covariance, rho = +1 is assumed. */
    }
    iup = ilo + 1;
    qlo = qcov[ilo];
    qup = qcov[iup];
    rlo = rho[ilo];
    rup = rho[iup];

    r = rlo + ((rup - rlo)/(qup - qlo))*(q - qlo);
    return r;
}
