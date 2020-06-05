/*
 * Functions used for the multilevel Van Vleck correction
 * Single precision (float) version
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vanvleckf.h"



float randnf(void)
{
	double v1, v2, r, fac;
	float vv1;
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



float adcf(float x, int nbit) {
    /* 
     * ADC, analog-to-digital converter
     * Inputs:
     *   x: analog quantity
     *   nbit: precision in bits.
     * Returns x after quantization.
     */
    int j, nlev;
    float aj, y;  /* Output */

    nlev = (1 << (nbit-1)) - 1;  /* = 2^(nbit-1)-1: Number of quant. levels */ 

    if (x >= -0.5) {
        y = nlev;
        for (j = 0; j < nlev; j++) {
            aj = (float) j;
            if (x < (aj + 0.5)) {
                y = aj;
                break;
            }
        }
    }

    else {
        y = -nlev;
        for (j = -1; j > -nlev; j--) {
            aj = (float) j;
            if (x >= (aj - 0.5)) {
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



int load_rho_tablef(char *dir, float *sig, float *rho, float (*rh)[NS][NR]) {

    FILE *fh;
    int len;
    char *srname = (char *) calloc(sizeof(char[512]), 1);
    char *rrname = (char *) calloc(sizeof(char[512]), 1);
    char *xyname = (char *) calloc(sizeof(char[512]), 1);

    if (dir == NULL) {
        srname = strdup("sig_rulerf.bin");
        rrname = strdup("rho_rulerf.bin");
        xyname = strdup("xy_tablef.bin");
    }
    else {
        sprintf(srname, "%s/%s", dir, "sig_rulerf.bin");
        sprintf(rrname, "%s/%s", dir, "rho_rulerf.bin");
        sprintf(xyname, "%s/%s", dir, "xy_tablef.bin");
    }

    //==================

    fh = fopen(srname, "rb");
    if (fh == NULL) {
        printf("Cannot open file 'sig_rulerf.bin'\n");
        return 1;
    }
    len = fread(sig, NS*sizeof(float), 1, fh);
    fclose(fh);

    //==================

    fh = fopen(rrname, "rb");
    if (fh == NULL) {
        printf("Cannot open file 'rho_rulerf.bin'\n");
        return 1;
    }
    len = fread(rho, NR*sizeof(float), 1, fh);
    fclose(fh);

    //==================

    fh = fopen(xyname, "rb");
    if (fh == NULL) {
        printf("Cannot open file 'xy_tablef.bin'\n");
        return 1;
    }
    len = fread(rh, NS*NS*NR*sizeof(float), 1, fh);
    fclose(fh);

    free(srname);
    free(rrname);
    free(xyname);

    return 0;

}  /* load_rho_tablef() */


int load_qsig_tablef(char *dir, float (*qsig)[NQC]) {
    FILE *fh;
    int len;

    char *qsname = (char *) calloc(sizeof(char[512]), 1);

    if (dir == NULL)
        qsname = strdup("qsigf.bin");
    else
        sprintf(qsname, "%s/%s", dir, "qsigf.bin");


    //==================

    fh = fopen(qsname, "rb");
    if (fh == NULL) {
        printf("Cannot open file 'qsigf.bin'\n");
        return 1;
    }
    len = fread(qsig, NQS*NQC*sizeof(float), 1, fh);
    fclose(fh);

    free(qsname);

    return 0;
}  /* load_qsig_tablef() */


float bilinf(float *sig, float (*rh)[NS][NR], float s1, float s2, int ir) {

    /* 
     * Evaluate rh[:,:,ir] at (s1,s2) using bilinear interpolation 
     */
    float dsig, res;
    int is1, is2;
    /*
     * Find (is1,is2) coordinates
     */
    dsig = sig[1] - sig[0];   /* Step in s1 and s2 dimensions */
    is1 = floor((s1 - sig[0])/dsig);
    if ((is1 >= NS-1) || (is1 < 0)) {   // sig[0] <= s1 <= sig[NS-1]
        //printf("In bilinf(): s1=%g outside sig\n", s1);
        return -1.0;
    }
    is2 = floor((s2 - sig[0])/dsig);
    if ((is2 >= NS-1) || (is2 < 0)) {   // sig[0] <= s2 <= sig[NS-1]
        //printf("In bilinf(): s2=%g outside sig\n", s2);
        return -1.0;
    }
    
    res = (1./pow(dsig,2))*      \
        (rh[is1]  [is2]  [ir]*(s1-sig[is1+1])*(s2-sig[is2+1])	   \
         - rh[is1+1][is2]  [ir]*(s1-sig[is1])*(s2-sig[is2+1])      \
         - rh[is1]  [is2+1][ir]*(s1-sig[is1+1])*(s2-sig[is2])      \
         + rh[is1+1][is2+1][ir]*(s1-sig[is1])*(s2-sig[is2]));
  
    return res;
  
}



float corr_estimf(float sig[], float rho[], float (*rh)[NS][NR], \
                  float s1, float s2, float xy) { 
    /*
     * Estimate the true correlation for the product of quantized 
     * signals xy = <xy>  via interpolation over the rh table
     */
    int xyNeg, ix0, ix1, ixm;
    float xy0, xy1, p, pm, corr;

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

    p = bilinf(sig, rh, s1, s2, ix1);  /* Approx rh at (is1,is2,nx-1) */

    if (xy > p) {
        //printf("In corr_estim(): input <xy>=%g is too large for\n", xy);
        //printf("the given values of s1=%g and s2=%g\n", s1, s2);
        //printf("Assumed correlation 0.9999999\n");
        if (xyNeg) 
            return -0.9999999;
        else
            return  0.9999999;
    }
  
    while(ix1 - ix0 > 1) {
        ixm = (ix0 + ix1)/2;
        pm = bilinf(sig, rh, s1, s2, ixm);  /* Approx rh at (is1,is2,ixm) */

        // print 'ix0=%d, ix1=%d, ix0+ix1=%d, ixm=%d, pm=%g' %	
        //       (ix0, ix1, ix0+ix1, ixm, pm)
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
     

    if (isnan(xy0)) xy0  = bilinf(sig, rh, s1, s2, ix0);
    if (isnan(xy1)) xy1  = bilinf(sig, rh, s1, s2, ix0+1);

    /*
     * Linear interpolation
     */
    corr = rho[ix0] + ((rho[ix1]-rho[ix0])/(xy1-xy0))*(xy - xy0);

    if (isnan(corr)) {
        //printf("In corr_estim(): corr == NaN!");
        return 0.0;
    }
    /*
     * Symmetry wrt the coordinate origin:
     */
    if (xyNeg) corr = -corr;

    return corr; 
}                               /* end corr_estimf() */



float std_estimf(float (*qsig)[NQC], int nbit, float squant) {
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
    float sanalog, s0, s1, sm, coef;
  
    /*
     * Find i such that squant is in (rh[s1,s2,i] .. rh[s1,s2,i+1])
     */   
    i0 = 0;
    i1 = NQS - 1;

    if (squant == 0.0) return 0.0;   /* Useless to search */

    if (nbit < 2 || nbit > 5) {
        //printf("In std_estim(): input nbit=%d is beyond the range [2..5].\n",
        //nbit);
        return  NAN;
    }
    if (squant > qsig[i1][ncol] || squant < qsig[i0][ncol]) {
        //printf("In std_estim(): input squant=%g is beyond the qsig table.\n", 
        //squant);
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


 
float qstdf(float sigma, int nbit) {
    /*
     * Theoretical prediction for the standard deviation of a normally
     * distributed signal quantized into nbit integers.
     * Output: predicted sigma of quantized_signal
     * Inputs:
     *   sigma: STD, i.e. sqrt(variance) of analog, non-quantized signal
     *   nbit:  precision of the quantization
     */
    int nlev, k;
    float cnorm, s1sqrt2, arg; 

    nlev = (1 << (nbit-1)) - 1;  /* = 2^(nbit-1)-1: Number of quant. levels */ 
    s1sqrt2 = sigma*sqrt(2.);
    cnorm = pow((float)nlev,2);
    for (k=0; k<nlev; k++) {
        arg = (float)(k+0.5)/s1sqrt2;
        cnorm = cnorm - (float)(2*k+1)*erf(arg);
    }
    return sqrt(cnorm);
}



float bilinearf(int ns, int nr, float *sig, float (*xy)[ns][nr], 
                float s1, float s2, int ir) {

	/* 
	 * Single precision version of bilinear().
	 * 
	 * Evaluate xy[:,:,ir] at (s1,s2) using bilinear interpolation 
	 */
	float dsig, res;
	int is1, is2;
	/*
	 * Find (is1,is2) coordinates
	 */
	dsig = sig[1] - sig[0];   /* Step in s1 and s2 dimensions */
	is1 = floor((s1 - sig[0])/dsig);
	if ((is1 >= ns-1) || (is1 < 0)) {   // sig[0] <= s1 <= sig[ns-1]
	    //printf("In bilin(): s1=%g outside sig\n", s1);
		return -1.0;
	}
	is2 = floor((s2 - sig[0])/dsig);
	if ((is2 >= ns-1) || (is2 < 0)) {   // sig[0] <= s2 <= sig[ns-1]
	    //printf("In bilin(): s2=%g outside sig\n", s2);
		return -1.0;
	}
    
	res = (1./pow(dsig,2))*      \
		(xy[is1]  [is2]  [ir]*(s1-sig[is1+1])*(s2-sig[is2+1])	   \
		 - xy[is1+1][is2]  [ir]*(s1-sig[is1])*(s2-sig[is2+1])          \
		 - xy[is1]  [is2+1][ir]*(s1-sig[is1+1])*(s2-sig[is2])          \
		 + xy[is1+1][is2+1][ir]*(s1-sig[is1])*(s2-sig[is2]));
  
	return res;
  
}



void corr_estimatef(int ns, int nr, float sig[], float rho[], 
                    float (*xytbl)[ns][nr], int nxy, 
                    float s1[nxy], float s2[nxy], 
                    float xys[nxy], float cors[nxy]) {
    /*
	 * Single precision version of corr_estimate().
	 *
     * Estimate the true correlations for the array of products of quantized 
     * signals xy = <xy>  via interpolation over the xytbl[ns,ns,nr] table.
     *
     * Input parameters
     * ns:      length of sig[] ruler (STDs). Currently 315.  
     * nr:      length of rho[] ruler (correlationss). Currently 101.
     * sig[ns]: STD ruler for first two dimensions of xytbl[ns][ns][nr].
     * rho[nr]: rho ruler for the third dimension of  xytbl[ns][ns][nr].
     * xytbl[ns][ns][nr]: 3D table of quantized covariances <xy>.
     * nxy:     Number of covariances to correct. Length of input and 
     *          output arrays.
     * s1[], s2[]: analog STDs of correlated signals.
     * xys[]: covariances of the signals to be searched for in xytbl.
     * 
     * Output parameter
     * cors[]: correlation estimate found as the third dimension of 
     *         the covariance found in xytbl 
     *
     */
    int xyNeg, ix0, ix1, ixm, ixy;
    float xy0, xy1, p, pm, xy, corr;
  
    for (ixy=0; ixy<nxy; ixy++) {
        xy = xys[ixy];   /* For speed */
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
         * Find ix in [0..nx-2] such that 
         * xy is in (xytbl[s1,s2,ix] .. xytbl[s1,s2,ix+1])
         */   
        ix0 = 0;
        ix1 = nr - 1;
        xy0 = xy1 = NAN;

        if (xy == 0.0) {   /* No need to search */
            cors[ixy] = 0.0;
            continue;   // ============================================== >>>
        }

        /* Approximate xytbl at (is1,is2,nr-1) */

        p = bilinearf(ns, nr, sig, xytbl, s1[ixy], s2[ixy], ix1);

        if (xy > p) {
            //printf("In corr_estim(): input <xy>=%g is too large for\n", xy);
            //printf("the given values of s1[%d]=%g and s2[%d]=%g\n", 
            //	   ixy, s1[ixy], ixy, s2[ixy]);
            //printf("Assumed correlation 0.9999999\n");
            if (xyNeg) 
                cors[ixy] = -0.9999999;
            else
                cors[ixy] =  0.9999999;
            continue;   // ============================================== >>>
        }
  
        while (ix1 - ix0 > 1) {               /* Bisection search */
            ixm = (ix0 + ix1)/2;
            /* Approx xytbl at (is1,is2,ixm) */
            pm = bilinearf(ns, nr, sig, xytbl, s1[ixy], s2[ixy], ixm);

            // print 'ix0=%d, ix1=%d, ix0+ix1=%d, ixm=%d, pm=%g' %	
            //       (ix0, ix1, ix0+ix1, ixm, pm)
            if (xy > pm) {
                ix0 = ixm;
                xy0 = pm;
            }
            else if (xy < pm) {
                ix1 = ixm;
                xy1 = pm;
            }
            else { /* if (xy == pm) */
                ix0 = ixm;  /* xy is found exactly at ix0 */
                xy0 = pm;
                break;
            }
        }
     

        if (isnan(xy0)) 
            xy0  = bilinearf(ns, nr, sig, xytbl, s1[ixy], s2[ixy], ix0);
        if (isnan(xy1)) 
            xy1  = bilinearf(ns, nr, sig, xytbl, s1[ixy], s2[ixy], ix0+1);

        /*
         * Linear interpolation
         */
        corr = rho[ix0] + ((rho[ix1]-rho[ix0])/(xy1-xy0))*(xy - xy0);

        /*
         * Symmetry wrt the coordinate origin:
         */
        if (xyNeg) corr = -corr;
	
        cors[ixy] = corr;
    }
    return; 
}                   /* corr_estimatef() */




int std_estimatef(int nqs, int nqc, float (*qsig)[nqc], int nbit,
                  int nsquant, float *squant, float *sanalog) {
    /*
	 * Single precision version of std_estimate().
	 * 
     * Estimate the 'analog' standard deviation of a signal. 
     * via linear interpolation over the qsig table.
     * The zeroth column of qsig table contains the STD ruler for all other
     * columns.
     *
     * Parameters:
     * nqs: qsig table height; number of distinct STDs.
     * nqc: qsig table width;  number of quantization bits (2,3,4,5) + 1 ruler.
     * qsig[nqs][nqc]: table, column 0 has the values of analog std,
     *                   columns 1 to 4 contain corresponding std's
     *                   of signals quantized with 2 to 5 bit precision.
     * nbit: number of bits of quantization
     * nsquant: number of STDs to correct (i.e. estimate their analog values).
     * squant[nsquant]: array of input quantized STDs.
     *
     * Returns results in
     * sanalog[nsquant], estimates of analog STDs obtained 
     *          by the lookup and linear interpolation of a colum in 
     *          the analog STDs table qsig[nqs][nqc].
     *          sanalog[nsquant] must be allocated before the call to
     *          std_estimate()
     */
    int i0, i1, im, isq, ncol = nbit - 1;
    float s0, s1, sm, coef, sqnt;
  
    for (isq=0; isq<nsquant; isq++) {
        sqnt = squant[isq];  /* For speed -- to not access the squant array */

        /*
         * Find i such that squant[isq] is within 
         * (qsig[i,ncol] .. qsig[i+1,ncol])
         */   
        i0 = 0;
        i1 = nqs - 1;

        if (sqnt == 0.0) {
            sanalog[isq] = 0.0;   /* Search not needed */
            continue;  // ============================================== >>>
        }

        if (sqnt > qsig[i1][ncol] || sqnt < qsig[i0][ncol]) {
            //printf("std_estimate(): input squant[%d]=%g is beyond the qsig " 
            //	   "table.\n", isq, sqnt);
            return 1;
        }

        while(i1 - i0 > 1) {
            im = (i0 + i1)/2;
            sm = qsig[im][ncol];  
            if (sqnt > sm) {
                i0 = im;
                s0 = sm;
            }
            else if (sqnt < sm) {
                i1 = im;
                s1 = sm;
            }
            else { /* sqnt == sm */
                i0 = im;  /* squant[isq] is found exactly at i0 */
                s0 = sm;
                break;
            }
        }

        i1 = i0 + 1; /* Found squant[isq] between i0 and i0+1. Set i1 to i0+1 */

        /*
         * Linear interpolation of qsig[:][0], the analog std. 
         */
        coef = (qsig[i1][0] - qsig[i0][0])/(s1 - s0);
        sanalog[isq] = qsig[i0][0] + coef*(squant[isq] - s0);
    }

    return 0;
}                  /* std_estimatef() */







