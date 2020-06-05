/*
 * save_sim_cuda.cu
 *
 * Calculate correlation of very long random sequences and make the
 * van Vleck correction of the correlation. The computation of ~10^10
 * sequences take ~3 minutes due to the use of Nvidia CUDA GPU.
 *
 * Compilation/linking:
 *
 * $ nvcc -g -arch=sm_30  vanvleck.cu calc_corr.cu save_sim_cuda.cu -lm -lgsl \
 *        -lgslcblas -o ssc
 *
 * Example:
 *
 * $ ./ssc  1e6  160  64  4  2.0 1.75
 *
 * -- use 160 blocks times 64 thread per block times 1^6 samples in each
 *    thread, which makes 10,240,000,000 samples on total.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <gsl/gsl_integration.h>
#include <locale.h>
#include "vanvleck.cuh"

#define pi M_PI
#define DtoH cudaMemcpyDeviceToHost
#define HtoD cudaMemcpyHostToDevice

double fun(double x, void *params);
double calc_vanvleck(double std[2], int nbit, int npoint, \
                     double *kappa, double *rho);


int main(int argc, char *argv[]) {

    long iseq, nseq;  /* Number of parallel sequences in GPU */
    int nblk;
    int nthr;
    //int nthr = nseq/nblk;
    double anblk, anthr;

    cudaError_t error;

    // int ns = 315;  /* Length of the sigma ruler */
    // int nr = 101;  /* Length of the rho ruler */
    // double *sigr = (double *) malloc(ns*sizeof(double)); /*Ruler for sigma */
    // double *rhor = (double *) malloc(nr*sizeof(double)); /* Ruler for rho */
    // double (*rho_3dtable)[NS][NR] = 
    //  (double (*)[NS][NR]) malloc(sizeof(double[ns][ns][nr])); /*<xy> table */
    //(double (*)[ns][nr]) malloc(sizeof(double[ns][ns][nr])); /* <xy> table */
    // double (*vvslp)[12][12] = 
    //   (double (*)[12][12]) malloc(sizeof(double[3][12][12])); /* vv slopes */

    //unsigned int seed;

    int npoint = 101;       /* Points in interpolated correlation table */
    int npos = npoint/2;
    int nbit, isn, ibit;
    long i;  //, j, k;
    long nrand, ntotal;
    double anrand, antotal;
    double *kap = (double *) malloc(sizeof(double[npoint]));
    double *rho = (double *) malloc(sizeof(double[npoint]));
    int nsnr = 8;
    /* Theoretical correlations thrho[8] */
    double thrho[] = {0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999};
    double snr[nsnr];
    // int nsig = 12;
    // double sigma[] = 
    //     {0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0};
    int nnb = 3, nbits[] = {2, 3, 4};

    double  r, qr, acx, acy, sx, sy, slp;

    double *vcor =  (double *) malloc(sizeof(double[nsnr]));
    double *cor =  (double *) malloc(sizeof(double[nsnr]));
    double *qcov = (double *) malloc(sizeof(double[nsnr]));
    double *sqcor = (double *) malloc(sizeof(double[nsnr]));
    double *dq = (double *) malloc(sizeof(double[nsnr]));
    double *dv = (double *) malloc(sizeof(double[nsnr]));

    calc_corr_input ci; 
    calc_corr_rnd   *rnd_h, *rnd_d; 
    calc_corr_result *cr_h, *cr_d; 

    int verbose = 1;
    char *cnrand, *csx, *csy;

    clock_t start_time, end_time;
    double cpu_time_used;
    long time_int;

    start_time = clock();

    setlocale(LC_NUMERIC, ""); /* To print like 1,000,000, */

    if (argc != 7)  {
        //printf("Usage:\n save_sim_real nrand nbit sig1 sig2 \n");
        printf("Usage:\n");         
        printf("save_sim_real nrand nblocks nthreads nbit sig1 sig2\n\n");
        printf("   nrand: number of random samples in one sequence (thread)\n");
        printf("   nblocks: number of blocks of threads\n");
        printf("   nthreads: number of threads per one block\n");
        printf("   nbit: ADC precision in number of bits\n");
        printf("   sig1, sig2: standard deviations of the correlated analog "
               "signals\n\n");
       return 0;
    }

    sscanf(argv[1], "%lf", &anrand);
    sscanf(argv[2], "%lf", &anblk);
    sscanf(argv[3], "%lf", &anthr);
    sscanf(argv[4], "%d", &nbit);
    sscanf(argv[5], "%lf", &sx);
    sscanf(argv[6], "%lf", &sy);
    cnrand = strdup(argv[1]);
    csx = strdup(argv[5]);
    csy = strdup(argv[6]);

    double std[] = {sx, sy};
    
    nrand = (long) anrand;
    nblk = (int) anblk;
    nthr = (int) anthr;
    nseq = nblk*nthr;
    antotal = nseq*anrand;
    ntotal = (long) antotal;

    if (nthr > 1024) {
        printf("Number of threads per block cannot exceed 1024 "
               "(you specified %d))\n", nthr);
        return 0;
    }
 
    printf("nrand=%s, nblk=%d, nthr=%d, nbit=%d, sx=%g, sy=%g\n", 
           cnrand, nblk, nthr, nbit, sx, sy);
    printf("nseq = nblk x nthr = %d, ntotal = nseq x nrand = %'ld\n", 
           nseq, ntotal);

    /*
     * Find index ibit of nbit in the nbits[nnb] table
     */
    ibit = -1;
    for (i=0; i<nnb; i++)
        if (nbit == nbits[i]) ibit = i;
    
    if (ibit < 0) {
        printf("ERROR: incorrect value in command line for nbit.\n"
               "  Only the following values for nbit  allowed:\n");
        for (i=0; i<nnb; i++) printf("%2d ", nbits[i]);
        printf("\n\n");
        exit(0);
    }

    // /*
    //  * Find indices isx and isy of sx and sy in the sigma[nsig] table
    //  */
    // isx = isy = -1;
    // for (i=0; i<nsig; i++) {
    //     if (fabs(sx - sigma[i]) < 1e-6) isx = i;
    //     if (fabs(sy - sigma[i]) < 1e-6) isy = i;
    // }
    // if (isx < 0 || isy < 0) {
    //     printf("ERROR: incorrect value in command line for STD sx or sy.\n"
    //            "  Only the following values for sx or sy  allowed:\n");
    //     for (i=0; i<nsig; i++) printf("%4.2f ", sigma[i]);
    //     printf("\n\n");
    //     exit(0);
    // }

    /*
     * Device memory allocation: structures of arrays IS THE PAIN!
     * So we don't use them.
     */
    // typedef struct calc_corr_input {
    //     int nseq;
    //     long nrand;
    //     int nbit;
    //     double sx;
    //     double sy;
    //     double a;
    //     double b;
    //     } calc_corr_input;

    ci.nseq = nseq;
    ci.nrand = nrand;
    ci.nbit = nbit;
    ci.sx = sx;
    ci.sy = sy;

    rnd_h = (calc_corr_rnd *) malloc(sizeof(calc_corr_rnd[nseq]));
    cudaMalloc((void **) &rnd_d, sizeof(calc_corr_rnd[nseq]));

    cr_h = (calc_corr_result *) malloc(sizeof(calc_corr_result[nseq]));
    cudaMalloc((void **) &cr_d, sizeof(calc_corr_result[nseq]));

    // /*
    //  * Find Van Vleck curve slope for converting quantized covariance 
    //  * into correlation 
    //  */
    // int ierr = load_vvslopes(vvslp);
    // if (ierr) return 0;
    
    // slp1 = vvslp[ibit][isx][isy];
 
    // /* Load rulers and the table of <xy>'s */ 
    // ierr = load_rho_table(sigr, rhor, rho_3dtable); 
    // if (ierr) return 0;

    /*
     * Create array of nseq random seeds 
     */
    FILE* frand = fopen("/dev/urandom", "r");
    for (iseq = 0; iseq < nseq; iseq++) {
        fread(&(rnd_h[iseq].seed), sizeof(unsigned long long), 1, frand);
    }
    fclose(frand);
    /* Copy the seeds to device */
    cudaMemcpy(rnd_d, rnd_h, sizeof(calc_corr_rnd[nseq]), HtoD);


    /* Signal-to-noise ratio from theoretical correlation */
    for (i=0; i<nsnr; i++) snr[i] = sqrt(thrho[i]/(1. - thrho[i])); 

    /* 
     * Create table rho[kappa] to be interpolated and
     * find the rho = f(kappa) curve slope near rho = 0. 
     */
    
    slp = calc_vanvleck(std, nbit, npoint, kap, rho);


    ////////////////////////////////////
    printf("\n");
    for (i=npos; i<npoint; i++)
        printf("i=%3d\t kap[i]=%g,\t rho[i]=%g\n", i, kap[i],  rho[i]);
    printf("\n");
    ////////////////////////////////////
    

    printf("std1 = %g, std2 = %g, slp = %g\n\n", std[0], std[1], slp);

    /*
     * Setup the random number generators on the GPU
     */
    setup_randgen<<<nblk,nthr>>>(rnd_d, nseq);
    

    for (isn = 0; isn < nsnr; isn++) {

        ci.a = snr[isn] / sqrt(1. + pow(snr[isn],2));
        ci.b = sqrt(1. - pow(ci.a,2));

        calc_corr<<<nblk,nthr>>>(ci, rnd_d, cr_d, nseq); /* GPU computation */
        cudaDeviceSynchronize();

        // check for error
        error = cudaGetLastError();
        if(error != cudaSuccess)
            {
                // print the CUDA error message and exit
                printf("CUDA error: %s\n", cudaGetErrorString(error));
                //exit(-1);
            }
        cudaMemcpy(cr_h, cr_d, sizeof(calc_corr_result[nseq]), DtoH);

        r = qr = acx = acy = 0;
        for (iseq = 0; iseq<nseq; iseq++) {
            r   += cr_h[iseq].r;
            qr  += cr_h[iseq].qr;
            acx += cr_h[iseq].acx;
            acy += cr_h[iseq].acy;
        }

        cor[isn] = r/sqrt(acx * acy);
        sqcor[isn] = slp * qr / antotal;
        qcov[isn] = qr/antotal;

        /*
         * Linear interpolation of rho = f(kappa)
         */
        vcor[isn] = lininterp_vv(npoint, kap, rho, qcov[isn]);

    }   /* for (isn = 0; isn < nsnr; isn++) */

    // struct {
    //     double r;
    //     double qr;
    //     double acx;
    //     double acy;
    // } calc_corr_result;

    /*
     * Relative errors (%)
     */
    for (i=0; i<nsnr; i++) dv[i] = 100.*(vcor[i] - cor[i])/cor[i];
    for (i=0; i<nsnr; i++) dq[i] = 100.*(sqcor[i] - cor[i])/cor[i];

    printf("thrho = "); 
    for (i=0; i<nsnr; i++) printf("%.8f ", thrho[i]);
    printf("\n"); 
    printf("snr =   "); 
    for (i=0; i<nsnr; i++) printf("%.8f ", snr[i]);
    printf("\n"); 
    printf("qcov =  "); 
    for (i=0; i<nsnr; i++) printf("%.8f ", qcov[i]);
    printf("\n"); 
    printf("cor =   "); 
    for (i=0; i<nsnr; i++) printf("%.8f ", cor[i]);
    printf("\n"); 
    printf("vcor =  "); 
    for (i=0; i<nsnr; i++) printf("%.8f ", vcor[i]);
    printf("\n"); 
    printf("sqcor = "); 
    for (i=0; i<nsnr; i++) printf("%.8f ", sqcor[i]);
    printf("\n"); 
    printf("dq =    "); 
    for (i=0; i<nsnr; i++) printf("%10.5f ", dq[i]);
    printf("\n"); 
    printf("dv =    "); 
    for (i=0; i<nsnr; i++) printf("%10.5f ", dv[i]);
    printf("\n"); 
        

    /*
     * Save 
     */
    char fname[512], cntot[64];
    /* save in cntot the number with commas, like '10,240,000,000' */
    snprintf(cntot, 63, "%'ld", ntotal);
    /* Replace commas with dots, like '10.240.000.000', for better file name */
    i = 0; while (cntot[i]) { if (cntot[i] == ',') cntot[i] = '.'; i++; }
    snprintf(fname, 511, "stat/vvstat_%s_rn%s_bk%'d_tr%d_b%d_sx%4.2f_sy%4.2f"
        ".txt", cntot, cnrand, nblk, nthr, nbit, sx, sy);

    /*
     * Create directory stat, if it does not exist
     */
    system("mkdir -p stat");
    
    FILE *fh = fopen(fname, "w");

    fprintf(fh, "# ntot=%'ld;\n", ntotal);
    fprintf(fh, "ntot=%ld; nrnd=%s; nblk=%d; nthr=%d; nb=%d; "
            "sx=%s; sy=%s; slp=%g\n\n", 
            ntotal, cnrand, nblk, nthr, nbit, csx, csy, slp);

    fprintf(fh, "thrho = array([%.3f", thrho[0]);
    for (i=1; i<nsnr; i++) fprintf(fh, ", %.3f", thrho[i]);
    fprintf(fh, "])\n");

    fprintf(fh, "snr =  array([%.8f", snr[0]);
    for (i=1; i<nsnr; i++) fprintf(fh, ", %.8f", snr[i]);
    fprintf(fh, "])\n");

    fprintf(fh, "qcov = array([%.12f", qcov[0]);
    for (i=1; i<nsnr; i++) fprintf(fh, ", %.12f", qcov[i]);
    fprintf(fh, "])\n");

    fprintf(fh, "cor =  array([%.12f", cor[0]);
    for (i=1; i<nsnr; i++) fprintf(fh, ", %.12f", cor[i]);
    fprintf(fh, "])\n");

    fprintf(fh, "vcor = array([%.12f", vcor[0]);
    for (i=1; i<nsnr; i++) fprintf(fh, ", %.12f", vcor[i]);
    fprintf(fh, "])\n");

    fprintf(fh, "sqcor = array([%.12f", sqcor[0]);
    for (i=1; i<nsnr; i++) fprintf(fh, ", %.12f", sqcor[i]);
    fprintf(fh, "])\n");

    fprintf(fh, "dq =   array([%14.7f", dq[0]);
    for (i=1; i<nsnr; i++) fprintf(fh, ", %14.7f", dq[i]);
    fprintf(fh, "])\n");

    fprintf(fh, "dv =   array([%14.7f", dv[0]);
    for (i=1; i<nsnr; i++) fprintf(fh, ", %14.7f", dv[i]);
    fprintf(fh, "])\n");

   
    if (verbose) {
        int hr, mn, sc, msc;
		end_time = clock();
		cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
		time_int = (long) cpu_time_used;
        msc = (int) rint(1000*(cpu_time_used - time_int));
        mn = time_int/60;
        sc = time_int%60;
        hr = mn/60;
        mn = mn%60;
        printf("\n");
        printf("CPU_time_used: %7.3fs; ", cpu_time_used);
        fprintf(fh, "\n# CPU_time_used: %7.3fs; ", cpu_time_used);
        if (hr == 0) {
            printf(" as MM:SS = %02d:%02d.%03d\n", mn, sc, msc);
            fprintf(fh, " as MM:SS = %02d:%02d.%03d\n", mn, sc, msc);
        }
        else {
            printf(" as HH:MM:SS = %02d:%02d:%02d.%03d\n", hr, mn, sc, msc);
            fprintf(fh, "as HH:MM:SS = %02d:%02d:%02d.%03d\n", hr, mn, sc, msc);
        }
        printf("\n");
    }
    
    fclose(fh);

    cudaFree(rnd_d);
    cudaFree(cr_d);

    free(rnd_h);
    free(cr_h);

    return 0;
}


double calc_vanvleck(double std[2], int nbit, int npoint, \
                     double *kappa, double *rho) {
    /* 
     * Find the Van Vleck correction curve kappa = f(rho),
     * kappa = <qx1,qx2> - quantized covariance (correlator output)
     * rho - true analog correlation,
     * Returns the slope of the van Vleck curve d_rho / d_kappa near kappa=0.
     *
     * Actually, rho is calculated below here, and kappa are computed via
     * integration of vvfun() at the rho points.
     * The results, both kappa and rho, are returned.
     */
    gsl_integration_workspace * w  = gsl_integration_workspace_alloc(1000);
    gsl_function F;
    double *p;
    double d_rho, d_kappa, slope = 0;

    int i;
    double xy, er;
    // int npoint = 101;
    int npos = npoint/2;
    double prec_order = 6;

    /* Prepare for GSL integration */ 
    F.function = &fun;
    F.params = malloc(3*sizeof(double));
    p = (double *) F.params;
    p[0] = std[0];
    p[1] = std[1];
    p[2] = (double) nbit;

    /*
     * True analog correlation rho.
     *
     * First, fill in rho starting from npos = npoint/2,
     * with (npos+1) numbers increasing exponentially from 1. to 10^prec_order.
     */
    log10space(0., prec_order, npos+1, rho+npos);

    /*
     * Now, using the large numbers in rho[npos+1..npoint-1], fill rho from
     * the middle in both directions with numbers (negative below npos) 
     * from -1+10^(-prec_order) to 1-10^(-prec_order)
     * They are denser towards +1.0 and -1.0.
     */
    for (i = npos; i < npoint; i++) {
        rho[i] = 1. - 1./rho[i];
        rho[npoint-i-1] = -rho[i];
    }
    rho[npos] = 0;

    /* 
     * Integrate vvfun() from 0 to rho to get xy, the estimate of <xy> 
     */
    for (i = npos+1; i < npoint; i++) {
        gsl_integration_qags (&F, rho[i-1], rho[i], 0, 1e-7, 1000, w, &xy, &er);
        kappa[i] = kappa[i-1] + xy;
        kappa[npoint-i-1] = -kappa[i]; /* kappa(rho) is an odd function */
        if (fabs(er) > 1e-6) 
           printf("+++ Integration error %g\n", fabs(er));
    }

    /*
     * Integrate vvfun() once again from 0 to some small d_rho = 0.0001 to 
     * calculate the rho = f(kappa) curve slope
     */
    d_rho = 0.0001;
    gsl_integration_qags (&F, 0., d_rho, 0, 1e-7, 1000, w, &d_kappa, &er);
    if (fabs(er) > 1e-6) 
        printf("+++ Integration error %g\n", fabs(er));

    slope = d_rho / d_kappa;
    
    gsl_integration_workspace_free(w);
    free(F.params);

    return slope;
}                          /* void calc_vanvleck() */



double fun(double x, void *params) {
    /* 
     * Driver for integration of the Van Vleck correction function vvfun()
     * with the use of gsl_integration_qags()
     * (QAGS adaptive integration with singularities) from GSL 
     * (GNU Science Library).
     * See https://www.gnu.org/software/gsl/manual/html_node/          \
     *                        QAGS-adaptive-integration-with-singularities.html 
     */
    double *p = (double *) params;
    double s1 = p[0];
    double s2 = p[1];
    double dnbit = p[2];
    int nbit = (int) dnbit;

    return vvfun(x, s1, s2, nbit); 
    }

