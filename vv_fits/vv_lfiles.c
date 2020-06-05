/*
 * vv_lfiles.c
 *
 * Van Vleck correction of the MWA correlator output in L-files format.
 *
 * Build the executable:
 *
 * (1) From the working copy from mwa-lfd repository:
 *
 * $ cd van_vleck/mwatools/build_lfiles
 * $ gcc -g -I../.. ../../vanvleckf.c vv_lfiles.c -o vv_lfiles -lm 
 *
 * (2) From the directory vv_fits/, downloaded (say, scp-ed) from 
 *     jansky.haystack.mit.edu:/data/vv_fits/
 *
 * $ cd vv_fits/
 * $ gcc -g vanvleckf.c vv_lfiles.c -o vv_lfiles -lm 
 *
 * Example:
 *
 * $ ./vv_lfiles  1130642936_20151104032841_gpubox11_00
 *
 * This will create the following L-files:
 * 
 * 1130642936_20151104032841_vv_gpubox11_00.LACSPC
 * 1130642936_20151104032841_vv_gpubox11_00.LCCSPC
 * 1130642936_20151104032841_vv_gpubox11_00.Lcorrs
 * 1130642936_20151104032841_vv_gpubox11_00.Lfactors
 * 1130642936_20151104032841_vv_gpubox11_00.Lstdevs
 *
 *  Created on: Aug 23, 2016
 *      Author: Leonid Benkevitch
 */

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <getopt.h>
#include <string.h>
#include <libgen.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include "vanvleckf.h"


void usage(void) {
    printf("\n");
    printf("vv_lfiles: Van Vleck correction of the MWA correlator output " \
           "in L-files format.\n");
    printf("\n");
    printf("Usage: \n");
    printf("./vv_lfiles L-filename_base\n");
    printf("\n");  
    exit(EXIT_FAILURE);
}



int main(int argc, char **argv) {

    int nst = 128, npol = 2;
    int nbase;  /* Number of baselines = nst*(nst+1)/2  */
    int nxpc;   /* Number of crosses per one channel = ninp*(ninp+1)/2 - ninp */
    int nchan = 32, ninp;
    int nauto, ncross;    /* Sizes of autos and crosses */

    char *output_file = NULL, *input_file = NULL;
    char *out_dir = NULL;
    char *bin_dir = "/usr/local/lib/van_vleck";

    int verbose = 1, factors = 1;  /* For devel only. Set to 0s for working */
    int n_samples = 20000;
    float t_intgr = 0.5;
    int n_files = 0;
    int create_out_dir = 0;

    clock_t start_time, end_time;
    double cpu_time_used;
    long time_int;

    if (argc == 1) usage();


    nbase = nst*(nst+1)/2;
    ninp = nst * npol;       /* = 256 */
    nxpc = ninp*(ninp+1)/2 - ninp;   /* Number of crosses per one channel */
    nauto = nchan*ninp*sizeof(float);
    ncross = nchan*nxpc*sizeof(float complex);

    printf("nst=%d, npol=%d, nchan=%d, nbase=%d\n", nst, npol, nchan, nbase);
    printf("ninp=%d, nxpc=%d, nauto=%d, ncross=%d\n", ninp,nxpc,nauto,ncross);
    printf("\n");

    /* Input arrays: variances and covariances */
    float (*autos)[nchan] = malloc(sizeof(float[ninp][nchan]));
    float complex (*cross)[nchan] = malloc(sizeof(float complex[nxpc][nchan]));

    /* Output arrays: van Vleck corrected variances and covariances */
    float (*vvars)[nchan] = malloc(sizeof(float[ninp][nchan]));
    float (*stdevs)[nchan] = malloc(sizeof(float[ninp][nchan]));
    float complex (*vvcov)[nchan] = malloc(sizeof(float complex[nxpc][nchan]));
    float complex (*vvcor)[nchan] = malloc(sizeof(float complex[nxpc][nchan]));
    float complex (*vvfact)[nchan] = malloc(sizeof(float complex[nxpc][nchan]));
    













    if (argc == 1) usage();

    int arg = 0;
    while ((arg = getopt(argc, argv, "b:i:o:t:n:f::hvc")) != -1) {

        switch (arg) {
        case 'h':
            usage();
            break;
        case 'b':
            bin_dir = strdup(optarg);
            break;
        case 'c':
            create_out_dir = 1;
            break;
        case 'n':
            n_files = atoi(optarg);
            break;
        case 'v':
            verbose = 1;
            break;
        default:
            usage();
            break;
        } 
    }



    if (optind+2 < argc) {
        fprintf(stderr, "More than two files specified.\n");
        exit(EXIT_FAILURE);
    }

    /*
     * Input and output can be specified without -i and -o
     */

    if (optind == argc-1) 
        input_file = strdup(argv[optind]);
    else if (optind == argc-2) {
        input_file = strdup(argv[optind]);
        output_file = strdup(argv[optind+1]);
    }
  

    
    if (input_file == NULL) {
        fprintf(stderr, "No input file provided.\n");
        exit(EXIT_FAILURE);
    }



    if (output_file == NULL) {
        /*
         * Build the output file name in the same directory as input file
         */
        output_file = (char *) calloc(sizeof(char[512]), 1);
        char *inpfn = strdup(input_file);
        char *gpuboxptr = strstr(inpfn, "gpubox");
        *(gpuboxptr - 1) = 0;  /* Terminate the filename before "gpubox_" */ 
        sprintf(output_file, "%s_vv_%s", inpfn, gpuboxptr);
    }
    else { /* Is output_file a directory? */
        DIR* dout = opendir(output_file);  
        if (dout) {                        /* Directory exists. */
            out_dir = strdup(output_file);
            closedir(dout);
            int len_out_dir = strlen(out_dir);
            if (out_dir[len_out_dir-1] == '/') out_dir[len_out_dir-1] = 0;
            /*
             * Build the output file name in the directory given at -o
             * as the output file name
             */
            free(output_file);
            output_file = (char *) calloc(sizeof(char[512]), 1);
            char *inpfn = strdup(input_file);
            char *outfn = basename(inpfn);
            char *gpuboxptr = strstr(outfn, "gpubox");
            *(gpuboxptr - 1) = 0;  /* Terminate the fname before "gpubox_" */ 
            sprintf(output_file, "%s/%s_vv_%s", out_dir, outfn, gpuboxptr);
        }

        else if (ENOENT == errno) {       /* Directory does not exist. */
            
            if (create_out_dir) {
                mode_t mode =  S_IRWXU | S_IRWXG | S_IRWXO;
                out_dir = strdup(output_file);
                fprintf(stderr, "Creating '%s' ...\n", out_dir);
                if ( mkdir(out_dir, mode) ) {
                    fprintf(stderr, "Failed to create directory '%s'.\n", 
                            out_dir);
                    exit(EXIT_FAILURE);
                }

                /*
                 * Build the output file name in the directory given at -o
                 * as the output file name
                 */
                free(output_file);
                output_file = (char *) calloc(sizeof(char[512]), 1);
                char *inpfn = strdup(input_file);
                char *outfn = basename(inpfn);
                char *gpuboxptr = strstr(outfn, "gpubox");
                /* Terminate the fname before "gpubox_" */
                *(gpuboxptr - 1) = 0;
                sprintf(output_file, "%s/%s_vv_%s", out_dir, outfn, gpuboxptr);
            }  /* if (create_out_dir) { */
            else { /* A file name */
              //remove(output_file); /* Delete old file if it already exists */
            }
        }
        else { /* Exists, but not a directory, hence a file name */
            //remove(output_file); /* Delete old file if it already exists */
        }
    }




    if (verbose) {
        printf("bin_dir=     '%s'\n", bin_dir);
        printf("input_file=    '%s'\n", input_file);
        printf("output_file =  '%s'\n", output_file);
    }









    return 0;


    //char *fbasename = strdup("1130642936_20151104032841_gpubox11_00");
    char *fbasename = argv[1];  /* The failename base is the only argument */
    char *vvbasename = malloc(sizeof(char[256]));

    /*
     * Build the output file names
     */
    char *bnam = strdup(fbasename);
    char *gpuboxptr = strstr(bnam, "gpubox");

    if (gpuboxptr == NULL) {
        fprintf(stderr, "\n");
        fprintf(stderr, "ERROR: filename base '%s' must contain " \
                "word 'gpubox'.\n", bnam);
        exit(EXIT_FAILURE);
    }

    *(gpuboxptr - 1) = 0;  /* Terminate the filename before "gpubox_" */ 
    sprintf(vvbasename, "%s_vv_%s", bnam, gpuboxptr);

    char *finp_autos_name = malloc(sizeof(char[256]));
    char *finp_cross_name = malloc(sizeof(char[256]));
    char *fout_autos_name = malloc(sizeof(char[256]));
    char *fout_cross_name = malloc(sizeof(char[256]));

    char *fout_stdevs_name = malloc(sizeof(char[256]));
    char *fout_corrs_name = malloc(sizeof(char[256]));
    char *fout_factors_name = malloc(sizeof(char[256]));

    sprintf(finp_autos_name, "%s.LACSPC", fbasename);
    sprintf(finp_cross_name, "%s.LCCSPC", fbasename);
    sprintf(fout_autos_name, "%s.LACSPC", vvbasename);
    sprintf(fout_cross_name, "%s.LCCSPC", vvbasename);

    sprintf(fout_stdevs_name,  "%s.Lstdevs",  vvbasename);
    sprintf(fout_corrs_name,   "%s.Lcorrs",   vvbasename);
    sprintf(fout_factors_name, "%s.Lfactors", vvbasename);


    FILE *finp_cross = fopen(finp_cross_name, "r");
    FILE *finp_autos = fopen(finp_autos_name, "r"); 

    if (finp_autos == NULL) {
        fprintf(stderr, "\n");
        fprintf(stderr, "ERROR: file '%s' does not exist.\n", finp_autos_name);
        exit(EXIT_FAILURE);
    }

    if (finp_cross == NULL) {
        fprintf(stderr, "\n");
        fprintf(stderr, "ERROR: file '%s' does not exist.\n", finp_cross_name);
        exit(EXIT_FAILURE);
    }


    if (verbose) {
        start_time = clock();
        printf("\n");
        printf("CPU timing started ...\n\n");
    }

    printf("fbasename= '%s'\n", fbasename);
    printf("vvbasename='%s'\n", vvbasename);
    printf("finp_autos_name='%s'\n", finp_autos_name);
    printf("finp_cross_name='%s'\n", finp_cross_name);
    printf("fout_autos_name='%s'\n", fout_autos_name);
    printf("fout_cross_name='%s'\n", fout_cross_name);
    printf("\n");
    printf("fout_stdevs_name='%s'\n",  fout_stdevs_name);
    printf("fout_corrs_name='%s'\n",   fout_corrs_name);
    printf("fout_factors_name='%s'\n", fout_factors_name);


    FILE *fout_autos = fopen(fout_autos_name, "w");
    FILE *fout_cross = fopen(fout_cross_name, "w");

    FILE *fout_stdevs =  fopen(fout_stdevs_name,  "w");
    FILE *fout_corrs =   fopen(fout_corrs_name,   "w");
    FILE *fout_factors = fopen(fout_factors_name, "w");


    /* 
     * Create tables for correction std and correlation 
     */
    float (*qsig)[NQC] = malloc(sizeof(float[NQS][NQC])); /* std table */
    float *sig_ruler = (float *) malloc(NS*sizeof(float)); 
    float *rho_ruler = (float *) malloc(NR*sizeof(float));
    float (*xy_table)[NS][NR] = malloc(sizeof(float[NS][NS][NR]));

    int nbit = 4;
    float an_samples = (float) n_samples;

    /*
     * Before the Van Vleck correction the sums of auto-products in the fits
     * file must be converted into true REAL VARIANCES:
     * - divided by n_samples
     * - divided by 2 because each component of complex product has 
     *   two equal parts
     * - multiplied by t_intgr, because it was normalized by t_intgr before
     *   saving in the fits file.
     *
     * Before the Van Vleck correction the sums of cross-products in the fits
     * file must be converted into true REAL COVARIANCES:
     * - divided by n_samples
     * - divided by 2 because each component of complex product has 
     *   two equal parts
     * - multiplied by t_intgr, because it was normalized by t_intgr before
     *   saving in the fits file.
     *
     * Both conversions are made via multiplication by the coefficient koef:
     *
     */
    float koef = 0.5 * t_intgr / an_samples;


    /*
     * Read into RAM the correction tables and rulers sig, rho. 
     */
    load_rho_tablef(bin_dir, sig_ruler, rho_ruler, xy_table);
    load_qsig_tablef(bin_dir, qsig);

    if (verbose) 
        printf("VV tables loaded from '%s'.\n", bin_dir);

    // return 0;

    int icnt = 0;
    int ichan, icross, inp, inp1, inp2;
    float qvar, qvar1, qvar2, qstd, qstd1, qstd2, sig1, sig2;
    float complex qcov, uqcov, cov, ucov, rho;
    float qcov_re, qcov_im, uqcov_re, uqcov_im, rho_re, rho_im, std, var, uvar;


    /*
     * MAIN LOOP
     */
    while(fread((float *)autos, nauto, 1, finp_autos) != 0) {

        fread((float complex *)cross, ncross, 1, finp_cross);

        /*
         * First estimate the true, anslog standard deviations
         * from the unnormalized "auto-covariances" (i.e. variances)
         */
        for (ichan = 0; ichan < nchan; ichan++) {
            for (inp = 0; inp < ninp; inp++) {

                qvar = autos[inp][ichan];
                qstd = sqrt(koef * qvar);
 
                /*
                 * Estimate the analog standard deviation 
                 * from the quantized variation and save 
                 * it in the stdevs[] array.
                 */
                std = std_estimf(qsig, nbit, qstd);
                stdevs[inp][ichan] = std;
            }
        } /* for (ichan = 0; ichan < nchan; ichan++) */



        /* 
         * Make the van Vleck correction
         */
        for (ichan = 0; ichan < nchan; ichan++) {
            icross = 0;
            for (inp1 = 0; inp1 < ninp-1; inp1++) {
                for (inp2 = inp1+1; inp2 < ninp; inp2++) {
                    
                    sig1 = stdevs[inp1][ichan];
                    sig2 = stdevs[inp2][ichan];
                    
                    //printf("sig1 = %g, sig2 = %g\n", sig1, sig2); 
                    
                    uqcov = cross[icross][ichan];
                    uqcov_re =  crealf(uqcov);
                    uqcov_im =  cimagf(uqcov);

                    //printf("koef=%g, cross[icross][ichan] = %g + I*%g\n", 
                    //       koef, uqcov_re, uqcov_im);
                    
                    qcov = koef * uqcov;

                    qcov_re =  crealf(qcov);
                    qcov_im =  cimagf(qcov);

                    rho_re = corr_estimf(sig_ruler, rho_ruler, xy_table, \
                                         sig1, sig2, qcov_re);
                    rho_im = corr_estimf(sig_ruler, rho_ruler, xy_table, \
                                         sig1, sig2, qcov_im);

                    rho = rho_re + I*rho_im;

                    //printf("qcov = %g + I*%g;  rho = %g + I*%g\n", 
                    //       qcov_re, qcov_im, rho_re, rho_im);

                    /*
                     * Turn corrected correlation rho 
                     * into corrected covariance cov 
                     */
                    cov = sig1 * sig2 * rho; /* True covariance, <xy> */
                    ucov = cov / koef;       /* Un-normalized covariance */

                    //printf("cov = %g + I*%g;  ucov = %g + I*%g\n", 
                    //       crealf(cov), cimagf(cov), 
                    //       crealf(ucov), cimagf(ucov));

                    /*
                     * Save the unnormalized covariance, correlation rho,
                     * and the factor in the output arrays
                     */
                    vvcov[icross][ichan] = ucov;
                    vvcor[icross][ichan] = rho;
                    if (abs(qcov) != 0.) 
                        vvfact[icross][ichan] = cov / qcov;
                    else
                        vvfact[icross][ichan] = 1.;   /* treat 0/0 = 1 */

                    //printf("icross = %d\n", icross);

                    icross++;
                }
            }
        }

        /*
         * Convert the true, anslog standard deviations
         * into the unnormalized "auto-covariances" (i.e. variances)
         */
        for (ichan = 0; ichan < nchan; ichan++) {
            for (inp = 0; inp < ninp; inp++) {

                std = stdevs[inp][ichan];
                var = std*std;
                uvar = var / koef;

                vvars[inp][ichan] = uvar;
            }
        } /* for (ichan = 0; ichan < nchan; ichan++) */

        /*
         * Save both corrected variances and covariances in output files
         */
        fwrite((float *)vvars, nauto, 1, fout_autos);
        fwrite((float complex *)vvcov, ncross, 1, fout_cross);
        /*
         * Save corrected standard deviations, correlations, and factors
         * in output files
         */
        fwrite((float *)stdevs, nauto, 1, fout_stdevs);
        fwrite((float complex *)vvcor, ncross, 1, fout_corrs);
        fwrite((float complex *)vvfact, ncross, 1, fout_factors);


        printf("record # %d\n", icnt); 
        icnt++;
    }                 /* while() */


    fclose(finp_autos);
    fclose(finp_cross);
    fclose(fout_autos);
    fclose(fout_cross);

    fclose(fout_stdevs);
    fclose(fout_corrs);
    fclose(fout_factors);

    free(autos);
    free(cross);
    free(vvars);
    free(vvcov);

    free(stdevs);
    free(vvcor);
    free(vvfact);

    if (verbose) {
        int hr, mn, sc, msc;
		end_time = clock();
		cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
        //printf("start_time=%g, end_time=%g, CLOCKS_PER_SEC=%d\n",
        //       (double) start_time, (double) end_time, CLOCKS_PER_SEC);
		time_int = (long) cpu_time_used;
        msc = (int) rint(1000*(cpu_time_used - time_int));
        mn = time_int/60;
        sc = time_int%60;
        hr = mn/60;
        mn = mn%60;
        printf("\nCPU time used: %7.3fs; ", cpu_time_used);
        if (hr == 0) {
            printf(" as MM:SS = %02d:%02d.%03d\n", mn, sc, msc);
        }
        else {
            printf(" as HH:MM:SS = %02d:%02d:%02d.%03d\n", hr, mn, sc, msc);
        }
    printf("\n");

    }

    return 0;
}
