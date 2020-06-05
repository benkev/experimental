/*
 * vv_fits_noshm.c
 *
 * This is a deprecated old version of vv_fits.c. It uses no shared memory
 * the van Vleck tables. Each copy of vv_fits loads its own copy of the 
 * tables taking ~400 MB of memory. Using the shared memory for the tables
 * allows to run hundreds of vv_fits copies simultaneously, each using 
 * the same tables.   
 *
 * Van Vleck correction of the MWA correlator output files in FITS format. 
 *
 * By default, the output format as that of the input fits file:
 * the data are saved as 32-bit integers with the scaling factor 0.5. When 
 * read, they are transparently converted into a specified type, e.g. TFLOAT 
 * or TDOUBLE.
 *
 * Note: this integer representation implies very bad precision for small
 * (less than hundreds or thousand) values of variances and covariances.
 *
 * Threfore, the option -t <type> has been added. It means "save as type" and
 * <type> can take three symbols: i32 (default, int32), f32 (float32)
 * or f64 (float64).
 *
 * I added the -n <# of HDUs> option, mostly to accelerate debugging. 
 * See Example below.
 *
 * Build the executable:
 *
 * $ cd vv_fits/
 * $ gcc -g   vanvleckf.c  mwac_utils.c antenna_mapping.c  vv_fits.c  \
 *             -o vv_fits  -lcfitsio  -lm      
 *
 * One can obtain the vv_fits/ from the Subversion repository using
 * the command:
 *
 * $ svn co svn+ssh://benkev@mwa-lfd.haystack.mit.edu/svn/MISC/vv_fits .     
 *
 *
 * Examples:
 *
 * $ ./vv_fits -i 1130642936_20151104032841_gpubox11_00.fits -f -t f32
 *
 * $ ./vv_fits -i 1130642936_20151104032841_gpubox11_00.fits \
 *             -o 1130642936_20151104032841_vv_gpubox11_00 -f -v 
 * 
 * For debugging, to prevent vv_fits from lengthy processing all the 120 HDUs,
 * one can limit the number of HDUs to write to the output, using -n <# of HFUs>
 * For example, for me 1 HDU is enough, and I invoke vv_fits this way:
 *
 * $ ./vv_fits -i 1130642936_20151104032841_gpubox11_00.fits \
 *             -v -f -o vv_out.fits -t f32 -n 1
 *
 *  Created on: Jul 20, 2016
 *      Author: Leonid Benkevitch
 * Based on build_lfiles.c by Steven Ord, Jul 20, 2012
 */

//#include <ctype.h>
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
#include "mwac_utils.h"
#include "antenna_mapping.h"
#include "vanvleckf.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>


//#include "fitsio.h"
#include <fitsio.h>

/* globals */
FILE *fpd;        //file hand for debug messages. Set this in main()
int npol;
int nsta;
int nfreq;
int debug=0;


void printerror(int status)
{
    /*****************************************************/
    /* Print out cfitsio error messages and exit program */
    /*****************************************************/

    char status_str[FLEN_STATUS], errmsg[FLEN_ERRMSG];

    if (status)
        fprintf(stderr, "\n*** Error occurred during program execution ***\n");

    fits_get_errstatus(status, status_str);   /* get the error description */
    fprintf(stderr, "\nstatus = %d: %s\n", status, status_str);

    /* get first message; null if stack is empty */
    if ( fits_read_errmsg(errmsg) )
        {
            fprintf(stderr, "\nError message stack:\n");
            fprintf(stderr, " %s\n", errmsg);

            while ( fits_read_errmsg(errmsg) )  /* get remaining messages */
                fprintf(stderr, " %s\n", errmsg);
        }

    exit( status );       /* terminate the program, returning error status */
}



void usage() {

    fprintf(stdout,"vv_fits:\n");
    fprintf(stdout,"Make van Vleck correction on a gpufits file, " \
            "the output of MWA GPU correlator\n");
    fprintf(stdout,"Usage:\n");
    fprintf(stdout,"\tvv_fits [options] <input> <output>\n");
    fprintf(stdout,"or\n");
    fprintf(stdout,"\tvv_fits [options] -i <input> -o <output>\n");
    fprintf(stdout,"Options:\n");
    fprintf(stdout,"\t-i\t <input visibility gpu fits file name> \n");
    fprintf(stdout,"\t-b\t <binary tables directory> [default: " \
            " /usr/local/lib/van_vleck\n");
    fprintf(stdout,"\t-o\t <output fits filename> [default: same as in -i "
            " with '_vv_' inserted before 'gpubox']\n"
            "\t\t   If -o is followed not by a file, " \
            "but by a directory name,\n" \
            "\t\t   then the default filename is written " \
            "to the -o <directory>\n");
    fprintf(stdout,"\t-c\t Create the directory for output specified " \
            "at -o, if it does not exist\n");
    fprintf(stdout,"\t-t\t <data type of output fits file> " \
            "[default: f32 (unlike i32 in input fits)\n");
    fprintf(stdout,"\t-n\t <number of data HDUs in output fits file> \n" \
            "\t\t    [default: -n 0 (write all HDUs. " \
            "As many as there are in the input fits file)]\n");
    fprintf(stdout,"\t-f\t Write a separate fits file with the correction " \
            "factors of float32 type.\n");
    fprintf(stdout,"\t\t   The factors file name is the same as in -i "
            " with '_vvfactors_' inserted before 'gpubox'.\n" \
            "\t\t   If the output directory is specified in -o, " \
            "then the factors file is written to the " \
            "specified directory.\n");
    fprintf(stdout,"\t\t   This option can provide a file name for the"
            " factors file, if used as -f=<name>.fits or -f<name>.fits.\n");
    fprintf(stdout,"\t-v\t print verbose info\n");
    fprintf(stdout,"\t-h\t print this message\n");

    exit(EXIT_FAILURE);
}



void bscale_bzero_in_header(fitsfile *fout, 
                            int *BSCALE_in_hdr, int *BZERO_in_hdr) {
    /*
     * Find out if the current header contains the BSCALE and BZERO
     * keywords 
     */
    int nkeystotal = 0, morekeys = 0, status = 0, ikey = 0;
    char *record =  malloc(128*sizeof(char));

    if ( fits_get_hdrspace(fout, &nkeystotal, &morekeys, &status) )
        printerror( status );

    //printf("after fits_get_hdrspace, nkeystotal=%d, morekeys=%d\n", 
    //       nkeystotal, morekeys);

    *BSCALE_in_hdr = 0;
    *BZERO_in_hdr = 0;

    for (ikey = 1; ikey <= nkeystotal; ikey++) {
        if ( fits_read_record(fout, ikey, record, &status) )
            printerror( status );
        if (strstr(record, "BSCALE")) *BSCALE_in_hdr = 1;
        if (strstr(record, "BZERO"))  *BZERO_in_hdr = 1;
        //printf("record # %d = '%s\n", ikey, record);
    }

}



int main(int argc, char **argv) {

    char *input_file = NULL, *output_file = NULL, *output_file2 = NULL;
    char *out_dir = NULL;
    //char *bin_dir = "/usr/local/lib/van_vleck";
    char *bin_dir = ".";
    char *fopt = NULL;
    fitsfile *finp;     /* FITS input file ptr, defined in fitsio.h */
    fitsfile *fout;     /* FITS output file ptr, defined in fitsio.h */
    fitsfile *fout2;    /* FITS output file ptr, defined in fitsio.h */
    float complex *cuda_matrix_h = NULL;
    float complex *cuda_matrix_vv = NULL;
    float complex *cuda_matrix_vvfactors = NULL;
    float complex *full_matrix_h = NULL;
    float *stdevs = NULL;
    int ier = 0, nhdus = 0;
    int status = 0; /*  CFITSIO status value MUST be initialized to zero!  */
    int hdutype = 0, ncols = 0; 
    long nrows=0;
    long dims[2];
    int ninput = 0;
    size_t nvis=0;
    size_t nbl=0;  /* Number of baselines, = nsta*(nsta+1)/2 */
    size_t matLength=0, lccspcLength=0, lacspcLength=0, fullLength=0;
    size_t autoLength = 0;
    long fpixel = 1;
    float nullval = 0; /* nullval: The value set for the undefined elements */
    /* anynull is set to True if any of the read elements are set undefined  */
    int anynull = 0x0;
    long naxes[2];
    long firstelem = 1;
    long nelems = 0;
    int ihdu;
    int verbose = 0, factors = 0;
    int n_samples = 20000;
    float t_intgr = 0.5;
    int n_hdus = 0;
    int i;
    int create_out_dir = 0; /* If the directory at -o does not exist, create */

    int BSCALE_in_hdr = 0;
    int BZERO_in_hdr = 0;
    int ikey = 0;
    char *keyname = malloc(128*sizeof(char)); 
    char *record =  malloc(128*sizeof(char));
    
 

    /* Load tables for correction std and correlation */
    float (*qsig)[NQC] = malloc(sizeof(float[NQS][NQC])); /* std table */
    float *sig_ruler = (float *) malloc(NS*sizeof(float)); 
    float *rho_ruler = (float *) malloc(NR*sizeof(float));
    float (*xy_table)[NS][NR] = malloc(sizeof(float[NS][NS][NR]));
    // double (*qsig)[NQC] = malloc(sizeof(double[NQS][NQC])); /* std table */
    // double *sig_ruler = (double *) malloc(NS*sizeof(double)); 
    // double *rho_ruler = (double *) malloc(NR*sizeof(double));
    // double (*xy_table)[NS][NR] = malloc(sizeof(double[NS][NS][NR]));

    // clock_t start_time, end_time;
    // double cpu_time_used;
    // long time_int;

    float koef;
    int nbit = 4;

    int numtype = -32;  /* == BITPIX. Default output fits format is float32 */
    char *save_as_type = strdup("f32");

    // if (verbose) {
    //    start_time = clock();
    //    printf("CPU timing started ...\n\n");
    //}

    npol = 2;

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
        case 'i':
            input_file = strdup(optarg);
            break;
        case 'o':
            output_file = strdup(optarg);
            break;
        case 'c':
            create_out_dir = 1;
            break;
        case 't':
            free(save_as_type);
            save_as_type = strdup(optarg);
            break;
        case 'n':
            n_hdus = atoi(optarg);
            break;
        case 'f':
            factors = 1;
            if (optarg != NULL) 
                fopt = strdup(optarg); 
            else
                fopt = NULL;
            break;
        case 'v':
            verbose = 1;
            break;
        default:
            usage();
            break;
        } 
    }


    /*
     * By default, getopt() permutes the contents of argv as it scans, 
     * so that eventually all the nonoptions are at the end.
     * Thus optind is the index in argv of the first argv-element 
     * that is not an option. 
     */

    if (optind+2 < argc) {
        fprintf(stderr, "More than two files specified.\n");
        exit(EXIT_FAILURE);
    }

    /*
     * Input and output can be specified without -i and -o
     */
    if (input_file == NULL) { /* If there were no -i option */
        if (optind == argc-1)  /* If only one non-option argument */
            input_file = strdup(argv[optind]); /* then it is input file */
        else if (optind == argc-2) { /* If there are 2 non-option args */
            input_file = strdup(argv[optind]); /* Then they are in and out f.*/
            if (output_file == NULL) output_file = strdup(argv[optind+1]);
        }
    }
    else { /* If the input file was specified at the -i option */
        if (optind == argc-1)  /* If only one non-optional argument */
            output_file = strdup(argv[optind]); /* then it is output file */
    }





    if (input_file == NULL) {
        fprintf(stderr, "No input file provided.\n");
        exit(EXIT_FAILURE);
    }
    else if (strstr(input_file, "_gpubox") == NULL) {
        fprintf(stderr, "Wrong input file name: it must contain '_gpubox'.\n");
        exit(EXIT_FAILURE);
    }
 
    /*
     * Decide what BITPIX will be in the output fits: int32 or float32 
     * or float64  
     */
    if      (strcmp(save_as_type, "i32") == 0) {
        numtype = 32;
    }
    else if (strcmp(save_as_type, "f32") == 0) {
        numtype = -32;
    }
    else if (strcmp(save_as_type, "f64") == 0) { 
        numtype = -64;
    }
    else {
        fprintf(stderr,"Wrong output type -t %s. " \
                "Only i32, f32, or f64 allowed.\n", save_as_type);
        return EXIT_FAILURE;
    }

    if (verbose) printf("Type of output fits file: %s\n", save_as_type);




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


    
    if (factors && fopt) {

        /* 
         * The factors file name specified explicitly as 
         * -f=name.fits or -fname.fits 
         */ 

        if (fopt[0] == '=')
            output_file2 = strdup(fopt+1);
        else
            output_file2 = strdup(fopt);
    }

    else if (factors) {  /* the -f option without factors file name */

        /*
         * Build the name of output file with factors
         */

        char *input_dir = dirname(strdup(input_file));
        char *output_dir = dirname(strdup(output_file));

        if (strcmp(input_dir, output_dir) != 0) {
            /* 
             * Write factors file to the directory where the corrected
             * output is written 
             */
            out_dir = strdup(output_dir);
        }

        if (out_dir) { /* Write to the directory specified at -o dirname */
            output_file2 = (char *) calloc(sizeof(char[512]), 1);
            char *inpfn = strdup(input_file);
            char *outfn2 = basename(inpfn);
            char *gpuboxptr = strstr(outfn2, "gpubox");
            /* Terminate the filename before "gpubox_" */
            *(gpuboxptr - 1) = 0; 
            sprintf(output_file2, "%s/%s_vvfactors_%s", 
                    out_dir, outfn2, gpuboxptr);
        }
        else { /* Output file name in the same directory as input file */
            output_file2 = (char *) calloc(sizeof(char[512]), 1);
            char *inpfn = strdup(input_file);
            char *outfn2 = strdup(inpfn);
            char *gpuboxptr = strstr(outfn2, "gpubox");
            *(gpuboxptr - 1) = 0; /* Terminate the filename before "gpubox_" */ 
            sprintf(output_file2, "%s_vvfactors_%s", outfn2, gpuboxptr);
        //  remove(output_file2);
        }
    }



    if (verbose) {
        printf("bin_dir=     '%s'\n", bin_dir);
        printf("input_file=    '%s'\n", input_file);
        printf("output_file =  '%s'\n", output_file);
        if (factors) 
            printf("factors file = '%s'\n", output_file2);
    }


    // return 0;

    /* 
     * Delete old files if they already exist
     */
    remove(output_file);
    if (factors) remove(output_file2);


    /*
     * Read into RAM the correction tables and rulers sig, rho. 
     */
    load_rho_tablef(bin_dir, sig_ruler, rho_ruler, xy_table);
    load_qsig_tablef(bin_dir, qsig);


    if (verbose) 
        printf("The correction tables loaded from '%s'.\n", bin_dir);

    // return 0;


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
    koef = 0.5 * t_intgr / an_samples;

    /*
     * Open input FITS file
     */
    ier = fits_open_file(&finp, input_file, READONLY, &status);
    if (ier != 0) {
        fprintf(stderr,"Cannot open file %s\n", input_file);
        printerror(status);
        return EXIT_FAILURE;
    }


    /*
     * How many HDUs are there?
     */
    if (fits_get_num_hdus(finp, &nhdus, &status)) {
        printerror(status);
        return EXIT_FAILURE;
    }
    if (verbose) printf("There are %d HDUs in file %s\n", nhdus, input_file);


    /*
     * Create a new fits file
     */
    ier = fits_create_file(&fout, output_file, &status);
    if (ier != 0) {
        fprintf(stderr,"Cannot create file %s\n", output_file);
        printerror(status);
        return EXIT_FAILURE;
    }

    if (factors) {
        /*
         * Create a new fits file with factors = vv_rho / rho
         */
        ier = fits_create_file(&fout2, output_file2, &status);
        if (ier != 0) {
            fprintf(stderr,"Cannot create file %s\n", output_file2);
            printerror(status);
            return EXIT_FAILURE;
        }
    }

    /*
     * Copy first HDU. Only header. No data in 1st HDU.
     */
    int morekeys = 0;     /* don't reserve space for additional keywords */
    int nkeystotal = 0;   /* how many keys are there in the header */
    if ( fits_copy_hdu(finp, fout, morekeys, &status) )
         printerror( status );

    /*
     * Find out if the current header contains the BSCALE and BZERO
     * keywords 
     */
    bscale_bzero_in_header(fout, &BSCALE_in_hdr, &BZERO_in_hdr);   

    if (numtype != 32) { /* i.e. if not i32, when changes are not needed */
        /*
         * Change the output type to 32 or 64-bit float numtype
         * Remove BSCALE and BZERO keywords
         */

        if (verbose) printf("Change: numtype=%d\n", numtype);

        morekeys = 0;     /* don't reserve space for additional keywords */
        if ( fits_update_key(fout, TINT, "BITPIX", &numtype, NULL, &status))
            printerror( status );

        /*
         * Delete BSCALE and BZERO,if they exist in the output HDU header
         */

        if (BSCALE_in_hdr) {
           if ( fits_delete_key(fout, "BSCALE", &status) )
               printerror( status );
        }
        if (BZERO_in_hdr) {
           if ( fits_delete_key(fout, "BZERO", &status) ) {
               printerror( status );
           }
        }
    }

    if (factors) {
        /*
         * Copy first HDU to factors file. Only header. No data in 1st HDU.
         */
        int bitpix = -32;     /* Single precision, 32 bit floats */
        morekeys = 0;     /* don't reserve space for additional keywords */
        if ( fits_copy_hdu(finp, fout2, morekeys, &status) )
            printerror( status );
        if ( fits_update_key(fout2, TINT, "BITPIX", &bitpix, NULL, &status))
            printerror( status );

        /*
         * Delete BSCALE and BZERO,if they exist in the output HDU header
         */
        if (BSCALE_in_hdr) {
            if ( fits_delete_key(fout2, "BSCALE", &status) )
                printerror( status );
        }
        if (BZERO_in_hdr) {
            if ( fits_delete_key(fout2, "BZERO", &status) ) {
                printerror( status );
            }
        }
    }

    /*
     * Get image dimensions (2 means number of elements of the dims array)
     */
    fits_movabs_hdu(finp, 2, &hdutype, &status);
    fits_get_img_size(finp, 2, dims, &status);
    if (status) {
       fits_report_error(stderr, status);
       return status;
    }
    if (verbose) printf("Found %ld x %ld image\n", dims[0], dims[1]);

    /* 
     * Update globals based on in-file metadata 
     */
    /* The number of correlation products is slightly different to the 
     * standard n(n+1)/2 formula since there is a small fraction of 
     * redundant info in the file. If n is the number of stations 
     * (not stations*pols), and N is the dimension of the data in the 
     * file then n = (-1 + sqrt(1+N))/2
     */
    nsta = (((int) round(sqrt(dims[0]+1)))-1)/2;
    ninput = nsta*npol;
    nbl = nsta*(nsta+1)/2;
    nfreq = dims[1];
    /* nvis = (nsta + 1)*(nsta/2)*npol*npol; */
    nvis = nbl*npol*npol;
    if (verbose) {
        printf("There are %d stations in the data; ", nsta);
        printf("\tnpol: %d. nfreq: %d, nvis: %ld\n", npol, nfreq, (long)nvis);
    }
    /*
     * Now that we know nsta, create mapping matrix 
     * map_t corr_mapping[NINPUT][NINPUT]; or 
     * map_t corr_mapping[256][256].
     * An element of the mapping matrix has 4 integers:
     * two antenna numbers an two polarizations, stn1, stn2, pol1, and pol2.
     * Instead of thinking of 128 antennas, 2 polarizations and 
     * 4 polarization cross products, it's easiest to just look at it 
     * as 256 inputs.
     * The indices of corr_mapping are these "inputs", numbers in [0..255].
     * The matrix corr_mapping associates two inputs, its indices, with 
     * true positions of the cross- or auto-covariance products in the 
     * full Hermitian matrix full_matrix_h[freq,st1,st2,pol1,pol2].
     */

    fill_mapping_matrix();


    matLength = nfreq * nvis; // cuda_matrix length (autos on the diagonal)
    fullLength = nfreq * ninput*ninput; // Full Hermitian matrix length
    autoLength = nfreq * ninput; // Auto-covariances length

    /* 
     * now we know the dimensions of the data, allocate arrays 
     */
    cuda_matrix_h = (float complex *) malloc(matLength*sizeof(float complex));
    full_matrix_h = (float complex *) malloc(fullLength*sizeof(float complex));
    stdevs = (float *) malloc(autoLength*sizeof(float));
    cuda_matrix_vv = (float complex *) malloc(matLength*sizeof(float complex));
    if (factors) cuda_matrix_vvfactors = 
                     (float complex *) malloc(matLength*sizeof(float complex));


    int inp, inp1, inp2; /* Antenna and its polarization, 0..255 */
    int ioffs, s1, s2, p1, p2;
    int ifreq;
    map_t mp;
    float *ptrinp = NULL, *ptrout = NULL, *ptrout2 = NULL;

    size_t k = 0, istd = 0, istd1 = 0, istd2 = 0, idx = 0;
    float var, uvar, qvar, qstd, std, sig1, sig2;
    float qcov_re, qcov_im, rho_re, rho_im;
    //float qvarimag;
    float complex qcov, ucov, cov, rho, factor;

    ptrinp = (float *) cuda_matrix_h;
    ptrout = (float *) cuda_matrix_vv;
    ptrout2 = (float *) cuda_matrix_vvfactors;

    /*
     * The visibility data in the input gpu fits file HDUs for each 
     * fine channel are packed in lower triangular matrices, cuda_matrix_h,
     * with 2x2 complex elements for 2x2 polarizations:
     * |RR RL|
     * |LR LL|
     *
     * Although the data may be treated as complex64, the description 
     * below considers the as consisting of pairs, re and im, float32 each.
     * Since the data are real, float32, the matrix elements are 2x4 dims:
     *     \          ipol2
     * ipol1\    0      1      2      3
     *       -------------------------------
     *   0   || RRre | RRim | RLre | RLim ||
     *       -------------------------------
     *   1   || LRre | LRim | LLre | LLim ||
     *       -------------------------------
     *
     *
     * The  matrix elements are shown below (antenna numbers ij)
     * Each matrix row starts with the index i(i+1)/2:
     *-------------------------------------------------------------------
     *  i | i(i+1)/2|  Matrix elements
     *-------------------------------------------------------------------
     *  0 |     0   |  00
     *  1 |     1   |  10 11
     *  2 |     3   |  20 21 22
     *  3 |     6   |  30 31 32 33
     *  4 |    10   |  40 41 42 43 44
     *  5 |    15   |  50 51 52 53 54 55
     * .  .  .  .  .  .  .  .
     * 126|   8001  |  126,0  126,1  .  .  .  .  .  126,125  126,126
     * 127|   8128  |  127,0  127,1  .  .  .  .  .  127,125  127,126  127,127
     *

     *
     * Indices of the diagonal elements:
     * 00, 11, 22, ... : i(i+1)/2 + i = i(i+3)2:
     * 0,    2,    5,    9,   14,   20,   27, ... 8000, 8127, 8255.   
     *
     * i128 = arange(128)
     * idia = i128*(i128 + 3)/2
     *
     * The diagonal elements are: 00 11 22 33 44 .  .  .  . 126,126  127,127
     * [pol,pol,complexity]
     *
     * The standard deviations are RR and LL,
     * real parts (imaginary are negligibly small).
     *
     */

    /*
     * Start processing from 2nd HDU (the 1st has no data)
     */
    ihdu = 2;

    while (1) {

        /* 
         * Move to the next hdu and get the HDU type 
         */
        if (fits_movabs_hdu(finp, ihdu, &hdutype, &status)) {
            if (status == END_OF_FILE) {
                status = 0;
                break; // ========== leave while(1) loop when done ======= >>> 
            }
            printerror(status);
        }
        
        /*
         * Get dimensions of the current data block
         */
        if (fits_get_img_size(finp, 2, naxes, &status)) {
            printerror(status);
        }
        status = 0;
        nelems = naxes[0] * naxes[1];

        //printf("naxes[0]=%ld, naxes[1]=%ld, nelems=%ld\n", 
        //       naxes[0], naxes[1], nelems);

        /* 
         * copy current header from the input file to the output file
         */
        if ( fits_copy_header(finp, fout, &status) )
            printerror( status );

        /*
         * Find out if the current header contains the BSCALE and BZERO
         * keywords 
         */
        bscale_bzero_in_header(fout, &BSCALE_in_hdr, &BZERO_in_hdr);    
  

        if (numtype != 32) {
            /* 
             * Change the output type to 32 or 64-bit float numtype
             * Remove BSCALE and BZERO from the header parameters
             */
            if ( fits_update_key(fout, TINT, "BITPIX", &numtype, 
                                 NULL, &status))
                printerror( status );
            /*
             * Delete BSCALE and BZERO if they exist in the output HDU header
             */
            if (BSCALE_in_hdr) {
                if ( fits_delete_key(fout, "BSCALE", &status) )
                    printerror( status );
            }
            if (BZERO_in_hdr) {
                if ( fits_delete_key(fout, "BZERO", &status) ) {
                    printerror( status );
                }
            }
        } /* if (numtype != 32) */

        if (factors) {
            /* 
             * Copy current header from the input file to the factors file
             * Remove BSCALE and BZERO from the header parameters
             */
            int bitpix = -32;  /* Single precision, 32 bit floats */
            if ( fits_copy_header(finp, fout2, &status) )
                printerror( status );
            if ( fits_update_key(fout2, TINT, "BITPIX", &bitpix, NULL, &status))
                printerror( status );

            /*
             * Delete BSCALE and BZERO,if they exist in the output HDU header
             */
            if (BSCALE_in_hdr) {
                if ( fits_delete_key(fout2, "BSCALE", &status) )
                    printerror( status );
            }
            if (BZERO_in_hdr) {
                if ( fits_delete_key(fout2, "BZERO", &status) )
                    printerror( status );
            }
        }

        /*
         * Read data from current HDU
         * Use fits_read_img so that this will also work transparently 
         * for compressed data
         */
        ier = fits_read_img(finp, TFLOAT, firstelem, nelems, &nullval,
                            ptrinp, &anynull, &status);
        if (ier) {
            printerror(status);
        }

        /*
         * Complement the lower triangle form, packed Hermitian matrix 
         * cuda_matrix_h, read from the gpufits file, to the full, square 
         * Hermitian matrix full_matrix_h nsta X nsta
         */
        extractMatrix(full_matrix_h, cuda_matrix_h);

        /* 
         * Make the standard deviations' correction
         */
        for (ifreq = 0; ifreq < nfreq; ifreq++) {
            for (inp = 0; inp < ninput; inp++) {       // real-world input
                mp = corr_mapping[inp][inp];     // internal fits file numbers
                s1 = mp.stn1;
                s2 = mp.stn2;
                p1 = mp.pol1;
                p2 = mp.pol2;

                // ioffs = ((s1*nsta + s2)*npol + p1)*npol + p2; ERROR!
                ioffs = (((ifreq*nsta + s1)*nsta + s2)*npol + \
                         p1)*npol + p2;
                qvar = crealf(full_matrix_h[ioffs]);
                //qvarimag = cimagf(full_matrix_h[ioffs]);
                /*
                 * Turn quantized variance into standard deviation
                 */
                qstd = sqrt(koef * qvar);

                /*
                 * Estimate the analog standard deviation 
                 * from the quantized variation and save 
                 * it in the stdevs[] array in real-world order.
                 */
                std = std_estimf(qsig, nbit, qstd);
                istd = ifreq*ninput + inp;           /* [ifreq,inp] */
                stdevs[istd] = std;

                /*
                 * Save the corrected variance in the
                 * lower-triangle packed Hermitian matrics
                 * to be written as HDU data in output file 
                 */
                var = std*std;
                uvar = var / koef;

                k = ifreq*nbl + s1*(s1+1)/2 + s2;
                idx = (k * npol + p1)*npol + p2;

                cuda_matrix_vv[idx] = uvar;

                if (factors) {
                    if (qvar == 0 || isnan(qvar)) 
                        factor = 1. + 0.*I;
                    else
                        factor = uvar / qvar + 0.*I;
                    cuda_matrix_vvfactors[idx] = factor;
                }


                
                
                //printf("inp=%d, s1=%d, s2=%d, p1=%d, p2=%d, nbit=%d\n", 
                //       inp, s1, s2, p1, p2, nbit);
                //printf("qvar=%f, qvarimag=%f\n", qvar, qvarimag);
                //printf("istd=%ld, qstd=%f, std=%f\n", istd, qstd, std);
            
            }
        
        }

        /* 
         * Make the van Vleck correction
         */
        for (ifreq = 0; ifreq < nfreq; ifreq++) {
            for (inp1 = 0; inp1 < ninput; inp1++) {      // real-world inputs 
                for (inp2 = 0; inp2 < ninput; inp2++) {  

                    if (inp1 == inp2) continue; // Cross-covariances only! >>>

                    mp = corr_mapping[inp1][inp2]; // fits file internal
                    s1 = mp.stn1;
                    s2 = mp.stn2;
                    p1 = mp.pol1;
                    p2 = mp.pol2;

                    if (s2 > s1) continue; // Entries of cuda matrix only >>>


                    istd1 = ifreq*ninput + inp1;  /* [ifreq,inp1] */
                    sig1 = stdevs[istd1];

                    istd2 = ifreq*ninput + inp2;  /* [ifreq,inp1] */
                    sig2 = stdevs[istd2];

                    //printf("istd1=%ld, istd2=%ld, sig1=%f, sig2=%f\n", 
                    //       istd1, istd2, sig1, sig2);

                    ioffs = (((ifreq*nsta + s1)*nsta + s2)*npol + \
                             p1)*npol + p2;
                    qcov = koef * full_matrix_h[ioffs];
                    qcov_re =  crealf(qcov);
                    qcov_im =  cimagf(qcov);
                    rho_re = corr_estimf(sig_ruler, rho_ruler, xy_table, \
                                         sig1, sig2, qcov_re);
                    rho_im = corr_estimf(sig_ruler, rho_ruler, xy_table, \
                                         sig1, sig2, qcov_im);
                    rho = rho_re + I*rho_im;


                    if (abs(rho_re) > 0.9999999 || abs(rho_im) > 0.9999999) {
                        printf("s1=%d, s2=%d, p1=%d, p2=%d\n", s1, s2, p1, p2);
                        printf("rho_re=%10.8f, rho_im=%10.8f\n", 
                               rho_re, rho_im);
                        printf("sig1=%10.6f, sig2=%10.6f, " \
                               "qcov_re=%10.2f, qcov_im=%10.2f\n", 
                               sig1, sig2, qcov_re, qcov_im);
                    }


                    /* 
                     * Turn corrected correlation rho 
                     * into corrected covariance cov 
                     */
                    cov = sig1 * sig2 * rho; /* True covariance, <xy> */
                    ucov = cov / koef;       /* Un-normalized covariance */
                    /*
                     * Save the corrected covariance in the
                     * lower-triangle packed Hermitian matrics
                     * to be written as HDU data in output file 
                     */
                    k = ifreq*nbl + s1*(s1+1)/2 + s2;
                    idx = (k * npol + p1)*npol + p2;

                    cuda_matrix_vv[idx] = ucov;

                    if (factors) {
                        if (qcov == 0 || isnan(qcov)) 
                            factor = 1. + 0j;
                        else
                            factor = cov / qcov;
                        cuda_matrix_vvfactors[idx] = factor;
                        //printf("factor = %e %e*I\n",
                        //       crealf(factor), cimagf(factor));
                    }
                    if (rho_re == 1. || rho_im == 1.) {
                        printf("sig1 = %f, sig2 = %f*I\n", sig1, sig2);
                        printf("qcov = %f %f*I\n", qcov_re, qcov_im);
                        printf("rho = %f %f*I\n", rho_re, rho_im);
                        printf("cov = %f %f*I\n", crealf(cov), cimagf(cov));
                        printf("ucov = %f %f*I\n", crealf(ucov), cimagf(ucov));
                    }

                }
            }
        } /* for (ifreq ... */

        //for (i = 0; i < nelems; i++) ptrout[i] = ptrinp[i];

        //printf("ptrout[:10] =\n");
        //for (i = 0; i < 10; i++) printf("%f ", ptrout[i]);
        //printf("\n\n");

       
        /*
         * Save the van Vleck corrected data in the output file HDU
         */
        if ( fits_write_img(fout, TFLOAT, firstelem, nelems,
                           ptrout, &status)) 
           printerror(status);

        if (factors) {
            /*
             * Save the factors = vv_rho / rho in the output file2 HDU
             */
            if ( fits_write_img(fout2, TFLOAT, firstelem, nelems,
                                ptrout2, &status)) 
                printerror(status);
        }

        if (verbose && (ihdu % 10 == 0)) printf("ihdu = %d\n", ihdu);

        if (n_hdus != 0 && ihdu > n_hdus) break; // ====================== >>>

        ihdu++;

    } /* while(1) */


    fits_close_file(finp, &status);
    fits_close_file(fout, &status);
    if (factors) 
        fits_close_file(fout2, &status);



    //if (verbose) {
    //    int hr, mn, sc, msc;
	//	end_time = clock();
	//	cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    //    //printf("start_time=%g, end_time=%g, CLOCKS_PER_SEC=%d\n",
    //    //       (double) start_time, (double) end_time, CLOCKS_PER_SEC);
	//	time_int = (long) cpu_time_used;
    //    msc = (int) rint(1000*(cpu_time_used - time_int));
    //    mn = time_int/60;
    //    sc = time_int%60;
    //    hr = mn/60;
    //    mn = mn%60;
    //    printf("\nCPU time used: %7.3fs; ", cpu_time_used);
    //    if (hr == 0) {
    //        printf(" as MM:SS = %02d:%02d.%03d\n", mn, sc, msc);
    //    }
    //    else {
    //        printf(" as HH:MM:SS = %02d:%02d:%02d.%03d\n", hr, mn, sc, msc);
    //    }
    // }


    return 0;

}


