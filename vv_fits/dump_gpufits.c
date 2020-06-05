/*
 * dump_gpufits.c
 *
 * Dump the MWA correlator output files in plain binary format. 
 *
 *
 * Build the executable:
 *
 * $ cd vv_fits/
 * $ gcc -g   mwac_utils_modif.c antenna_mapping.c  dump_gpufits.c  \
 *             -o dump_gpufits  -lcfitsio  -lm      
 *
 * One can obtain the vv_fits/ from the Subversion repository using
 * the command:
 *
 * $ svn co svn+ssh://benkev@mwa-lfd.haystack.mit.edu/svn/MISC/vv_fits .     
 *
 *
 * Examples:
 *
 * $ ./dump_gpufits  1130642936_20151104032841_gpubox11_00.fits -f -t f32
 *
 * The result will have name 1130642936_20151104032841_f32_gpubox11_00.dmp
 *
 *  Created on: Sep 15, 2016
 *      Author: Leonid Benkevitch
 * Based on build_lfiles.c by Steven Ord, Jul 20, 2012
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
#include "mwac_utils.h"
#include "antenna_mapping.h"
#include "vanvleckf.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

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

    fprintf(stdout,"dump_gpufits:\n");
    fprintf(stdout," Dump the MWA correlator output file " \
            "in plain binary format.\n");
    fprintf(stdout,"Usage:\n");
    fprintf(stdout,"\tdump_gpufits [options] <input> <output>\n");
    fprintf(stdout,"Files:\n");
    fprintf(stdout,"\t<input>: GPU fits filename. Must contain '_gpubox' "
            "substring.\n");
    fprintf(stdout,"\t<output>: filename [default: same as <input>, "
            " with '_<mxtag>_<type>_' inserted before '_gpubox', "
            "and the extension '.dmp']\n"
            "\t\t <mxtag> depends on '-f' option.\n" \
            "\t\t If '-f' specified, <mxtag> is 'fulsq' " \
            "(full square matrix)\n"                                    \
            "\t\t Otherwise, <mxtag> is 'lwtri' (lower-triangular matrix)\n"\
            "\t\t Currently, <type> is always 'f32'.\n" \
            "\t\t If <output> is a directory name,\n" \
            "\t\t then the default filename is written " \
            "to this directory.\n");
    fprintf(stdout,"Options:\n");
    fprintf(stdout,"\t-c\t Create the directory specified " \
            "as <output>, if it does not exist\n");
    fprintf(stdout,"\t-n\t <number of data HDUs to output> \n" \
            "\t\t    [default: -n 0 (write all HDUs.]\n");
    fprintf(stdout,"\t-f\t Write full square 256 x 256 Hermitian matrices " \
            "obtained from the lower-triangular packed format in gpu fits "
            "file.\n");
    fprintf(stdout,"\t-v\t print verbose info\n");
    fprintf(stdout,"\t-h\t print this message\n");

    exit(EXIT_FAILURE);
}



void bscale_bzero_in_header(fitsfile *fitsfl, 
                            int *BSCALE_in_hdr, int *BZERO_in_hdr) {
    /*
     * Find out if the current header contains the BSCALE and BZERO
     * keywords 
     */
    int nkeystotal = 0, morekeys = 0, status = 0, ikey = 0;
    char *record =  malloc(128*sizeof(char));

    if ( fits_get_hdrspace(fitsfl, &nkeystotal, &morekeys, &status) )
        printerror( status );

    //printf("after fits_get_hdrspace, nkeystotal=%d, morekeys=%d\n", 
    //       nkeystotal, morekeys);

    *BSCALE_in_hdr = 0;
    *BZERO_in_hdr = 0;

    for (ikey = 1; ikey <= nkeystotal; ikey++) {
        if ( fits_read_record(fitsfl, ikey, record, &status) )
            printerror( status );
        if (strstr(record, "BSCALE")) *BSCALE_in_hdr = 1;
        if (strstr(record, "BZERO"))  *BZERO_in_hdr = 1;
        //printf("record # %d = '%s\n", ikey, record);
    }

}



int main(int argc, char **argv) {

    char *input_file = NULL, *output_file = NULL;
    char *out_dir = NULL;
    fitsfile *finp;     /* FITS input file ptr, defined in fitsio.h */
    float complex *cuda_matrix_h = NULL;
    float complex *full_matrix_h = NULL;
    int ier = 0, nhdus = 0;
    int status = 0; /*  CFITSIO status value MUST be initialized to zero!  */
    int hdutype = 0; 
    long dims[2];
    int ninput = 0;
    size_t nvis=0;
    size_t nbl=0;  /* Number of baselines, = nsta*(nsta+1)/2 */
    size_t matLength=0, fullLength=0;
    float nullval = 0; /* nullval: The value set for the undefined elements */
    /* anynull is set to True if any of the read elements are set undefined  */
    int anynull = 0x0;
    long naxes[2];
    long firstelem = 1;
    long nelems = 0;
    int ihdu;
    int verbose = 0, fullMatrix = 0;
    int n_hdus = 0;
    int create_out_dir = 0; /* If the directory at -o does not exist, create */

    int BSCALE_in_hdr = 0;
    int BZERO_in_hdr = 0;
 
    // int numtype = -32; /* == BITPIX. Default output fits format is float32 */



    char *save_as_type = strdup("f32");
    char ftypetag[16], mxtag[16];

    sprintf(mxtag, "lwtri"); /* Default: otput low-triangle packed matr. */ 

    npol = 2;

    if (argc == 1) usage();

    int arg = 0;
    while ((arg = getopt(argc, argv, "b:i:o:t:n:f::hvc")) != -1) {

        switch (arg) {
        case 'h':
            usage();
            break;
        case 'c':
            create_out_dir = 1;
            break;
        //case 't':
        //    free(save_as_type);
        //    save_as_type = strdup(optarg);
        //    break;
        case 'n':
            n_hdus = atoi(optarg);
            break;
        case 'f':
            fullMatrix = 1;
            sprintf(mxtag, "fulsq");  /* Output full Hermitian matrices */ 
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
    else if (optind == argc) {
        fprintf(stderr, "No input file specified.\n");
        exit(EXIT_FAILURE);
    }


    /*
     * Input and output file names
     */

    if (input_file == NULL) {
        if (optind == argc-1) 
            input_file = strdup(argv[optind]);
        else if (optind == argc-2) {
            input_file = strdup(argv[optind]);
            output_file = strdup(argv[optind+1]);
        }
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
    // if      (strcmp(save_as_type, "i32") == 0) {
    //     numtype = 32;
    // }
    // else if (strcmp(save_as_type, "f32") == 0) {
    //    numtype = -32;
    // }
    // else if (strcmp(save_as_type, "f64") == 0) { 
    //     numtype = -64;
    // }
    // else {
    //     fprintf(stderr,"Wrong output type -t %s. "               
    //            "Only i32, f32, or f64 allowed.\n", save_as_type);
    //    return EXIT_FAILURE;
    // }

    // if (verbose) printf("Type of output fits file: %s\n", save_as_type);
    if (verbose) printf("Type of output file: f32\n");

    sprintf(ftypetag, "%s_%s", mxtag, save_as_type);
    printf("ftypetag = '%s'\n", ftypetag);


    if (output_file == NULL) {
        /*
         * Build the output file name in the same directory as input file
         */
        output_file = (char *) calloc(sizeof(char[512]), 1);
        char *inpfn = strdup(input_file);
        char *gpuboxptr = strstr(inpfn, "gpubox");
        *(gpuboxptr - 1) = 0;  /* Terminate the filename before "gpubox_" */
        char *dotptr = strrchr(gpuboxptr, '.'); /* Find the last dot */
        *dotptr = 0;  /* Cut the 'fits' extension away from the gpuboxptr */
        sprintf(output_file, "%s_%s_%s.dmp", 
                             inpfn, ftypetag, gpuboxptr);
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
            char *dotptr = strrchr(gpuboxptr, '.'); /* Find the last dot */
            *dotptr = 0;  /* Cut the 'fits' extension away from the gpuboxptr */
            sprintf(output_file, "%s/%s_%s_%s.dmp", 
                    out_dir, outfn, ftypetag, gpuboxptr);
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
                char *dotptr = strrchr(gpuboxptr, '.'); /* Find the last dot */
                *dotptr = 0;  /* Cut the extension away from the gpuboxptr */
                sprintf(output_file, "%s/%s_%s_%s.dmp", out_dir, outfn, 
                        ftypetag, gpuboxptr);
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
        printf("input_file=    '%s'\n", input_file);
        printf("output_file =  '%s'\n", output_file);
    }


    // return 0;


    /* 
     * Delete old files if they already exist
     */
    remove(output_file);



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
     * Open the output file
     */
    FILE *fout = fopen(output_file, "w");


    

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


    if (verbose) {
        if (fullMatrix) {
            printf("An output block has %d full Hermitian matrices.\n", nfreq);
            printf("Data block length = %ld, element size = %ld\n", 
                   fullLength, sizeof(float complex));
        }
        else {
            printf("An output block has %d lower-triangle matrices.\n", nfreq);
            printf("Data block length = %ld, element size = %ld\n", 
                   matLength, sizeof(float complex));
        }
    }



    /* 
     * now we know the dimensions of the data, allocate arrays 
     */
    cuda_matrix_h = (float complex *) malloc(matLength*sizeof(float complex));
    full_matrix_h = (float complex *) malloc(fullLength*sizeof(float complex));


    // float *ptrinp = NULL, *ptrout = NULL;

    float *ptrinp = (float *) cuda_matrix_h;

    // ptrout = (float *) full_matrix_h;

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
        nelems = naxes[0] * naxes[1];

        /*
         * Find out if the current header contains the BSCALE and BZERO
         * keywords 
         */
        bscale_bzero_in_header(finp, &BSCALE_in_hdr, &BZERO_in_hdr);    
  

        /*
         * Read data from current HDU
         * Use fits_read_img so that this will also work transparently 
         * for compressed data
         */
        if (fits_read_img(finp, TFLOAT, firstelem, nelems, &nullval,
                          ptrinp, &anynull, &status) )
            printerror(status);


        if (fullMatrix) {
            /*
             * Complement the lower triangle form, packed Hermitian matrix 
             * cuda_matrix_h, read from the gpufits file, to the full, square 
             * Hermitian matrix full_matrix_h nsta X nsta
             */
            extractMatrix(full_matrix_h, cuda_matrix_h);

            /*
             * Save the full matrix in the output binary file
             */
            fwrite(full_matrix_h, sizeof(float complex), fullLength, fout);
        }
        else 
            /*
             * Save the lower-triangular HDU.data in the output binary file
             */
            fwrite(cuda_matrix_h, sizeof(float complex), matLength, fout);
 


        if (verbose && (ihdu % 10 == 0)) printf("ihdu = %d\n", ihdu);

        if (n_hdus != 0 && ihdu > n_hdus) break; // ====================== >>>

        ihdu++;

    } /* while(1) */


    fits_close_file(finp, &status);
    fclose(fout);

    return 0;

}


