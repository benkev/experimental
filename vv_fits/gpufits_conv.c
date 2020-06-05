/*
 * gpufits_conv.c
 *
 * Copy a GPU fits file with the data changed to float32 or float64,
 * i.e. float or double.
 *
 *
 * Build the executable:
 *
 * $ gcc -g  -lcfitsio -lm  gpufits_conv.c   -o gpufits_conv
 *
 * Example 1:
 *
 * $ ./gpufits_conv 1130642936_20151104032841_gpubox11_00.fits
 *
 *    -- convert file into f32 saving in 
 *       1130642936_20151104032841_f32_gpubox11_00.fits
 *
 * Example 2:
 *
 * $ ./gpufits_conv -v -t f64 1130642936_20151104032841_gpubox11_00.fits \
 *                  -o out.fits
 *
 *    -- convert file into f64 saving in out.fits. Verbose output.
 *
 */


#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <getopt.h>
#include <fitsio.h>

void printerror( int status) {

    /*****************************************************/
    /* Print out cfitsio error messages and exit program */
    /*****************************************************/

    if (status) {
       fits_report_error(stderr, status); /* print error report */
       exit( status );    /* terminate the program, returning error status */
    }
    return;
}



void usage() {

    fprintf(stdout,"gpufits_conv:\n");
    fprintf(stdout,"Copy a GPU fits file with the data changed " \
            "to float32 or float64\n");
    fprintf(stdout,"Usage:\n");
    fprintf(stdout,"./gpufits_conv [options] file_name.fits\n\n");
    fprintf(stdout,"Options:\n");
    fprintf(stdout,"\t-o\t <output fits filename> [default: same as input "
            " with _i32_, _f32_, or _f64_ inserted before 'gpubox']\n");
    fprintf(stdout,"\t-t\t <output fits file data type> " \
            "[default: f32 (convert into 32-bit float)\n");
    fprintf(stdout,"\t-n\t <number of data HDUs in output fits file> \n" \
            "\t\t    [default: -n 0 (write all HDUs. " \
            "As many as there are in the input fits file)]\n");
    fprintf(stdout,"\t-v\t print verbose info\n");
    fprintf(stdout,"\t-h\t print this message\n\n");
    fprintf(stdout,"Example 1:\n");
    fprintf(stdout,
            "\t./gpufits_conv 1130642936_20151104032841_gpubox11_00.fits\n");
    fprintf(stdout,"\t\t -- convert file into f32 saving in 1130642936_" \
            "20151104032841_f32_gpubox11_00.fits\n");
    fprintf(stdout,"Example 2:\n");
    fprintf(stdout, "\t./gpufits_conv -v -t f64 1130642936_20151104032841"
            "_gpubox11_00.fits -o out.fits\n");
    fprintf(stdout,"\t\t -- convert file into f64 saving in out.fits. " \
            "Verbose output.\n");

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

    char *input_file = NULL, *output_file = NULL;
    fitsfile *finp;     /* FITS input file ptr, defined in fitsio.h */
    fitsfile *fout;     /* FITS output file ptr, defined in fitsio.h */
    int verbose = 0;
    int status = 0; /*  CFITSIO status value MUST be initialized to zero!  */
    int nhdus = 0, hdutype = 0, n_hdus = 0;
    long dims[2], nelems = 0, firstelem = 1;
    float nullval = 0; /* nullval: The value set for the undefined elements */
    /* anynull is set to True if any of the read elements are set undefined  */
    int anynull = 0x0;

    int ihdu;

    int numtype = -32;  /* == BITPIX. Default output fits format is float32 */
    char *save_as_type = strdup("f32");

    int BSCALE_in_hdr = 0;
    int BZERO_in_hdr = 0;

    clock_t start_time, end_time;
    double cpu_time_used;
    long time_int;

    if (verbose) {
        start_time = clock();
        printf("CPU timing started ...\n\n");
    }

    if (argc == 1) usage();

    int arg = 0;
    while ((arg = getopt(argc, argv, "i:o:t:n:fhv")) != -1) {

        switch (arg) {
        case 'h':
            usage();
            break;
        case 'o':
            output_file = strdup(optarg);
            break;
        case 't':
            free(save_as_type);
            save_as_type = strdup(optarg);
            break;
        case 'n':
            n_hdus = atoi(optarg);
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
     * The first non-option argument must be the input file name
     */
    if (optind < argc) 
        input_file = strdup(argv[optind]);
    else {
        fprintf(stderr,"No file specified\n");
        return EXIT_FAILURE;
    }


    /* printf("opterr=%d, optopt=%d, optind=%d, argc=%d\n",  */
    /*        opterr, optopt, optind, argc); */
    /* printf("argv[%d]=%s\n", optind, argv[optind]); */
    /* printf("argv[%d]=%s\n", optind+1, argv[optind+1]); */
    /* printf("argv[%d]=%s\n", optind+2, argv[optind+2]); */

    // exit(EXIT_FAILURE);
    
 
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

    printf("Type of output fits file: %s\n", save_as_type);

    // exit(EXIT_FAILURE);

    /*
     * Open input FITS file
     */
    if ( fits_open_file(&finp, input_file, READONLY, &status) ) {
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
    printf("There are %d HDUs in file %s\n", nhdus, input_file);


    if (output_file == NULL) {
        /*
         * Build the output file name
         */
        output_file = (char *) calloc(sizeof(char[512]), 1);
        char *inpfn = strdup(input_file);
        char *gpuboxptr = strstr(inpfn, "gpubox");
        *(gpuboxptr - 1) = 0;  /* Terminate the filename before "gpubox_" */ 
        sprintf(output_file, "%s_%s_%s", inpfn, save_as_type, gpuboxptr);
    }
    
    printf("output_file =  '%s'\n", output_file);

    /* 
     * Delete old file(s) if it already exists 
     */
    remove(output_file);

    /*
     * Create a new fits file
     */
    if ( fits_create_file(&fout, output_file, &status) ) {
        fprintf(stderr,"Cannot create file %s\n", output_file);
        printerror(status);
        return EXIT_FAILURE;
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

        printf("Change type to: numtype=%d (i.e. '%s')\n", 
               numtype, save_as_type);

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

    /*
     * Get image dimensions (2 means number of elements of the dims array)
     */
    fits_movabs_hdu(finp, 2, &hdutype, &status);
    fits_get_img_size(finp, 2, dims, &status);
    if (status) {
       fits_report_error(stderr, status);
       return status;
    }
    printf("Found %ld x %ld image\n", dims[0], dims[1]);
 
    int nrows = dims[0], nchan = dims[1];
    float (*dat)[nchan] = malloc(sizeof(float[nrows][nchan]));



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
        if (fits_get_img_size(finp, 2, dims, &status)) {
            printerror(status);
        }
        status = 0;
        nelems = dims[0] * dims[1];


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
  

        if (numtype != 32) { /* i.e. if not i32, when changes are not needed */
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



        /*
         * Read data from current HDU
         * Use fits_read_img so that this will also work transparently 
         * for compressed data
         */
        if ( fits_read_img(finp, TFLOAT, firstelem, nelems, &nullval,
                           dat, &anynull, &status) ) {
            printerror(status);
        }



        /*
         * Save the data in the output file HDU
         */
        if ( fits_write_img(fout, TFLOAT, firstelem, nelems,
                           dat, &status)) 
           printerror(status);


        if (verbose && (ihdu % 10 == 0)) printf("ihdu = %d\n", ihdu);

        if (n_hdus != 0 && ihdu > n_hdus) break; // ===================== >>>

        ihdu++;

    } /* while(1) */










    fits_close_file(finp, &status);
    fits_close_file(fout, &status);

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
        printf("\nCPU time used: %7.3fs; ", cpu_time_used);
        if (hr == 0) {
            printf(" as MM:SS = %02d:%02d.%03d\n", mn, sc, msc);
        }
        else {
            printf(" as HH:MM:SS = %02d:%02d:%02d.%03d\n", hr, mn, sc, msc);
        }
    }


    return 0;

}



 
