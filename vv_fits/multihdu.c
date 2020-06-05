/*
 * multihdu.c
 *
 * Read from existing and write to a new fits file copies of headers 
 * and modified data for a multi-HDU fits files
 *
 */


#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <fitsio.h>

void printerror( int status)
{
    /*****************************************************/
    /* Print out cfitsio error messages and exit program */
    /*****************************************************/


    if (status)
    {
       fits_report_error(stderr, status); /* print error report */

       exit( status );    /* terminate the program, returning error status */
    }
    return;
}


int main(int argc, char *argv[])
{
    /********************************************************************/
    /* copy the 1st and 3rd HDUs from the input file to a new FITS file */
    /********************************************************************/
    fitsfile *infptr;      /* pointer to the FITS file, defined in fitsio.h */
    fitsfile *outfptr;                 /* pointer to the new FITS file      */

    char infilename[]  = "1130642936_20151104032841_gpubox11_00.fits";
    char outfilename[] = "btestfil.fits";

    int status, morekeys, hdutype;

    float pi = 3.141592653589793115;

    status = 0;

    remove(outfilename);            /* Delete old file if it already exists */

    /* open the existing FITS file */
    if ( fits_open_file(&infptr, infilename, READONLY, &status) ) 
         printerror( status );

    if (fits_create_file(&outfptr, outfilename, &status)) /*create FITS file*/
         printerror( status );           /* call printerror if error occurs */

    /* copy the primary array from the input file to the output file */
    morekeys = 0;     /* don't reserve space for additional keywords */
    if ( fits_copy_hdu(infptr, outfptr, morekeys, &status) )
         printerror( status );

    int ihdu = 2;
    long firstelem = 1;
    float nullval = 0; /* nullval: The value set for the undefined elements */
    /* anynull is set to True if any of the read elements are set undefined  */
    int anynull = 0x0;   
    long naxes[2];
    long nelems = 0;
    float complex *cuda_matrix = NULL, *modif_matrix = NULL;
    float *ptrinp = NULL, *ptrout = NULL;
     /* 32 freq chans by # of baselines by 4 polarization combinations: */
    size_t matLength = 4*32*128*(128+1)/2;
    cuda_matrix =  (float complex *) malloc(matLength*sizeof(float complex));
    modif_matrix = (float complex *) malloc(matLength*sizeof(float complex));


    ptrinp = (float *) cuda_matrix;
    ptrout = (float *) modif_matrix;

    while(1) {
        size_t i = 0;

        /* move to the next HDU in the input file */
        if ( fits_movabs_hdu(infptr, ihdu, &hdutype, &status) ) {
            if (status == END_OF_FILE) 
                break; // ===== all read; leave the while(1) loop ======= >>>
            printerror( status );
        }

        if (fits_get_img_size(infptr, 2, naxes, &status)) {
            printf("Printing fits_get_img_size() status, status = %d\n", 
                   status);
            printerror(status);
        }
        status = 0;
        nelems = naxes[0] * naxes[1];

        printf("naxes[0]=%ld, naxes[1]=%ld, nelems=%ld\n", 
               naxes[0], naxes[1], nelems);


        /* copy current header from the input file to the output file */
        if ( fits_copy_header(infptr, outfptr, &status) )
            printerror( status );

        printf("copying header %d ...\n", ihdu);

        if (fits_read_img(infptr, TFLOAT, firstelem, nelems, &nullval,
                          (float *)ptrinp, &anynull, &status))
            printerror(status);

        for (i = 0; i < nelems; i++) ptrout[i] = pi*ptrinp[i];

        if ( fits_write_img(outfptr, TFLOAT, firstelem, nelems,
                            (float *)ptrout, &status)) 
            printerror(status);

        ihdu++;
    }

    status = 0;

    fits_close_file(outfptr, &status);
    fits_close_file(infptr, &status);

    //if (fits_close_file(outfptr, &status) ||
    //    fits_close_file(infptr, &status)) /* close files */ {
    //    printf("Printing fits_close_file() status, status = %d\n", status);
    //    printerror( status );
    //}

    return 0;
}
