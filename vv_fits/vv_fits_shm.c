/*
 * vv_fits_shm.c
 *
 * Van Vleck correction of the MWA correlator output files in FITS format. 
 * This is a variant of the van Vleck correction code, vv_fits.
 *
 * vv_fits_shm  uses a single copy of vv tables in a shared RAM segment. This
 * allows running any number of vv_fits_shm instances to correct many (24?
 * 48? 196?..) different gpu fits files simultaneously even though the machine
 * has limited amount of RAM.
 *
 * The interface is the same as that of vv_fits. Only two options have been
 * added: 
 * "-l" - load the correction tables to shared memory, and 
 * "-u" - unload them to free the shared RAM.
 *
 * Use it in three steps:
 *
 * 1. To load the tables, run
 * $ vv_fits_shm -l
 *
 * 2. I recommend to use something similar to the following one-line bash
 * command to run, say, 24 vv_fits_shm copies on 24 files in parallel:
 *
 * $ for f in *_*032841_gpubox??_00.fits; do vv_fits_shm $f -t f32 -c outdir  &
 *                                                                        done
 * In some time you will get 24 corrected *_vv_*.fits files in the outdir/
 * directory.
 *
 * 3. Unload the tables and free ~400 MB they occupied:
 * $ vv_fits_shm -u
 *
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
 * $ gcc -g   vanvleckf.c  mwac_utils.c antenna_mapping.c  vv_fits_shm.c  \
 *             -o vv_fits_shm  -lcfitsio  -lm  -lrt  
 *
 * One can obtain the vv_fits/ from the Subversion repository using
 * the command:
 *
 * $ svn co svn+ssh://benkev@mwa-lfd.haystack.mit.edu/svn/MISC/vv_fits .     
 *
 *
 * Examples:
 *
 * $ ./vv_fits_shm -i 1130642936_20151104032841_gpubox11_00.fits -f -t f32
 *
 * $ ./vv_fits_shm -i 1130642936_20151104032841_gpubox11_00.fits \
 *             -o 1130642936_20151104032841_vv_gpubox11_00 -f -v 
 *
 * To run multiple copies of vv_fits_shm, I recommend to use the following 
 * bash commands:
 *
 * $ for f in *_gpubox??_00.fits ; do ./vv_fits_shm $f -t f32 -c ~/ftdir & done
 *
 * 
 * For debugging, to prevent vv_fits_shm from lengthy processing all 
 * the HDUs (typically, amounting hundreds of), one can limit the number 
 * of HDUs to write to the output, using -n <# of HFUs>
 * For example, for me 1 HDU is enough, and I can invoke vv_fits_shm this way:
 *
 * $ ./vv_fits_shm -i 1130642936_20151104032841_gpubox11_00.fits \
 *             -v -f -o vv_out.fits -t f32 -n 1
 *
 * Created on: Sep 23, 2016
 *      Author: Leonid Benkevitch
 * Based on build_lfiles.c by Steven Ord, Jul 20, 2012
 */

//#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/mman.h> 
#include <fcntl.h> 
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

#define XY_TABLE_SHARED "/vanVleck_xy_table" 

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

    printf("vv_fits_shm:\n");
    printf("Make van Vleck correction on a gpufits file, " \
            "the output of MWA GPU correlator.\n\n");
    printf("This version of vv_fits uses a shared memory segment\n");
    printf("to hold the bulky (~400MB) van Vleck tables and.\n");
    printf("The tables are loaded only once with 'vv_fits_shm -l'\n");
    printf("Multiple copies of vv_fits_shm working on many different \n");
    printf("gpufits files share a single copy of the tables.\n");
    printf("This technique allows running hundreds vv_fits_shm processes\n");
    printf("having quite frugal memory budget.\n");
    printf("After the usage the shared memory should be freed (see -u).\n\n");
    printf("Usage:\n");
    printf("\tvv_fits_shm [options] <input> <output>\n");
    printf("or\n");
    printf("\tvv_fits_shm [options] -i <input> -o <output>\n");
    printf("Options:\n");
    printf("\t-l\t Load the van Vleck correction tables to shared memory.\n");
    printf("\t\t   Can be used with -b to point at the tables location.\n");
    printf("\t-u\t Unload the van Vleck correction tables.\n");
    printf("\t\t   Simply frees the RAM occupied by shared memory.\n");
    printf("\t-i\t <input visibility gpu fits file name> \n");
    printf("\t-b\t <binary tables directory> [default: " \
            " /usr/local/lib/van_vleck\n");
    printf("\t-o\t <output fits filename> [default: same as in -i "
            " with '_vv_' inserted before 'gpubox']\n"
            "\t\t   If -o is followed not by a file, " \
            "but by a directory name,\n" \
            "\t\t   then the default filename is written " \
            "to the -o <directory>\n");
    printf("\t-c\t Create the directory for output specified " \
            "at -o, if it does not exist\n");
    printf("\t-t\t <data type of output fits file> " \
            "[default: f32 (unlike i32 in input fits)\n");
    printf("\t-n\t <number of data HDUs in output fits file> \n" \
            "\t\t    [default: -n 0 (write all HDUs. " \
            "As many as there are in the input fits file)]\n");
    printf("\t-f\t Write a separate fits file with the correction " \
            "factors of float32 type.\n");
    printf("\t\t   The factors file name is the same as in -i "
            " with '_vvfactors_' inserted before 'gpubox'.\n" \
            "\t\t   If the output directory is specified in -o, " \
            "then the factors file is written to the " \
            "specified directory.\n");
    printf("\t\t   This option can provide a file name for the"
            " factors file, if used as -f=<name>.fits or -f<name>.fits.\n");
    printf("\t-v\t print verbose info\n");
    printf("\t-h\t print this message\n");

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
    // char *bin_dir = "/usr/local/lib/van_vleck";
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
    int i, j;
    int create_out_dir = 0; /* If the directory at -o does not exist, create */

    int BSCALE_in_hdr = 0;
    int BZERO_in_hdr = 0;
    int ikey = 0;
    char *keyname = malloc(128*sizeof(char)); 
    char *record =  malloc(128*sizeof(char));
    
 

    /* Tables for correction std and correlation */
    float (*qsig)[NQC] = NULL; /* std table */
    float *sig_ruler = NULL; 
    float *rho_ruler = NULL;
    float (*xy_table)[NS][NR] = NULL;

    char *xyname = (char *) calloc(sizeof(char[512]), 1);
    char *srname = (char *) calloc(sizeof(char[512]), 1);
    char *rrname = (char *) calloc(sizeof(char[512]), 1);
    char *qsname = (char *) calloc(sizeof(char[512]), 1);

    int shm; /* Shared memory segment descriptor */
    size_t tblen, xylen, srlen, rrlen, qslen;
    int load_vvtables = 0, unload_vvtables = 0, vvtables_loaded = 0;
    struct stat buf;

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
    while ((arg = getopt(argc, argv, "lub:i:o:t:n:f::hvc")) != -1) {

        switch (arg) {
        case 'h':
            usage();
            break;
        case 'l':
            load_vvtables = 1;
            break;
        case 'u':
            unload_vvtables = 1;
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


    // printf("argc=%d, optind=%d\n", argc, optind);
    // for (i=0; i<argc; i++) printf("argv[%d]='%s'\n", i, argv[i]);

    // return 0;

    
    /*
     * Check if the vv tables are already loaded to the shared memory
     */
    vvtables_loaded = 1; /* Assume already loaded */
    if ( stat("/dev/shm/vanVleck_xy_table", &buf) ) {
        if (errno == ENOENT) 
            vvtables_loaded = 0;
        else {
            printf("Failure of stat(\"/dev/shm/vanVleck_xy_table\", &buf)");
            return -1;
        }
    }


    /*
     * Load the vv tables into the shared memory
     */
    if (load_vvtables) {
        if (vvtables_loaded) {
            fprintf(stderr, "van Vleck tables already loaded.\n");
            fprintf(stderr, "To unload and free shared memory segment, use\n");
            fprintf(stderr, "$ %s -u\n", argv[0]);
            return -1;
        }
        else {

            sprintf(xyname, "%s/%s", bin_dir, "xy_tablef.bin");
            sprintf(srname, "%s/%s", bin_dir, "sig_rulerf.bin");
            sprintf(rrname, "%s/%s", bin_dir, "rho_rulerf.bin");
            sprintf(qsname, "%s/%s", bin_dir, "qsigf.bin");

            printf("NS=%d, NR=%d, NS*NS*NR=%d\n", NS, NR, NS*NS*NR);

            tblen = 0;

            /*
             * Measure the total size of the vv binary tables 
             * in the files bin_dir
             */
            stat(xyname, &buf);
            xylen = buf.st_size;
            printf("xyname='%s', buf.st_size=%ld\n", xyname, xylen);
            tblen += xylen;

            stat(srname, &buf);
            srlen = buf.st_size;
            printf("srname='%s', buf.st_size=%ld\n", srname, srlen);
            tblen += srlen;

            stat(rrname, &buf);
            rrlen = buf.st_size;
            printf("rrname='%s', buf.st_size=%ld\n", rrname, rrlen);
            tblen += rrlen;

            stat(qsname, &buf);
            qslen = buf.st_size;
            printf("qsname='%s', buf.st_size=%ld\n", qsname, qslen);
            tblen += qslen;

            printf("tblen=%ld\n", tblen);

            /*
             * Create the shared memory region to load vv tables into.
             * Initially, it has zero size.
             */
            if ((shm = shm_open(XY_TABLE_SHARED, O_RDWR | O_CREAT, 0660)) == -1)
                printf("shm_open error\n");

            /*
             * Make the size of shared memory region sufficient to
             * hold the vv tables (i.e. total size tablen)
             */
            if (ftruncate(shm, tblen) == -1) 
                error ("ftruncate"); 

            /*
             * Map a memory segment of the current process on the
             * shared memory segment placing pointer to its beginning
             * into xy_table.
             */
            if ((xy_table = mmap(NULL, tblen, 
                                 PROT_READ | PROT_WRITE, MAP_SHARED, 
                                 shm, 0)) == MAP_FAILED) 
                error ("mmap"); 

            /*
             * Place other tables in the shared memory after xy_table
             */
            sig_ruler = (float *) xy_table + xylen/sizeof(float);
            rho_ruler = (float *) xy_table + (xylen + srlen)/sizeof(float);
            qsig = (float (*)[NQC]) ( (float *) xy_table + 
                                     (xylen + srlen + rrlen)/sizeof(float) );

            printf("xy_table=%p, sig_ruler=%p, rho_ruler=%p, qsig=%p\n", 
                   xy_table, sig_ruler, rho_ruler, qsig);

            FILE *finp = fopen(xyname, "rb");
            fread(xy_table, NS*NS*NR*sizeof(float), 1, finp);
            fclose(finp);
            finp = fopen(srname, "rb");
            fread(sig_ruler, NS*sizeof(float), 1, finp);
            fclose(finp);
            finp = fopen(rrname, "rb");
            fread(rho_ruler, NR*sizeof(float), 1, finp);
            fclose(finp);
            finp = fopen(qsname, "rb");
            fread(qsig, NQC*NQS*sizeof(float), 1, finp);
            fclose(finp);

            /*
             * Make the shared memory segment write-protected (read-only)
             */
            if ( mprotect(xy_table, tblen, PROT_READ) )
                perror("mprotect() error\n");

        }
        printf("van Vleck binary tables have been successfully loaded \n");
        printf("from %s directory to the shared memory segment", bin_dir);
        printf("/dev/shm/vanVleck_xy_table\n");

        return 0;
    } /* if (load_vvtables) */



    /*
     * Unload the vv tables: remove the shared memory segment 
     * /dev/shm/vanVleck_xy_table
     */
    if (unload_vvtables) {
        if (vvtables_loaded) {
            if ( shm_unlink(XY_TABLE_SHARED) == -1) 
                error("shm_unlink");
            printf("Shared memory segment /dev/shm/vanVleck_xy_table "  \
           "successfully removed.\n");
            return 0;
        }
        else {
            printf("Shared memory segment /dev/shm/vanVleck_xy_table "  \
                   "does not exist.\n");
            printf("The the vv tables already unloaded. To load the " \
                   "vv tabes to the shared memory use\n");
            printf("$ %s -l [-b <bin_dir_path>]\n", argv[0]);
            return -1;
        }
    }


    if (vvtables_loaded) {
        /*
         * Map the van Vleck tables on the shared memoty segment 
         */
        tblen = (NS*NS*NR + NS + NR + NQC*NQS)*sizeof(float);
        
        printf("tblen=%ld\n", tblen);


        if ((shm = shm_open(XY_TABLE_SHARED, O_RDONLY, 0440)) == -1) {
            perror("shm_open() error\n");
            exit(EXIT_FAILURE);
        }

        if ((xy_table = mmap (NULL, tblen, PROT_READ, MAP_SHARED, 
                              shm, 0)) == MAP_FAILED) {
            perror("mmap() error"); 
            exit(EXIT_FAILURE);
        }
        
        sig_ruler = (float *) xy_table + NS*NS*NR;
        rho_ruler = (float *) xy_table + NS*NS*NR + NS;
        qsig = (float (*)[NQC]) ((float *) xy_table + NS*NS*NR + NS + NR);

        if (verbose) {
            printf("xy_table=%p, sig_ruler=%p, rho_ruler=%p, qsig=%p\n", 
                   xy_table, sig_ruler, rho_ruler, qsig);

            printf("xy_table[89,123,17] = %.8f\n", xy_table[89][123][17]);

            printf("sig_ruler=");
            for (i=0; i<NS; i++) {
                if (i%20 == 0) printf("\n");
                printf("%.2f  ", sig_ruler[i]);
            }
            printf("\n\n");
            printf("sig_ruler=");
            for (i=0; i<NR; i++) {
                if (i%10 == 0) printf("\n");
                printf("%.8f  ", rho_ruler[i]);
            }
            printf("\n\n");
            printf("qsig=\n");
            for (i=0; i<NQS; i++) {
                for (j=0; j<NQC; j++) printf("%.8f  ", qsig[i][j]);
                printf("\n");
            }
            printf("\n");
        }
    }

    else { /* vv tables not loaded */
        fprintf(stderr, "Please, first load the van Vleck tables "  \
                "into shared memory using\n");
        fprintf(stderr, "$ %s -l [-b <bin_dir_path>]\n", argv[0]);
        exit(EXIT_FAILURE);
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
     * 
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


