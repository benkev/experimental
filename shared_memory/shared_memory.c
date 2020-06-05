/* 
 * 
 * shared_memory.c: Use shared memory to access van Vleck correction tables
 *                  from many processes
 * Build command:
 *
 * $ gcc shared_memory.c -o shared_memory -lrt
 *
 */ 
#include <stdio.h> 
#include <stdlib.h> 
#include <sys/stat.h> 
#include <sys/mman.h> 
#include <fcntl.h> 
#include "vanvleckf.h"

#define XY_TABLE_SHARED "/vanVleck_xy_table" 


int main(int argc, char *argv[]) {

    int shm, i;
    char *bin_dir = "/usr/local/lib/van_vleck";
    size_t tblen, xylen, srlen, rrlen;
    float (*xy_table)[NS][NR] = NULL;  
    float *sig_ruler = NULL;
    float *rho_ruler = NULL;
    struct stat finfo, shmi;

    char *xyname = (char *) calloc(sizeof(char[512]), 1);
    char *srname = (char *) calloc(sizeof(char[512]), 1);
    char *rrname = (char *) calloc(sizeof(char[512]), 1);

    sprintf(xyname, "%s/%s", bin_dir, "xy_tablef.bin");
    sprintf(srname, "%s/%s", bin_dir, "sig_rulerf.bin");
    sprintf(rrname, "%s/%s", bin_dir, "rho_rulerf.bin");

    printf("NS=%d, NR=%d, NS*NS*NR=%ld\n", NS, NR, NS*NS*NR);

    tblen = 0;

    stat(xyname, &finfo);
    xylen = finfo.st_size;
    printf("xyname='%s', finfo.st_size=%ld\n", xyname, xylen);
    tblen += xylen;

    stat(srname, &finfo);
    srlen = finfo.st_size;
    printf("srname='%s', finfo.st_size=%ld\n", srname, srlen);
    tblen += srlen;

    stat(rrname, &finfo);
    rrlen = finfo.st_size;
    printf("rrname='%s', finfo.st_size=%ld\n", rrname, rrlen);
    tblen += rrlen;

    printf("tblen=%ld\n", tblen);
   
    //if ((shm = shm_open(XY_TABLE_SHARED, O_RDWR, 0660)) == -1)

    if ((shm = shm_open(XY_TABLE_SHARED, O_RDWR | O_CREAT, 0660)) == -1)
        printf("shm_open error\n");

    if (ftruncate(shm, tblen) == -1) 
        error ("ftruncate"); 

    if ((xy_table = mmap (NULL, tblen, PROT_READ | PROT_WRITE, MAP_SHARED, 
                                shm, 0)) == MAP_FAILED) 
        error ("mmap"); 
    
    sig_ruler = (float *) xy_table + xylen/sizeof(float);
    rho_ruler = (float *) xy_table + (xylen + srlen)/sizeof(float);

    printf("xy_table=%p, sig_ruler=%p, rho_ruler=%p\n", 
           xy_table, sig_ruler, rho_ruler);
    printf("rho_ruler + rrlen/sizeof(float) - (float *)xy_table = %ld\n", 
           rho_ruler + rrlen/sizeof(float) - (float *)xy_table);


    int nrr = rrlen/sizeof(float),  nsr = srlen/sizeof(float);

    FILE *finp = fopen(xyname, "rb");
    fread(xy_table, NS*NS*NR*sizeof(float), 1, finp);
    fclose(finp);
    finp = fopen(srname, "rb");
    fread(sig_ruler, NS*sizeof(float), 1, finp);
    fclose(finp);
    finp = fopen(rrname, "rb");
    fread(rho_ruler, NR*sizeof(float), 1, finp);
    fclose(finp);


    //mprotect(xy_table, tblen, PROT_READ);

    printf("xy_table[100,100,10] = %.8f\n", xy_table[100][100][10]);
    printf("xy_table[89,123,17] = %.8f\n", xy_table[89][123][17]);

    xy_table[89][123][17] = -999.123;

    printf("xy_table[89,123,17] = %.8f\n", xy_table[89][123][17]);

    printf("sig_ruler=\n");
    for (i=0; i<nsr; i++) printf("%.2f  ", sig_ruler[i]);
    printf("\n\n");
    printf("sig_ruler=\n");
    for (i=0; i<nrr; i++) printf("%.8f  ", rho_ruler[i]);
    printf("\n");



    return 0;
}
