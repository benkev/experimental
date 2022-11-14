#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

const uint ND = 2;

uint bmul(uint a, uint b[], int nd);
uint badd(uint a, uint b[], int nd);
int ibig2str(uint b[], int nd, char str[]);


int main(int argc, char *argv[]) {
    int nd = ND, nc;
    //uint bint[] = {65535, 0};
    uint bint[] = {65535, 0};
    uint c, a = 65535;
    ulong bint_id, res = 0;
    ulong abc, bl;
    char str[80];

   
    /*
     * Add bint = bint + a
     */

    printf("Add bint = bint + a:\n\n");
    printf("Given: a = %5u, bint[] = %u  ", a, c);
    for (int id=nd-1; id>-1; id--) printf("%5u  ", bint[id]);
    printf("\n");
    bl = 65536*bint[1] + bint[0];
    printf("a = %d, b = %lu\n\n", a, bl);

    c = badd(a, bint, nd);
    
    res = 0;
    res = 65536*bint[1] + bint[0];
    
    printf("Result:           bint[] = %u  ", c);
    for (int id=nd-1; id>-1; id--) printf("%5u  ", bint[id]);
    printf("\n");
    printf("res = %lu\n", res);

  
    printf("\n");

    
    
    // return 0;

    /*
     * Multiply bint = a * bint
     */

    printf("==========================================\n\n");
    
    printf("Multiply bint = a * bint:\n\n");
    
    /* a = 99193; */
    /* bint[1] = 278; bint[0] = 28297; */
    a = 6537;
    bint[1] = 1; bint[0] = 5535;
    
    printf("Given: a = %5u, bint[] = %u  ", a, c);
    for (int id=nd-1; id>-1; id--) printf("%5u  ", bint[id]);
    printf("\n");
    bl = 65536*bint[1] + bint[0];
    printf("a = %d, b = %lu\n\n", a, bl);
    

    /* c = 0; // Carry */
    /* for (int id=0; id < nd; id++) { */
    /*     abc = a * bint[id] + c; */
    /*     bint[id] = abc % 65536; */
    /*     c =        abc / 65536; */
        
    /*     printf("id = %d; bint[] = ", id); */
    /*     for (int id=nd-1; id>-1; id--) printf("%5u  ", bint[id]); */
    /*     printf("\n"); */
    /* } */
    /* printf("\n\n"); */

    c = bmul(a, bint, nd);

    res = 0;
    res = 65536*bint[1] + bint[0];
    
    printf("bint[] = ");
    for (int id=nd-1; id>-1; id--) printf("%5u  ", bint[id]);
    printf("\n");
    printf("c = %d, res = %lu\n", c, res);


    printf("==========================================\n");


    return 0;   
}


uint badd(uint a, uint b[], int nd) {

    uint c = 0; // Carry
    uint bc;    // = b + c

    b[0] += a;    
    for (int i = 0; i < nd; i++) {
        bc = b[i] + c;
        b[i] = bc % 65536;
        c =    bc / 65536;
    }
    
    return c;   
}


uint bmul(uint a, uint b[], int nd) {
    
    uint c = 0; // Carry
    uint abc;   // = a*b + c

    for (int i = 0; i < nd; i++) {
        abc = a * b[i] + c;
        b[i] = abc % 65536;
        c =    abc / 65536;
    }
    
    return c;   
}


int ibig2str(uint b[], int nd, char str[]) {
    int nc;
    uint d;
    char *pstr = (char *) str;
    uint *dtmp = (uint *) malloc(nd*sizeof(uint));
    
    for (int id = nd-1; id > -1; id--) {
        nc = sprintf(pstr, "%u", b[id]);
        bmul(65536, dtmp, nd);
        pstr += nc;
        
    }
    return nc;
}


int str2ibig(uint b[], int nd, char str[]) {
    int nc;
    uint d;
    char *pstr = (char *) str;
    uint *dtmp = (uint *) malloc(nd*sizeof(uint));
    
    for (int id = nd-1; id > -1; id--) {
        nc = sprintf(pstr, "%u", b[id]);
        bmul(65536, dtmp, nd);
        pstr += nc;
        
    }
    return nc;
}
