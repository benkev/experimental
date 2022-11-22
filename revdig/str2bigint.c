#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <string.h>

typedef	unsigned int		uint;
typedef	unsigned long		ulong;

uint bmul(uint c, uint a, uint b[], int nd);
uint badd(uint c, uint a, uint b[], int nd);
uint bdiv(uint a[], uint b, uint q[], int nd);
uint bdiv_bigendian(uint a[], uint b, uint q[], int nd);
int str2bint(uint b[], int nd, char str[]);
char* bint2str(uint b[], int nd);
int str2bint(uint b[], int nd, char str[]);

int main(int argc, char *argv[]) {

    // char str[] = "65535";
    //char str[] = "34143735976";
    //char str[] = "3414373597";
    //char str[] = "1234567";

    char str[] = "356901";

    /* int nd = 2; */
    /* uint b[2] = {0, 0};  /\* b[0] is the minor digit *\/ */
    /* uint q[2] = {0, 0}; */

    int nd = 10;
    uint b[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};  /* b[0] is the minor digit */
    uint q[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    uint c;  /* Carry. Non-zero if str does not fit into  */
    uint r;
    ulong bl = 0, res = 0;
    uint d;

    uint a = 10, s;
    
    int ns = strlen(str);

    printf("str = \"%s\", ns = %d\n\n", str, ns);

    printf("c = str2bint(b, nd, str);\n");
    
    c = str2bint(b, nd, str);
    
    bl = 65536*b[1] + b[0];
    printf("c = %d, bl = %lu\n\n", c, bl);

    // return 0;


    // Division:

    printf("Division:\n");

    // bl = 12345678 = 0xbc614e = 0xbc, 0xb14e = 188, 24910
    b[1] = 188;
    b[0] = 24910;

    r = bdiv(b, 10, q, nd);

    //res = 65536*q[0] + q[1];
    res = 65536*q[1] + q[0];
    printf("q = %u, %u\n", q[1], q[0]);
    printf("res = %lu, r = %u\n\n", res, r);


    // bl = 8791 = 0x2257 = 0, 8791
    b[1] = 0;
    b[0] = 8791;

    r = bdiv(b, 10, q, nd);

    res = 65536*q[1] + q[0];
    printf("q = %u, %u\n", q[1], q[0]);
    printf("res = %lu, r = %u\n\n", res, r);


    // bl = 4042386655 = 0xf0f1ecdf = 0xf0f1, 0xecdf = 61681, 60639
    b[1] = 61681;
    b[0] = 60639;
    
    r = bdiv(b, 10, q, nd);

    res = 65536*q[1] + q[0];
    printf("q = %u, %u\n", q[1], q[0]);
    printf("res = %lu, r = %u\n\n", res, r);

    
    // bl = 356901 = 0x57225 = 0x5, 0x7225 = 5, 29221
    b[1] = 5;
    b[0] = 29221;
    
    r = bdiv(b, 4, q, nd);

    res = 65536*q[1] + q[0];
    printf("q = %u, %u\n", q[1], q[0]);
    printf("res = %lu, r = %u\n\n", res, r);
    
    //=================================================

    printf("//=================================================\n");
    
    printf("str = \"%s\"\n", str);

    printf("c = str2bint(b, nd, str);\n");
   
    c = str2bint(b, nd, str);
    
    bl = 65536*b[1] + b[0];
    printf("c = %d, bl = %lu\n\n", c, bl);
    
    //=================================================

    printf("//=================================================\n");

    d = 4;
    printf("Divide %s / %d:\n", str, d);
    
    r = bdiv(b, d, q, nd);

    res = 65536*q[1] + q[0];
    printf("q = %u, %u\n", q[1], q[0]);
    printf("res = %lu, r = %u\n\n", res, r);

    //=================================================

    printf("//=================================================\n");

    bl = 4294563795; // = 0xfff9d7d3 = 0xfff9, 0xd7d3 = 65529, 55251
    d = 46723;
    
    b[1] = 65529;
    b[0] = 55251;

    printf("Divide %lu / %u:\n", bl, d);

    r = bdiv(b, d, q, nd);

    res = 65536*q[1] + q[0];
    printf("q = %u, %u\n", q[1], q[0]);
    printf("res = %lu, r = %u\n\n", res, r);



    
    //=================================================

    printf("//=================================================\n");

    bl = 356901;

    b[1] = 5;
    b[0] = 0x7225;
    
    printf("Convert bl = %lu to string:\n", bl);
    printf("char *bstr = bint2str(uint b, int nd);\n");
    
    char *bstr = bint2str(b, nd);

    printf("bstr = \"%s\"\n", bstr);

    free(bstr);
    
    bstr = bint2str(b, nd);

    printf("bstr = \"%s\"\n", bstr);

    //=================================================

    printf("//=================================================\n");

    bl = 4294901760;

    b[1] = 65535;
    b[0] = 0;
    
    printf("Convert bl = %lu to string:\n", bl);
    printf("char *bstr = bint2str(uint b, int nd);\n");
    
    bstr = bint2str(b, nd);

    printf("bstr = \"%s\"\n", bstr);

    free(bstr);
    
    bstr = bint2str(b, nd);

    printf("bstr = \"%s\"\n", bstr);

    //=================================================

    printf("//=================================================\n");

    char str1[] = "2147489998";
    
    printf("str1 = \"%s\"\n", str1);

    printf("c = str2bint(b, nd, str);\n");
   
    c = str2bint(b, nd, str1);
    
    bl = 65536*b[1] + b[0];
    printf("c = %d, bl = %lu\n\n", c, bl);

    free(bstr);
    
    printf("Convert bl = %lu to string:\n", bl);
    printf("char *bstr = bint2str(b, nd);\n");
    
    bstr = bint2str(b, nd);

    printf("bstr = \"%s\"\n", bstr);

     //=================================================

    printf("//=================================================\n");

    char str2[] = "281474976710655281474976710655";
    
    printf("str2 = \"%s\"\n", str2);

    printf("c = str2bint(b, nd, str2);\n");
   
    c = str2bint(b, nd, str2);
    
    bl = 65536*(65536*b[2] + b[1]) + b[0];
    printf("c = %d, bl = %lu\n\n", c, bl);
    printf("bl[2,1,0] = %u, %u, %u %u, %u, %u, %u\n",
           b[6], b[5], b[4], b[3], b[2], b[1], b[0]);
    printf("bl[2,1,0] = %x, %x, %x, %x, %x, %x %x\n", 
           b[6], b[5], b[4], b[3], b[2], b[1], b[0]);

    free(bstr);
    
    printf("Convert bl = %lu to string:\n", bl);
    printf("char *bstr = bint2str(b, nd);\n");
    
    bstr = bint2str(b, nd);

    printf("bstr = \"%s\"\n", bstr);
    
    free(bstr);
    

    
    
    return 0;

}


uint bdiv(uint a[], uint b, uint q[], int nd) {
    /*
     * Short division of big integer a by one digit b.
     * nd: number of a and b digits.
     * The big integer b[nd] is little endian, i.e. the least significant digit
     * is in b[0].
     *
     * The quotient is saved in q[nd].
     *
     * Returns the remainder.
     */
    
    uint p = 0, r = 0, c = 0; // Carry

    for (int i = 0; i < nd; i++) q[i] = 0;
    
    r = 0;

    for (int i = nd-1; i >= 0; i--) {
        p = 65536 * r + a[i];
        q[i] = p / b;
        r =    p % b;
    }
    
    return r;   
}




uint bdiv_bigendian(uint a[], uint b, uint q[], int nd) {
    /*
     * Short division of big integer b by one digit a.
     * nd: number of a and b digits.
     * The big integer b[nd] is big endian, i.e. the most significant digit
     * is in b[0].
     *
     * Returns the remainder.
     */
    
    uint p, r, c = 0; // Carry

    r = 0;
    
    for (int i = 0; i < nd; i++) {
        p = 65536 * r + a[i];
        q[i] = p / b;
        r =    p % b;
    }
    
    return r;   
}



uint badd(uint c, uint a, uint b[], int nd) {
    /*
     * Short add one digit to a big integer b.
     * c: carry from previous operations.
     * nd: number of b digits.
     * Returns non-zero carry if the result does not fit into nd digits.
     */

    // uint c = 0; // Carry
    uint bc;    // = b + c

    b[0] += a;    
    for (int i = 0; i < nd; i++) {
        bc = b[i] + c;             // ????????????????? CHECK CORRECTNESS!
        b[i] = bc % 65536;
        c =    bc / 65536;
        if (c != 0) printf("badd c = %x\n", c);
        // printf("badd: bc = %6x, b[i] = %6x, c = %6x\n", bc, b[i], c); 
    }
    
    return c;   
}



uint bmul(uint c, uint a, uint b[], int nd) {
    /*
     * Short multiply of big integer b by one digit a.
     * c: carry from previous operations.
     * nd: number of b digits.
     * Returns non-zero carry if the result does not fit into nd digits.
     */
    
    // uint c = 0; // Carry
    uint abc;   // = a*b + c

    for (int i = 0; i < nd; i++) {
        abc = a * b[i] + c;
        b[i] = abc % 65536;
        c =    abc / 65536;
        // if (c != 0) printf("bmul c = %x\n", c);
    }
    
    return c;   
}



int str2bint(uint b[], int nd, char str[]) {
    /*
     * Convert numeric string str into big integer b.
     * nd: number of b digits.
     * Returns non-zero carry if the result does not fit into nd digits.
     */
    ulong bl=0;
    
    uint base = 10, s, c = 0; 
    int ns = strlen(str);

    for (int i = 0; i < nd; i++) b[i] = 0;  // Zeroize b: b = 0
        
    for (int i = 0; i < ns; i++) {
        s = str[i] - '0';
        
        c = bmul(c, base, b, nd);  // Move b up by one decimal digit: b = 10*b
        
        if (c != 0) {
            printf("str2bint, bmul c = %x\n", c);
            return c;
        }
        
        c = badd(c, s, b, nd); // Place str[i] into the least significant digit
        if (c != 0) {
            return c;
        }
    }

    return c;
}





char* bint2str(uint b[], int nd) {
    /*
     * Convert big integer b into numeric string str.
     * nd: number of b digits.
     * Returns NULL if the result does not fit into nd digits.
     */
    ulong bl=0;

    int ns = 128, nchar_max = 0, nchar = 0, i, j;
    uint base = 10, r = 0, s = 0, c = 0;
    uint q[nd];

    char *rts = (char *) malloc(ns*sizeof(char));
    char *str = (char *) malloc(ns*sizeof(char));
      
    nchar = 0;
    nchar_max = ns - 1;  // Maximum number of characters in string
    while (1) {

        r = bdiv(b, base, q, nd);

        if (nchar < nchar_max) {
            rts[nchar] = '0' + r;
        }
        else {
            free(rts);
            free(str);
            return 0;
        }
            
       nchar++;

        if (q[0] == 0) {
            break;
        }

        for (int j = 0; j < nd; j++) b[j] = q[j];  // b = q

    }
    
    rts[nchar] = 0;
    str[nchar] = 0;

    /*
     * Reverse chars in rts to str
     */
    j = nchar - 1;
    i = 0;
    
    while (i < nchar) str[i++] = rts[j--];
    
    free(rts);

    return str;
}
    
