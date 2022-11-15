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
int str2bint(uint b[], int nd, char str[]);


int main(int argc, char *argv[]) {

    // char str[] = "65535";
    char str[] = "34143735976";
    //char str[] = "3414373597";
    //char str[] = "1234567";
    int nd = 2;
    uint b[2] = {0, 0};  /* b[0] is the minor digit */
    uint q[2] = {0, 0};
    uint c;  /* Carry. Non-zero if str does not fit into  */
    uint r;
    ulong bl = 0, res = 0;

    uint a = 10, s;
    
    int ns = strlen(str);

    printf("str = \"%s\", ns = %d\n\n", str, ns);

    
    /* // for (int i = ns-1; i > -1; i--) { */
    /* for (int i = 0; i < ns; i++) { */
    /*     s = str[i] - '0'; */
    /*     printf("i = %d, s = %d, str[i] = %c\n", i, s, str[i]); */

    /*     c = bmul(a, b, nd); */
    /*     for (int id=nd-1; id>-1; id--) printf("%5u  ", b[id]); */

    /*     bl = 65536*b[1] + b[0]; */
    /*     printf("c = %d, bl = %lu\n\n", c, bl); */

    /*     printf(" -- mul\n"); */
        
    /*     c = badd(s, b, nd); */
    /*     for (int id=nd-1; id>-1; id--) printf("%5u  ", b[id]); */
    /*     printf(" -- add\n"); */
    /* } */

    c = str2bint(b, nd, str);
    
    bl = 65536*b[1] + b[0];
    printf("c = %d, bl = %lu\n\n", c, bl);

    // return 0;


    // Division:

    // bl = 12345678 = 0xbc614e = 0xbc, 0xb14e = 188, 24910
    b[0] = 188;
    b[1] = 24910;

    r = bdiv(b, 10, q, nd);

    res = 65536*q[0] + q[1];
    printf("q = %lu, %lu\n", q[0], q[1]);
    printf("res = %lu, r = %lu\n\n", res, r);


    // bl = 8791 = 0x2257 = 0, 8791
    b[0] = 0;
    b[1] = 8791;

    r = bdiv(b, 10, q, nd);

    res = 65536*q[0] + q[1];
    printf("q = %lu, %lu\n", q[0], q[1]);
    printf("res = %lu, r = %lu\n\n", res, r);


    // bl = 4042386655 = 0xf0f1ecdf = 0xf0f1, 0xecdf = 61681, 60639
    b[0] = 61681;
    b[1] = 60639;
    
    r = bdiv(b, 10, q, nd);

    res = 65536*q[0] + q[1];
    printf("q = %lu, %lu\n", q[0], q[1]);
    printf("res = %lu, r = %lu\n\n", res, r);

    
    return 0;

}


//
// ?????????????? REVERSE ORDER OF B[]!!!!!!!!!!!!
//

uint bdiv(uint a[], uint b, uint q[], int nd) {
    /*
     * Short division of big integer b by one digit a.
     * nd: number of a and b digits.
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
        bc = b[i] + c;
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
        if (c != 0) printf("bmul c = %x\n", c);
    }
    
    return c;   
}



int str2bint(uint b[], int nd, char str[]) {
    /*
     * Convert numeric string str into big integer b.
     * nd: number of b digits.
     * Returns non-zero carry if the result does not fit into nd digits.
     */

    uint base = 10, s, c = 0; 
    int ns = strlen(str);

    
    for (int i = 0; i < ns; i++) {
        s = str[i] - '0';
        c = bmul(c, base, b, nd);
        if (c != 0) {
            printf("str2bint, bmul c = %x\n", c);
            return c;
        }
        c = badd(c, s, b, nd);
        if (c != 0) {
            printf("str2bint, badd c = %x\n", c);
            return c;
        }
    }
    
}







        
    /* // for (int i = ns-1; i > -1; i--) { */
    /* for (int i = 0; i < ns; i++) { */
    /*     s = str[i] - '0'; */
    /* printf("i = %d, s = %d, str[i] = %c, res = %lu\n", i, s, str[i], res); */

    /*     c = bmul(a, b, nd); */
    /*     for (int id=nd-1; id>-1; id--) printf("%5u  ", b[id]); */
    /*     printf(" -- mul\n"); */
        
    /*     c = badd(s, b, nd); */
    /*     for (int id=nd-1; id>-1; id--) printf("%5u  ", b[id]); */
    /*     printf(" -- add\n"); */
    /* } */
    
