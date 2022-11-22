#include <stdio.h>
#include <stdlib.h>
#include <limits.h>


uint bmul(uint c, uint a, uint b[], int nd);
uint badd(uint c, uint a, uint b[], int nd);
int str2bint(uint b[], int nd, char str[]);

int reverse(int x);

int main(int argc, char *argv[]) {
    
    long x0 = atol(argv[1]);
    int x = (int) x0;

    printf("x0 = %ld, x = %d\n", x0, x);
    
    if (x != x0) {
        printf("Too large number to fit 32-bit int: %ld\n", x0);
        printf("INT_MIN = %d, INT_MAX = %d\n\n", INT_MIN, INT_MAX);
        return -1;
    }
    else
        x = (int) x0;
    
    
    int y = reverse(x);

    printf("%d ==>> %d\n", x, y);

    return 0;
}



int reverse(int x) {
    
    int y = 0, r = 0, sgn = 0;
    uint ybig[2] = {0, 0};  /* ybig[0] is the lowest-order digit */
    uint carry = 0;

    if (x < 0) {
        sgn = 1;
        x = -x;
    }
    
    while(x != 0){

        r = x%10;  /* Extract the lowest-order digit of x */
        
        // y = y*10 + r;
        
        carry = bmul(carry, 10, ybig, 2);  /* ybig_i+1 = 10*ybig_i */

        if (carry)                 /* 65536*ybig[1] + ybig[0] > 2^32 */
            return 0;
        if (ybig[1] & 0xffff8000)  /* 65536*ybig[1] + ybig[0] > INT_MAX */
            return 0;


        carry = badd(carry, r, ybig, 2);   /* ybig_i+1 = 10*ybig_i + r_i */

        if (carry)
            return 0;              /* 65536*ybig[1] + ybig[0] > 2^32 */
        if (ybig[1] & 0xffff8000)  /* 65536*ybig[1] + ybig[0] > INT_MAX */
            return 0;


        x /= 10;
    }

    y = 65536*ybig[1] + ybig[0];   /* y = ybig */

    if (sgn)
        y = -y;
        
    return y;
}



uint badd(uint c, uint a, uint b[], int nd) {
    /*
     * "Short add" one digit a to a big integer b, 0 <= a <= 65536.
     * c: carry from previous operations.
     * nd: number of b digits.
     * Returns non-zero carry if the result does not fit into nd digits.
     */

    uint bc;    // bc = b + c

    b[0] += a;    
    for (int i = 0; i < nd; i++) {
        bc = b[i] + c;
        b[i] = bc % 65536;
        c =    bc / 65536;
        if (c != 0) printf("badd c = %x\n", c);
    }
    
    return c;   
}



uint bmul(uint c, uint a, uint b[], int nd) {
    /*
     * "Short multiply" of big integer b by one digit a, 0 <= a <= 65536.
     * c: carry from previous operation.
     * nd: number of b digits.
     * Returns non-zero carry if the result does not fit into nd digits.
     */
    
    uint abc;   // abc = a*b + c

    for (int i = 0; i < nd; i++) {
        abc = a * b[i] + c;
        b[i] = abc % 65536;
        c =    abc / 65536;
    }
    
    return c;   
}

