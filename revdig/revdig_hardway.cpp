// https://leetcode.com/problems/reverse-integer/
//
// $ g++ revdig_hardway.cpp -o revdig_hardway_cpp
//          or
// $ g++ -std=c++11 revdig_hardway.cpp -o revdig_hardway_cpp
//


#include <climits>
#include <iostream>
#include <climits>

using namespace std;

uint bmul(uint c, uint a, uint b[], int nd);
uint badd(uint c, uint a, uint b[], int nd);
int str2bint(uint b[], int nd, char str[]);


class Solution {
public:
    int reverse(int x) {
        
        //
        // This digit-reversal algorithm is based on two pieces of 
        // arbitrary-long integer arithmetic to programmatically catch 
        // the integer overflow: badd() and madd(), addition and 
        // multiplication.
        // 
        // An arbitrary long integer ("big integer") is comprised of digits
        // with radix or base 0x10000 = 65536, which means each digit
        // can have values 0 through 65535 = 0xffff. The big integer is
        // represented as an array of 32-bit ints, in each only the lower 16
        // bits containing a digit. This format allows catching the integer
        // overflow and correct carrying to the upper digits. If an
        // operation overflows the whole big integer, the extra carry bits are 
        // returned.
        //
        // The big integers are "little-endian".
        //
        // Here "int ybig[2]" has only 2 16-bit digits to represent the 
        // standard 32-bit int type.
        //
    
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
};



int main(int argc, char *argv[]) {
    
    long x0 = stol(argv[1]);
    int x = (int) x0;

    std::cout << "x0 = " << x0 << ", x = " << x << "\n";
    
    if (x != x0) {
        printf("Too large number to fit 32-bit int: %ld\n", x0);
        printf("INT_MIN = %d, INT_MAX = %d\n\n", INT_MIN, INT_MAX);
        return -1;
    }
    else
        x = (int) x0;
    
    Solution revdig;
    
    int y = revdig.reverse(x);

    std::cout << x << " ==>> " << y << ".\n";
    
    return 0;
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




