#include <stdio.h>

class Solution {
public:
    int reverse(int x_in) {
        
        int x = x_in;
        
        printf("Given: x = %d\n", x);
        
        if (x == 0)
            return 0;
        
        int neg = 0;     // Assume x >= 0
        if (x < 0) {
            x = -x; 
            neg = 1;      // Mark x negative
        }
            
        int d = 0, y = 0;
        while (x > 0) {
            d = x % 10; // Move x's least significant decimal digit to d
            
            if (y >= 214748364) {
                printf("Overflow: %d*10 >= 2147483647 (2^31 - 1).\n\n", y);
                return 0;
            }
                
            y = 10*y + d;  
            x /= 10;    // Shift x right by one decimal digit
            printf("d = %d, y = %d, x = %d\n", d, y, x);
        }
        if (neg)
            y = -y;     // x is negative, so is y
        
        printf("Result: %d is reverse of %d\n\n", y, x_in);
        
        return y;
    }
};
