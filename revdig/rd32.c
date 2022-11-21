// https://leetcode.com/problems/reverse-integer/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <time.h>


int main(int argc, char *argv[]) {
    int x0 = 23767;  //atoi(argv[1]);
    int x, y = 0, q, r, j;
    int y_by_10, y_by_10_div_10, y_minus_r;
    int oflw = 0, sgn;

    if (argc > 0)
        x = x0 = atoi(argv[1]);
    else
        x = x0;

    if (x < 0) {
        sgn = -1;
        x = -x;
    }
    
    
    j = 0;
    while (x && j < 16) {
        q = x / 10;
        r = x % 10;

        y_by_10 = 10*y;
        y_by_10_div_10 = y_by_10/10;
        
        printf("%2d: y=%6d, 10*y  = %6d\n", j, y, 10*y);
        
        if (y_by_10/10 != y) {
            printf("Overflow! (10*y)/10 != y: %d != %d.\n",
                   y_by_10/10, y);
            oflw = 1;
            break;
        }

        y = y_by_10 + r;

        y_minus_r = y - r;

        if (y_minus_r != y_by_10) {
            printf("Overflow! y_minus_r != y_by_10: %d != %d.\n",
                   y_minus_r, y_by_10);
            oflw = 1;
            break;
        }
        
        
        x = q;
        j = j + 1;
    }

    if (oflw) {
        printf("Reversal impossible!\n");
        return -1;
    }

    if (sgn < 0)
        y = -y;
    
    printf("%d ==> %d\n", x0, y); 
    
    return 0;






}

    
