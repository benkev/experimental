
// https://leetcode.com/problems/reverse-integer/

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
//#include <string.h>


int reverse(int x) {
    
    int limit1=INT_MAX;
    int limit2=INT_MIN;
    int y=0;

    while(x!=0){
        
        if ((limit1/10 < y) || (limit1/10 == y && x%10 > limit1%10))
            return 0;
        if ((limit2/10 > y) || (limit2/10 == y && x%10 < limit2%10))
            return 0;
        
        y = y*10 + x%10;
        x /= 10;
    }
    return y;
        
}

int main(int argc, char *argv[]) {

    long x0 = atol(argv[1]);
    int x = x0;

    if (x != x0) {
        printf("Too large number to fit 32-bit int: %L\n", x0);
        return -1;
    }
    else
        x = (int) x0;
    
    
    int y = reverse(x);

    // printf("INT_MIN = %d, INT_MAX = %d\n", INT_MIN, INT_MAX);
    printf("%d ==>> %d\n", x, y);

    return 0;
}

