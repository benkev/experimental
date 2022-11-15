#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <time.h>


int main(int argc, char *argv[]) {
    short x = 23767;  //atoi(argv[1]);
    short y = 0, q, r, j;
    short const sten = 10;

    j = 12;
    while (x && j) {
        q = x / 10;
        r = x % 10;

        //        if (y != 0 && ((sten*y)/y != sten)) {
        if (sten*y < 0) {
            printf("Overflow! 10*y  = %6hd > 32767.\n", sten*y);
            break;
        }
            
        y = sten*y + r;
        printf("x=%6hd; q=%6hd; r=%6hd; y=%6hd, 10*y = %6hd, (10*y)/y=%6hd\n", \
               x, q, r, y, sten*y, (sten*y)/y);
        x = q;
        j = j - 1;
    }

    printf("x=%6hd; q=%6hd; r=%6hd; y=%6hd\n", x, q, r, y);

    return 0;






}

    
