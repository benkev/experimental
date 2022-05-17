/*
 * Illustrate contiguity of a 2D array
 *
 * Compile:
 * $ gcc -g pointers_to_arrays.c -o pointers_to_arrays
 */

#include <stdio.h>
#include <stdlib.h>


int main() {

    int i, a[5][4];

    for (i=0; i<5; i++) {
        printf("a[%1d] = %x\n", i, a[i]);
    }

    printf("\n");
    
    for (i=1; i<5; i++) {
        printf("a[%1d] - a[%1d] = %x\n", i, i-1, a[i] - a[i-1]);
    }

}


