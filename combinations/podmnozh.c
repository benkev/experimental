/*
 * podmnozh.c
 *
 * For a given set M with w elements print out all its 2^w subsets.
 *
 * Compile & build:
 *
 * $ gcc -g podmnozh.c -lm -o podmnozh
 *
 */



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 
int main()
{
    /* int M[5] = {0, 2, 5, 6, 9}; // Set */
    /* int w = 5;                  // Number of elements of the set M */
 
    int M[4] = {2, 5, 6, 9}; // Set
    int w = 4;               // Number of elements of the set M
 
    int i, j, n;
 
    n = pow(2, w);
    for ( i = 0; i < n; i++ ) // Enumerate all the bit patterns in w bits
    {
        printf("{");
        for ( j = 0; j < w; j++ )   // Enumerate the bits in the pattern
            if ( i & (1 << j) )     // If the j-th bit is set
               printf("%d ", M[j]); // then output j-th lement of the set
        printf("}\n");
    }
    return 0;
} 
