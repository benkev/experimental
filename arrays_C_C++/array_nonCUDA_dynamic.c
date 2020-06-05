/*
 * Compile:
 * 
 * $ gcc -g array_nonCUDA_dynamic.c -o ancd
 *
 * warning: initialization from incompatible pointer type [enabled by default]
 *   int (*cube)[m][n] = arr;
 *                       ^
 */


#include <stdio.h>
#include <stdlib.h>


void array_dynamic(int l, int m, int n, int *arr) {

    int (*cube)[m][n] = arr;
    int i, j, k, iseq=0;
    
    for (i=0; i < l; ++i)
        for (j=0; j < m; ++j)
            for (k=0; k < n; ++k)
                cube[i][j][k] = iseq++;

    printf("l=%d, m=%d, n=%d\n\n", l, m, n);

    for (i=0; i<l; i++) {
        for (j=0; j<m; j++) {
            for (k=0; k<n; k++) {
                printf("%d  ", cube[i][j][k]);
            }
            printf("\n");
        }
        printf("=================\n");
    }

    printf("\n\n");

    

}

int main() {
    int i, lmn = 3*5*4;
    int *ar = (int *) malloc(lmn*sizeof(int));

    for (i=0; i<lmn; i++) ar[i] = i;
    for (i=0; i<lmn; i++) printf("%d  ", ar[i]);
    printf("\n\n");
    
    array_dynamic(3, 5, 4, ar);
    
  return 0;
}
