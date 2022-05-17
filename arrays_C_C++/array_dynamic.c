/*
 * array_dynamic.c
 *
 * How to dynamically allocate a variable dimension array
 * Array A[m,n] is allocated and 1-dimensional array B overlaps A. 
 *   Two-dimensional array C[n,m] overlaps A[n,m]. 
 *   Now A may be treated as 1D (ie B) and 2D. 
 *
 * Array P[m,n,k] is allocated. Array Q[n][m] overlaps 3D array P[m][n][k].
 *   One-D array R overlaps P[m,n,k], and 2D array S overlaps P[m,n,k].
 */

#include <stdio.h>
#include <stdlib.h>


int main() {
  int i, j, l, m = 5, n = 7, k = 3;
  int (*A)[n] = malloc(sizeof(int[m][n]));  /* 2D array A[m][n] */
  int mn = m*n;
  int *B = (int *)A;         /* 1D array overlapping A[m][n]  */
  int (*C)[m];        /* 2D array C[n][m] overlapping 2D array A[m][n] */

  //int (*P)[n][k] = malloc(sizeof(int[m][n][k]));
  int (*P)[n][k] = malloc(m*n*k*sizeof(int)); // ?????????? n, k, where is m ?
  int mnk = m*n*k;
  int nk = n*k;
  int (*Q)[n][k];        /* 3D array Q[n][m] overlapping 3D array P[m][n][k] */
  int *R = (int *)P;         /* 1D array overlapping P[m][n][k]  */
  
  int (*S)[nk] = (int (*)[(size_t)(nk)])P;

  int ***T;
  
  size_t ifrm, idt, ich, ifrmdat, iqua;
  long (*qua)[16][4]; /* 4 quantiles of the experiment data for 16 channels */
  long *pqua;
  int nfrm = 3;

  printf("Overlapping 1 and 2-dim arrays:\n\n");

  C = A;
  T = (int ***)Q;
  
  for (i=0; i<mn; i++) {
    B[i] = i + 1;
  }


  printf("\n");

  for (j=0; j<mn; j++) 
      printf("%d  ", B[j]);
  printf("\n\n");
  
  printf("sizeof(int[m][n]) = %d\n", sizeof(int[m][n]));

  /* return 0; */
  
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      printf("%d  ", A[i][j]);
    }
    printf("\n");
  }

  printf("\n\n");

  for (i=0; i<n; i++) {
    for (j=0; j<m; j++) {
      printf("%d  ", C[i][j]);
    }
    printf("\n");
  }

  printf("\n");
  printf("Overlapping 1, 2, and 3-dim arrays:\n\n");

  Q = P;
  R = (int *)P;

  for (i=0; i<mnk; i++) {
    R[i] = i + 1;
  }

  for (j=0; j<mnk; j++) 
      printf("%d  ", R[j]);
  printf("\n\n");

  printf("m=%d, n=%d, k=%d, mnk=%d\n\n", m, n, k, mnk);

  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      for (l=0; l<k; l++) {
	printf("%d  ", P[i][j][l]);
      }
      printf("\n");
    }
    printf("=================\n");
  }

  printf("\n\n");

  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      for (l=0; l<k; l++) {
	printf("%d  ", Q[i][j][l]);
      }
      printf("\n");
    }
    printf("=================\n");
  }

  printf("\n\n");

  printf("Interpret 3D P[m][n][k] as a 2D array S[m][n*k] \n\n");

  for (i=0; i<m; i++) {
    for (j=0; j<n*k; j++) 
      printf("%d  ", S[i][j]);
    printf("\n");
    printf("=================\n");
  }

  printf("\n\n");

  
  printf("A 3D array qua[nfrm][16][4] coincides with 1D array pqua.\n");

  qua = malloc(sizeof(long[nfrm][16][4]));
  pqua = (long *)qua;
  nfrm = 3;
  
  i = 1;
  for (ifrm=0; ifrm<nfrm; ifrm++)
      for (ich=0; ich<16; ich++)
          for (iqua=0; iqua<4; iqua++) {
              qua[ifrm][ich][iqua] = i;
              i += 1;
          }

  printf("Print 3D array qua[nfrm][16][4]:\n");
 
  for (ifrm=0; ifrm<nfrm; ifrm++) {
      for (ich=0; ich<16; ich++) {
          for (iqua=0; iqua<4; iqua++) {
              printf("%3ld ", qua[ifrm][ich][iqua]);
          }
          printf("\n");
      }
      printf("=================\n");
  }

  printf("Print 1D array pqua[nfrm*16*4]:\n");
 
  for (i=0; i<192; i++)
      printf("%3ld ", pqua[i]);
  printf("\n=================\n");


  
}

