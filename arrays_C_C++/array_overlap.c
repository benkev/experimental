#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 

int main() {

  int M=20, N=20;
  int i, j;

  typedef double mat_t[M][N];

  /* Make arrays aa[400] and matr[M,N] overlay */
  double aa[M*N], x;
  mat_t *A = (mat_t *) aa; /* A[any][M][N] */

  FILE *fh = fopen("image.txt", "r");


  /*
   * Read image into A
   */
  printf("\n\n");
  printf("Original:\n\n");
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      fscanf(fh, "%lf", &x);
      A[0][i][j] = x;
      printf("%3.0lf ", A[0][i][j]);
    }  /* for (j = 0; j < N; j++) */
    printf("\n");
  }  /* for (i = 0; i < M; i++)  */
  fclose(fh);


  printf("\n\n");
  printf("One-dimensional: \n\n");
  for (i = 0; i < M*N; i++) printf("%3.0lf ", aa[i]);
  printf("\n\n");

  return 0;
}

