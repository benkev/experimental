#include <stdio.h>

static void transpose(double **m, double **m_tr, int cols, int rows) {
  /*
   * Transpose array m onto array m_transposed
   */
  int i, j;
  for (i = 0; i < cols; i++) {
    for (j = 0; j < rows; j++) {
      m_tr[i][j] = m[j][i];
    }
  }
}

int main(int argc, char *argv[]) {

  double a[2][3] = {{1, 2, 3}, {-3, 5, 1}};
  double *c[2] = {&a[0], &a[1]};
  double b[3][2];
  int i, j;

  //transpose(a, b, 3, 2);

  printf("c = \n");
  for (i = 0; i < 2; i++) {
    for (j = 0; j < 3; j++) {
      printf("%g ", c[i][j]);
    }
    printf("\n");
  }
  printf("\n");

  printf("b = \n");
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 2; j++) {
      printf("%g ", b[i][j]);
    }
    printf("\n\n");
  }
  printf("\n");

}
