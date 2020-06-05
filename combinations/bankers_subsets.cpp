/*
 * bankers_subsets.cpp
 *
 *
 * J. Loughry, J.I. van Hemert, and L. Schoofs, \
 *      “Efficiently Enumerating the Subsets of a Set.”
 *
 * Compile & build:
 *
 * $ g++ -g bankers_subsets.cpp -o bankers_subsets
 * or
 * $ gcc -g bankers_subsets.cpp -lstdc++ -o bankers_subsets
 *
 *
 *
 */


#include <iostream>
#include <stdio.h>
#include <stdlib.h>

//using namespace System;
using namespace std;

void output(int set[], int m);
void gen(int set[], int p, int m);
void gen1(int set[], int n, int m, int p);
void gen2(int set[], int n, int m, int p, int c[100][3], int ic);

int n;

// This function takes "set", which contains a description
// of what bits should be set in the output, and writes the
// corresponding binary representation to the terminal.
// The variable "m" is the effective last position.

void output(int set[], int m)
{
  int * temp_set = new int[n];
  int index = 0;
  int i;

  for (i = 0; i < n; i++) {
    if ((index < m) && (set[index] == i)) {
      temp_set[i] = 1;
      index++;
    }
    else
      temp_set[i] = 0;
  }
  for (i = 0; i < n; i++)
    //cout << temp_set[n-i-1];
    cout << temp_set[i];
  cout << endl;

  delete [] temp_set;
}

//
//Recursively generate the banker's sequence.
//
void gen(int set[], int p, int m) {
  int i;
  if (p < m) {

    if (p == 0) {
      for(i = 0; i < n; i++) {
	set[0] = i;
	gen(set, 1, m);
      }
    }
    else { /* Here p > 0: */
      for(i = set[p-1]+1; i < n; i++) {
	set[p] = i;
	gen(set, p+1, m);
      }
    }

  } /* if (p < m) */
  else { /*  Here p == m: */
    printf("p = %i; set = ", m);
    for (int k = 0; k < n; k++) 
      printf("%i ", set[k]);
    printf("\n");
    output(set, m);
  }
}


//
// Recursively generate the banker's sequence.
// 
void gen1(int set[], int n, int m, int p) {
  int i;
  if (p < m) {

    if (p == 0) {
      for(i = 0; i < n; i++) {
	set[0] = i;
	printf("1 i = %d; p = 0; set = ", i);
	for (int k = 0; k < n; k++) printf("%i ", set[k]); printf("\n");
	gen1(set, n, m, 1);
	printf("2 i = %d; p = 0; set = ", i);
	for (int k = 0; k < n; k++) printf("%i ", set[k]); printf("\n");
      }
    }
    else { /* Here p > 0: */
      for(i = set[p-1]+1; i < n; i++) {
	set[p] = i;
	printf("3 i = %d; p = %d; set = ", i, p);
	for (int k = 0; k < n; k++) printf("%i ", set[k]); printf("\n");
	gen1(set, n, m, p+1);
	printf("4 i = %d; p = %d; set = ", i, p);
	for (int k = 0; k < n; k++) printf("%i ", set[k]); printf("\n");
      }
    }

  } /* if (p < m) */
  else { /*  Here p == m: */
    // printf("f = %d; set = ", p);
    // printf("f = %d; set = ", p);
    // for (int k = 0; k < n; k++) printf("%i ", set[k]); printf("\n");
    output(set, m);
  }
}


//
// Recursively generate the banker's sequence.
// 
void gen2(int set[], int n, int m, int p, int c[100][3], int *ic) {
  int i;
  if (p < m) {

    if (p == 0) {
      for(i = 0; i < n; i++) {
	set[0] = i;
	printf("1 i = %d; p = 0; set = ", i);
	for (int k = 0; k < n; k++) printf("%i ", set[k]); printf("\n");
	gen2(set, n, m, 1, c, ic);
	printf("2 i = %d; p = 0; set = ", i);
	for (int k = 0; k < n; k++) printf("%i ", set[k]); printf("\n");
      }
    }
    else { /* Here p > 0: */
      for(i = set[p-1]+1; i < n; i++) {
	set[p] = i;
	printf("3 i = %d; p = %d; set = ", i, p);
	for (int k = 0; k < n; k++) printf("%i ", set[k]); printf("\n");
	gen2(set, n, m, p+1, c, ic);
	printf("4 i = %d; p = %d; set = ", i, p);
	for (int k = 0; k < n; k++) printf("%i ", set[k]); printf("\n");
      }
    }

  } /* if (p < m) */
  else { /*  Here p == m: */
    // printf("f = %d; set = ", p);
    // printf("f = %d; set = ", p);
    // for (int k = 0; k < n; k++) printf("%i ", set[k]); printf("\n");
    //output(set, m);
    for (i = 0; i < p; i++) c[*ic][i] = set[i];
    ic++;
  }
}







// Main program accepts one parameter: the number of elements
// in the set n. It loops over the allowed number of ones, from
// zero to n.
int main (int argc, char ** argv) {

  int c[100][3], ic, i, j;
  if (argc != 2) {
    cout << "Usage: " << argv[0] << " n" << endl;
    exit(1);
  }

  n = atoi(argv[1]);

  //for (int i = 0; i <= n; i++) {
  int m = 3; /* Number of a subset elements */
  int *set = new int[n];

  //for (int i = 0; i > n; i++) set[i] = 9999;
  //gen1(set, n, m, 0);
  //gen(set, 0, m);
  ic = 0;
  gen2(set, n, m, 0, c, &ic);

  printf("ic = %d\n", &ic);
  for (i = 0; i < ic; i++) {
    for (j = 0; j < 3; j++) 
      printf("%d", c[i][j]);
    printf("\n");
  }
  delete [] set;
  //}
  return (0);
}
