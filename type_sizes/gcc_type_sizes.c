/*
 * Compile:
 *
 * gcc -g gcc_type_sizes.c -o ts
 *
 */

#include <stdio.h>
 
int main() {
  printf("sizeof(int) = \t\t%ld bytes = %ld bits\n", sizeof(int), 
	 8*sizeof(int)); 
  printf("sizeof(long) = \t\t%ld bytes = %ld bits\n", sizeof(long), 
	 8*sizeof(long)); 
  printf("sizeof(long long) = \t%ld bytes = %ld bits\n", sizeof(long long), 
	 8*sizeof(long long)); 
  printf("sizeof(float) = \t%ld bytes = %ld bits\n", sizeof(float), 
	 8*sizeof(float)); 
  printf("sizeof(double) = \t%ld bytes = %ld bits\n", sizeof(double), 
	 8*sizeof(double)); 
  printf("sizeof(long double) = \t%ld bytes = %ld bits\n", sizeof(long double), 
	 8*sizeof(long double)); 
  printf("\n");
  printf("sizeof(size_t) = \t%ld bytes = %ld bits\n", sizeof(size_t), 
	 8*sizeof(size_t)); 
  printf("sizeof(off_t) = \t%ld bytes = %ld bits\n", sizeof(off_t), 
	 8*sizeof(long)); 

  return 0;
}





