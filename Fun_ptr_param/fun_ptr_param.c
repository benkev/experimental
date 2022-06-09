#include <stdio.h>

void SquareByAddress(int * n) 
 {  *n = (*n) * (*n);  } 

int main() 
 { 
   int num = 4; 
   printf("Original = %d\n", num); 
   SquareByAddress(&num); 
   printf("New: %d\n", num); 
 } 
 
