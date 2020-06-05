#include <iostream>
#include <stdio.h>
#include <stdlib.h>

//using namespace System;
using namespace std;

void output(int string[], int position);
void generate(int string[], int position, int positions);
int length;
// This function takes "string", which contains a description
// of what bits should be set in the output, and writes the
// corresponding binary representation to the terminal.
// The variable "position" is the effective last position.

void output(int string[], int position)
{
	int * temp_string = new int[length];
	int index = 0;
	int i;
	for (i = 0; i < length; i++)
	{
		if ((index < position) && (string[index] == i))
		{
			temp_string[i] = 1;
			index++;
		}
		else
			temp_string[i] = 0;
	}
	for (i = 0; i < length; i++)
		cout << temp_string[length-i-1];
	delete [] temp_string;
	cout << endl;
}
// Recursively generate the banker's sequence.
void generate(int string[], int position, int positions)
{
	if (position < positions)
	{
		if (position == 0)
		{
			for (int i = 0; i < length; i++)
			{
				string[position] = i;
				generate(string, position + 1, positions);
			}
		}
		else
		{
			for (int i = string[position - 1] + 1; i < length; i++)
			{
				string[position] = i;
				generate(string, position + 1, positions);
			}
		}
	}
	else 
	{
		printf("pos = %i; string = ", positions);
		for (int k = 0; k < length; k++) 
			printf("%i ", string[k]);
		printf("\n");
		output(string, positions);
	}
}


// Recursively generate the banker's sequence.
void gen(int row[], int pos, int nsub) {
  int i;
  if (pos < nsub) {
    if (pos == 0) {
      for(i = 0; i < length; i++) {
	  row[pos] = i;
	  gen(row, pos+1, nsub);
      }
    }
    else { /* pos != 0 */
      for(i = row[pos-1]+1; i < length; i++) {
	row[pos] = i;
	gen(row, pos+1, nsub);
      }
    }
  } /* if (pos < nsub) */
  else {
    //printf("pos = %i; row = ", nsub);
    //for (int k = 0; k < length; k++) 
    //  printf("%i ", row[k]);
    //printf("\n");
    output(row, nsub);
  }
}







// Main program accepts one parameter: the number of elements
// in the set. It loops over the allowed number of ones, from
// zero to n.
int
main (int argc, char ** argv)
{
  if (argc != 2) {
    cout << "Usage: " << argv[0] << " n" << endl;
    exit(1);
  }
  length = atoi(argv[1]);
  //for (int i = 0; i <= length; i++) {
  int i=2;
  int * row = new int[length];
  gen(row, 0, 2);
  delete [] row;
  //}
  return (0);
}
