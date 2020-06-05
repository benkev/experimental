/*
 * powerset.c
 *
 * For a given set with w elements print out all its 2^w subsets.
 *
 * Compile & build:
 *
 * $ gcc -g powerset.c -o powerset
 *
 * Run examples:
 *
 * $ ./powerset 5 9 
 * {}
 * {5}
 * {9}
 * {5, 9}
 *
 * $ ./powerset a p z
 * {}
 * {z}
 * {a}
 * {z, a}
 * {p}
 * {z, p}
 * {a, p}
 * {z, a, p}
 *
 * $ ./powerset яблоко груша слива велосипед
 * {} 
 * {яблоко} 
 * {груша} 
 * {яблоко, груша} 
 * {слива} 
 * {яблоко, слива} 
 * {груша, слива} 
 * {яблоко, груша, слива} 
 * {велосипед} 
 * {яблоко, велосипед} 
 * {груша, велосипед} 
 * {яблоко, груша, велосипед} 
 * {слива, велосипед} 
 * {яблоко, слива, велосипед} 
 * {груша, слива, велосипед} 
 * {яблоко, груша, слива, велосипед} 
 * 
 */

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
 
static void powerset(int argc, char** argv)
{
  unsigned int i, j, bits, i_max = 1U << argc; // = 2^(number of set elements)
 
  if (argc >= sizeof(i) * CHAR_BIT) {
    fprintf(stderr, "Error: set too large\n");
    exit(1);
  }
 
  for (i = 0; i < i_max ; ++i) {
    printf("{");
    for (bits = i, j = 0; bits; bits >>= 1, ++j) {
      if (bits & 1)
	printf(bits > 1 ? "%s, " : "%s", argv[j]);
    }
    printf("}\n");
  }
}
 
int main(int argc, char* argv[])
{
  powerset(argc-1, argv+1);
  return 0;
}
