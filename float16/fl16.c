#include <stdio.h>
#include <math.h>

int main() {
  short float a=1., b=2., c;
  /* half e=101.5, f=99.5, g; */
  __float16 e=101.5, f=99.5, g;
  c = a + b;
  g = e + f;
  printf("short float: %g + %g = %g",a,b,c);
  printf("half:        %g + %g = %g",e,f,g);
}

/* $ gcc -g fl16.c -o fl16 */
/* fl16.c: In function ‘main’: */

/* fl16.c:5:9: error: both ‘short’ and ‘float’ in declaration specifiers */
/*    short float a=1., b=2., c; */
/*          ^ */
/* fl16.c:7:3: error: unknown type name ‘__float16’ */
/*    __float16 e=101.5, f=99.5, g; */
/*    ^ */







