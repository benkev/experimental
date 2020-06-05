#include <stdio.h>

struct str1 {
  float a[10];
  int ia[10];  
};

struct str2 {
  int ib[10];
  float b[10];  
};

int subr(void *a);

int main() {

  struct str1 x;
  int y, i;

  for (i = 0; i < 10; i++) {
    x.ia[i] = i;
    x.a[i] = i*i;
  }
  y = subr(&x);
  printf("y = %d\n", y);
  for (i = 0; i < 10; i++) printf("%d %g\n", x.ia[i], x.a[i]);

}


int subr(void *a) {
  struct str2 *p;
  p = a;
  p->b[5] = 1234;
  p->ib[5] = 1234;
  return 12345;
  
}
