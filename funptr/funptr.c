#include <stdio.h>

void aaa() {
  printf("aaa\n");
}

void bbb() {
  printf("bbb\n");
}

int main(int argc, char *argv[]) {
  void (*fun)(void);

  if (argc == 1) return 1;

  int i = atoi(argv[1]);
  if (i == 1) 
    fun = aaa;
  else if (i == 2)
    fun = bbb;
  else
    return 2;

  (*fun)();

  return 0;
}






