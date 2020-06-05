#include <stdio.h>
#include <stdlib.h>

typedef struct node {
  char *key;          /* 0: empty node */
  char *data;
  struct node *link;  /* 0: data (if any) inside table, no linked list */
} node_t;

int main() {
  size_t i, N = 100;
  node_t **table;
  table = (node_t **) malloc(N*sizeof(node_t *));

  for (i = 0; i < N; i++) 
    table[i] = NULL;
  return 0;
}
