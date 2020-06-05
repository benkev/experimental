#include <iostream>
#include <cstring>

using std::cout;
using std::endl;

class Hash {
public:
  typedef struct node {
    char *key;          /* 0: empty node */
    char *data;
    struct node *link;  /* 0: data (if any) inside table, no linked list */
  } node_t;

  Hash(size_t N = 256) {
    len = N;
    table = new node_t* [N];
    if (ptable == NULL) {
      cout << "hash_insert: memory allocation error" << endl; 
      //exit(1);
    }
    for (size_t i = 0; i < N; i++) 
      table[i] = 0;
  }
  void insert(char *key, char *data);
  char *find(char *key);
  int remove(char *key);

private:
  size_t len;
  node_t **table;
  int hash_fun(char *key, int len);
};

int main() {
  Hash h;
  return 0;
}
