#include <stdio.h>
#include <string.h>
#include <stdlib.h>


/*
 * Keys: any character strings
 * Data: character strings
 */
#define N  200

struct node {
  char *key;          /* 0: empty node */
  char *data;
  struct node *link;  /* 0: data (if any) inside table, no linked list */
};

struct node table[N];

int hash_fun(char *key, int len) {
  unsigned char *pkey = key;
  unsigned int idx = 0;
  unsigned int word;

  pkey = key;
  while (*pkey) {
    word = 0;
    //printf("word = %X \n", word);
    int i;
    for (i = 0; i < 4; i++) {
      if (*pkey == 0) break;
      word = word << 8; /* leave room for next char */
      word = word | (unsigned int)*pkey++;
      //printf("i = %d, *pkey = %X = %c, word = %X\n", i, *pkey, *pkey, word);
    }
    idx += word;
    idx &= 0x7FFFFFFF; /* Make sure idx >= 0 */
  }
  return (int) idx%len;
}

void hash_insert(char *key, char *data) {
  int idx = hash_fun(key, N); /* Address in table */
  char *dat = 0, *kee = 0; /* Copies of data and key */
  struct node *nod, *eol, *pred;

  if (((dat = strdup(data)) == 0) ||
      ((kee = strdup(key)) == 0))
    {printf("hash_insert: memory allocation error\n"); exit(1);}

  if (table[idx].key == 0) {
    table[idx].data = dat; /* Fill in the vacant (idx'th) slot */
    table[idx].key = kee;
  }
      
  else { /* Append the new data to the linked list rooted at idx'th slot */
    if (idx == 58) printf("COLLISION: idx = %d\n", idx);
    if ((nod = (struct node *) malloc(sizeof(struct node))) == 0) 
      {printf("hash_insert:  memory allocation error\n"); exit(1);}
    pred = &(table[idx]);
    eol = table[idx].link;
    while (eol) {
      pred = eol;
      eol = eol->link;    /* Scan the list for the last element  */
    }
    nod->key = kee;
    nod->data = dat;
    nod->link = 0;
    pred->link = nod;      /* Append node instead of NIL */
  }
  return;
}


char *hash_find(char *key) {
  int idx = hash_fun(key, N);    /* Address in table */
  struct node *nod;

  if (idx == 58) 
    printf("COLLISION: idx = %d\n", idx);

  nod = &table[idx];
  if (nod->link == 0) { /* The data is inside the table, not on list */
    return table[idx].data; /* Return pointer to the data */
  }
      
  /* Scan the list for the key string */
  int found = 0;  /* Assume "not found" */
  while (nod) {
    found = (strcmp(nod->key, key) == 0);
    if (found) break;
    nod = nod->link; 
  }   
  if (found) 
    return nod->data;  /* Return pointer to the data */
  else 
    return 0; /* key not found; error */
}



int hash_remove(char *key) {
  int idx = hash_fun(key, N); /* Address in table */
  struct node *nod, *pred;

  nod = &table[idx];
  if ((nod->link == 0) || /* The data is inside the table, not on list */
      (strcmp(nod->key, key) == 0)) {
    table[idx].key = 0;
    table[idx].data = 0; /* Make the node vacant */
    return 0;
  }

  int found = 0;  /* Assume "not found" */
  while (nod) {  /* Scan the list for the key string */
    found = (strcmp(nod->key, key) == 0);
    if (found) break;
    pred = nod;        /* Predecessor atom pointer */
    nod = nod->link;    
  }
  if (found) {
    pred->link = nod->link; /* Bypass the node */
    free(nod->key);
    free(nod->data);
    free(nod);
    return 0;
  }
  else
    return 1; /* Indicates "key not found" */
}


/*
 * Initialize the hash table.
 * data = NULL means vacant slot (or node).
 * link = NULL means NIL (end of list).
 */

void hash_init() {
  int i;
  for (i = 0; i < N; i++) {
    table[i].link = 0;
    table[i].data = 0;
    table[i].key = 0;
  }
}




int main() {
  char key[256], dat[256];
  char *s = 0;
  FILE *fh = NULL;

  /*
   * Initialize the hash table.
   */
  hash_init();

  fh = fopen("key_data.txt", "r");
  while(fscanf(fh, "%s %s", key, dat) != EOF) {
    printf("key = '%s', dat = '%s'\n", key, dat); 
    hash_insert(key, dat);
    //printf("table[58] = '%s', '%s', %p\n", table[58].key, table[58].data, 
    //	   table[58].link);
  }
  fclose(fh);

  s = "Benkevitch,Leonid";
  printf("table['%s'] = '%s'\n", s, hash_find(s));
  if (hash_remove(s) == 0)
    printf("'%s' removed.\n", s);
  else     
    printf("'%s': error in remove.\n", s);
  if (hash_remove(s) == 0)
    printf("'%s' removed.\n", s);
  else     
    printf("'%s': error in remove.\n", s);

  /* s = "Sousa,Don"; */
  /* printf("table['%s'] = '%s'\n", s, hash_find(s)); */
  /* s = "Cassidy,Karen"; */
  /* printf("table['%s'] = '%s'\n", s, hash_find(s)); */
  /* s = "Letters,Liz"; */
  /* printf("table['%s'] = '%s'\n", s, hash_find(s)); */
  /* s = "Rogers,William"; */
  /* printf("table['%s'] = '%s'\n", s, hash_find(s)); */

  return 0;
}
