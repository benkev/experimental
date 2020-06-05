#include <stdio.h>
#include <string.h>
#include <stdlib.h>


/*
 * Keys: any character strings
 * Data: character strings
 */
#define N  200
#define WORDLEN sizeof(size_t)


typedef struct node {
  char *key;          /* 0: empty node */
  char *data;
  struct node *link;  /* 0: data (if any) inside table, no linked list */
} node_t;


/*
 * Hash table.
 * It is an array of pointers to "nodes", containing 
 * a data for a key. The index into the table is obtained 
 * from the hash function that maps any character string key
 * on the interval from 0 to N-1, where N is the table length.
 * In case that different keys generate the same index 
 * into the hash table (collision condition) the nodes make
 * linked list of nodes with the head at the table entry.
 */
node_t *table[N];

/*
 * Function prototypes
 */
size_t hash_fun(const char *key, const size_t len);
void hash_insert(const char *key, const char *data);
char *hash_find(const char *key);
int hash_remove(const char *key);
void hash_init();
node_t *rm_list(node_t *head);
void hash_clean();


/*
 * Hash function maps a character string key 
 * on the natural number subset [0:len],
 * where len is the hash table length.
 * Method: every WORDLEN bytes of the key string
 * are treated as a WORDLEN-byte size_t word.
 * The most significant bytes of key are padded
 * with zeroes on the left. These words are
 * added up in the idx integer variable.
 * The index is obtained as the reminder of
 * division of idx by the length len of the hash
 * table.
 * 
 */
size_t hash_fun(const char *key, const size_t len) {
  unsigned char *pkey = (unsigned char *) key;
  size_t idx = 0;
  size_t word;

  
  while (*pkey) {
    word = 0;
    size_t  i;
    for (i = 0; i < WORDLEN; i++) {
      if (*pkey == 0) break;
      word = word << 8; /* leave room for next 8-bit char */
      word = word | (size_t)*pkey++;
    }
    idx += word;
  }
  return (size_t) idx%len;
}

/*
 * Insert a data addressed by key.
 * 
 */
void hash_insert(const char *key, const char *data) {
  size_t idx = hash_fun(key, N); /* Address in table */
  char *dat = 0, *kee = 0; /* Copies of data and key */
  node_t *nod, *eol, *pred;

  eol = table[idx];

  /* 
   * Scan the list for the last element
   */
  int key_exists = 0; /* Assume key is not in table */ 
  while (eol) {
    pred = eol;
    if (strcmp(key, eol->key) == 0) {
      key_exists = 1;
      break;
    } 
    eol = eol->link;
  }
  if (key_exists) {
    /* Replace data in current node */
    free(pred->data);
    if ((dat = strdup(data)) == 0) 
      {printf("hash_insert: memory allocation error\n"); exit(1);}
    pred->data = dat;
  }
  else { /* key is not in table */
    /* 
     * Create new node and key and data strings 
     * Save pointer to the new node in nod.
     */
    if (((dat = strdup(data)) == 0) ||
	((kee = strdup(key)) == 0) ||
	((nod = (node_t *) malloc(sizeof(node_t))) == 0)) 
      {printf("hash_insert: memory allocation error\n"); exit(1);}
    nod->data = dat;
    nod->key = kee;
    nod->link = NULL;
    if (table[idx] == 0) { /* Empty slot in table */
      table[idx] = nod;
    }
    else { /* Slot NOT empty */
      pred->link = nod; /* Append the new node */
    }
  }
  return;
}


/*
 * Find the data entry addressed by the key.
 * If the data is found, returns the pointer to data,
 * otherwise returns NULL.
 */
char *hash_find(const char *key) {
  size_t idx = hash_fun(key, N);    /* Address in table */
  node_t *nod;

  /* Scan the list for the key string */
  nod = table[idx];
  int found = 0;  /* Assume "not found" */
  while (nod) {
    found = (strcmp(nod->key, key) == 0);
    if (found) break;
    nod = nod->link;
  }
  if (found)
    return nod->data;  /* Return pointer to the data */
  else
    return NULL; /* key not found; error */
}


/*
 * Delete the data entry addressed by key from the hash table.
 * If the data is found, returns 0,
 * otherwise returns 1.
 */
int hash_remove(const char *key) {
  size_t idx = hash_fun(key, N); /* Address in table */
  node_t *nod, *pred;

  nod = table[idx];

  if (nod == 0) 
    return 1; /* Indicates "key not found" */

  /*
   * Check the first atom
   */
  if (strcmp(nod->key, key) == 0) {
    table[idx] = nod->link; /* Point at the next atom (or put 0 into table */
    free(nod->key);
    free(nod->data);
    free(nod);
    return 0;  /* Indicates "key found" */
  }

  /*
   * Search the list to end checking the nodes for key
   */
  int found = 0;  /* Assume "key not found in table" */
  while (nod) {  /* Scan the list for the key string */
    found = (strcmp(nod->key, key) == 0);
    if (found) break;
    pred = nod;        /* Predecessor atom pointer */
    nod = nod->link;
  }

  /* If 'found' is true, 'nod' cannot be NULL! */
  if (found) {  /* Bypass the node, linking pred and nod */
    pred->link = nod->link;
    free(nod->key);
    free(nod->data);
    free(nod);
    return 0; /* Indicates "key found" */
  }
  else { /* Key not found */
    return 1; /* Indicates "key not found" */
  }
}


/*
 * Initialize the hash table.
 */

void hash_init() {
  int i;
  for (i = 0; i < N; i++) {
    table[i] = NULL;
  }
}

/*
 * Delete entire list from head
 * Returns pointer to next or NULL if the last atom
 */
node_t *rm_list(node_t *head) {
  if (head) {
    free(head->key);
    free(head->data);

    rm_list(head->link); /* Recursive diving ... */

    free(head);
    return 0;
  }
  else
    return 0;
}


/* 
 * Find all the stored keys and data and delete them
 */
void hash_clean() {
  size_t i;
  for (i = 0; i < N; i++) {
    if (table[i] == 0) continue;
    rm_list(table[i]);
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

  /*
   * Read test set of pairs (key, data)
   */
  fh = fopen("key_data.txt", "r");
  while(fscanf(fh, "%s %s", key, dat) != EOF) {
    printf("key = '%s', \tdat = '%s', \tidx = %ld\n", 
	   key, dat, hash_fun(key,N)); 
    //printf("key = '%s', dat = '%s'\n", key, dat); 
    hash_insert(key, dat);
  }
  fclose(fh);

  /*
   * Try functions hash_find(s) and hash_remove(s)
   */
  s = "Benkevitch,Leonid";
  printf("table['%s'] = '%s'\n", s, hash_find(s));
  if (hash_remove(s) == 0)
    printf("'%s' removed.\n", s);
  else
    printf("'%s': not in table.\n", s);
  if (hash_remove(s) == 0)
    printf("'%s' removed.\n", s);
  else
    printf("'%s': not in table.\n", s);
  if (hash_find(s) == 0)
    printf("table['%s'] not found'\n", s);


  s = "Letters,Liz";
  printf("table['%s'] = '%s'\n", s, hash_find(s));
  if (hash_remove(s) == 0)
    printf("'%s' removed.\n", s);
  else
    printf("'%s': not in table.\n", s);
  if (hash_remove(s) == 0)
    printf("'%s' removed.\n", s);
  else
    printf("'%s': not in table.\n", s);
  if (hash_find(s) == 0)
    printf("table['%s'] not found'\n", s);

  s = "Sousa,Don";
  printf("table['%s'] = '%s', idx =  %ld\n", s, hash_find(s), hash_fun(s,N));
  s = "Cassidy,Karen";
  printf("table['%s'] = '%s', idx =  %ld\n", s, hash_find(s), hash_fun(s,N));
  s = "Letters,Liz";
  printf("table['%s'] = '%s', idx =  %ld\n", s, hash_find(s), hash_fun(s,N));
  s = "Rogers,William";
  printf("table['%s'] = '%s', idx =  %ld\n", s, hash_find(s), hash_fun(s,N));
  s = "Knight";
  if (hash_find(s) == 0)
    printf("table['%s'] not found'\n", s);

  /* Repeated insertion with the same key */
  s = "Benkevitch,Leonid";
  hash_insert(s, "abcdef");
  printf("table['%s'] = '%s', idx =  %ld\n", s, hash_find(s), hash_fun(s,N));
  hash_insert(s, "ghijklmnop");
  printf("table['%s'] = '%s', idx =  %ld\n", s, hash_find(s), hash_fun(s,N));
  hash_insert(s, "603-577-1909");
  printf("table['%s'] = '%s', idx =  %ld\n", s, hash_find(s), hash_fun(s,N));
  
  hash_clean(); /* Erase entire table */

  /* After cleaning no information saved */
  s = "Sousa,Don";
  printf("table['%s'] = '%s', idx =  %ld\n", s, hash_find(s), hash_fun(s,N));
  s = "Cassidy,Karen";
  printf("table['%s'] = '%s', idx =  %ld\n", s, hash_find(s), hash_fun(s,N));

  printf("WORDLEN = %ld\n",  WORDLEN);
  return 0;
}
