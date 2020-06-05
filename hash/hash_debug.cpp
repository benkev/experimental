#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdio>

using std::cout;
using std::endl;
using std::cin;
using std::ifstream;

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
    if (table == NULL) {
      cout << "hash_insert: memory allocation error" << endl; 
      //exit(1);
    }
    for (size_t i = 0; i < N; i++) table[i] = NULL;
  }

  ~Hash() {
    /* 
     * Find all the stored keys and data and delete them
     */
    for (size_t i = 0; i < len; i++)
      if (table[i]) 
	printf("destr: [%ld], @(%p) ->key = '%s', ->data = '%s'\n", 
	       i, table[i], table[i]->key, table[i]->data);

      for (size_t i = 0; i < len; i++) {
      if (table[i] == 0) continue;
      rm_list(table[i]);
    }
    delete[] table;
  }

  void insert(char *key, char *data);
  char *find(char *key);
  int remove(char *key);
  char *operator[] (char *key);

private:
  size_t len;
  node_t **table;
  int hash_fun(char *key, int len);
  /*
   * Delete entire list from head
   * Returns pointer to next or NULL if the last atom
   */
  node_t *rm_list(node_t *head) {
    cout << "Entered rm_list;\t";
    printf("head = %p, ", head);
    if (head) printf("head->key = '%s'\n", head->key);
    else cout << endl;
    if (head) {
      delete[] head->key;
      delete[] head->data;
      rm_list(head->link);
      delete head;
      return 0;
    }
    else
      return 0;
  }
};


//---------------------------------------------------------------------------


/*
 * Hash function maps a character string key 
 * on the natural number subset [0:len],
 * where len is the hash table length.
 * Method: every 4 bytes of the key string
 * are treated as a 4-byte integer word, padded
 * with zeroes on the left. These words are
 * added up in the idx integer variable.
 * The index is obtained as the reminder of
 * dividing idx by the length len of the hash
 * table.
 * 
 */
int Hash::hash_fun(char *key, int len) {
  unsigned char *pkey = (unsigned char *)key;
  unsigned int idx = 0;
  unsigned int word;

  while (*pkey) {
    word = 0;
    int i;
    for (i = 0; i < 4; i++) {
      if (*pkey == 0) break;
      word = word << 8; /* leave room for next char */
      word = word | (unsigned int)*pkey++;
    }
    idx += word;
    idx &= 0x7FFFFFFF; /* Make sure idx >= 0 */
  }
  return (int) idx%len;
}


/*
 * Insert a data addressed by key.
 * 
 */
void Hash::insert(char *key, char *data) {
  int idx = hash_fun(key, len); /* Address in table */
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
    delete[] pred->data;
    dat = new char[strlen(data)+1];
    if (dat == 0) 
      cout << "Hash::insert: memory allocation error" << endl;
    pred->data = dat;
  }
  else { /* key is not in table */
    /* 
     * Create new node and key and data strings 
     * Save pointer to the new node in nod.
     */
    dat = new char[strlen(data)+1];
    kee = new char[strlen(key)+1];
    nod = new node_t;
    if (dat == 0 || kee == 0 || nod == 0) {
      cout << "Hash::insert: memory allocation error" << endl;
      //exit(1);
    }
    strcpy(dat, data);
    strcpy(kee, key);
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
char *Hash::find(char *key) {
  int idx = hash_fun(key, len);    /* Address in table */
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
 * Access the data by key as hash["key"]
 */
char *Hash::operator[] (char *key) {
  return this->find(key);
}



/*
 * Delete the data entry addressed by key from the hash table.
 * If the data is found, returns 0,
 * otherwise returns 1.
 */
int Hash::remove(char *key) {
  int idx = hash_fun(key, len); /* Address in table */
  node_t *nod, *pred;

  nod = table[idx];

  if (nod == 0) 
    return 1; /* Indicates "key not found" */

  /*
   * Check the first atom
   */
  if (strcmp(nod->key, key) == 0) {
    table[idx] = nod->link; /* Point at the next atom (or put 0 into table */
    delete[] nod->key;
    delete[] nod->data;
    delete nod;
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
    delete[] nod->key;
    delete[] nod->data;
    delete nod;
    return 0; /* Indicates "key found" */
  }
  else { /* Key not found */
    return 1; /* Indicates "key not found" */
  }
}



int main() {
  //Hash h(100);
  Hash *h = new Hash(100);
  char key[256], dat[256];
  char *s = 0;
  //FILE *fh = NULL;
  ifstream fh;
  bool eof;

  /*
   * Read test set of pairs (key, data)
   */
  fh.open("key_data.txt");
  do {
    //fscanf(fh, "%s %s", key, dat) != EOF;
    fh >> key >> dat;
    eof = fh.eof();
    cout << "key = '" << key << "', \tdat = '" << dat << "'" << endl; 
    h->insert(key, dat);
  }  while(!eof);
  fh.close();

  /*
   * Try functions hash_find(s) and hash_remove(s)
   */
  s = "Benkevitch,Leonid";
  cout << "table['" << s <<"'] = " << h->find(s) << endl;

  if (h->remove(s) == 0)
    cout << "'" << s << "' removed.\n";
  else
    cout << "'" << s << "': error in remove.\n";

  if (h->remove(s) == 0)
    cout << "'" << s << "' removed.\n";
  else
    cout << "'" << s << "': not in hash table.\n";

  if (h->find(s) == 0)
    cout << "table['" << s << "'] not found.\n";

  s = "Letters,Liz";
  cout << "table['" << s <<"'] = " << h->find(s) << endl;

  // if (h->remove(s) == 0)
  //   cout << "'" << s << "' removed.\n";
  // else
  //   cout << "'" << s << "': error in remove.\n";

  // if (h->remove(s) == 0)
  //   cout << "'" << s << "' removed.\n";
  // else
  //   cout << "'" << s << "': not in hash table.\n";

  if (h->find(s) == 0)
    cout << "table['" << s << "'] not found.\n";

  s = "Sousa,Don";
  cout << "table['" << s << "'] = '" << h->find(s) << "'\n";
  s = "Cassidy,Karen";
  cout << "table['" << s << "'] = '" << h->find(s) << "'\n";
  s = "Letters,Liz";
  if (h->find(s))
    cout << "table['" << s << "'] = '" << h->find(s) << "'\n";
  s = "Rogers,William";
  if (h->find(s))
    cout << "table['" << s << "'] = '" << h->find(s) << "'\n";
  else 
    cout << "'" << s << "': not in hash table.\n";
  s = "Knight";
  if (h->find(s))
    cout << "table['" << s << "'] = '" << h->find(s) << "'\n";
  else 
    cout << "'" << s << "': not in hash table.\n";
  s = "Ivanov,Sergey";
  if (h->find(s))
    cout << "table['" << s << "'] = '" << h->find(s) << "'\n";
  else 
    cout << "'" << s << "': not in hash table.\n";

  s = (*h)["Coster,Anthea"];
  cout << "table['Coster,Anthea'] = '" << s << "'\n";

  delete h;

  return 0;
}

