This directory contains two demonstration programs implementing the hash
addressing:

hashc (source hash.c) is written in C.
hashcpp (source hash.cpp) is written in C++.

The hash table in both is restricted to only character string type of both key
and stored data. The hash table contains pointers to nodes. In case of a
collision, the new node is created and put on the tail of a linked list of
nodes. The node contains pointers to the key string and to the data string. The
hash function is documented in the comments.

  



    	   hash.c:
	  ========

Provides the functions to work with a hash table. The table is declared as
external. Its length is na integer literal. The following functions are provided
to work with the hash table:

size_t   hash_fun(const char *key, const size_t len);
void     hash_insert(const char *key, const char *data);
char    *hash_find(const char *key);
int      hash_remove(const char *key);
void     hash_init();
node_t  *rm_list(node_t *head);
void     hash_clean();

They are documented in the comments.





     	 hash.cpp:
	==========

Provides a class Hash. It is initialized by the number of table entries. For
example,

Hash dict(1000);

The table is created dynamically and the user has no direct access to it. The
only class member user can use is the length of the table, len:

cout << x.len << endl;

The Hash class provides several methods to work with the table:

size_t   hash_fun(const char *key, const size_t len);
void     insert(const char *key, const char *data);
char    *find(const char *key);
int      remove(const char *key);


The brackets operator, [], is overloaded to provide access to data by key. For
example, the operator

char *data = x[key];

is equivalent to

char *data = x.find(key);

The methods are documented in the comments.
